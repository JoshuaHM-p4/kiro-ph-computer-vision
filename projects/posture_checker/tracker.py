"""Pure posture analysis logic — no camera, no display, no I/O.

Classifies posture into GOOD_POSTURE, SLOUCHING, HEAD_TILTED, or NO_PERSON
using MediaPipe Pose landmarks. Works from **both frontal and side-view**
camera angles by computing view-independent metrics.

Key landmarks used (MediaPipe Pose indices):
    0  - Nose
    7  - Left ear
    8  - Right ear
    11 - Left shoulder
    12 - Right shoulder
    23 - Left hip
    24 - Right hip

Metrics computed:
    - Neck angle: angle of (ear-midpoint → nose) from vertical — detects
      forward head posture from both front and side views.
    - Shoulder alignment: vertical tilt between left and right shoulders.
    - Forward head offset: how far nose/ear is ahead of the shoulder line,
      normalized by torso length. This is the PRIMARY side-view metric.
    - Spine angle: angle of (hip-midpoint → shoulder-midpoint) from vertical —
      detects overall trunk slouch from side view.

View detection:
    When shoulder width is small relative to torso length, the user is likely
    in side/profile view. The tracker adapts its scale reference and weighting
    accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class PostureState(Enum):
    """Detected posture states."""
    GOOD_POSTURE = "Good Posture"
    SLOUCHING = "Slouching"
    HEAD_TILTED = "Head Tilted"
    NO_PERSON = "No Person"


@dataclass
class Config:
    """All tunable thresholds, in one place.

    Metrics are normalized by torso length (shoulder-midpoint to hip-midpoint)
    which stays roughly constant regardless of whether the camera sees the
    front or side of the body.
    """

    # --- Neck angle (degrees from vertical) ---
    # Applies in both views: nose below/ahead of ears = large angle
    neck_angle_slouch: float = 30.0
    neck_angle_good: float = 22.0       # hysteresis release

    # --- Forward head offset (normalized by torso length) ---
    # In side view: horizontal distance from ear to shoulder
    # In front view: approximated from depth (z) or nose-drop
    forward_head_slouch: float = 0.30
    forward_head_good: float = 0.20

    # --- Spine angle (degrees from vertical, hip→shoulder) ---
    # Large angle = trunk leaning forward (side view) or collapsing (front)
    spine_angle_slouch: float = 20.0
    spine_angle_good: float = 14.0

    # --- Shoulder tilt (|Δy| / torso_length) ---
    shoulder_tilt_threshold: float = 0.10
    shoulder_tilt_good: float = 0.06

    # --- Slouch alert ---
    slouch_alert_seconds: float = 3.0

    # --- Calibration ---
    calibration_margin_degrees: float = 8.0   # added to angle thresholds
    calibration_margin_ratio: float = 0.10    # added to ratio thresholds

    # --- View detection ---
    # If shoulder_width / torso_length < this, treat as side view
    side_view_ratio: float = 0.45


@dataclass
class Baseline:
    """Calibrated baseline posture measurements."""
    neck_angle: float = 0.0
    forward_head: float = 0.0
    spine_angle: float = 0.0
    shoulder_tilt: float = 0.0
    is_calibrated: bool = False


@dataclass
class Metrics:
    """Computed posture metrics for one frame."""
    neck_angle: float       # degrees from vertical
    forward_head: float     # normalized offset
    spine_angle: float      # degrees from vertical
    shoulder_tilt: float    # normalized |Δy|
    is_side_view: bool      # whether we detected a profile angle


class PostureTracker:
    """Tracks posture over time with view-adaptive classification.

    Works from both frontal and side/profile camera angles by detecting
    which view is active and weighting metrics accordingly.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.baseline = Baseline()

        # State
        self.current_state: PostureState = PostureState.NO_PERSON
        self.posture_score: float = 100.0
        self.slouch_start: Optional[float] = None
        self.slouch_alert_active: bool = False
        self.is_side_view: bool = False

        # Score smoothing
        self._score_history: List[float] = []
        self._max_history: int = 30

    def update(
        self,
        landmarks: Optional[List[Tuple[float, float, float]]],
        timestamp: float,
    ) -> dict:
        """Process landmarks and return posture state.

        Args:
            landmarks: 33 pose landmarks (x, y, z) normalized 0..1, or None.
            timestamp: Monotonic seconds.

        Returns:
            State dict with posture, score, metrics, and alert info.
        """
        if landmarks is None or len(landmarks) < 25:
            self.current_state = PostureState.NO_PERSON
            self.slouch_start = None
            self.slouch_alert_active = False
            return self._make_state(None, 0.0, timestamp)

        metrics = self._compute_metrics(landmarks)
        if metrics is None:
            self.current_state = PostureState.NO_PERSON
            self.slouch_start = None
            self.slouch_alert_active = False
            return self._make_state(None, 0.0, timestamp)

        self.is_side_view = metrics.is_side_view

        # Classify
        new_state = self._classify(metrics)
        self.current_state = new_state

        # Score
        raw_score = self._compute_score(metrics)
        self._score_history.append(raw_score)
        if len(self._score_history) > self._max_history:
            self._score_history.pop(0)
        self.posture_score = sum(self._score_history) / len(self._score_history)

        # Slouch timer
        slouch_seconds = self._update_slouch_timer(new_state, timestamp)

        return self._make_state(metrics, slouch_seconds, timestamp)

    def calibrate_baseline(
        self, landmarks: Optional[List[Tuple[float, float, float]]]
    ) -> bool:
        """Save current posture as 'good' baseline."""
        if landmarks is None or len(landmarks) < 25:
            return False

        metrics = self._compute_metrics(landmarks)
        if metrics is None:
            return False

        self.baseline = Baseline(
            neck_angle=metrics.neck_angle,
            forward_head=metrics.forward_head,
            spine_angle=metrics.spine_angle,
            shoulder_tilt=metrics.shoulder_tilt,
            is_calibrated=True,
        )

        self._score_history = [100.0]
        self.posture_score = 100.0
        self.slouch_start = None
        self.slouch_alert_active = False
        return True

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self, landmarks: List[Tuple[float, float, float]]
    ) -> Optional[Metrics]:
        """Extract view-adaptive posture metrics."""
        nose = landmarks[0]
        left_ear = landmarks[7]
        right_ear = landmarks[8]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # Midpoints
        ear_mid = _midpoint(left_ear, right_ear)
        shoulder_mid = _midpoint(left_shoulder, right_shoulder)
        hip_mid = _midpoint(left_hip, right_hip)

        # Scale references
        shoulder_width = _dist_2d(left_shoulder, right_shoulder)
        torso_length = _dist_2d(shoulder_mid, hip_mid)

        # Guard against degenerate poses
        if torso_length < 1e-5:
            return None

        # Detect side view
        is_side_view = (shoulder_width / torso_length) < self.config.side_view_ratio

        # --- Neck angle ---
        # Vector from ear-midpoint to nose, angle from vertical (0,-1)
        neck_angle = _angle_from_vertical(ear_mid, nose)

        # --- Forward head offset ---
        # How far the ear/nose is horizontally ahead of the shoulder line
        # In side view: this is the X-axis offset
        # In front view: use vertical drop (nose-y relative to ear-y) as proxy
        if is_side_view:
            # Horizontal offset of ear midpoint from shoulder midpoint
            forward_head = abs(ear_mid[0] - shoulder_mid[0]) / torso_length
        else:
            # Vertical proxy: how much nose drops below ear level
            nose_drop = max(0.0, nose[1] - ear_mid[1])
            forward_head = nose_drop / torso_length

        # --- Spine angle ---
        # Angle of (hip_mid → shoulder_mid) from vertical
        spine_angle = _angle_from_vertical(hip_mid, shoulder_mid)

        # --- Shoulder tilt ---
        shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) / torso_length

        return Metrics(
            neck_angle=neck_angle,
            forward_head=forward_head,
            spine_angle=spine_angle,
            shoulder_tilt=shoulder_tilt,
            is_side_view=is_side_view,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, metrics: Metrics) -> PostureState:
        """Classify posture with hysteresis and calibration offsets."""
        cfg = self.config
        is_bad = self.current_state in (PostureState.SLOUCHING, PostureState.HEAD_TILTED)

        # Effective thresholds (with calibration offset)
        if self.baseline.is_calibrated:
            neck_slouch = self.baseline.neck_angle + cfg.calibration_margin_degrees
            neck_good = self.baseline.neck_angle + cfg.calibration_margin_degrees * 0.6
            fwd_slouch = self.baseline.forward_head + cfg.calibration_margin_ratio
            fwd_good = self.baseline.forward_head + cfg.calibration_margin_ratio * 0.6
            spine_slouch = self.baseline.spine_angle + cfg.calibration_margin_degrees
            spine_good = self.baseline.spine_angle + cfg.calibration_margin_degrees * 0.6
        else:
            neck_slouch = cfg.neck_angle_slouch
            neck_good = cfg.neck_angle_good
            fwd_slouch = cfg.forward_head_slouch
            fwd_good = cfg.forward_head_good
            spine_slouch = cfg.spine_angle_slouch
            spine_good = cfg.spine_angle_good

        tilt_thresh = cfg.shoulder_tilt_good if is_bad else cfg.shoulder_tilt_threshold

        # 1. Shoulder tilt (head tilted) — mainly relevant in frontal view
        if not metrics.is_side_view and metrics.shoulder_tilt > tilt_thresh:
            return PostureState.HEAD_TILTED

        # 2. Slouching — combine neck angle, forward head, and spine angle
        neck_thresh = neck_good if is_bad else neck_slouch
        fwd_thresh = fwd_good if is_bad else fwd_slouch
        spine_thresh = spine_good if is_bad else spine_slouch

        if metrics.is_side_view:
            # Side view: forward head and spine angle are the primary signals
            if metrics.forward_head > fwd_thresh or metrics.spine_angle > spine_thresh:
                return PostureState.SLOUCHING
        else:
            # Front view: neck angle and forward head proxy
            if metrics.neck_angle > neck_thresh or metrics.forward_head > fwd_thresh:
                return PostureState.SLOUCHING

        # Also check spine in frontal (catches general trunk collapse)
        if metrics.spine_angle > spine_thresh:
            return PostureState.SLOUCHING

        return PostureState.GOOD_POSTURE

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_score(self, metrics: Metrics) -> float:
        """Compute 0–100 posture score."""
        cfg = self.config
        ref_neck = self.baseline.neck_angle if self.baseline.is_calibrated else 0.0
        ref_fwd = self.baseline.forward_head if self.baseline.is_calibrated else 0.0
        ref_spine = self.baseline.spine_angle if self.baseline.is_calibrated else 0.0

        neck_dev = max(0.0, metrics.neck_angle - ref_neck)
        fwd_dev = max(0.0, metrics.forward_head - ref_fwd)
        spine_dev = max(0.0, metrics.spine_angle - ref_spine)
        tilt_dev = metrics.shoulder_tilt

        neck_score = max(0.0, 100.0 - (neck_dev / cfg.neck_angle_slouch) * 50.0)
        fwd_score = max(0.0, 100.0 - (fwd_dev / cfg.forward_head_slouch) * 50.0)
        spine_score = max(0.0, 100.0 - (spine_dev / cfg.spine_angle_slouch) * 50.0)
        tilt_score = max(0.0, 100.0 - (tilt_dev / cfg.shoulder_tilt_threshold) * 40.0)

        if metrics.is_side_view:
            # In side view, weight forward head and spine more
            score = fwd_score * 0.35 + spine_score * 0.35 + neck_score * 0.20 + tilt_score * 0.10
        else:
            score = neck_score * 0.30 + fwd_score * 0.25 + spine_score * 0.25 + tilt_score * 0.20

        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _update_slouch_timer(self, state: PostureState, timestamp: float) -> float:
        """Update slouch timer. Returns seconds of continuous slouching."""
        if state in (PostureState.SLOUCHING, PostureState.HEAD_TILTED):
            if self.slouch_start is None:
                self.slouch_start = timestamp
            slouch_seconds = timestamp - self.slouch_start
            if slouch_seconds >= self.config.slouch_alert_seconds:
                self.slouch_alert_active = True
            return slouch_seconds
        else:
            self.slouch_start = None
            self.slouch_alert_active = False
            return 0.0

    # ------------------------------------------------------------------
    # State output
    # ------------------------------------------------------------------

    def _make_state(
        self,
        metrics: Optional[Metrics],
        slouch_seconds: float,
        timestamp: float,
    ) -> dict:
        """Build the state dict."""
        if metrics is None:
            return {
                "posture": self.current_state.value,
                "score": round(self.posture_score, 1),
                "neck_angle": 0.0,
                "forward_head": 0.0,
                "spine_angle": 0.0,
                "shoulder_tilt": 0.0,
                "slouch_seconds": 0.0,
                "alert": self.slouch_alert_active,
                "calibrated": self.baseline.is_calibrated,
                "side_view": self.is_side_view,
            }
        return {
            "posture": self.current_state.value,
            "score": round(self.posture_score, 1),
            "neck_angle": round(metrics.neck_angle, 1),
            "forward_head": round(metrics.forward_head, 3),
            "spine_angle": round(metrics.spine_angle, 1),
            "shoulder_tilt": round(metrics.shoulder_tilt, 3),
            "slouch_seconds": round(slouch_seconds, 1),
            "alert": self.slouch_alert_active,
            "calibrated": self.baseline.is_calibrated,
            "side_view": metrics.is_side_view,
        }


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _dist_2d(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance in 2D (ignores z)."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _midpoint(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, float, float]:
    """Midpoint of two 3D points."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def _angle_from_vertical(base: Tuple[float, ...], tip: Tuple[float, ...]) -> float:
    """Angle (degrees) of the vector base→tip measured from vertical (0,-1).

    0° = tip directly above base. 90° = tip at same height, fully horizontal.
    """
    dx = tip[0] - base[0]
    dy = tip[1] - base[1]  # Y increases downward in image coords
    vec_len = math.sqrt(dx * dx + dy * dy)
    if vec_len < 1e-6:
        return 0.0
    # Vertical in image = (0, -1). cos(θ) = dot(v, up) / |v| = -dy / vec_len
    cos_angle = -dy / vec_len
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))
