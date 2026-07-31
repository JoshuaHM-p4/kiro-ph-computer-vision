"""Pure posture analysis logic. No camera, no display, no Flask.

Takes normalized landmarks (0..1) and a timestamp, returns posture state.
All thresholds are scale-relative (fractions of torso/shoulder measurements).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import PostureConfig


class ViewType(Enum):
    """Detected camera view relative to the person."""

    FRONT = "front"
    FRONT_LEFT = "front_left"    # Camera is angled from the left
    FRONT_RIGHT = "front_right"  # Camera is angled from the right
    LEFT_SIDE = "left_side"      # Camera sees right side (left ear hidden)
    RIGHT_SIDE = "right_side"    # Camera sees left side (right ear hidden)
    UPPER = "upper"              # Camera is above looking down
    UNKNOWN = "unknown"


@dataclass
class PostureState:
    """Output state from a single frame analysis."""

    score: float = 100.0  # 0-100, higher is better
    view: ViewType = ViewType.UNKNOWN
    paused: bool = False  # True when no person is detected
    head_forward: bool = False
    shoulders_uneven: bool = False
    torso_leaning: bool = False
    # Detail metrics: how far off and which direction
    head_forward_degrees: float = 0.0  # degrees off from ideal (180)
    shoulder_tilt_degrees: float = 0.0  # degrees of tilt
    shoulder_higher_side: str = ""  # "left" or "right"
    torso_lean_degrees: float = 0.0  # degrees of forward lean
    good_streak_seconds: float = 0.0
    bad_streak_seconds: float = 0.0
    is_slouching: bool = False
    slouch_event_triggered: bool = False  # True on the frame a new slouch clip starts
    messages: list[str] = field(default_factory=list)


@dataclass
class _Landmark:
    """A single pose landmark with normalized coordinates."""

    x: float
    y: float
    z: float
    visibility: float


def _angle_between_points(a: tuple[float, float], b: tuple[float, float],
                          c: tuple[float, float]) -> float:
    """Compute angle at point b (in degrees) formed by points a-b-c."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return 180.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _angle_from_vertical(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Angle (degrees) of line a->b from vertical (0 = perfectly upright)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    # Vertical is (0, -1) in image coords (y increases downward)
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle


def _slope_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Angle (degrees) of line from a to b relative to horizontal."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


class PostureChecker:
    """Stateful posture analysis engine.

    Call `update(landmarks, timestamp)` each frame. Returns a PostureState.
    Landmarks should be a list/dict of 33 MediaPipe Pose landmarks,
    each with x, y, z, visibility in normalized [0..1] coordinates.
    """

    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self, config: Optional[PostureConfig] = None) -> None:
        self.config = config or PostureConfig()
        self._head_forward_active = False
        self._shoulders_uneven_active = False
        self._torso_lean_active = False
        self._good_streak_start: Optional[float] = None
        self._bad_streak_start: Optional[float] = None
        self._last_good_duration: float = 0.0
        self._last_bad_duration: float = 0.0
        self._last_slouch_event: float = -self.config.slouch_cooldown_seconds
        self._is_bad_posture = False

    def update(self, landmarks: list[dict], timestamp: float) -> PostureState:
        """Analyze one frame of pose landmarks.

        Args:
            landmarks: List of 33 dicts with keys x, y, z, visibility (all 0..1).
            timestamp: Monotonic time in seconds.

        Returns:
            PostureState with current analysis results.
        """
        state = PostureState()

        if not landmarks or len(landmarks) < 25:
            state.messages.append("No person detected — paused")
            state.paused = True
            # Freeze streaks: shift start times forward by elapsed pause time
            # so the streak doesn't grow while nobody is on screen
            if self._good_streak_start is not None:
                self._good_streak_start = timestamp - self._last_good_duration
            if self._bad_streak_start is not None:
                self._bad_streak_start = timestamp - self._last_bad_duration
            return state

        lm = [_Landmark(**l) for l in landmarks]

        # Detect the camera view
        state.view = self._detect_view(lm)

        # Compute posture metrics based on view
        head_forward_angle = self._measure_head_forward(lm, state.view)
        shoulder_tilt, shoulder_higher = self._measure_shoulder_tilt(lm, state.view)
        torso_lean = self._measure_torso_lean(lm, state.view)

        # Populate detail metrics
        state.head_forward_degrees = round(max(0.0, 180.0 - head_forward_angle), 1)
        state.shoulder_tilt_degrees = round(shoulder_tilt, 1)
        state.shoulder_higher_side = shoulder_higher
        state.torso_lean_degrees = round(torso_lean, 1)

        # Apply hysteresis for each metric
        self._head_forward_active = self._hysteresis(
            self._head_forward_active,
            head_forward_angle,
            self.config.head_forward_angle_bad,
            self.config.head_forward_angle_good,
            invert=True,  # bad when angle is BELOW threshold
        )

        self._shoulders_uneven_active = self._hysteresis(
            self._shoulders_uneven_active,
            shoulder_tilt,
            self.config.shoulder_tilt_bad,
            self.config.shoulder_tilt_good,
            invert=False,  # bad when tilt is ABOVE threshold
        )

        self._torso_lean_active = self._hysteresis(
            self._torso_lean_active,
            torso_lean,
            self.config.torso_lean_bad,
            self.config.torso_lean_good,
            invert=False,  # bad when lean is ABOVE threshold
        )

        state.head_forward = self._head_forward_active
        state.shoulders_uneven = self._shoulders_uneven_active
        state.torso_leaning = self._torso_lean_active

        # Build messages
        if state.head_forward:
            state.messages.append("Head too far forward")
        if state.shoulders_uneven:
            state.messages.append("Shoulders uneven")
        if state.torso_leaning:
            state.messages.append("Torso leaning forward")

        # Compute score (0-100)
        state.score = self._compute_score(
            head_forward_angle, shoulder_tilt, torso_lean
        )

        # Determine overall posture quality
        currently_bad = state.head_forward or state.shoulders_uneven or state.torso_leaning
        state.is_slouching = currently_bad

        # Initialize streak on first frame
        if self._good_streak_start is None and self._bad_streak_start is None:
            if currently_bad:
                self._bad_streak_start = timestamp
            else:
                self._good_streak_start = timestamp
            self._is_bad_posture = currently_bad

        # Update streaks
        if currently_bad:
            if not self._is_bad_posture:
                # Transition good -> bad
                self._bad_streak_start = timestamp
                self._good_streak_start = None
            self._is_bad_posture = True
            # Use explicit None check (0.0 is a valid start time)
            start = self._bad_streak_start if self._bad_streak_start is not None else timestamp
            state.bad_streak_seconds = timestamp - start
            state.good_streak_seconds = 0.0
            self._last_bad_duration = state.bad_streak_seconds
            self._last_good_duration = 0.0
        else:
            if self._is_bad_posture:
                # Transition bad -> good
                self._good_streak_start = timestamp
                self._bad_streak_start = None
            self._is_bad_posture = False
            start = self._good_streak_start if self._good_streak_start is not None else timestamp
            state.good_streak_seconds = timestamp - start
            state.bad_streak_seconds = 0.0
            self._last_good_duration = state.good_streak_seconds
            self._last_bad_duration = 0.0

        # Check for slouch event trigger
        state.slouch_event_triggered = False
        if (
            currently_bad
            and state.bad_streak_seconds >= self.config.slouch_trigger_seconds
            and (timestamp - self._last_slouch_event) >= self.config.slouch_cooldown_seconds
        ):
            state.slouch_event_triggered = True
            self._last_slouch_event = timestamp

        if not state.messages:
            state.messages.append("Good posture!")

        return state

    def _detect_view(self, lm: list[_Landmark]) -> ViewType:
        """Determine camera angle from landmark visibility, positions, and ratios."""
        left_ear = lm[self.LEFT_EAR]
        right_ear = lm[self.RIGHT_EAR]
        left_shoulder = lm[self.LEFT_SHOULDER]
        right_shoulder = lm[self.RIGHT_SHOULDER]
        nose = lm[self.NOSE]
        left_hip = lm[self.LEFT_HIP]
        right_hip = lm[self.RIGHT_HIP]

        threshold = self.config.side_view_ear_visibility_threshold

        # Pure side view: one ear completely hidden
        if left_ear.visibility < threshold and right_ear.visibility >= threshold:
            return ViewType.LEFT_SIDE
        if right_ear.visibility < threshold and left_ear.visibility >= threshold:
            return ViewType.RIGHT_SIDE

        # Check shoulder width (very narrow = side view)
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        if shoulder_width < 0.05:
            if left_ear.visibility > right_ear.visibility:
                return ViewType.RIGHT_SIDE
            return ViewType.LEFT_SIDE

        # Upper view: camera above looking down
        # Detected when shoulders are much closer to top of frame AND
        # nose is significantly above shoulders (compressed vertical distance)
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        mid_hip_y = (left_hip.y + right_hip.y) / 2
        torso_height = mid_hip_y - mid_shoulder_y
        if torso_height < 0.08 and mid_shoulder_y < 0.4:
            return ViewType.UPPER

        # Angled front views: detect by shoulder width asymmetry and nose offset
        # In an angled view, one shoulder appears wider (closer to camera)
        # and the nose shifts toward the closer side
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2

        # Distance from each shoulder to center
        left_dist = abs(left_shoulder.x - mid_shoulder_x)
        right_dist = abs(right_shoulder.x - mid_shoulder_x)

        # Asymmetry ratio: 1.0 = symmetric, >1 = one side appears wider
        if min(left_dist, right_dist) > 0.01:
            asymmetry = max(left_dist, right_dist) / min(left_dist, right_dist)
        else:
            asymmetry = 1.0

        # Nose offset from shoulder midpoint (indicates which side camera favors)
        nose_offset = nose.x - mid_shoulder_x

        # Also check ear visibility difference for partial angles
        ear_vis_diff = left_ear.visibility - right_ear.visibility

        # Front-angled detection: significant asymmetry OR partial ear visibility diff
        if asymmetry > 1.4 or abs(ear_vis_diff) > 0.2:
            # Determine which side the camera is angled from
            # If left shoulder appears narrower (closer to center) -> camera from left
            # Also factor in ear visibility and nose offset
            left_score = 0.0
            right_score = 0.0

            # Shoulder asymmetry: the side with less distance to center is farther
            if left_dist < right_dist:
                left_score += 1.0  # Left shoulder compressed -> camera from left
            else:
                right_score += 1.0

            # Ear visibility: less visible ear is farther from camera
            if ear_vis_diff < -0.15:
                left_score += 0.8  # Left ear less visible -> camera from left
            elif ear_vis_diff > 0.15:
                right_score += 0.8

            # Nose offset: nose shifts toward camera side
            if nose_offset < -0.02:
                left_score += 0.5
            elif nose_offset > 0.02:
                right_score += 0.5

            if left_score > right_score:
                return ViewType.FRONT_LEFT
            elif right_score > left_score:
                return ViewType.FRONT_RIGHT

        return ViewType.FRONT

    def _measure_head_forward(self, lm: list[_Landmark], view: ViewType) -> float:
        """Measure the ear-shoulder-hip angle (side view) or ear-shoulder vertical (front).

        Returns angle in degrees. 180 = perfectly aligned, lower = head forward.
        """
        if view in (ViewType.LEFT_SIDE, ViewType.RIGHT_SIDE):
            # LEFT_SIDE means left ear is hidden (camera sees right side) -> use right landmarks
            # RIGHT_SIDE means right ear is hidden (camera sees left side) -> use left landmarks
            if view == ViewType.LEFT_SIDE:
                ear = lm[self.RIGHT_EAR]
                shoulder = lm[self.RIGHT_SHOULDER]
                hip = lm[self.RIGHT_HIP]
            else:
                ear = lm[self.LEFT_EAR]
                shoulder = lm[self.LEFT_SHOULDER]
                hip = lm[self.LEFT_HIP]

            if ear.visibility < self.config.min_landmark_visibility:
                return 180.0  # Can't measure, assume good

            return _angle_between_points(
                (ear.x, ear.y),
                (shoulder.x, shoulder.y),
                (hip.x, hip.y),
            )
        elif view in (ViewType.FRONT_LEFT, ViewType.FRONT_RIGHT):
            # Angled front view: use the ear on the camera's side (more visible/reliable)
            # and that side's shoulder+hip for the angle
            if view == ViewType.FRONT_LEFT:
                # Camera from left: left side is more visible
                ear = lm[self.LEFT_EAR]
                shoulder = lm[self.LEFT_SHOULDER]
                hip = lm[self.LEFT_HIP]
            else:
                ear = lm[self.RIGHT_EAR]
                shoulder = lm[self.RIGHT_SHOULDER]
                hip = lm[self.RIGHT_HIP]

            if ear.visibility < self.config.min_landmark_visibility:
                return 180.0

            angle = _angle_between_points(
                (ear.x, ear.y),
                (shoulder.x, shoulder.y),
                (hip.x, hip.y),
            )
            # Perspective distortion makes angles appear smaller from an angle;
            # apply a correction factor to be more lenient
            correction = 5.0
            return min(180.0, angle + correction)
        elif view == ViewType.UPPER:
            # Upper view can't reliably measure head forward; assume OK
            return 180.0
        else:
            # Front view: use nose position relative to midpoint of shoulders
            nose = lm[self.NOSE]
            mid_shoulder_x = (lm[self.LEFT_SHOULDER].x + lm[self.RIGHT_SHOULDER].x) / 2
            mid_shoulder_y = (lm[self.LEFT_SHOULDER].y + lm[self.RIGHT_SHOULDER].y) / 2
            mid_hip_x = (lm[self.LEFT_HIP].x + lm[self.RIGHT_HIP].x) / 2
            mid_hip_y = (lm[self.LEFT_HIP].y + lm[self.RIGHT_HIP].y) / 2

            return _angle_between_points(
                (nose.x, nose.y),
                (mid_shoulder_x, mid_shoulder_y),
                (mid_hip_x, mid_hip_y),
            )

    def _measure_shoulder_tilt(self, lm: list[_Landmark], view: ViewType) -> tuple[float, str]:
        """Measure shoulder tilt from horizontal.

        From angled views, perspective distortion makes the far shoulder appear
        higher, so we apply a compensation to avoid false positives.

        Returns:
            Tuple of (degrees from horizontal, which shoulder is higher: "left"/"right"/"").
        """
        left = lm[self.LEFT_SHOULDER]
        right = lm[self.RIGHT_SHOULDER]

        if (left.visibility < self.config.min_landmark_visibility or
                right.visibility < self.config.min_landmark_visibility):
            return 0.0, ""

        # Upper view: shoulder tilt measurement is meaningless
        if view == ViewType.UPPER:
            return 0.0, ""

        # Compute tilt as the angle the shoulder line makes with the horizontal.
        # Use abs(dx) so mirror-flipped images don't produce ~180° angles.
        dx = abs(right.x - left.x)
        dy = right.y - left.y  # positive = right is lower on screen

        if dx < 1e-6:
            angle = 90.0 if abs(dy) > 1e-6 else 0.0
        else:
            angle = abs(math.degrees(math.atan2(dy, dx)))

        # From an angled view, perspective naturally creates apparent tilt.
        # Subtract a compensation so minor perspective tilt isn't flagged.
        if view in (ViewType.FRONT_LEFT, ViewType.FRONT_RIGHT):
            angle = max(0.0, angle - 4.0)

        # Determine which shoulder is higher (lower y = higher on screen)
        if abs(left.y - right.y) < 0.005:
            higher_side = ""
        elif left.y < right.y:
            higher_side = "left"
        else:
            higher_side = "right"

        return angle, higher_side

    def _measure_torso_lean(self, lm: list[_Landmark], view: ViewType) -> float:
        """Measure torso forward lean. Returns degrees from vertical (0 = upright)."""
        if view == ViewType.UPPER:
            # Upper view can't reliably measure forward lean
            return 0.0

        if view in (ViewType.LEFT_SIDE, ViewType.RIGHT_SIDE):
            # LEFT_SIDE means camera sees right side; RIGHT_SIDE means camera sees left side
            if view == ViewType.LEFT_SIDE:
                shoulder = lm[self.RIGHT_SHOULDER]
                hip = lm[self.RIGHT_HIP]
            else:
                shoulder = lm[self.LEFT_SHOULDER]
                hip = lm[self.LEFT_HIP]
        elif view in (ViewType.FRONT_LEFT, ViewType.FRONT_RIGHT):
            # Angled view: use the side closer to camera for better accuracy
            if view == ViewType.FRONT_LEFT:
                shoulder = lm[self.LEFT_SHOULDER]
                hip = lm[self.LEFT_HIP]
            else:
                shoulder = lm[self.RIGHT_SHOULDER]
                hip = lm[self.RIGHT_HIP]
        else:
            # Front view: use midpoints
            shoulder = _Landmark(
                x=(lm[self.LEFT_SHOULDER].x + lm[self.RIGHT_SHOULDER].x) / 2,
                y=(lm[self.LEFT_SHOULDER].y + lm[self.RIGHT_SHOULDER].y) / 2,
                z=0, visibility=1.0,
            )
            hip = _Landmark(
                x=(lm[self.LEFT_HIP].x + lm[self.RIGHT_HIP].x) / 2,
                y=(lm[self.LEFT_HIP].y + lm[self.RIGHT_HIP].y) / 2,
                z=0, visibility=1.0,
            )

        if (shoulder.visibility < self.config.min_landmark_visibility or
                hip.visibility < self.config.min_landmark_visibility):
            return 0.0

        lean = _angle_from_vertical((hip.x, hip.y), (shoulder.x, shoulder.y))

        # Angled views introduce apparent lateral offset that looks like lean;
        # apply a small correction
        if view in (ViewType.FRONT_LEFT, ViewType.FRONT_RIGHT):
            lean = max(0.0, lean - 3.0)

        return lean

    def _compute_score(self, head_angle: float, shoulder_tilt: float,
                       torso_lean: float) -> float:
        """Compute posture score 0-100 from raw measurements."""
        cfg = self.config

        # Head: 180 is perfect, lower is worse. Map to 0-1 (1=good)
        head_score = max(0.0, min(1.0,
            (head_angle - 130.0) / (180.0 - 130.0)
        ))

        # Shoulders: 0 is perfect, higher is worse. Map to 0-1 (1=good)
        shoulder_score = max(0.0, min(1.0,
            1.0 - (shoulder_tilt / 20.0)
        ))

        # Torso: 0 is perfect, higher is worse. Map to 0-1 (1=good)
        torso_score = max(0.0, min(1.0,
            1.0 - (torso_lean / 30.0)
        ))

        weighted = (
            head_score * cfg.weight_head_forward
            + shoulder_score * cfg.weight_shoulder_tilt
            + torso_score * cfg.weight_torso_lean
        )

        return round(weighted * 100.0, 1)

    @staticmethod
    def _hysteresis(active: bool, value: float, enter_threshold: float,
                    release_threshold: float, invert: bool = False) -> bool:
        """Apply hysteresis to a measurement.

        If invert=False: triggers when value > enter, releases when value < release.
        If invert=True: triggers when value < enter, releases when value > release.
        """
        if invert:
            if not active and value < enter_threshold:
                return True
            if active and value > release_threshold:
                return False
        else:
            if not active and value > enter_threshold:
                return True
            if active and value < release_threshold:
                return False
        return active
