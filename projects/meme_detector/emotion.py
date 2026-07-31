"""Pure emotion detection logic — no camera, no display, no I/O.

Classifies facial expressions into Happy, Surprised, Neutral, or Sad using
ratios derived from MediaPipe Face Mesh landmarks. All thresholds are
scale-relative (normalized by interocular distance) so distance from the camera
does not affect detection.

Landmark indices used (468-point Face Mesh):
- Mouth corners: 61 (left), 291 (right)
- Upper lip top: 13
- Lower lip bottom: 14
- Upper eyelid: 159 (left), 386 (right)
- Lower eyelid: 145 (left), 374 (right)
- Left eye outer: 33, inner: 133
- Right eye outer: 263, inner: 362
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class Emotion(Enum):
    """Detected emotions."""
    HAPPY = "Happy"
    SURPRISED = "Surprised"
    SAD = "Sad"
    NEUTRAL = "Neutral"


@dataclass
class Config:
    """Tunables for emotion classification — all in one place.

    Ratios are relative to interocular distance (eye-to-eye span), making them
    independent of face size / camera distance.
    """

    # Mouth aspect ratio (vertical opening / mouth width) thresholds
    mouth_open_ratio_surprised: float = 0.45  # above this → surprised
    mouth_open_ratio_happy: float = 0.15      # above this (with smile) → happy

    # Smile curvature: how much the mouth corners are above the lip midpoint,
    # normalized by interocular distance. Positive = smile, negative = frown.
    smile_threshold: float = 0.02   # above → happy
    sad_threshold: float = -0.015   # below → sad

    # Eye aspect ratio (vertical / horizontal per eye). Wide eyes reinforce surprise.
    eye_open_ratio_surprised: float = 0.30

    # Hysteresis offset applied when leaving a state (prevents flicker)
    hysteresis: float = 0.005

    # Minimum consecutive frames before committing to a new emotion
    settle_frames: int = 3


class EmotionAnalyzer:
    """Analyzes facial landmarks and returns the detected emotion.

    Follows the same shape as the starter app's BrightnessTracker:
    - State lives in the object
    - ``update()`` takes measurements and returns a state dict
    - Pure logic, no I/O
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.current_emotion: Emotion = Emotion.NEUTRAL
        self._candidate: Emotion = Emotion.NEUTRAL
        self._candidate_count: int = 0

    def update(self, landmarks: List[Tuple[float, float, float]], timestamp: float) -> dict:
        """Classify the emotion from 468 Face Mesh landmarks.

        Args:
            landmarks: List of (x, y, z) normalized landmarks (0..1).
            timestamp: Monotonic timestamp (seconds). Reserved for future
                       temporal smoothing.

        Returns:
            State dict with keys: emotion, smile_ratio, mouth_ratio, eye_ratio.
        """
        metrics = self._compute_metrics(landmarks)
        raw_emotion = self._classify(metrics)

        # Settle: require N consecutive frames of the same candidate before switching
        if raw_emotion != self._candidate:
            self._candidate = raw_emotion
            self._candidate_count = 1
        else:
            self._candidate_count += 1

        if (
            self._candidate != self.current_emotion
            and self._candidate_count >= self.config.settle_frames
        ):
            self.current_emotion = self._candidate

        return self.state(metrics)

    def state(self, metrics: dict | None = None) -> dict:
        """Current state as a serializable dict."""
        base = {"emotion": self.current_emotion.value}
        if metrics:
            base.update(metrics)
        return base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_metrics(self, landmarks: List[Tuple[float, float, float]]) -> dict:
        """Extract facial ratios from raw landmarks."""
        # Interocular distance (scale reference)
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        interocular = _dist_2d(left_eye_outer, right_eye_outer)

        # Guard against degenerate landmarks
        if interocular < 1e-6:
            return {"smile_ratio": 0.0, "mouth_ratio": 0.0, "eye_ratio": 0.0}

        # --- Mouth metrics ---
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]

        mouth_width = _dist_2d(mouth_left, mouth_right)
        mouth_height = _dist_2d(upper_lip, lower_lip)
        mouth_ratio = mouth_height / max(mouth_width, 1e-6)

        # Smile curvature: average Y of corners vs midpoint of lips
        # In normalized coords, Y increases downward, so corners ABOVE midpoint
        # means corners have LOWER y → positive curvature = smile.
        lip_mid_y = (upper_lip[1] + lower_lip[1]) / 2.0
        corners_avg_y = (mouth_left[1] + mouth_right[1]) / 2.0
        # Positive when corners are above lip center (smiling)
        smile_ratio = (lip_mid_y - corners_avg_y) / interocular

        # --- Eye metrics ---
        # Left eye
        left_upper = landmarks[159]
        left_lower = landmarks[145]
        left_inner = landmarks[133]
        left_outer = landmarks[33]
        left_eye_h = _dist_2d(left_upper, left_lower)
        left_eye_w = _dist_2d(left_inner, left_outer)

        # Right eye
        right_upper = landmarks[386]
        right_lower = landmarks[374]
        right_inner = landmarks[362]
        right_outer = landmarks[263]
        right_eye_h = _dist_2d(right_upper, right_lower)
        right_eye_w = _dist_2d(right_inner, right_outer)

        eye_ratio = (
            (left_eye_h / max(left_eye_w, 1e-6))
            + (right_eye_h / max(right_eye_w, 1e-6))
        ) / 2.0

        return {
            "smile_ratio": round(smile_ratio, 4),
            "mouth_ratio": round(mouth_ratio, 4),
            "eye_ratio": round(eye_ratio, 4),
        }

    def _classify(self, metrics: dict) -> Emotion:
        """Map facial metrics to an emotion label."""
        cfg = self.config
        smile = metrics["smile_ratio"]
        mouth = metrics["mouth_ratio"]
        eye = metrics["eye_ratio"]

        # Apply hysteresis: require a slightly larger threshold to ENTER a state
        # than to LEAVE it.
        hyst = cfg.hysteresis if self.current_emotion != Emotion.SURPRISED else -cfg.hysteresis

        # 1. Surprised: mouth wide open AND eyes wide
        if mouth > (cfg.mouth_open_ratio_surprised + hyst) and eye > cfg.eye_open_ratio_surprised:
            return Emotion.SURPRISED

        # 2. Happy: noticeable smile curvature
        hyst_smile = cfg.hysteresis if self.current_emotion != Emotion.HAPPY else -cfg.hysteresis
        if smile > (cfg.smile_threshold + hyst_smile):
            return Emotion.HAPPY

        # 3. Sad: mouth corners droop
        hyst_sad = cfg.hysteresis if self.current_emotion != Emotion.SAD else -cfg.hysteresis
        if smile < (cfg.sad_threshold - hyst_sad):
            return Emotion.SAD

        # 4. Default
        return Emotion.NEUTRAL


def _dist_2d(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance in 2D (ignores z)."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
