"""Tests for the EmotionAnalyzer — no camera needed.

Builds synthetic Face Mesh landmarks (468 points) with controlled geometry to
verify that the emotion classifier returns the expected emotion for each
expression.

Run from the repository root:
    pytest projects/MacXenix/test_emotion.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure imports work when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emotion import Config, Emotion, EmotionAnalyzer, _dist_2d  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers: build synthetic 468-landmark arrays
# ---------------------------------------------------------------------------

def _base_landmarks() -> list:
    """Return 468 landmarks with a neutral face geometry.

    Key points are placed explicitly; the rest fill with a default position.
    The face is roughly centered at (0.5, 0.5) in normalized coordinates.
    """
    # Default all 468 to center
    landmarks = [(0.5, 0.5, 0.0)] * 468

    # --- Eyes (defines interocular distance) ---
    # Left eye outer (33) and inner (133)
    landmarks[33] = (0.35, 0.45, 0.0)   # left eye outer
    landmarks[133] = (0.42, 0.45, 0.0)  # left eye inner

    # Right eye outer (263) and inner (362)
    landmarks[263] = (0.65, 0.45, 0.0)  # right eye outer
    landmarks[362] = (0.58, 0.45, 0.0)  # right eye inner

    # Eyelids (neutral: slightly open)
    landmarks[159] = (0.385, 0.44, 0.0)  # left upper eyelid
    landmarks[145] = (0.385, 0.46, 0.0)  # left lower eyelid
    landmarks[386] = (0.615, 0.44, 0.0)  # right upper eyelid
    landmarks[374] = (0.615, 0.46, 0.0)  # right lower eyelid

    # --- Mouth (neutral: closed, straight) ---
    landmarks[61] = (0.43, 0.65, 0.0)   # mouth left corner
    landmarks[291] = (0.57, 0.65, 0.0)  # mouth right corner
    landmarks[13] = (0.50, 0.64, 0.0)   # upper lip top
    landmarks[14] = (0.50, 0.66, 0.0)   # lower lip bottom

    return landmarks


def _make_happy_landmarks() -> list:
    """Landmarks with raised mouth corners (smile)."""
    landmarks = _base_landmarks()
    # Move mouth corners UP relative to lip center → positive smile ratio
    # Lip center Y ≈ 0.65, move corners to 0.62 (above center)
    landmarks[61] = (0.43, 0.61, 0.0)
    landmarks[291] = (0.57, 0.61, 0.0)
    landmarks[13] = (0.50, 0.64, 0.0)
    landmarks[14] = (0.50, 0.67, 0.0)
    return landmarks


def _make_surprised_landmarks() -> list:
    """Landmarks with wide-open mouth and wide eyes."""
    landmarks = _base_landmarks()
    # Wide mouth opening
    landmarks[13] = (0.50, 0.60, 0.0)   # upper lip way up
    landmarks[14] = (0.50, 0.74, 0.0)   # lower lip way down
    # Keep corners neutral
    landmarks[61] = (0.43, 0.67, 0.0)
    landmarks[291] = (0.57, 0.67, 0.0)
    # Wide eyes (large vertical gap)
    landmarks[159] = (0.385, 0.42, 0.0)  # left upper far up
    landmarks[145] = (0.385, 0.48, 0.0)  # left lower far down
    landmarks[386] = (0.615, 0.42, 0.0)  # right upper far up
    landmarks[374] = (0.615, 0.48, 0.0)  # right lower far down
    return landmarks


def _make_sad_landmarks() -> list:
    """Landmarks with drooping mouth corners (frown)."""
    landmarks = _base_landmarks()
    # Move mouth corners DOWN below lip center → negative smile ratio
    landmarks[61] = (0.43, 0.69, 0.0)
    landmarks[291] = (0.57, 0.69, 0.0)
    landmarks[13] = (0.50, 0.64, 0.0)
    landmarks[14] = (0.50, 0.66, 0.0)
    return landmarks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDistHelper:
    """Test the distance utility function."""

    def test_same_point(self):
        assert _dist_2d((0.5, 0.5, 0.0), (0.5, 0.5, 0.0)) == 0.0

    def test_known_distance(self):
        # 3-4-5 triangle scaled to 0.03, 0.04
        d = _dist_2d((0.0, 0.0, 0.0), (0.03, 0.04, 0.0))
        assert abs(d - 0.05) < 1e-9

    def test_ignores_z(self):
        d1 = _dist_2d((0.1, 0.2, 0.0), (0.4, 0.6, 0.0))
        d2 = _dist_2d((0.1, 0.2, 99.0), (0.4, 0.6, -50.0))
        assert abs(d1 - d2) < 1e-9


class TestEmotionAnalyzer:
    """Test emotion classification with synthetic landmarks."""

    def _settle(self, analyzer: EmotionAnalyzer, landmarks: list, frames: int = 5) -> dict:
        """Feed the same landmarks for multiple frames to pass the settle filter."""
        state = {}
        for i in range(frames):
            state = analyzer.update(landmarks, float(i) * 0.033)
        return state

    def test_neutral_baseline(self):
        """Neutral landmarks should classify as Neutral."""
        analyzer = EmotionAnalyzer(Config(settle_frames=1))
        landmarks = _base_landmarks()
        state = analyzer.update(landmarks, 0.0)
        assert state["emotion"] == "Neutral"

    def test_happy_detection(self):
        """Raised mouth corners should classify as Happy."""
        analyzer = EmotionAnalyzer(Config(settle_frames=2))
        landmarks = _make_happy_landmarks()
        state = self._settle(analyzer, landmarks, frames=4)
        assert state["emotion"] == "Happy"

    def test_surprised_detection(self):
        """Wide mouth and eyes should classify as Surprised."""
        analyzer = EmotionAnalyzer(Config(settle_frames=2))
        landmarks = _make_surprised_landmarks()
        state = self._settle(analyzer, landmarks, frames=4)
        assert state["emotion"] == "Surprised"

    def test_sad_detection(self):
        """Drooping mouth corners should classify as Sad."""
        analyzer = EmotionAnalyzer(Config(settle_frames=2))
        landmarks = _make_sad_landmarks()
        state = self._settle(analyzer, landmarks, frames=4)
        assert state["emotion"] == "Sad"

    def test_settle_frames_prevents_flicker(self):
        """A single frame should NOT change the emotion (settle filter)."""
        analyzer = EmotionAnalyzer(Config(settle_frames=5))
        # Start neutral
        analyzer.update(_base_landmarks(), 0.0)
        # Single happy frame — should NOT switch yet
        state = analyzer.update(_make_happy_landmarks(), 0.033)
        assert state["emotion"] == "Neutral"

    def test_emotion_transitions(self):
        """Verify transitions between emotions work correctly."""
        analyzer = EmotionAnalyzer(Config(settle_frames=2))

        # Start happy
        self._settle(analyzer, _make_happy_landmarks(), frames=4)
        assert analyzer.current_emotion == Emotion.HAPPY

        # Transition to surprised
        state = self._settle(analyzer, _make_surprised_landmarks(), frames=4)
        assert state["emotion"] == "Surprised"

        # Transition to sad
        state = self._settle(analyzer, _make_sad_landmarks(), frames=4)
        assert state["emotion"] == "Sad"

        # Back to neutral
        state = self._settle(analyzer, _base_landmarks(), frames=4)
        assert state["emotion"] == "Neutral"

    def test_state_contains_metrics(self):
        """State dict should include smile_ratio, mouth_ratio, eye_ratio."""
        analyzer = EmotionAnalyzer(Config(settle_frames=1))
        state = analyzer.update(_base_landmarks(), 0.0)
        assert "smile_ratio" in state
        assert "mouth_ratio" in state
        assert "eye_ratio" in state
        assert "emotion" in state

    def test_degenerate_landmarks_no_crash(self):
        """All-zero landmarks should not crash (guards against division by zero)."""
        analyzer = EmotionAnalyzer(Config(settle_frames=1))
        landmarks = [(0.0, 0.0, 0.0)] * 468
        state = analyzer.update(landmarks, 0.0)
        # Should default to Neutral without crashing
        assert state["emotion"] == "Neutral"

    def test_config_customization(self):
        """Custom config thresholds should affect classification."""
        # Very low smile threshold → even a small smile registers
        config = Config(smile_threshold=0.001, settle_frames=1)
        analyzer = EmotionAnalyzer(config)
        landmarks = _base_landmarks()
        # Slightly raise corners
        landmarks[61] = (0.43, 0.645, 0.0)
        landmarks[291] = (0.57, 0.645, 0.0)
        state = analyzer.update(landmarks, 0.0)
        assert state["emotion"] == "Happy"
