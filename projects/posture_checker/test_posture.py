"""Tests for the Posture Tracker — no camera needed.

Uses synthetic 33-point Pose landmarks to verify posture classification
from both frontal and side/profile views, scoring, slouch timer,
calibration, and view detection.

Run from the repository root:
    pytest projects/posture_checker/test_posture.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracker import (  # noqa: E402
    Config, Metrics, PostureState, PostureTracker,
    _angle_from_vertical, _dist_2d, _midpoint,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _base_landmarks_front() -> list:
    """33 landmarks: good upright posture, FRONTAL view.

    Shoulder width is large relative to torso → detected as front view.
    """
    lm = [(0.5, 0.5, 0.0)] * 33

    # Head: nose above ears
    lm[0] = (0.50, 0.20, 0.0)    # nose
    lm[7] = (0.44, 0.25, 0.0)    # left ear
    lm[8] = (0.56, 0.25, 0.0)    # right ear

    # Shoulders: wide, level
    lm[11] = (0.30, 0.40, 0.0)   # left shoulder
    lm[12] = (0.70, 0.40, 0.0)   # right shoulder

    # Arms
    lm[13] = (0.24, 0.55, 0.0)   # left elbow
    lm[14] = (0.76, 0.55, 0.0)   # right elbow
    lm[15] = (0.22, 0.70, 0.0)   # left wrist
    lm[16] = (0.78, 0.70, 0.0)   # right wrist

    # Hips: level, below shoulders
    lm[23] = (0.38, 0.70, 0.0)   # left hip
    lm[24] = (0.62, 0.70, 0.0)   # right hip

    # Knees
    lm[25] = (0.38, 0.85, 0.0)   # left knee
    lm[26] = (0.62, 0.85, 0.0)   # right knee

    return lm


def _base_landmarks_side() -> list:
    """33 landmarks: good upright posture, SIDE/PROFILE view.

    Shoulder width is small (both shoulders nearly overlap in X) relative to
    torso length → detected as side view.
    """
    lm = [(0.5, 0.5, 0.0)] * 33

    # Side view: all points stacked vertically, shoulder width ≈ 0
    lm[0] = (0.50, 0.18, 0.0)    # nose
    lm[7] = (0.49, 0.23, 0.0)    # left ear (almost same X as right)
    lm[8] = (0.51, 0.23, 0.0)    # right ear

    # Shoulders: nearly overlapping in X (profile)
    lm[11] = (0.49, 0.38, 0.0)   # left shoulder
    lm[12] = (0.51, 0.38, 0.0)   # right shoulder (tiny width: 0.02)

    # Arms (hanging down)
    lm[13] = (0.45, 0.52, 0.0)
    lm[14] = (0.55, 0.52, 0.0)
    lm[15] = (0.43, 0.65, 0.0)
    lm[16] = (0.57, 0.65, 0.0)

    # Hips: directly below shoulders
    lm[23] = (0.49, 0.65, 0.0)   # left hip
    lm[24] = (0.51, 0.65, 0.0)   # right hip (torso_length ≈ 0.27)

    lm[25] = (0.49, 0.82, 0.0)
    lm[26] = (0.51, 0.82, 0.0)

    return lm


def _slouching_front() -> list:
    """Frontal view: nose drops near ear level (forward lean)."""
    lm = _base_landmarks_front()
    # Nose drops to ear level → large neck angle
    lm[0] = (0.50, 0.28, 0.0)   # was 0.20, now at 0.28 (close to ears at 0.25)
    return lm


def _slouching_side() -> list:
    """Side view: head pushed forward (ear ahead of shoulder)."""
    lm = _base_landmarks_side()
    # Move ears and nose far ahead of shoulders in X
    lm[0] = (0.65, 0.18, 0.0)   # nose pushed forward
    lm[7] = (0.63, 0.23, 0.0)   # left ear pushed forward
    lm[8] = (0.65, 0.23, 0.0)   # right ear pushed forward
    # Shoulders stay at 0.50
    return lm


def _spine_slouch_side() -> list:
    """Side view: trunk leaning forward (shoulders ahead of hips)."""
    lm = _base_landmarks_side()
    # Push shoulders forward (larger X) while hips stay
    lm[11] = (0.62, 0.38, 0.0)
    lm[12] = (0.64, 0.38, 0.0)
    # Hips remain at 0.50
    lm[0] = (0.68, 0.18, 0.0)
    lm[7] = (0.66, 0.23, 0.0)
    lm[8] = (0.68, 0.23, 0.0)
    return lm


def _tilted_front() -> list:
    """Frontal view: uneven shoulders."""
    lm = _base_landmarks_front()
    lm[11] = (0.30, 0.35, 0.0)  # left shoulder up
    lm[12] = (0.70, 0.50, 0.0)  # right shoulder way down
    return lm


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_dist_2d_zero(self):
        assert _dist_2d((0.5, 0.5, 0.0), (0.5, 0.5, 0.0)) == 0.0

    def test_dist_2d_known(self):
        d = _dist_2d((0.0, 0.0, 0.0), (0.3, 0.4, 0.0))
        assert abs(d - 0.5) < 1e-9

    def test_midpoint(self):
        m = _midpoint((0.2, 0.4, 0.0), (0.8, 0.6, 0.0))
        assert abs(m[0] - 0.5) < 1e-9
        assert abs(m[1] - 0.5) < 1e-9

    def test_angle_from_vertical_straight_up(self):
        # Tip directly above base → 0 degrees
        angle = _angle_from_vertical((0.5, 0.5, 0.0), (0.5, 0.3, 0.0))
        assert angle < 1.0

    def test_angle_from_vertical_horizontal(self):
        # Tip at same height, to the right → 90 degrees
        angle = _angle_from_vertical((0.5, 0.5, 0.0), (0.8, 0.5, 0.0))
        assert abs(angle - 90.0) < 1.0

    def test_angle_from_vertical_downward(self):
        # Tip directly below base → 180 degrees
        angle = _angle_from_vertical((0.5, 0.3, 0.0), (0.5, 0.7, 0.0))
        assert abs(angle - 180.0) < 1.0


# ---------------------------------------------------------------------------
# View detection tests
# ---------------------------------------------------------------------------

class TestViewDetection:
    def test_frontal_detected(self):
        tracker = PostureTracker(Config())
        state = tracker.update(_base_landmarks_front(), 0.0)
        assert state["side_view"] is False

    def test_side_view_detected(self):
        tracker = PostureTracker(Config())
        state = tracker.update(_base_landmarks_side(), 0.0)
        assert state["side_view"] is True


# ---------------------------------------------------------------------------
# Frontal view classification
# ---------------------------------------------------------------------------

class TestFrontalClassification:
    def _settle(self, tracker, landmarks, frames=3):
        state = {}
        for i in range(frames):
            state = tracker.update(landmarks, float(i) * 0.033)
        return state

    def test_good_posture_front(self):
        tracker = PostureTracker(Config())
        state = tracker.update(_base_landmarks_front(), 0.0)
        assert state["posture"] == "Good Posture"

    def test_slouching_front(self):
        tracker = PostureTracker(Config())
        state = self._settle(tracker, _slouching_front(), frames=5)
        assert state["posture"] == "Slouching"

    def test_head_tilted_front(self):
        tracker = PostureTracker(Config())
        state = self._settle(tracker, _tilted_front(), frames=5)
        assert state["posture"] == "Head Tilted"

    def test_no_person(self):
        tracker = PostureTracker(Config())
        state = tracker.update(None, 0.0)
        assert state["posture"] == "No Person"

    def test_too_few_landmarks(self):
        tracker = PostureTracker(Config())
        state = tracker.update([(0.5, 0.5, 0.0)] * 10, 0.0)
        assert state["posture"] == "No Person"


# ---------------------------------------------------------------------------
# Side-view classification
# ---------------------------------------------------------------------------

class TestSideViewClassification:
    def _settle(self, tracker, landmarks, frames=3):
        state = {}
        for i in range(frames):
            state = tracker.update(landmarks, float(i) * 0.033)
        return state

    def test_good_posture_side(self):
        tracker = PostureTracker(Config())
        state = tracker.update(_base_landmarks_side(), 0.0)
        assert state["posture"] == "Good Posture"

    def test_forward_head_side(self):
        """Head pushed forward in side view should detect slouching."""
        tracker = PostureTracker(Config())
        state = self._settle(tracker, _slouching_side(), frames=5)
        assert state["posture"] == "Slouching"

    def test_spine_lean_side(self):
        """Trunk leaning forward in side view should detect slouching."""
        tracker = PostureTracker(Config())
        state = self._settle(tracker, _spine_slouch_side(), frames=5)
        assert state["posture"] == "Slouching"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:
    def test_good_posture_high_score(self):
        tracker = PostureTracker(Config())
        state = tracker.update(_base_landmarks_front(), 0.0)
        assert state["score"] >= 70.0

    def test_slouching_lower_score(self):
        tracker = PostureTracker(Config())
        for i in range(15):
            state = tracker.update(_slouching_front(), float(i) * 0.033)
        assert state["score"] < 85.0

    def test_score_in_range(self):
        tracker = PostureTracker(Config())
        for lm in [_base_landmarks_front(), _slouching_front(), _tilted_front(),
                   _base_landmarks_side(), _slouching_side()]:
            state = tracker.update(lm, 0.0)
            assert 0.0 <= state["score"] <= 100.0


# ---------------------------------------------------------------------------
# Slouch timer
# ---------------------------------------------------------------------------

class TestSlouchTimer:
    def test_timer_starts_on_slouch(self):
        tracker = PostureTracker(Config())
        tracker.update(_slouching_front(), 0.0)
        state = tracker.update(_slouching_front(), 1.5)
        assert state["slouch_seconds"] >= 0.0

    def test_timer_resets_on_good(self):
        tracker = PostureTracker(Config())
        tracker.update(_slouching_front(), 0.0)
        tracker.update(_slouching_front(), 2.0)
        state = tracker.update(_base_landmarks_front(), 3.0)
        assert state["slouch_seconds"] == 0.0

    def test_alert_fires_after_threshold(self):
        config = Config(slouch_alert_seconds=2.0)
        tracker = PostureTracker(config)
        # Use side view slouch (more reliable to exceed threshold)
        for i in range(80):
            state = tracker.update(_slouching_side(), float(i) * 0.05)
        assert state["alert"] is True

    def test_no_alert_for_brief(self):
        config = Config(slouch_alert_seconds=3.0)
        tracker = PostureTracker(config)
        for i in range(20):
            state = tracker.update(_slouching_side(), float(i) * 0.033)
        assert state["alert"] is False

    def test_alert_clears_on_good(self):
        config = Config(slouch_alert_seconds=1.0)
        tracker = PostureTracker(config)
        for i in range(60):
            tracker.update(_slouching_side(), float(i) * 0.05)
        state = tracker.update(_base_landmarks_front(), 10.0)
        assert state["alert"] is False


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibrate_success(self):
        tracker = PostureTracker(Config())
        assert tracker.calibrate_baseline(_base_landmarks_front()) is True
        assert tracker.baseline.is_calibrated is True

    def test_calibrate_side_view(self):
        tracker = PostureTracker(Config())
        assert tracker.calibrate_baseline(_base_landmarks_side()) is True

    def test_calibrate_no_person(self):
        tracker = PostureTracker(Config())
        assert tracker.calibrate_baseline(None) is False

    def test_calibrate_resets_score(self):
        tracker = PostureTracker(Config())
        for i in range(20):
            tracker.update(_slouching_front(), float(i) * 0.033)
        tracker.calibrate_baseline(_base_landmarks_front())
        state = tracker.update(_base_landmarks_front(), 10.0)
        assert state["score"] >= 90.0

    def test_state_reports_calibration(self):
        tracker = PostureTracker(Config())
        state = tracker.update(_base_landmarks_front(), 0.0)
        assert state["calibrated"] is False
        tracker.calibrate_baseline(_base_landmarks_front())
        state = tracker.update(_base_landmarks_front(), 1.0)
        assert state["calibrated"] is True


# ---------------------------------------------------------------------------
# State dict structure
# ---------------------------------------------------------------------------

class TestStateDict:
    def test_all_keys_present(self):
        tracker = PostureTracker(Config())
        state = tracker.update(_base_landmarks_front(), 0.0)
        expected = {
            "posture", "score", "neck_angle", "forward_head", "spine_angle",
            "shoulder_tilt", "slouch_seconds", "alert", "calibrated", "side_view",
        }
        assert expected.issubset(state.keys())

    def test_types(self):
        tracker = PostureTracker(Config())
        state = tracker.update(_base_landmarks_front(), 0.0)
        assert isinstance(state["posture"], str)
        assert isinstance(state["score"], float)
        assert isinstance(state["neck_angle"], float)
        assert isinstance(state["forward_head"], float)
        assert isinstance(state["spine_angle"], float)
        assert isinstance(state["shoulder_tilt"], float)
        assert isinstance(state["slouch_seconds"], float)
        assert isinstance(state["alert"], bool)
        assert isinstance(state["calibrated"], bool)
        assert isinstance(state["side_view"], bool)

    def test_degenerate_landmarks(self):
        """All-same-point landmarks should not crash."""
        tracker = PostureTracker(Config())
        lm = [(0.5, 0.5, 0.0)] * 33
        state = tracker.update(lm, 0.0)
        assert state["posture"] == "No Person"
