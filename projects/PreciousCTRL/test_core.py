"""Tests for posture checker core logic using synthetic landmarks.

No webcam required. All landmarks are fabricated to test specific posture states.
"""

import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import pytest

from config import PostureConfig
from core import PostureChecker, PostureState, ViewType


# --- Landmark Factories ---

def _make_landmark(x: float, y: float, z: float = 0.0,
                   visibility: float = 1.0) -> dict:
    """Create a single landmark dict."""
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def make_33_landmarks() -> list[dict]:
    """Create a baseline set of 33 landmarks (all at origin, fully visible)."""
    return [_make_landmark(0.5, 0.5) for _ in range(33)]


def make_good_posture_front() -> list[dict]:
    """Good posture from the front: shoulders level, head centered above torso."""
    lm = make_33_landmarks()
    # Nose centered above shoulders
    lm[0] = _make_landmark(0.50, 0.25)
    # Both ears visible and symmetric
    lm[7] = _make_landmark(0.44, 0.27, visibility=0.95)   # left ear
    lm[8] = _make_landmark(0.56, 0.27, visibility=0.95)   # right ear
    # Level shoulders
    lm[11] = _make_landmark(0.38, 0.40, visibility=0.99)  # left shoulder
    lm[12] = _make_landmark(0.62, 0.40, visibility=0.99)  # right shoulder
    # Hips centered below shoulders
    lm[23] = _make_landmark(0.42, 0.65, visibility=0.95)  # left hip
    lm[24] = _make_landmark(0.58, 0.65, visibility=0.95)  # right hip
    return lm


def make_bad_posture_front_uneven_shoulders() -> list[dict]:
    """Bad posture: one shoulder significantly higher than the other."""
    lm = make_good_posture_front()
    # Tilt shoulders: right shoulder much higher
    lm[11] = _make_landmark(0.38, 0.44, visibility=0.99)  # left lower
    lm[12] = _make_landmark(0.62, 0.34, visibility=0.99)  # right higher
    return lm


def make_good_posture_side() -> list[dict]:
    """Good posture from the side: ear directly above shoulder and hip."""
    lm = make_33_landmarks()
    # Side view: only right side visible
    lm[0] = _make_landmark(0.50, 0.25)
    lm[7] = _make_landmark(0.48, 0.27, visibility=0.2)    # left ear hidden
    lm[8] = _make_landmark(0.50, 0.27, visibility=0.95)   # right ear visible
    # Ear directly above shoulder, shoulder above hip (aligned vertically)
    lm[12] = _make_landmark(0.50, 0.40, visibility=0.99)  # right shoulder
    lm[24] = _make_landmark(0.50, 0.65, visibility=0.95)  # right hip
    lm[11] = _make_landmark(0.50, 0.40, visibility=0.3)   # left shoulder (hidden)
    lm[23] = _make_landmark(0.50, 0.65, visibility=0.3)   # left hip (hidden)
    return lm


def make_bad_posture_side_head_forward() -> list[dict]:
    """Bad posture from side: head jutting forward (ear far in front of shoulder)."""
    lm = make_33_landmarks()
    lm[0] = _make_landmark(0.35, 0.22)
    lm[7] = _make_landmark(0.33, 0.24, visibility=0.2)    # left ear hidden
    lm[8] = _make_landmark(0.35, 0.24, visibility=0.95)   # right ear forward
    lm[12] = _make_landmark(0.50, 0.40, visibility=0.99)  # right shoulder
    lm[24] = _make_landmark(0.50, 0.65, visibility=0.95)  # right hip
    lm[11] = _make_landmark(0.50, 0.40, visibility=0.3)
    lm[23] = _make_landmark(0.50, 0.65, visibility=0.3)
    return lm


def make_bad_posture_side_torso_lean() -> list[dict]:
    """Bad posture from side: torso leaning forward significantly."""
    lm = make_33_landmarks()
    lm[0] = _make_landmark(0.40, 0.22)
    lm[7] = _make_landmark(0.38, 0.24, visibility=0.2)
    lm[8] = _make_landmark(0.40, 0.24, visibility=0.95)
    # Shoulder significantly in front of hip
    lm[12] = _make_landmark(0.38, 0.40, visibility=0.99)  # shoulder forward
    lm[24] = _make_landmark(0.52, 0.65, visibility=0.95)  # hip back
    lm[11] = _make_landmark(0.38, 0.40, visibility=0.3)
    lm[23] = _make_landmark(0.52, 0.65, visibility=0.3)
    return lm


# --- Tests ---

class TestViewDetection:
    """Test that the checker correctly identifies front vs side views."""

    def test_front_view_both_ears_visible(self):
        checker = PostureChecker()
        lm = make_good_posture_front()
        state = checker.update(lm, 0.0)
        assert state.view == ViewType.FRONT

    def test_side_view_left_ear_hidden(self):
        checker = PostureChecker()
        lm = make_good_posture_side()
        # Left ear hidden, right ear visible -> LEFT_SIDE view
        state = checker.update(lm, 0.0)
        assert state.view == ViewType.LEFT_SIDE

    def test_side_view_right_ear_hidden(self):
        checker = PostureChecker()
        lm = make_good_posture_side()
        # Swap: right ear hidden, left ear visible -> RIGHT_SIDE
        lm[7] = _make_landmark(0.50, 0.27, visibility=0.95)  # left ear visible
        lm[8] = _make_landmark(0.50, 0.27, visibility=0.2)   # right ear hidden
        state = checker.update(lm, 0.0)
        assert state.view == ViewType.RIGHT_SIDE


class TestGoodPosture:
    """Test that good posture produces high scores and no warnings."""

    def test_front_good_posture_high_score(self):
        checker = PostureChecker()
        lm = make_good_posture_front()
        state = checker.update(lm, 0.0)
        assert state.score >= 80.0
        assert not state.head_forward
        assert not state.shoulders_uneven
        assert not state.torso_leaning
        assert not state.is_slouching

    def test_side_good_posture_high_score(self):
        checker = PostureChecker()
        lm = make_good_posture_side()
        state = checker.update(lm, 0.0)
        assert state.score >= 80.0
        assert not state.head_forward
        assert not state.torso_leaning
        assert not state.is_slouching

    def test_good_posture_message(self):
        checker = PostureChecker()
        lm = make_good_posture_front()
        state = checker.update(lm, 0.0)
        assert "Good posture!" in state.messages


class TestBadPosture:
    """Test that bad posture is detected correctly."""

    def test_uneven_shoulders_detected(self):
        checker = PostureChecker()
        lm = make_bad_posture_front_uneven_shoulders()
        state = checker.update(lm, 0.0)
        assert state.shoulders_uneven
        assert state.is_slouching
        assert "Shoulders uneven" in state.messages

    def test_head_forward_side_view(self):
        checker = PostureChecker()
        lm = make_bad_posture_side_head_forward()
        state = checker.update(lm, 0.0)
        assert state.head_forward
        assert "Head too far forward" in state.messages

    def test_torso_lean_detected(self):
        checker = PostureChecker()
        lm = make_bad_posture_side_torso_lean()
        state = checker.update(lm, 0.0)
        assert state.torso_leaning
        assert "Torso leaning forward" in state.messages

    def test_bad_posture_lower_score(self):
        checker = PostureChecker()
        good_state = checker.update(make_good_posture_front(), 0.0)

        checker2 = PostureChecker()
        bad_state = checker2.update(make_bad_posture_front_uneven_shoulders(), 0.0)

        assert bad_state.score < good_state.score


class TestStreaks:
    """Test good/bad posture streak timers."""

    def test_good_streak_accumulates(self):
        checker = PostureChecker()
        lm = make_good_posture_front()
        checker.update(lm, 0.0)
        state = checker.update(lm, 5.0)
        assert state.good_streak_seconds == pytest.approx(5.0)
        assert state.bad_streak_seconds == 0.0

    def test_bad_streak_accumulates(self):
        checker = PostureChecker()
        lm = make_bad_posture_front_uneven_shoulders()
        checker.update(lm, 0.0)
        state = checker.update(lm, 5.0)
        assert state.bad_streak_seconds == pytest.approx(5.0)
        assert state.good_streak_seconds == 0.0

    def test_streak_resets_on_transition(self):
        checker = PostureChecker()
        good = make_good_posture_front()
        bad = make_bad_posture_front_uneven_shoulders()

        # Start good
        checker.update(good, 0.0)
        state = checker.update(good, 3.0)
        assert state.good_streak_seconds == pytest.approx(3.0)

        # Transition to bad
        state = checker.update(bad, 4.0)
        assert state.bad_streak_seconds == pytest.approx(0.0)  # Just started
        assert state.good_streak_seconds == 0.0

        # Continue bad
        state = checker.update(bad, 7.0)
        assert state.bad_streak_seconds == pytest.approx(3.0)


class TestSlouchEvents:
    """Test slouch event triggering and cooldown."""

    def test_slouch_triggers_after_threshold(self):
        config = PostureConfig(slouch_trigger_seconds=3.0, slouch_cooldown_seconds=10.0)
        checker = PostureChecker(config)
        bad = make_bad_posture_front_uneven_shoulders()

        # Frame 1: bad posture starts
        state = checker.update(bad, 0.0)
        assert not state.slouch_event_triggered

        # Frame 2: 2 seconds in, not enough
        state = checker.update(bad, 2.0)
        assert not state.slouch_event_triggered

        # Frame 3: 4 seconds in, should trigger (>= 3s of bad)
        state = checker.update(bad, 4.0)
        assert state.slouch_event_triggered

    def test_slouch_cooldown_prevents_rapid_triggers(self):
        config = PostureConfig(slouch_trigger_seconds=3.0, slouch_cooldown_seconds=10.0)
        checker = PostureChecker(config)
        bad = make_bad_posture_front_uneven_shoulders()

        checker.update(bad, 0.0)
        checker.update(bad, 4.0)  # triggers first event

        # Should NOT trigger again at 8s (only 4s since last trigger, < 10s cooldown)
        state = checker.update(bad, 8.0)
        assert not state.slouch_event_triggered

        # Should trigger again at 15s (11s since first trigger, > 10s cooldown)
        state = checker.update(bad, 15.0)
        assert state.slouch_event_triggered

    def test_no_slouch_event_with_good_posture(self):
        checker = PostureChecker()
        good = make_good_posture_front()

        for t in range(0, 60, 1):
            state = checker.update(good, float(t))
            assert not state.slouch_event_triggered


class TestHysteresis:
    """Test that hysteresis prevents flickering at threshold boundaries."""

    def test_does_not_trigger_below_threshold(self):
        """A value just below the bad threshold should not trigger."""
        config = PostureConfig(shoulder_tilt_bad=8.0, shoulder_tilt_good=5.0)
        checker = PostureChecker(config)

        # Create landmarks with 7 degree tilt (below 8 threshold)
        lm = make_good_posture_front()
        # Mild tilt: about 7 degrees
        lm[11] = _make_landmark(0.38, 0.42, visibility=0.99)
        lm[12] = _make_landmark(0.62, 0.39, visibility=0.99)

        state = checker.update(lm, 0.0)
        assert not state.shoulders_uneven

    def test_stays_active_between_thresholds(self):
        """Once triggered, should stay active until below release threshold."""
        config = PostureConfig(shoulder_tilt_bad=8.0, shoulder_tilt_good=5.0)
        checker = PostureChecker(config)

        # First trigger with high tilt
        bad = make_bad_posture_front_uneven_shoulders()
        checker.update(bad, 0.0)

        # Now reduce to 6 degrees (between 5 and 8): should STAY active
        lm = make_good_posture_front()
        # About 6 degrees of tilt
        lm[11] = _make_landmark(0.38, 0.425, visibility=0.99)
        lm[12] = _make_landmark(0.62, 0.385, visibility=0.99)

        state = checker.update(lm, 1.0)
        # The hysteresis should keep it active since we haven't dropped below 5
        assert state.shoulders_uneven


class TestNoLandmarks:
    """Test behaviour with missing or insufficient landmarks."""

    def test_no_landmarks_returns_default(self):
        checker = PostureChecker()
        state = checker.update([], 0.0)
        assert state.view == ViewType.UNKNOWN
        assert state.paused
        assert "No person detected" in state.messages[0]

    def test_too_few_landmarks(self):
        checker = PostureChecker()
        lm = [_make_landmark(0.5, 0.5) for _ in range(10)]
        state = checker.update(lm, 0.0)
        assert state.paused
        assert "No person detected" in state.messages[0]
