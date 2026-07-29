"""Tests for logic.py. No webcam, no window, no sleeping.

Run from the repository root:

    python -m pytest projects/your_example_app
"""

from __future__ import annotations

from logic import BrightnessTracker, Config


def feed(tracker: BrightnessTracker, brightness: float, *, start: float, frames: int = 6) -> dict:
    """Push the same reading for a while, advancing a fake clock."""
    state: dict = {}
    for index in range(frames):
        state = tracker.update(brightness, start + index * 0.1)
    return state


def test_starts_dark():
    assert BrightnessTracker().state(0.0)["bright"] is False


def test_a_bright_scene_registers():
    tracker = BrightnessTracker()
    assert feed(tracker, 0.9, start=0.0)["bright"] is True


def test_a_single_bright_frame_is_not_enough():
    """The settle time exists so one flash does not flip the state."""
    tracker = BrightnessTracker()
    assert tracker.update(0.9, 0.0)["bright"] is False


def test_values_between_the_thresholds_hold_the_state():
    """This is the hysteresis test. Without two thresholds it flickers."""
    config = Config()
    tracker = BrightnessTracker(config)
    feed(tracker, 0.9, start=0.0)
    midpoint = (config.bright_enter + config.bright_release) / 2
    assert feed(tracker, midpoint, start=1.0)["bright"] is True


def test_noise_at_the_boundary_never_counts_a_change():
    config = Config()
    tracker = BrightnessTracker(config)
    now = 0.0
    for index in range(60):
        tracker.update(config.bright_enter - 0.01 if index % 2 else config.bright_release + 0.01, now)
        now += 0.05
    assert tracker.changes == 0


def test_going_dark_again_counts_a_second_change():
    tracker = BrightnessTracker()
    feed(tracker, 0.9, start=0.0)
    feed(tracker, 0.1, start=1.0)
    assert tracker.changes == 2
    assert tracker.state(0.1)["bright"] is False


def test_fixture_values_straddle_the_thresholds():
    """Guards the test data: change a threshold and this fails loudly."""
    config = Config()
    assert 0.9 > config.bright_enter
    assert 0.1 < config.bright_release
