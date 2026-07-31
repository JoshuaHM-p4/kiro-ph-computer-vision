"""Tests for the Don't Blink game logic — no camera needed."""

import pytest

from .config import GameConfig
from .core import (
    Distraction,
    Game,
    Phase,
    compute_ear,
    compute_ear_both_eyes,
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_config(**overrides) -> GameConfig:
    """Create a config with test-friendly defaults."""
    defaults = {
        "ear_blink_threshold": 0.21,
        "ear_open_threshold": 0.25,
        "blink_frames": 2,
        "countdown_seconds": 3,
        "distraction_start_delay": 2.0,
        "distraction_interval_initial": 3.0,
        "distraction_interval_min": 0.5,
        "distraction_interval_decay": 0.9,
        "distraction_duration": 0.3,
    }
    defaults.update(overrides)
    return GameConfig(**defaults)


def make_landmarks_with_ear(ear: float) -> list[tuple[float, float]]:
    """Build a fake 468-point landmark list that produces a given EAR.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    We set horizontal distance = 0.1, so vertical distances = ear * 0.1

    We only need the indices used by LEFT_EYE_IDX and RIGHT_EYE_IDX.
    """
    # Create 468 dummy landmarks
    landmarks = [(0.5, 0.5)] * 468

    # For each eye, place landmarks to produce the target EAR
    for indices in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
        # p1 (left corner) and p4 (right corner) — horizontal = 0.1
        landmarks[indices[0]] = (0.4, 0.5)
        landmarks[indices[3]] = (0.5, 0.5)

        # p2, p6 vertical pair — distance = ear * 0.1
        vert = ear * 0.1 / 2
        landmarks[indices[1]] = (0.45, 0.5 - vert)  # p2 above
        landmarks[indices[5]] = (0.45, 0.5 + vert)  # p6 below

        # p3, p5 vertical pair — same
        landmarks[indices[2]] = (0.47, 0.5 - vert)  # p3 above
        landmarks[indices[4]] = (0.47, 0.5 + vert)  # p5 below

    return landmarks


# ------------------------------------------------------------------
# Tests: EAR computation
# ------------------------------------------------------------------

class TestEAR:
    """Test Eye Aspect Ratio computation."""

    def test_open_eyes_high_ear(self):
        landmarks = make_landmarks_with_ear(0.30)
        ear = compute_ear_both_eyes(landmarks)
        assert 0.28 <= ear <= 0.32

    def test_closed_eyes_low_ear(self):
        landmarks = make_landmarks_with_ear(0.10)
        ear = compute_ear_both_eyes(landmarks)
        assert 0.08 <= ear <= 0.12

    def test_zero_horizontal_returns_zero(self):
        # Both eye corners at the same point
        landmarks = [(0.5, 0.5)] * 468
        ear = compute_ear(landmarks, LEFT_EYE_IDX)
        assert ear == 0.0


# ------------------------------------------------------------------
# Tests: Game phases
# ------------------------------------------------------------------

class TestGamePhases:
    """Test the game lifecycle."""

    def test_starts_in_waiting(self):
        game = Game(make_config(), seed=42)
        state = game.update(0.3, True, 0.0)
        assert state.phase == Phase.WAITING

    def test_start_begins_countdown(self):
        game = Game(make_config(), seed=42)
        state = game.start(0.0)
        assert state.phase == Phase.COUNTDOWN
        assert state.countdown_value == 3

    def test_countdown_progresses(self):
        game = Game(make_config(countdown_seconds=3), seed=42)
        game.start(0.0)

        state = game.update(0.3, True, 1.0)
        assert state.countdown_value == 2

        state = game.update(0.3, True, 2.0)
        assert state.countdown_value == 1

    def test_countdown_ends_starts_playing(self):
        game = Game(make_config(countdown_seconds=3), seed=42)
        game.start(0.0)

        state = game.update(0.3, True, 3.0)
        assert state.phase == Phase.PLAYING

    def test_face_lost_during_play(self):
        """Face leaving the screen is now instant death."""
        game = Game(make_config(countdown_seconds=0), seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)  # start playing

        state = game.update(0.3, False, 1.0)
        assert state.phase == Phase.BLINKED
        assert state.death_reason == "face_lost"
        assert game.is_game_over()

    def test_face_regained_resumes(self):
        """Face lost = game over, no resuming."""
        game = Game(make_config(countdown_seconds=0), seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)
        game.update(0.3, False, 1.0)  # dead

        # Can't resume — game is over
        state = game.update(0.3, True, 2.0)
        assert state.phase == Phase.BLINKED


# ------------------------------------------------------------------
# Tests: Blink detection
# ------------------------------------------------------------------

class TestBlinkDetection:
    """Test the blink detection with hysteresis."""

    def test_single_low_frame_no_blink(self):
        """Need blink_frames consecutive frames to trigger."""
        game = Game(make_config(countdown_seconds=0, blink_frames=2), seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)  # start playing

        # One frame below threshold — not enough
        state = game.update(0.15, True, 1.0)
        assert state.phase == Phase.PLAYING

    def test_consecutive_frames_triggers_blink(self):
        game = Game(make_config(countdown_seconds=0, blink_frames=2), seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)  # start playing

        # Two consecutive frames below threshold
        game.update(0.15, True, 1.0)
        state = game.update(0.15, True, 1.1)
        assert state.phase == Phase.BLINKED
        assert game.is_game_over()

    def test_interrupted_close_resets_counter(self):
        """If eyes open briefly between frames, counter resets."""
        game = Game(make_config(countdown_seconds=0, blink_frames=3), seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)

        game.update(0.15, True, 1.0)  # frame 1 closed
        game.update(0.15, True, 1.1)  # frame 2 closed
        game.update(0.30, True, 1.2)  # open! resets
        game.update(0.15, True, 1.3)  # frame 1 closed again

        state = game.update(0.15, True, 1.4)  # frame 2 — still needs one more
        assert state.phase == Phase.PLAYING

    def test_survival_time_recorded(self):
        game = Game(make_config(countdown_seconds=0, blink_frames=1), seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)  # start playing

        # Survive 5 seconds then blink
        game.update(0.3, True, 5.1)
        state = game.update(0.15, True, 5.2)
        assert state.phase == Phase.BLINKED
        assert 5.0 <= state.survival_time <= 5.2


# ------------------------------------------------------------------
# Tests: Distractions
# ------------------------------------------------------------------

class TestDistractions:
    """Test distraction scheduling and escalation."""

    def test_no_distraction_before_delay(self):
        config = make_config(countdown_seconds=0, distraction_start_delay=5.0)
        game = Game(config, seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)

        state = game.update(0.3, True, 3.0)
        assert state.active_distraction is None

    def test_distraction_fires_after_delay(self):
        config = make_config(
            countdown_seconds=0,
            distraction_start_delay=2.0,
            distraction_interval_initial=2.0,
        )
        game = Game(config, seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)  # start playing, game_start = 0.1

        # next_distraction_time = 0.1 + 2.0 = 2.1
        state = game.update(0.3, True, 2.2)
        assert state.active_distraction is not None

    def test_distraction_expires(self):
        config = make_config(
            countdown_seconds=0,
            distraction_start_delay=1.0,
            distraction_duration=0.3,
            distraction_interval_initial=10.0,  # long so only one fires
        )
        game = Game(config, seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)

        # Trigger distraction
        game.update(0.3, True, 1.2)
        # Wait for it to expire (0.3s duration)
        state = game.update(0.3, True, 1.6)
        assert state.active_distraction is None

    def test_distractions_get_faster(self):
        config = make_config(
            countdown_seconds=0,
            distraction_start_delay=1.0,
            distraction_interval_initial=2.0,
            distraction_interval_decay=0.5,
            distraction_interval_min=0.5,
            distraction_duration=0.1,
        )
        game = Game(config, seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)

        # First distraction at ~1.1
        game.update(0.3, True, 1.2)
        # After first, interval becomes 2.0 * 0.5 = 1.0
        # Next at ~1.2 + 1.0 = 2.2
        game.update(0.3, True, 1.4)  # expire first
        state = game.update(0.3, True, 2.3)
        assert state.active_distraction is not None


# ------------------------------------------------------------------
# Tests: High score and restart
# ------------------------------------------------------------------

class TestHighScore:
    """Test high score tracking and restart."""

    def test_high_score_updates(self):
        game = Game(make_config(countdown_seconds=0, blink_frames=1), seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)

        # Survive 3s
        game.update(0.3, True, 3.1)
        game.update(0.15, True, 3.2)  # blink
        assert game._high_score >= 3.0

    def test_restart_resets_state(self):
        game = Game(make_config(countdown_seconds=0, blink_frames=1), seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)
        game.update(0.15, True, 1.0)  # blink

        state = game.restart(2.0)
        assert state.phase == Phase.WAITING
        assert state.survival_time == 0.0

    def test_high_score_persists_after_restart(self):
        game = Game(make_config(countdown_seconds=0, blink_frames=1), seed=42)
        game.start(0.0)
        game.update(0.3, True, 0.1)
        game.update(0.3, True, 5.1)
        game.update(0.15, True, 5.2)  # blink at ~5s

        old_high = game._high_score
        game.restart(6.0)
        assert game._high_score == old_high
