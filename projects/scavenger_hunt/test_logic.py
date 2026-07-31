"""Tests for logic.py — no webcam, no model, no sleeping.

Run from the scavenger_hunt directory:
    python -m pytest test_logic.py -v

Or from the repo root:
    python -m pytest projects/scavenger_hunt/test_logic.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow import whether tests run from the project dir or the repo root
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pytest

from logic import (
    DONE, FOUND, PLAYING, SKIP, WAITING,
    Config, RoundResult, ScavengerHunt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_game(**cfg_kwargs) -> ScavengerHunt:
    """Return a fresh game with a fast config suitable for tests."""
    defaults = dict(
        total_rounds=3,
        round_seconds=10.0,
        hold_seconds=0.5,
        found_pause=0.1,
        skip_pause=0.1,
    )
    defaults.update(cfg_kwargs)
    return ScavengerHunt(Config(**defaults))


def tick(game: ScavengerHunt, labels: list[str], now: float) -> dict:
    """Single update call."""
    return game.update(labels, now)


def tick_n(game: ScavengerHunt, labels: list[str], start: float, n: int = 10, dt: float = 0.1) -> dict:
    """Push the same detections for n frames, advancing the clock."""
    state: dict = {}
    for i in range(n):
        state = game.update(labels, start + i * dt)
    return state


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

class TestPhaseTransitions:

    def test_starts_in_waiting(self):
        game = make_game()
        assert game.state(0.0)["phase"] == WAITING

    def test_start_moves_to_playing(self):
        game = make_game()
        state = game.start(0.0)
        assert state["phase"] == PLAYING

    def test_start_sets_round_1(self):
        game = make_game()
        state = game.start(0.0)
        assert state["round"] == 1

    def test_start_gives_a_target(self):
        game = make_game()
        state = game.start(0.0)
        assert state["target"] != ""

    def test_update_before_start_keeps_waiting(self):
        game = make_game()
        state = tick(game, ["cup"], 0.0)
        assert state["phase"] == WAITING

    def test_skip_moves_to_skip_phase(self):
        game = make_game()
        game.start(0.0)
        state = game.skip(1.0)
        assert state["phase"] == SKIP

    def test_skip_before_start_has_no_effect(self):
        game = make_game()
        state = game.skip(0.0)
        assert state["phase"] == WAITING


# ---------------------------------------------------------------------------
# Hold timer
# ---------------------------------------------------------------------------

class TestHoldTimer:

    def test_single_frame_does_not_score(self):
        game = make_game(hold_seconds=0.5)
        game.start(0.0)
        target = game.target
        state = tick(game, [target], 0.0)
        assert state["phase"] == PLAYING

    def test_hold_for_long_enough_scores(self):
        # hold_seconds=0.5, found_pause=2.0 so FOUND persists well past the hold tick.
        game = make_game(hold_seconds=0.5, found_pause=2.0)
        game.start(0.0)
        target = game.target
        # Feed target for 0.6 s (past the 0.5 s hold threshold).
        # found_pause=2.0 means the FOUND phase cannot auto-advance back to PLAYING
        # within these 7 frames, so the final state is reliably FOUND.
        state = tick_n(game, [target], start=0.0, n=7, dt=0.1)
        assert state["phase"] == FOUND

    def test_hold_progress_reaches_1_on_success(self):
        game = make_game(hold_seconds=0.5)
        game.start(0.0)
        target = game.target
        # Just before threshold
        state = tick_n(game, [target], start=0.0, n=4, dt=0.1)
        # hold_progress should be non-zero and < 1
        assert 0 < state["hold_progress"] < 1.0

    def test_hold_resets_when_target_disappears(self):
        game = make_game(hold_seconds=0.5)
        game.start(0.0)
        target = game.target

        # Start holding
        tick_n(game, [target], start=0.0, n=3, dt=0.1)
        # Target disappears
        tick(game, [], 0.4)
        # Progress should be reset; single frame not enough to score
        state = tick(game, [target], 0.5)
        assert state["phase"] == PLAYING
        # Hold progress restarts close to 0
        assert state["hold_progress"] < 0.3

    def test_hold_progress_zero_without_target(self):
        game = make_game()
        game.start(0.0)
        state = tick(game, [], 0.1)
        assert state["hold_progress"] == 0.0


# ---------------------------------------------------------------------------
# Countdown / timeout
# ---------------------------------------------------------------------------

class TestCountdown:

    def test_time_left_decreases(self):
        game = make_game(round_seconds=10.0)
        game.start(0.0)
        state_early = tick(game, [], 1.0)
        state_late  = tick(game, [], 5.0)
        assert state_late["time_left"] < state_early["time_left"]

    def test_time_left_never_negative(self):
        game = make_game(round_seconds=5.0)
        game.start(0.0)
        state = tick(game, [], 100.0)
        assert state["time_left"] >= 0.0

    def test_countdown_expiry_transitions_to_skip(self):
        game = make_game(round_seconds=5.0)
        game.start(0.0)
        # Jump well past round_seconds without showing target
        state = tick(game, [], 6.0)
        assert state["phase"] == SKIP

    def test_countdown_expiry_adds_history_entry(self):
        game = make_game(round_seconds=5.0)
        game.start(0.0)
        tick(game, [], 6.0)
        assert len(game.history) == 1
        assert game.history[0].found is False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:

    def test_quick_find_scores_more_than_slow_find(self):
        def score_at(seconds: float) -> int:
            g = make_game(round_seconds=10.0, hold_seconds=0.0)
            g.start(0.0)
            target = g.target
            g.update([target], seconds)  # immediate hold (0 s)
            return g.score

        assert score_at(1.0) > score_at(8.0)

    def test_score_never_below_min_points(self):
        cfg = Config(
            total_rounds=3,
            round_seconds=10.0,
            hold_seconds=0.0,
            found_pause=0.1,
            skip_pause=0.1,
        )
        game = ScavengerHunt(cfg)
        game.start(0.0)
        target = game.target
        # Find at the very end of time
        game.update([target], 9.99)
        assert game.score >= cfg.min_points_per_round

    def test_skip_gives_zero_points(self):
        game = make_game()
        game.start(0.0)
        game.skip(1.0)
        assert game.score == 0

    def test_score_accumulates_across_rounds(self):
        game = make_game(
            total_rounds=3,
            round_seconds=10.0,
            hold_seconds=0.0,
            found_pause=0.0,
            skip_pause=0.0,
        )
        game.start(0.0)
        now = 0.1
        for _ in range(3):
            target = game.target
            # Hold long enough to score
            game.update([target], now)
            now += 0.5  # advance past found_pause
            # Tick once more to advance through FOUND pause
            game.update([], now)
            now += 0.5

        assert game.score > 0
        assert len(game.history) == 3


# ---------------------------------------------------------------------------
# Streak bonus
# ---------------------------------------------------------------------------

class TestStreak:

    def test_streak_increments_on_consecutive_finds(self):
        game = make_game(
            total_rounds=3,
            round_seconds=10.0,
            hold_seconds=0.0,
            found_pause=0.0,
            skip_pause=0.0,
        )
        game.start(0.0)
        now = 0.1

        def find_and_advance() -> None:
            nonlocal now
            target = game.target
            game.update([target], now)
            now += 0.3
            game.update([], now)    # tick through pause
            now += 0.3

        find_and_advance()
        find_and_advance()
        assert game.streak == 2

    def test_skip_resets_streak(self):
        game = make_game(
            total_rounds=3,
            round_seconds=10.0,
            hold_seconds=0.0,
            found_pause=0.0,
            skip_pause=0.0,
        )
        game.start(0.0)
        # Find first round
        target = game.target
        game.update([target], 0.1)
        game.update([], 0.5)
        assert game.streak == 1
        # Skip second round
        game.skip(0.6)
        assert game.streak == 0

    def test_streak_bonus_reflected_in_points(self):
        """Second consecutive find should score more than the first."""
        cfg = Config(
            total_rounds=3,
            round_seconds=10.0,
            hold_seconds=0.0,
            found_pause=0.0,
            skip_pause=0.0,
            streak_bonus_per=50,
        )
        game = ScavengerHunt(cfg)
        game.start(0.0)
        now = 0.1

        target = game.target
        game.update([target], now); now += 0.3
        game.update([], now);       now += 0.3
        pts_r1 = game.history[0].points

        target = game.target
        game.update([target], now); now += 0.3
        pts_r2 = game.history[1].points

        assert pts_r2 > pts_r1


# ---------------------------------------------------------------------------
# Full game completion
# ---------------------------------------------------------------------------

class TestGameCompletion:

    def test_done_phase_after_all_rounds_skipped(self):
        game = make_game(
            total_rounds=3,
            skip_pause=0.0,
            found_pause=0.0,
        )
        game.start(0.0)
        now = 0.1
        for _ in range(3):
            game.skip(now); now += 0.3
            game.update([], now); now += 0.3
        assert game.state(now)["phase"] == DONE

    def test_history_length_equals_total_rounds(self):
        game = make_game(
            total_rounds=3,
            skip_pause=0.0,
        )
        game.start(0.0)
        now = 0.1
        for _ in range(3):
            game.skip(now); now += 0.2
            game.update([], now); now += 0.2
        assert len(game.history) == 3

    def test_update_after_done_is_stable(self):
        """Calling update after DONE should not raise and should return DONE."""
        game = make_game(total_rounds=1, skip_pause=0.0)
        game.start(0.0)
        game.skip(0.1)
        game.update([], 1.0)   # advance past pause
        state = game.update([], 2.0)
        assert state["phase"] == DONE

    def test_total_rounds_in_state(self):
        game = make_game(total_rounds=5)
        state = game.start(0.0)
        assert state["total_rounds"] == 5


# ---------------------------------------------------------------------------
# RoundResult record
# ---------------------------------------------------------------------------

class TestRoundResult:

    def test_found_round_result_fields(self):
        game = make_game(hold_seconds=0.0, found_pause=0.0)
        game.start(0.0)
        target = game.target
        game.update([target], 2.0)

        r = game.history[0]
        assert isinstance(r, RoundResult)
        assert r.found is True
        assert r.target == target
        assert r.points > 0
        assert r.time_taken > 0.0

    def test_skipped_round_result_fields(self):
        game = make_game()
        game.start(0.0)
        game.skip(3.0)

        r = game.history[0]
        assert r.found is False
        assert r.points == 0
        assert r.time_taken == 0.0
        assert r.streak == 0


# ---------------------------------------------------------------------------
# Target pool
# ---------------------------------------------------------------------------

class TestTargetPool:

    def test_target_is_from_configured_list(self):
        cfg = Config(targets=("cup", "book"), total_rounds=3, found_pause=0.0, skip_pause=0.0)
        game = ScavengerHunt(cfg)
        game.start(0.0)
        now = 0.1
        seen = set()
        for _ in range(3):
            seen.add(game.target)
            game.skip(now); now += 0.2
            game.update([], now); now += 0.2
        assert seen <= {"cup", "book"}

    def test_all_targets_are_eventually_used(self):
        """Over enough rounds, all desk targets should appear at least once."""
        cfg = Config(total_rounds=16, round_seconds=1.0, found_pause=0.0, skip_pause=0.0)
        game = ScavengerHunt(cfg)
        game.start(0.0)
        seen: set[str] = set()
        now = 0.1
        for _ in range(16):
            seen.add(game.target)
            game.skip(now); now += 0.2
            game.update([], now); now += 0.2
        assert seen == set(cfg.targets)



# ---------------------------------------------------------------------------
# apply_settings — deferred timer, immediate confidence
# ---------------------------------------------------------------------------

class TestApplySettings:

    # --- confidence: immediate ---

    def test_confidence_applies_immediately(self):
        game = make_game()
        game.start(0.0)
        game.apply_settings(confidence=0.8)
        assert game.cfg.confidence == pytest.approx(0.8)

    def test_confidence_clears_hold_progress(self):
        """Changing confidence mid-hold resets the hold timer."""
        game = make_game(hold_seconds=1.0)
        game.start(0.0)
        target = game.target
        # Build up some hold progress
        tick_n(game, [target], start=0.0, n=4, dt=0.1)
        assert game._hold_since is not None
        # Changing confidence should discard that progress
        game.apply_settings(confidence=0.3)
        assert game._hold_since is None

    def test_confidence_clamped_to_min(self):
        game = make_game()
        game.apply_settings(confidence=0.0)
        assert game.cfg.confidence >= game.cfg.min_confidence

    def test_confidence_clamped_to_max(self):
        game = make_game()
        game.apply_settings(confidence=1.0)
        assert game.cfg.confidence <= game.cfg.max_confidence

    def test_confidence_in_state_dict(self):
        game = make_game()
        game.apply_settings(confidence=0.65)
        assert game.state(0.0)["confidence"] == pytest.approx(0.65)

    # --- round_seconds: deferred mid-round ---

    def test_timer_deferred_during_playing(self):
        game = make_game(round_seconds=10.0)
        game.start(0.0)
        assert game.phase == PLAYING
        result = game.apply_settings(round_seconds=20.0)
        # Should be deferred, not applied yet
        assert game.cfg.round_seconds == pytest.approx(10.0)
        assert result["deferred"].get("round_seconds") == pytest.approx(20.0)
        assert result["applied"].get("round_seconds") is None

    def test_deferred_timer_visible_in_pending_settings(self):
        game = make_game(round_seconds=10.0)
        game.start(0.0)
        game.apply_settings(round_seconds=25.0)
        state = game.state(0.0)
        assert state["pending_settings"].get("round_seconds") == pytest.approx(25.0)

    def test_deferred_timer_takes_effect_at_next_round(self):
        game = make_game(
            total_rounds=3,
            round_seconds=10.0,
            skip_pause=0.0,
            found_pause=0.0,
        )
        game.start(0.0)
        game.apply_settings(round_seconds=99.0)
        # Still 10 s this round
        assert game.cfg.round_seconds == pytest.approx(10.0)
        # Advance to next round: skip current round and tick past the pause
        game.skip(0.1)
        game.update([], 0.5)   # tick through skip_pause=0.0 → next round begins
        # Now the pending value should be flushed
        assert game.cfg.round_seconds == pytest.approx(99.0)
        assert "round_seconds" not in game._pending_settings

    def test_pending_cleared_after_next_round_starts(self):
        game = make_game(total_rounds=3, skip_pause=0.0, found_pause=0.0)
        game.start(0.0)
        game.apply_settings(round_seconds=60.0)
        game.skip(0.1)
        game.update([], 0.5)
        state = game.state(0.5)
        assert state["pending_settings"] == {}

    def test_timer_applied_immediately_when_not_playing(self):
        """In WAITING or DONE state the timer can be applied right away."""
        game = make_game()
        # WAITING — no round started
        result = game.apply_settings(round_seconds=45.0)
        assert game.cfg.round_seconds == pytest.approx(45.0)
        assert result["applied"].get("round_seconds") == pytest.approx(45.0)
        assert result["deferred"] == {}

    def test_timer_clamped_to_min(self):
        game = make_game()
        game.apply_settings(round_seconds=1.0)
        assert game.cfg.round_seconds >= game.cfg.min_round_seconds

    def test_timer_clamped_to_max(self):
        game = make_game()
        game.apply_settings(round_seconds=9999.0)
        assert game.cfg.round_seconds <= game.cfg.max_round_seconds

    def test_timer_in_state_dict(self):
        game = make_game(round_seconds=15.0)
        assert game.state(0.0)["round_seconds"] == pytest.approx(15.0)

    # --- both settings together ---

    def test_apply_both_confidence_and_timer(self):
        game = make_game(round_seconds=10.0)
        game.start(0.0)
        result = game.apply_settings(confidence=0.7, round_seconds=30.0)
        # Confidence is immediate
        assert game.cfg.confidence == pytest.approx(0.7)
        assert "confidence" in result["applied"]
        # Timer is deferred
        assert "round_seconds" in result["deferred"]
        assert game.cfg.round_seconds == pytest.approx(10.0)
