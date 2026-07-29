"""Tests for the 6-7 counter: wrist see-saw sign flips, prepare mode, tempo."""

from __future__ import annotations

import numpy as np
import pytest

from demos.common import landmarks as lm
from demos.six_seven_counter.config import CounterConfig
from demos.six_seven_counter.core import (
    COUNTING,
    PREPARE,
    SixSevenCore,
    TiltTracker,
    wrist_tilt,
)
from demos.tests.fixtures import make_frame, make_pose

# Wrist heights in torso lengths above the shoulder line. The pair only has to
# differ by more than CounterConfig.tilt_enter; absolute height is irrelevant to
# this algorithm, which is the point of it.
# Kept modest so that even with a long torso, or with a drift offset applied, the
# wrists stay inside the frame: the visibility gate rejects landmarks that leave
# it, which would otherwise make these tests pass for the wrong reason.
HIGH = 0.15
LOW = -0.15
# Kept for the app tests, which drive frames through the same helpers.
UP, DOWN = HIGH, LOW


def frame_at(left: float, right: float) -> lm.LandmarkFrame:
    """A pose frame with the given wrist heights."""
    return make_frame(pose=make_pose(left_wrist_up=left, right_wrist_up=right))


def tilted(higher: str, *, offset: float = 0.0) -> lm.LandmarkFrame:
    """One hand clearly above the other. ``offset`` raises *both* wrists."""
    left, right = (HIGH, LOW) if higher == "left" else (LOW, HIGH)
    return frame_at(left + offset, right + offset)


class Clock:
    """Feeds frames with a steadily advancing timestamp."""

    def __init__(self, core: SixSevenCore, step: float = 0.1):
        self.core = core
        self.step = step
        self.now = 0.0

    def feed(self, frame: lm.LandmarkFrame, times: int = 4) -> dict:
        state: dict = {}
        for _ in range(times):
            state = self.core.update(frame, self.now)
            self.now += self.step
        return state

    def level(self, times: int = 4) -> dict:
        return self.feed(frame_at(0.0, 0.0), times)

    def swap(self, higher: str, *, times: int = 4, offset: float = 0.0) -> dict:
        return self.feed(tilted(higher, offset=offset), times)

    def idle(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def core() -> SixSevenCore:
    """A counter past the prepare hold, so tests exercise counting."""
    counter = SixSevenCore(CounterConfig())
    counter.handle_command("start", {})
    return counter


class TestWristTilt:
    def test_sign_says_which_wrist_is_higher(self):
        assert wrist_tilt(make_pose(left_wrist_up=HIGH, right_wrist_up=LOW)) > 0
        assert wrist_tilt(make_pose(left_wrist_up=LOW, right_wrist_up=HIGH)) < 0

    def test_level_wrists_are_zero(self):
        assert wrist_tilt(make_pose(left_wrist_up=0.2, right_wrist_up=0.2)) == pytest.approx(0.0)

    def test_magnitude_is_the_height_difference_in_torso_lengths(self):
        tilt = wrist_tilt(make_pose(left_wrist_up=0.3, right_wrist_up=-0.3))
        assert tilt == pytest.approx(0.6, rel=1e-6)

    def test_is_scale_invariant(self):
        near = wrist_tilt(make_pose(left_wrist_up=HIGH, right_wrist_up=LOW, shoulder_y=0.2, hip_y=0.95))
        far = wrist_tilt(make_pose(left_wrist_up=HIGH, right_wrist_up=LOW, shoulder_y=0.45, hip_y=0.60))
        assert near == pytest.approx(far, rel=1e-6)

    def test_raising_both_wrists_together_does_not_change_it(self):
        base = wrist_tilt(make_pose(left_wrist_up=HIGH, right_wrist_up=LOW))
        lifted = wrist_tilt(make_pose(left_wrist_up=HIGH + 0.5, right_wrist_up=LOW + 0.5))
        assert base == pytest.approx(lifted, rel=1e-6)


class TestBodyScale:
    def test_prefers_shoulder_to_hip_when_the_hips_are_visible(self):
        from demos.six_seven_counter.core import body_scale
        from demos.common import gestures as gs

        pose = make_pose()
        assert body_scale(pose) == pytest.approx(gs.torso_scale(pose))

    def test_falls_back_to_shoulder_width_without_hips(self):
        from demos.six_seven_counter.core import body_scale

        pose = make_pose()
        for name in ("left_hip", "right_hip"):
            pose.points[lm.POSE[name]] = (0.5, 1.9)
        config = CounterConfig()
        expected = 0.28 * config.shoulder_to_torso  # fixture shoulder width
        assert body_scale(pose, config) == pytest.approx(expected, rel=1e-3)

    def test_fallback_is_still_scale_invariant(self):
        from demos.six_seven_counter.core import wrist_tilt

        def tilt(center_scale: float) -> float:
            pose = make_pose(left_wrist_up=HIGH, right_wrist_up=LOW)
            # Shrink the whole body toward its centre, hips pushed out of frame.
            pose.points = [(0.5 + (x - 0.5) * center_scale, 0.5 + (y - 0.5) * center_scale)
                           for x, y in pose.points]
            for name in ("left_hip", "right_hip"):
                pose.points[lm.POSE[name]] = (0.5, 1.9)
            return wrist_tilt(pose)

        assert tilt(1.0) == pytest.approx(tilt(0.5), rel=1e-6)


class TestTiltTracker:
    def test_first_side_establishes_a_baseline_without_a_flip(self):
        tracker = TiltTracker(CounterConfig())
        assert tracker.update(0.6, 0.0) is None
        assert tracker.side == "left"

    def test_crossing_to_the_other_side_reports_a_flip(self):
        tracker = TiltTracker(CounterConfig())
        for step in range(6):
            tracker.update(0.6, step * 0.1)
        flipped = None
        for step in range(6):
            flipped = tracker.update(-0.6, 1.0 + step * 0.1) or flipped
        assert flipped == "right"
        assert tracker.side == "right"

    def test_deadband_holds_the_current_side(self):
        config = CounterConfig(tilt_alpha=1.0)
        tracker = TiltTracker(config)
        tracker.update(0.6, 0.0)
        for value in (0.10, 0.0, -0.10, 0.05):
            assert tracker.update(value, 1.0) is None
        assert tracker.side == "left"

    def test_noise_inside_the_deadband_never_flips(self):
        config = CounterConfig(tilt_alpha=1.0)
        tracker = TiltTracker(config)
        tracker.update(0.6, 0.0)
        flips = 0
        for step in range(60):
            value = 0.14 if step % 2 else -0.14
            if tracker.update(value, 1.0 + step * 0.05):
                flips += 1
        assert flips == 0

    def test_missing_tilt_holds_state(self):
        tracker = TiltTracker(CounterConfig())
        tracker.update(0.6, 0.0)
        assert tracker.update(None, 1.0) is None
        assert tracker.side == "left"

    def test_min_swap_seconds_rejects_an_impossibly_fast_flip(self):
        config = CounterConfig(tilt_alpha=1.0, min_swap_seconds=1.0)
        tracker = TiltTracker(config)
        tracker.update(0.6, 0.0)
        assert tracker.update(-0.6, 0.1) is None, "0.1 s is faster than a real swap"
        assert tracker.update(-0.6, 2.0) == "right"

    def test_magnitude_grows_past_the_deadband(self):
        config = CounterConfig(tilt_alpha=1.0)
        tracker = TiltTracker(config)
        tracker.update(0.0, 0.0)
        assert tracker.magnitude == pytest.approx(0.0)
        tracker.update(config.tilt_enter, 0.1)
        assert tracker.magnitude == pytest.approx(0.5)
        tracker.update(config.tilt_enter * 4, 0.2)
        assert tracker.magnitude == pytest.approx(1.0)

    def test_reset_clears_the_side(self):
        tracker = TiltTracker(CounterConfig())
        tracker.update(0.6, 0.0)
        tracker.reset()
        assert tracker.side is None
        assert tracker.tilt is None


class TestCounting:
    def test_first_raise_does_not_count(self, core):
        clock = Clock(core)
        clock.swap("left")
        assert core.count == 0
        assert core.side == "left"

    def test_each_swap_counts_once(self, core):
        clock = Clock(core)
        clock.swap("left")
        clock.swap("right")
        assert core.count == 1
        clock.swap("left")
        assert core.count == 2

    def test_ten_swaps_count_ten(self, core):
        clock = Clock(core)
        clock.swap("left")  # baseline
        for index in range(10):
            clock.swap("right" if index % 2 == 0 else "left")
        assert core.count == 10

    def test_holding_one_hand_up_does_not_accumulate(self, core):
        clock = Clock(core)
        clock.swap("left", times=40)
        assert core.count == 0

    def test_same_hand_twice_cannot_count(self, core):
        """Intrinsic to the algorithm: a flip needs both hands to take part."""
        clock = Clock(core)
        clock.swap("left")
        clock.level()
        clock.swap("left")
        clock.level()
        clock.swap("left")
        assert core.count == 0

    def test_level_hands_never_count(self, core):
        clock = Clock(core)
        clock.level(times=40)
        assert core.count == 0
        assert core.side is None

    def test_returning_through_level_still_counts_the_swap(self, core):
        clock = Clock(core)
        clock.swap("left")
        clock.level()
        clock.swap("right")
        assert core.count == 1

    def test_moving_the_whole_body_up_does_not_count(self, core):
        """Both wrists rising together cancels out: only the difference matters."""
        clock = Clock(core)
        clock.swap("left")
        for offset in (0.1, 0.2, 0.3, 0.4):
            clock.swap("left", offset=offset)
            assert core.state(clock.now)["poseVisible"] is True
        assert core.count == 0

    def test_swaps_count_while_the_body_drifts(self, core):
        clock = Clock(core)
        clock.swap("left", offset=0.0)
        clock.swap("right", offset=0.2)
        clock.swap("left", offset=0.4)
        assert core.count == 2

    def test_expected_hand_is_the_other_one(self, core):
        clock = Clock(core)
        assert core.expected_hand is None
        clock.swap("left")
        assert core.expected_hand == "right"
        clock.swap("right")
        assert core.expected_hand == "left"

    def test_hint_names_the_next_hand(self, core):
        clock = Clock(core)
        state = clock.swap("left")
        assert "RIGHT" in state["hint"]

    def test_side_and_hands_agree(self, core):
        clock = Clock(core)
        state = clock.swap("left")
        assert state["side"] == "left"
        assert state["hands"]["left"]["up"] is True
        assert state["hands"]["right"]["up"] is False


class TestTempo:
    def test_last_swap_duration_is_measured(self, core):
        clock = Clock(core)
        clock.swap("left")
        clock.swap("right")
        clock.swap("left")
        assert core.last_swap_seconds is not None
        assert core.last_swap_seconds > 0

    def test_tempo_needs_two_counted_swaps(self, core):
        clock = Clock(core)
        clock.swap("left")
        clock.swap("right")
        assert core.reps_per_minute() is None
        clock.swap("left")
        assert core.reps_per_minute() is not None
        assert core.reps_per_minute() > 0

    def test_history_is_bounded(self, core):
        clock = Clock(core)
        clock.swap("left")
        for index in range(70):
            clock.swap("right" if index % 2 == 0 else "left")
        assert len(core.history) <= 30


class TestVisibility:
    def test_no_pose_freezes_the_count(self, core):
        clock = Clock(core)
        clock.swap("left")
        clock.swap("right")
        assert core.count == 1
        for _ in range(5):
            core.update(make_frame(), clock.now)
            clock.now += 0.1
        assert core.count == 1
        assert core.state(clock.now)["poseVisible"] is False

    def test_low_visibility_is_treated_as_no_pose(self, core):
        pose = make_pose(left_wrist_up=HIGH, right_wrist_up=LOW)
        pose.visibility = [0.1] * len(pose.visibility)
        assert core.update(make_frame(pose=pose), 0.0)["poseVisible"] is False

    def test_counting_survives_a_brief_dropout(self, core):
        clock = Clock(core)
        clock.swap("left")
        core.update(make_frame(), clock.now)
        clock.now += 0.2
        clock.swap("right")
        assert core.count == 1

    def test_thresholds_are_scale_invariant(self):
        def run(shoulder_y: float, hip_y: float) -> int:
            counter = SixSevenCore(CounterConfig())
            counter.handle_command("start", {})
            now = 0.0
            for index in range(5):
                higher = "left" if index % 2 == 0 else "right"
                left, right = (HIGH, LOW) if higher == "left" else (LOW, HIGH)
                for _ in range(4):
                    counter.update(
                        make_frame(
                            pose=make_pose(
                                left_wrist_up=left,
                                right_wrist_up=right,
                                shoulder_y=shoulder_y,
                                hip_y=hip_y,
                            )
                        ),
                        now,
                    )
                    now += 0.1
            return counter.count

        assert run(0.20, 0.95) == run(0.45, 0.60) == 4


class TestPrepareMode:
    def _core(self, **overrides) -> SixSevenCore:
        return SixSevenCore(CounterConfig(**overrides))

    def test_starts_in_prepare(self):
        counter = self._core()
        assert counter.phase == PREPARE
        state = counter.update(make_frame(), 0.0)
        assert state["phase"] == PREPARE
        assert state["hint"] == "STEP INTO FRAME"

    def test_prepare_completes_after_the_hold(self):
        counter = self._core(prepare_seconds=1.0)
        clock = Clock(counter)
        clock.level(times=5)
        assert counter.phase == PREPARE
        assert 0.0 < counter.prepare_progress(clock.now) < 1.0
        clock.level(times=8)
        assert counter.phase == COUNTING

    def test_swaps_during_prepare_do_not_count(self):
        counter = self._core(prepare_seconds=5.0)
        clock = Clock(counter)
        clock.swap("left")
        clock.swap("right")
        clock.swap("left")
        assert counter.phase == PREPARE
        assert counter.count == 0

    def test_counting_starts_cleanly_after_prepare(self):
        counter = self._core(prepare_seconds=0.5)
        clock = Clock(counter)
        clock.level(times=8)
        assert counter.phase == COUNTING
        clock.swap("left")
        clock.swap("right")
        assert counter.count == 1

    def test_hips_are_not_required(self):
        """Standing close enough to fill the frame must still work."""
        counter = self._core()
        pose = make_pose(left_wrist_up=LOW, right_wrist_up=LOW)
        for name in ("left_hip", "right_hip"):
            pose.points[lm.POSE[name]] = (0.5, 1.6)  # below the frame
        state = counter.update(make_frame(pose=pose), 0.0)
        assert state["poseVisible"] is True
        assert state["missing"] == []

    def test_missing_hand_names_the_problem(self):
        counter = self._core()
        pose = make_pose(left_wrist_up=LOW, right_wrist_up=LOW)
        pose.points[lm.POSE["left_wrist"]] = (0.5, 1.4)
        state = counter.update(make_frame(pose=pose), 0.0)
        assert state["hint"] == "SHOW BOTH HANDS"
        assert "left_wrist" in state["missing"]

    def test_missing_shoulder_names_the_problem(self):
        counter = self._core()
        pose = make_pose(left_wrist_up=LOW, right_wrist_up=LOW)
        pose.points[lm.POSE["right_shoulder"]] = (1.5, 0.3)
        state = counter.update(make_frame(pose=pose), 0.0)
        assert "SHOULDERS" in state["hint"]

    def test_offscreen_wrist_blocks_counting(self):
        counter = self._core()
        pose = make_pose(left_wrist_up=HIGH, right_wrist_up=LOW)
        pose.points[lm.POSE["right_wrist"]] = (1.4, 0.5)
        state = counter.update(make_frame(pose=pose), 0.0)
        assert state["poseVisible"] is False
        assert "HANDS" in state["hint"]

    def test_counting_works_with_the_hips_out_of_frame(self):
        """The scale reference falls back to shoulder width."""
        counter = self._core(prepare_seconds=0.0)
        now = 0.0
        count_before = counter.count
        for index in range(5):
            higher = "left" if index % 2 == 0 else "right"
            left, right = (HIGH, LOW) if higher == "left" else (LOW, HIGH)
            for _ in range(4):
                pose = make_pose(left_wrist_up=left, right_wrist_up=right)
                for name in ("left_hip", "right_hip"):
                    pose.points[lm.POSE[name]] = (0.5, 1.8)
                counter.update(make_frame(pose=pose), now)
                now += 0.1
        assert counter.count > count_before
        assert counter.count == 4

    def test_brief_dropout_does_not_force_a_new_hold(self):
        counter = self._core(prepare_seconds=0.5, lost_grace_seconds=0.8)
        clock = Clock(counter)
        clock.level(times=8)
        assert counter.phase == COUNTING
        counter.update(make_frame(), clock.now)
        clock.now += 0.1
        assert counter.phase == COUNTING

    def test_long_dropout_requires_a_new_hold(self):
        counter = self._core(prepare_seconds=0.5, lost_grace_seconds=0.5)
        clock = Clock(counter)
        clock.level(times=8)
        for _ in range(10):
            counter.update(make_frame(), clock.now)
            clock.now += 0.1
        assert counter.phase == PREPARE

    def test_prepare_command_pauses_counting(self):
        counter = self._core(prepare_seconds=5.0)
        counter.handle_command("start", {})
        assert counter.phase == COUNTING
        counter.handle_command("prepare", {})
        assert counter.phase == PREPARE

    def test_count_survives_a_return_to_prepare(self):
        counter = self._core(prepare_seconds=0.5, lost_grace_seconds=0.1)
        clock = Clock(counter)
        clock.level(times=8)
        clock.swap("left")
        clock.swap("right")
        assert counter.count == 1
        for _ in range(5):
            counter.update(make_frame(), clock.now)
            clock.now += 0.1
        assert counter.phase == PREPARE
        assert counter.count == 1, "leaving the frame must not lose reps"


class TestCommandsAndState:
    def test_command_map(self, core):
        clock = Clock(core)
        clock.swap("left")
        clock.swap("right")
        assert core.handle_command("add", {})["count"] == 2
        assert core.handle_command("reset", {})["count"] == 0
        assert core.handle_command("bogus", {})["ok"] is False

    def test_reset_clears_everything(self, core):
        clock = Clock(core)
        clock.swap("left")
        clock.swap("right")
        core.reset()
        assert core.count == 0
        assert core.side is None
        assert core.phase == PREPARE

    def test_state_is_json_safe(self, core):
        import json

        clock = Clock(core)
        clock.swap("left")
        json.dumps(core.state(clock.now))

    def test_state_exposes_the_tilt_signal(self, core):
        clock = Clock(core)
        state = clock.swap("left")
        assert state["tilt"] > state["tiltEnter"]
        assert 0.0 <= state["tiltMagnitude"] <= 1.0


class TestRendering:
    def test_canvas_shows_the_count(self, core):
        canvas = core.render_canvas()
        assert canvas.shape == (core.config.canvas_height, core.config.canvas_width, 3)
        blank = int((canvas > 100).sum())
        clock = Clock(core)
        clock.swap("left")
        clock.swap("right")
        assert int((core.render_canvas() > 100).sum()) != blank

    def test_overlay_draws_the_seesaw(self, core):
        from demos.six_seven_counter.core import draw_overlay

        frame = np.full((360, 640, 3), 60, dtype=np.uint8)
        clock = Clock(core)
        state = clock.swap("left")
        out = draw_overlay(frame.copy(), core, state)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_overlay_without_a_pose_prompts_the_user(self, core):
        from demos.six_seven_counter.core import draw_overlay

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        state = core.update(make_frame(), 0.0)
        assert draw_overlay(frame, core, state).max() > 0
