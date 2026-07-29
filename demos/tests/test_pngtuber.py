"""Tests for the PNGTuber classifier, sprite set, and placeholder generator."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from demos.pngtuber.config import EXPRESSIONS, YAW_BUCKETS, PngTuberConfig
from demos.pngtuber.core import (
    ANGRY,
    HAPPY,
    NEUTRAL,
    SURPRISED,
    Baseline,
    PngTuberCore,
    Ratios,
    SpriteSet,
    YawBucketer,
    classify_expression,
)
from demos.tests.fixtures import make_face, make_frame
from demos.tools.make_placeholder_sprites import generate, make_sprite

# Fixture faces per expression, chosen to sit clearly past the thresholds.
FACE_ARGS = {
    NEUTRAL: {},
    HAPPY: {"corner_lift": 0.05, "mouth_open": 0.10},
    SURPRISED: {"mouth_open": 0.40, "brow_raise": 0.10, "eye_open": 1.4},
    ANGRY: {"brow_raise": -0.06, "eye_open": 0.55},
}


@pytest.fixture()
def sprites_dir(tmp_path: Path) -> Path:
    generate(tmp_path, size=96)
    return tmp_path


@pytest.fixture()
def core(sprites_dir: Path) -> PngTuberCore:
    return PngTuberCore(PngTuberConfig(sprites_dir=sprites_dir, calibration_seconds=0.5))


def calibrate(core: PngTuberCore, *, start: float = 0.0) -> float:
    """Feed neutral frames through the calibration window; returns the new clock."""
    now = start
    while now <= start + core.config.calibration_seconds:
        core.update(make_frame(face=make_face()), now)
        now += 0.1
    return now


def hold(core: PngTuberCore, face_args: dict, now: float, seconds: float = 0.6) -> dict:
    """Hold one expression long enough to pass the settle window."""
    state: dict = {}
    end = now + seconds
    while now <= end:
        state = core.update(make_frame(face=make_face(**face_args)), now)
        now += 0.1
    return state


class TestRatiosAndBaseline:
    def test_measure_returns_all_four_ratios(self):
        ratios = Ratios.measure(make_face())
        assert ratios.mouth_open >= 0
        assert ratios.eye > 0
        assert set(ratios.to_json()) == {"mouthOpen", "smile", "brow", "eye"}

    def test_baseline_averages_samples(self):
        baseline = Baseline()
        baseline.add(Ratios(0.0, 0.0, 0.2, 1.0))
        baseline.add(Ratios(0.2, 0.1, 0.3, 1.2))
        assert baseline.samples == 2
        assert baseline.mouth_open == pytest.approx(0.1)
        assert baseline.brow == pytest.approx(0.25)

    def test_deltas_are_relative_to_the_baseline(self):
        baseline = Baseline()
        baseline.add(Ratios(0.1, 0.0, 0.25, 1.0))
        deltas = baseline.deltas(Ratios(0.3, 0.05, 0.30, 0.9))
        assert deltas.mouth_open == pytest.approx(0.2)
        assert deltas.eye == pytest.approx(-0.1)


class TestExpressionClassifier:
    config = PngTuberConfig()

    def test_no_change_is_neutral(self):
        assert classify_expression(Ratios(0, 0, 0, 0), self.config) == NEUTRAL

    def test_smile_is_happy(self):
        deltas = Ratios(mouth_open=0.05, smile=0.08, brow=0.0, eye=0.0)
        assert classify_expression(deltas, self.config) == HAPPY

    def test_open_mouth_with_raised_brows_is_surprised(self):
        deltas = Ratios(mouth_open=0.4, smile=0.0, brow=0.12, eye=0.3)
        assert classify_expression(deltas, self.config) == SURPRISED

    def test_lowered_brows_are_angry(self):
        deltas = Ratios(mouth_open=0.0, smile=0.0, brow=-0.08, eye=-0.1)
        assert classify_expression(deltas, self.config) == ANGRY

    def test_squint_alone_is_angry(self):
        deltas = Ratios(mouth_open=0.0, smile=0.0, brow=0.0, eye=-0.2)
        assert classify_expression(deltas, self.config) == ANGRY

    def test_a_grin_beats_slightly_lowered_brows(self):
        """A broad smile often pulls the brows down a little; happy should win."""
        deltas = Ratios(mouth_open=0.2, smile=0.09, brow=-0.04, eye=-0.06)
        assert classify_expression(deltas, self.config) == HAPPY

    def test_raised_brows_without_an_open_mouth_still_read_as_surprised(self):
        deltas = Ratios(mouth_open=0.0, smile=0.0, brow=0.09, eye=0.1)
        assert classify_expression(deltas, self.config) == SURPRISED

    @pytest.mark.parametrize("expression", list(EXPRESSIONS))
    def test_fixture_faces_classify_as_intended(self, expression):
        baseline = Baseline()
        baseline.add(Ratios.measure(make_face()))
        deltas = baseline.deltas(Ratios.measure(make_face(**FACE_ARGS[expression])))
        assert classify_expression(deltas, self.config) == expression


class TestYawBucketer:
    def test_center_by_default(self):
        assert YawBucketer(PngTuberConfig()).bucket == "center"

    def test_crossing_the_enter_threshold_switches(self):
        bucketer = YawBucketer(PngTuberConfig())
        assert bucketer.update(25.0) == "right"
        assert bucketer.update(-25.0) == "left"

    def test_small_angles_stay_centered(self):
        bucketer = YawBucketer(PngTuberConfig())
        for yaw in (0.0, 5.0, -5.0, 17.0, -17.0):
            assert bucketer.update(yaw) == "center"

    def test_hovering_at_the_boundary_does_not_flicker(self):
        config = PngTuberConfig()
        bucketer = YawBucketer(config)
        bucketer.update(30.0)
        # Values between release and enter must hold the current bucket.
        for yaw in (17.0, 13.0, 16.0, 12.0, 17.5):
            assert bucketer.update(yaw) == "right", f"flickered at {yaw}"
        assert bucketer.update(config.yaw_release - 1) == "center"

    def test_can_cross_straight_from_one_side_to_the_other(self):
        bucketer = YawBucketer(PngTuberConfig())
        bucketer.update(30.0)
        assert bucketer.update(-30.0) == "left"

    def test_reset_returns_to_center(self):
        bucketer = YawBucketer(PngTuberConfig())
        bucketer.update(30.0)
        bucketer.reset()
        assert bucketer.bucket == "center"


class TestCalibrationFlow:
    def test_calibrating_reports_neutral_and_collects_samples(self, core):
        state = core.update(make_frame(face=make_face(**FACE_ARGS[HAPPY])), 0.0)
        assert state["calibrating"] is True
        assert state["expression"] == NEUTRAL
        assert state["baseline"]["samples"] >= 1

    def test_expressions_are_classified_after_calibration(self, core):
        now = calibrate(core)
        state = hold(core, FACE_ARGS[HAPPY], now)
        assert state["calibrating"] is False
        assert state["expression"] == HAPPY
        assert state["sprite"] == "center_happy"

    @pytest.mark.parametrize("expression", list(EXPRESSIONS))
    def test_each_expression_selects_its_sprite(self, core, expression):
        now = calibrate(core)
        state = hold(core, FACE_ARGS[expression], now)
        assert state["expression"] == expression

    def test_baseline_absorbs_a_resting_face(self, sprites_dir):
        """A user whose relaxed face has a slight smile still reads as neutral."""
        core = PngTuberCore(PngTuberConfig(sprites_dir=sprites_dir, calibration_seconds=0.5))
        resting = {"corner_lift": 0.05, "mouth_open": 0.10}
        now = 0.0
        while now <= 0.5:
            core.update(make_frame(face=make_face(**resting)), now)
            now += 0.1
        state = hold(core, resting, now)
        assert state["expression"] == NEUTRAL

    def test_recalibration_resets_the_baseline(self, core):
        now = calibrate(core)
        hold(core, FACE_ARGS[HAPPY], now)
        core.calibrate(now=10.0)
        assert core.calibrating is True
        assert core.baseline.samples == 0
        assert core.expression == NEUTRAL

    def test_single_frame_flicker_does_not_switch_the_sprite(self, core):
        now = calibrate(core)
        hold(core, FACE_ARGS[NEUTRAL], now)
        # One surprised frame, well under expression_hold.
        core.update(make_frame(face=make_face(**FACE_ARGS[SURPRISED])), now + 0.05)
        assert core.expression == NEUTRAL

    def test_missing_face_holds_the_last_sprite(self, core):
        now = calibrate(core)
        hold(core, FACE_ARGS[HAPPY], now)
        state = core.update(make_frame(), now + 2.0)
        assert state["faceVisible"] is False
        assert state["sprite"] == "center_happy"

    def test_yaw_buckets_pick_side_sprites(self, core):
        now = calibrate(core)
        for yaw, bucket in ((40.0, "right"), (-40.0, "left"), (0.0, "center")):
            state = hold(core, {"yaw": yaw}, now, seconds=0.4)
            now += 0.5
            assert state["yawBucket"] == bucket
            assert state["sprite"].startswith(bucket)


class TestSpriteSet:
    def test_all_sprites_present(self, sprites_dir):
        sprites = SpriteSet(sprites_dir, PngTuberConfig())
        assert sprites.missing == []
        assert len(sprites.available()) == 12

    def test_sprites_load_as_bgra(self, sprites_dir):
        sprites = SpriteSet(sprites_dir, PngTuberConfig())
        image = sprites.get("center", NEUTRAL)
        assert image is not None
        assert image.shape[2] == 4

    def test_falls_back_to_neutral_then_center(self, sprites_dir):
        (sprites_dir / "left_angry.png").unlink()
        sprites = SpriteSet(sprites_dir, PngTuberConfig())
        fallback = sprites.get("left", ANGRY)
        assert fallback is not None
        assert np.array_equal(fallback, sprites.get("left", NEUTRAL))
        assert "left_angry" in sprites.missing

    def test_empty_directory_returns_none(self, tmp_path):
        sprites = SpriteSet(tmp_path, PngTuberConfig())
        assert sprites.get("center", NEUTRAL) is None
        assert len(sprites.missing) == 12

    def test_reload_picks_up_new_art(self, tmp_path):
        config = PngTuberConfig()
        sprites = SpriteSet(tmp_path, config)
        assert sprites.get("center", NEUTRAL) is None
        generate(tmp_path, size=64)
        sprites.reload()
        assert sprites.get("center", NEUTRAL) is not None


class TestGenerator:
    def test_writes_twelve_rgba_sprites(self, tmp_path):
        written = generate(tmp_path, size=64)
        assert len(written) == 12
        for path in written:
            assert path.is_file()
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert image.shape[2] == 4, f"{path.name} must have an alpha channel"
            assert image[:, :, 3].max() == 255
            assert image[0, 0, 3] == 0, "corners should be transparent"

    def test_filenames_follow_the_documented_pattern(self, tmp_path):
        names = {p.name for p in generate(tmp_path, size=64)}
        expected = {f"{b}_{e}.png" for b in YAW_BUCKETS for e in EXPRESSIONS}
        assert names == expected

    def test_expressions_look_different(self):
        rendered = {name: make_sprite("center", name, 96) for name in EXPRESSIONS}
        for first in EXPRESSIONS:
            for second in EXPRESSIONS:
                if first < second:
                    assert not np.array_equal(rendered[first], rendered[second])

    def test_yaw_shift_moves_the_features(self):
        left = make_sprite("left", NEUTRAL, 96)
        right = make_sprite("right", NEUTRAL, 96)
        assert not np.array_equal(left, right)


class TestRendering:
    def test_canvas_has_configured_size(self, core):
        canvas = core.render_canvas()
        assert canvas.shape == (core.config.canvas_height, core.config.canvas_width, 3)
        assert canvas.dtype == np.uint8

    def test_sprite_is_composited_onto_a_frame(self, core):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = core.render_sprite(frame.copy(), now=0.0)
        assert not np.array_equal(frame, out)

    def test_bob_moves_the_sprite_over_time(self, core):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        first = core.render_sprite(frame.copy(), now=0.0)
        later = core.render_sprite(frame.copy(), now=0.25)
        assert not np.array_equal(first, later)

    def test_missing_sprites_render_guidance(self, tmp_path):
        core = PngTuberCore(PngTuberConfig(sprites_dir=tmp_path))
        canvas = core.render_canvas()
        assert canvas.max() > 0

    def test_debug_overlay_draws(self, core):
        from demos.pngtuber.core import draw_debug

        now = calibrate(core)
        state = hold(core, FACE_ARGS[SURPRISED], now)
        frame = np.full((360, 640, 3), 50, dtype=np.uint8)
        out = draw_debug(frame.copy(), core, state)
        assert not np.array_equal(frame, out)


class TestCommandsAndState:
    def test_command_map(self, core):
        assert core.handle_command("calibrate", {"now": 1.0})["ok"] is True
        assert core.handle_command("reload", {})["ok"] is True
        assert core.handle_command("bogus", {})["ok"] is False

    def test_sprites_command_reports_availability(self, core):
        body = core.handle_command("sprites", {})
        assert body["missing"] == []
        assert len(body["available"]) == 12

    def test_expression_override(self, core):
        assert core.handle_command("expression", {"name": ANGRY})["sprite"] == "center_angry"
        assert core.handle_command("expression", {"name": "confused"})["ok"] is False

    def test_state_is_json_safe(self, core):
        import json

        now = calibrate(core)
        state = hold(core, FACE_ARGS[HAPPY], now)
        json.dumps(state)

    def test_reset_clears_calibration_and_expression(self, core):
        now = calibrate(core)
        hold(core, FACE_ARGS[HAPPY], now)
        core.reset()
        assert core.expression == NEUTRAL
        assert core.baseline.samples == 0
        assert core.yaw is None
        assert core.yaw_bucket == "center"


class TestYawCompensation:
    """Turning the head must not change the detected expression.

    Every ratio divides a vertical distance by a horizontal one, and yaw
    foreshortens the horizontal denominator by cos(yaw). Without compensation a
    relaxed face at 40 degrees reads as surprised.
    """

    def _core(self, sprites_dir: Path, *, compensate: bool) -> PngTuberCore:
        return PngTuberCore(
            PngTuberConfig(
                sprites_dir=sprites_dir,
                calibration_seconds=0.4,
                yaw_compensation=compensate,
            )
        )

    @pytest.mark.parametrize("yaw", [0.0, 20.0, 40.0, 55.0, -40.0])
    def test_neutral_stays_neutral_at_any_yaw(self, sprites_dir, yaw):
        core = self._core(sprites_dir, compensate=True)
        now = calibrate(core)
        state = hold(core, {"yaw": yaw}, now, seconds=0.8)
        assert state["expression"] == NEUTRAL

    def test_without_compensation_a_turned_head_misfires(self, sprites_dir):
        """Documents the failure mode the compensation exists to prevent."""
        core = self._core(sprites_dir, compensate=False)
        now = calibrate(core)
        state = hold(core, {"yaw": 40.0}, now, seconds=0.8)
        assert state["expression"] != NEUTRAL

    @pytest.mark.parametrize("expression", [HAPPY, SURPRISED, ANGRY])
    def test_expressions_still_register_while_turned(self, sprites_dir, expression):
        core = self._core(sprites_dir, compensate=True)
        now = calibrate(core)
        state = hold(core, {**FACE_ARGS[expression], "yaw": 30.0}, now, seconds=0.8)
        assert state["expression"] == expression

    def test_factor_is_floored_for_a_near_profile_face(self, sprites_dir):
        core = self._core(sprites_dir, compensate=True)
        core.yaw = 85.0
        assert core._yaw_factor() == pytest.approx(core.config.min_yaw_cosine)

    def test_factor_is_one_when_disabled(self, sprites_dir):
        core = self._core(sprites_dir, compensate=False)
        core.yaw = 40.0
        assert core._yaw_factor() == 1.0

    def test_factor_is_one_before_any_face(self, sprites_dir):
        assert self._core(sprites_dir, compensate=True)._yaw_factor() == 1.0
