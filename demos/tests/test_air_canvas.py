"""Tests for the air canvas paint engine.

Everything is driven by scripted landmark frames, so gesture behaviour is
verified without a camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from demos.air_canvas.config import PALETTE_COLORS, AirCanvasConfig
from demos.air_canvas.core import DRAW, ERASE, HOVER, IDLE, AirCanvasCore, Stroke
from demos.common import landmarks as lm
from demos.tests.fixtures import make_frame, make_hand


def _hand_with_tip(tip, *, extended, pinch=1.0):
    """Build a hand whose index fingertip sits at ``tip``."""
    hand = make_hand(extended=extended, pinch=pinch, center=(0.5, 0.5), span=0.08)
    offset = (tip[0] - hand.index_tip[0], tip[1] - hand.index_tip[1])
    hand.points = [(x + offset[0], y + offset[1]) for x, y in hand.points]
    return hand


def draw_at(tip) -> lm.LandmarkFrame:
    return make_frame(hands=[_hand_with_tip(tip, extended=("index",))])


def hover_at(tip) -> lm.LandmarkFrame:
    return make_frame(hands=[_hand_with_tip(tip, extended=("index", "middle"))])


def pinch_at(tip) -> lm.LandmarkFrame:
    return make_frame(hands=[_hand_with_tip(tip, extended=("index",), pinch=0.05)])


def empty() -> lm.LandmarkFrame:
    return make_frame(hands=[])


class TestGestureClassification:
    def test_pointing_draws(self):
        core = AirCanvasCore()
        state = core.update(draw_at((0.5, 0.5)), 0.0)
        assert state["tool"] == DRAW

    def test_peace_hovers(self):
        core = AirCanvasCore()
        assert core.update(hover_at((0.5, 0.5)), 0.0)["tool"] == HOVER

    def test_pinch_erases(self):
        core = AirCanvasCore()
        assert core.update(pinch_at((0.5, 0.5)), 0.0)["tool"] == ERASE

    def test_fist_is_idle(self):
        core = AirCanvasCore()
        frame = make_frame(hands=[make_hand(extended=())])
        assert core.update(frame, 0.0)["tool"] == IDLE

    def test_no_hand_is_idle_with_no_cursor(self):
        core = AirCanvasCore()
        state = core.update(empty(), 0.0)
        assert state["tool"] == IDLE
        assert state["cursor"] is None


class TestStrokeLifecycle:
    def test_drawing_starts_a_stroke(self):
        core = AirCanvasCore()
        core.update(draw_at((0.4, 0.4)), 0.0)
        assert len(core.strokes) == 1
        assert core.active is not None

    def test_stroke_accumulates_points(self):
        core = AirCanvasCore()
        for i in range(5):
            core.update(draw_at((0.30 + i * 0.05, 0.5)), i * 0.05)
        assert len(core.strokes) == 1
        assert len(core.strokes[0].points) == 5

    def test_jitter_below_min_move_is_dropped(self):
        core = AirCanvasCore()
        core.update(draw_at((0.5, 0.5)), 0.0)
        for i in range(10):
            core.update(draw_at((0.5, 0.5)), 0.01 * (i + 1))
        assert len(core.strokes[0].points) == 1

    def test_releasing_the_gesture_ends_the_stroke(self):
        core = AirCanvasCore()
        core.update(draw_at((0.4, 0.4)), 0.0)
        core.update(draw_at((0.5, 0.4)), 0.05)
        core.update(empty(), 0.1)
        assert core.active is None
        assert len(core.strokes) == 1

    def test_redrawing_starts_a_second_stroke(self):
        core = AirCanvasCore()
        core.update(draw_at((0.3, 0.3)), 0.0)
        core.update(empty(), 0.1)
        core.update(draw_at((0.7, 0.7)), 0.2)
        assert len(core.strokes) == 2

    def test_stroke_timeout_closes_a_stalled_stroke(self):
        core = AirCanvasCore()
        core.update(draw_at((0.4, 0.4)), 0.0)
        # Same point for longer than stroke_timeout: no new ink, so it closes.
        core.update(draw_at((0.4, 0.4)), 5.0)
        assert core.active is None

    def test_stroke_cap_discards_the_oldest(self):
        config = AirCanvasConfig()
        config.max_strokes = 3
        core = AirCanvasCore(config)
        for i in range(6):
            core.update(draw_at((0.2 + i * 0.05, 0.5)), i * 1.0)
            core.update(empty(), i * 1.0 + 0.5)
        assert len(core.strokes) <= 3


class TestBrushProperties:
    def test_stroke_captures_current_color_size_opacity(self):
        core = AirCanvasCore()
        core.set_color("magenta")
        core.set_size(0.03)
        core.set_opacity(0.4)
        core.update(draw_at((0.5, 0.5)), 0.0)
        stroke = core.strokes[0]
        assert stroke.color == core.config.color_for("magenta")
        assert stroke.size == pytest.approx(0.03)
        assert stroke.opacity == pytest.approx(0.4)

    def test_setters_clamp_to_configured_range(self):
        core = AirCanvasCore()
        core.set_size(99.0)
        assert core.size == core.config.size_max
        core.set_size(-1.0)
        assert core.size == core.config.size_min
        core.set_opacity(99.0)
        assert core.opacity == core.config.opacity_max
        core.set_opacity(-1.0)
        assert core.opacity == core.config.opacity_min

    def test_eraser_stroke_is_wider_and_opaque(self):
        core = AirCanvasCore()
        core.set_size(0.01)
        core.set_opacity(0.3)
        core.update(pinch_at((0.5, 0.5)), 0.0)
        stroke = core.strokes[0]
        assert stroke.erase is True
        assert stroke.size == pytest.approx(0.01 * core.config.eraser_scale)
        assert stroke.opacity == 1.0

    def test_switching_between_draw_and_erase_splits_strokes(self):
        core = AirCanvasCore()
        core.update(draw_at((0.4, 0.4)), 0.0)
        core.update(pinch_at((0.45, 0.4)), 0.1)
        assert len(core.strokes) == 2
        assert core.strokes[0].erase is False
        assert core.strokes[1].erase is True


class TestPaletteDwell:
    def _hover(self, core, key, *, start=0.0, hold=1.6, steps=12):
        """Hover a cell's centre for ``hold`` seconds.

        The hold has to outlast both the dwell threshold and the cursor
        smoothing: the smoothed fingertip drifts through neighbouring cells on
        its way, restarting the dwell each time it crosses a boundary.
        """
        cell = core.config.cell(key)
        assert cell is not None, key
        for i in range(steps):
            core.update(hover_at(cell.center), start + hold * i / (steps - 1))
        return cell

    def test_dwelling_a_swatch_changes_color(self):
        core = AirCanvasCore()
        assert core.color_name == "cyan"
        self._hover(core, "color:magenta")
        assert core.color_name == "magenta"
        assert core.color == core.config.color_for("magenta")

    def test_sweeping_past_swatches_selects_nothing(self):
        core = AirCanvasCore()
        cells = [c for c in core.config.cells if c.kind == "color"]
        for index, cell in enumerate(cells):
            core.update(hover_at(cell.center), index * 0.05)  # 50 ms each
        assert core.color_name == "cyan"

    def test_dwell_progress_is_reported(self):
        core = AirCanvasCore()
        cell = core.config.cell("color:lime")
        core.update(hover_at(cell.center), 0.0)
        state = core.update(hover_at(cell.center), core.config.dwell_seconds * 0.5)
        assert 0.4 < state["dwell"] < 0.6
        assert state["hover"] == "color:lime"

    def test_hovering_does_not_paint(self):
        core = AirCanvasCore()
        self._hover(core, "color:red")
        assert core.strokes == []

    def test_undo_cell_removes_the_last_stroke(self):
        core = AirCanvasCore()
        core.update(draw_at((0.4, 0.4)), 0.0)
        core.update(empty(), 0.1)
        core.update(draw_at((0.6, 0.6)), 0.2)
        core.update(empty(), 0.3)
        assert len(core.strokes) == 2
        self._hover(core, "action:undo", start=1.0)
        assert len(core.strokes) == 1

    def test_clear_cell_removes_everything(self):
        core = AirCanvasCore()
        core.update(draw_at((0.4, 0.4)), 0.0)
        core.update(empty(), 0.1)
        self._hover(core, "action:clear", start=1.0)
        assert core.strokes == []

    def test_eraser_cell_makes_draw_gesture_erase(self):
        core = AirCanvasCore()
        self._hover(core, "tool:eraser")
        core.update(draw_at((0.5, 0.5)), 2.0)
        assert core.strokes[0].erase is True

    def test_selecting_a_color_leaves_eraser_mode(self):
        core = AirCanvasCore()
        self._hover(core, "tool:eraser")
        self._hover(core, "color:lime", start=2.0)
        core.update(draw_at((0.5, 0.5)), 4.0)
        assert core.strokes[0].erase is False


class TestSliders:
    def _grab(self, core, key, fraction, *, start=0.0, steps=14):
        """Dwell to grab a slider, then hold at ``fraction`` of its height.

        ``fraction`` 1.0 is the top of the bar (maximum). The target is inset a
        hair from the ends so the smoothed cursor settles inside the bar rather
        than straddling its boundary.
        """
        cell = core.config.cell(key)
        x1, y1, x2, y2 = cell.rect
        x = (x1 + x2) / 2
        inset = 0.002
        grab_y = (y1 + inset) + ((y2 - inset) - (y1 + inset)) * (1.0 - fraction)
        for i in range(steps):
            core.update(hover_at((x, grab_y)), start + i * 0.2)
        return cell

    def test_grabbing_the_size_slider_sets_size(self):
        core = AirCanvasCore()
        span = core.config.size_max - core.config.size_min
        self._grab(core, "slider:size", 1.0)
        assert core.size == pytest.approx(core.config.size_max, abs=span * 0.05)
        self._grab(core, "slider:size", 0.0, start=4.0)
        assert core.size == pytest.approx(core.config.size_min, abs=span * 0.05)

    def test_grabbing_the_opacity_slider_sets_opacity(self):
        core = AirCanvasCore()
        span = core.config.opacity_max - core.config.opacity_min
        self._grab(core, "slider:opacity", 0.0)
        assert core.opacity == pytest.approx(core.config.opacity_min, abs=span * 0.05)
        self._grab(core, "slider:opacity", 1.0, start=4.0)
        assert core.opacity == pytest.approx(core.config.opacity_max, abs=span * 0.05)

    def test_value_tracks_the_finger_after_the_grab(self):
        core = AirCanvasCore()
        cell = self._grab(core, "slider:size", 0.0)
        assert core._grabbed_slider == "slider:size"
        low = core.size
        x1, y1, x2, y2 = cell.rect
        # No second dwell needed: the value follows the finger while grabbed.
        for i in range(10):
            core.update(hover_at(((x1 + x2) / 2, y1 + 0.002)), 4.0 + i * 0.05)
        assert core.size > low
        assert core.size == pytest.approx(
            core.config.size_max, abs=(core.config.size_max - core.config.size_min) * 0.05
        )
        assert core._grabbed_slider == "slider:size", "the grab should survive the move"

    def test_leaving_the_bar_releases_the_grab(self):
        core = AirCanvasCore()
        self._grab(core, "slider:size", 0.5)
        assert core._grabbed_slider == "slider:size"
        for i in range(6):
            core.update(hover_at((0.5, 0.5)), 5.0 + i * 0.05)
        assert core._grabbed_slider is None

    def test_sliders_report_fractions_in_state(self):
        core = AirCanvasCore()
        core.set_size(core.config.size_max)
        core.set_opacity(core.config.opacity_min)
        state = core.update(empty(), 0.0)
        assert state["sizeFraction"] == pytest.approx(1.0)
        assert state["opacityFraction"] == pytest.approx(0.0)


class TestCommandsAndState:
    def test_command_map(self):
        core = AirCanvasCore()
        core.update(draw_at((0.4, 0.4)), 0.0)
        assert core.handle_command("color", {"name": "amber"})["ok"] is True
        assert core.color_name == "amber"
        assert core.handle_command("size", {"value": 0.02})["ok"] is True
        assert core.size == pytest.approx(0.02)
        assert core.handle_command("opacity", {"value": 0.5})["ok"] is True
        assert core.opacity == pytest.approx(0.5)
        assert core.handle_command("undo", {})["ok"] is True
        assert core.handle_command("clear", {})["ok"] is True
        assert core.handle_command("bogus", {})["ok"] is False

    def test_sync_command_returns_all_strokes(self):
        core = AirCanvasCore()
        core.update(draw_at((0.4, 0.4)), 0.0)
        core.update(draw_at((0.5, 0.4)), 0.1)
        result = core.handle_command("sync", {})
        assert len(result["strokes"]) == 1
        assert len(result["strokes"][0]["points"]) == 2
        assert result["strokes"][0]["color"] == list(core.color)

    def test_revision_advances_on_structural_change(self):
        core = AirCanvasCore()
        start = core.revision
        core.update(draw_at((0.4, 0.4)), 0.0)
        core.update(empty(), 0.2)
        assert core.revision > start
        mid = core.revision
        core.clear()
        assert core.revision > mid

    def test_state_is_json_safe(self):
        import json

        core = AirCanvasCore()
        core.update(draw_at((0.4, 0.4)), 0.0)
        json.dumps(core.state(0.0))
        json.dumps(core.palette_json())

    def test_active_stroke_tail_is_capped(self):
        core = AirCanvasCore()
        for i in range(20):
            core.update(draw_at((0.2 + i * 0.02, 0.5)), i * 0.05)
        assert len(core.state(1.0)["activeStroke"]["points"]) <= 2

    def test_reset_restores_defaults(self):
        core = AirCanvasCore()
        core.set_color("red")
        core.set_size(0.05)
        core.update(draw_at((0.4, 0.4)), 0.0)
        core.reset()
        assert core.strokes == []
        assert core.color_name == "cyan"
        assert core.size == core.config.size_default
        assert core.cursor is None

    def test_palette_json_covers_every_cell(self):
        core = AirCanvasCore()
        payload = core.palette_json()
        assert len(payload) == len(core.config.cells)
        assert {entry["kind"] for entry in payload} == {"color", "tool", "action", "slider"}
        colors = [entry for entry in payload if entry["kind"] == "color"]
        assert len(colors) == len(PALETTE_COLORS)
        assert colors[0]["css"].startswith("rgb(")


class TestRendering:
    def test_canvas_has_configured_size_and_background(self):
        core = AirCanvasCore()
        canvas = core.render_canvas()
        assert canvas.shape == (core.config.canvas_height, core.config.canvas_width, 3)
        assert canvas.dtype == np.uint8
        assert tuple(canvas[0, 0]) == core.config.background

    def test_drawn_stroke_appears_on_the_canvas(self):
        core = AirCanvasCore()
        core.set_color("white")
        core.set_size(0.04)
        for i in range(6):
            core.update(draw_at((0.3 + i * 0.06, 0.5)), i * 0.05)
        canvas = core.render_canvas()
        row = canvas[core.config.canvas_height // 2]
        assert row.max() > 200, "the stroke should be visible"

    def test_opacity_produces_a_dimmer_stroke(self):
        def peak(opacity: float) -> int:
            core = AirCanvasCore()
            core.set_color("white")
            core.set_size(0.04)
            core.set_opacity(opacity)
            for i in range(6):
                core.update(draw_at((0.3 + i * 0.06, 0.5)), i * 0.05)
            return int(core.render_canvas().max())

        assert peak(0.25) < peak(1.0)

    def test_eraser_removes_coverage(self):
        core = AirCanvasCore()
        core.set_color("white")
        core.set_size(0.05)
        for i in range(8):
            core.update(draw_at((0.2 + i * 0.07, 0.5)), i * 0.05)
        core.update(empty(), 1.0)
        before = int((core.render_layer(320, 180)[:, :, 3] > 10).sum())

        core.set_size(0.05)
        for i in range(8):
            core.update(pinch_at((0.2 + i * 0.07, 0.5)), 2.0 + i * 0.05)
        after = int((core.render_layer(320, 180)[:, :, 3] > 10).sum())
        assert after < before

    def test_layer_is_bgra(self):
        core = AirCanvasCore()
        layer = core.render_layer(64, 48)
        assert layer.shape == (48, 64, 4)

    def test_stroke_size_scales_with_canvas_height(self):
        stroke = Stroke(color=(255, 255, 255), size=0.1, opacity=1.0, points=[(0.5, 0.5)])
        core = AirCanvasCore()
        core.strokes = [stroke]
        small = (core.render_layer(100, 100)[:, :, 3] > 10).sum()
        large = (core.render_layer(400, 400)[:, :, 3] > 10).sum()
        # Four times the height means roughly sixteen times the area.
        assert large > small * 8
