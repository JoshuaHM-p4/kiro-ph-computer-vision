"""Tests for the air canvas desktop renderer and web app.

The desktop renderer is exercised with a fake loop object, so no camera or window
is involved; the web app goes through the Flask test client.
"""

from __future__ import annotations

import numpy as np
import pytest

from demos.air_canvas import desktop, web
from demos.air_canvas.core import AirCanvasCore
from demos.common.geometry import FPSCounter
from demos.common.webapp import create_app
from demos.tests.fixtures import make_face, make_frame, make_hand
from demos.tests.test_air_canvas import draw_at, hover_at


class FakeLoop:
    """Stands in for CameraLoop: only the attributes the renderer reads."""

    def __init__(self, *, debug: bool = True, elapsed: float = 0.0, frame_index: int = 0):
        self.debug = debug
        self.elapsed = elapsed
        self.frame_index = frame_index
        self.fps = FPSCounter()
        self.mirror = True


@pytest.fixture()
def frame() -> np.ndarray:
    return np.full((360, 640, 3), 90, dtype=np.uint8)


class TestDesktopRenderer:
    def test_renders_and_keeps_frame_geometry(self, frame):
        core = AirCanvasCore()
        render = desktop.make_renderer(core, desktop.DesktopOptions())
        out = render(frame, draw_at((0.5, 0.5)), FakeLoop())
        assert out.shape == (360, 640, 3)
        assert out.dtype == np.uint8

    def test_drawing_gesture_creates_a_stroke_through_the_renderer(self, frame):
        core = AirCanvasCore()
        render = desktop.make_renderer(core, desktop.DesktopOptions())
        loop = FakeLoop()
        for i in range(5):
            loop.elapsed = i * 0.05
            render(frame.copy(), draw_at((0.3 + i * 0.05, 0.5)), loop)
        assert len(core.strokes) == 1
        assert len(core.strokes[0].points) == 5

    def test_dim_camera_darkens_the_background(self, frame):
        core = AirCanvasCore()
        bright = desktop.make_renderer(core, desktop.DesktopOptions(dim_camera=0.0))(
            frame.copy(), make_frame(), FakeLoop(debug=False)
        )
        dim = desktop.make_renderer(core, desktop.DesktopOptions(dim_camera=0.8))(
            frame.copy(), make_frame(), FakeLoop(debug=False)
        )
        assert dim.mean() < bright.mean()

    def test_face_and_hand_overlays_can_be_disabled(self, frame):
        core = AirCanvasCore()
        landmark_frame = make_frame(hands=[make_hand(extended=("index",))], face=make_face())
        on = desktop.make_renderer(core, desktop.DesktopOptions())(
            frame.copy(), landmark_frame, FakeLoop(debug=False)
        )
        off = desktop.make_renderer(
            AirCanvasCore(), desktop.DesktopOptions(show_face=False, show_hands=False)
        )(frame.copy(), landmark_frame, FakeLoop(debug=False))
        assert not np.array_equal(on, off)

    def test_debug_overlay_is_toggleable(self, frame):
        core = AirCanvasCore()
        render = desktop.make_renderer(core, desktop.DesktopOptions())
        with_debug = render(frame.copy(), make_frame(), FakeLoop(debug=True))
        without = render(frame.copy(), make_frame(), FakeLoop(debug=False))
        assert not np.array_equal(with_debug, without)

    def test_hover_renders_the_dwell_ring(self, frame):
        core = AirCanvasCore()
        render = desktop.make_renderer(core, desktop.DesktopOptions())
        loop = FakeLoop()
        cell = core.config.cell("color:magenta")
        for i in range(4):
            loop.elapsed = i * 0.1
            out = render(frame.copy(), hover_at(cell.center), loop)
        assert out.shape == frame.shape

    def test_missing_landmarks_render_fine(self, frame):
        core = AirCanvasCore()
        render = desktop.make_renderer(core, desktop.DesktopOptions())
        out = render(frame, make_frame(), FakeLoop())
        assert out.shape == (360, 640, 3)


class TestDesktopKeys:
    def _handler(self):
        core = AirCanvasCore()
        options = desktop.DesktopOptions()
        return core, options, desktop.make_key_handler(core, options)

    def test_clear_and_undo(self):
        core, _, on_key = self._handler()
        core.update(draw_at((0.4, 0.4)), 0.0)
        core.update(make_frame(), 0.2)
        assert on_key(ord("z"), FakeLoop()) is True
        assert core.strokes == []

        core.update(draw_at((0.4, 0.4)), 1.0)
        assert on_key(ord("c"), FakeLoop()) is True
        assert core.strokes == []

    def test_size_and_opacity_keys(self):
        core, _, on_key = self._handler()
        start = core.size
        on_key(ord("]"), FakeLoop())
        assert core.size > start
        on_key(ord("["), FakeLoop())
        assert core.size == pytest.approx(start)

        core.set_opacity(0.5)
        on_key(ord("="), FakeLoop())
        assert core.opacity == pytest.approx(0.6)
        on_key(ord("-"), FakeLoop())
        assert core.opacity == pytest.approx(0.5)

    def test_size_keys_respect_limits(self):
        core, _, on_key = self._handler()
        for _ in range(40):
            on_key(ord("]"), FakeLoop())
        assert core.size == core.config.size_max
        for _ in range(80):
            on_key(ord("["), FakeLoop())
        assert core.size == core.config.size_min

    def test_overlay_toggles(self):
        _, options, on_key = self._handler()
        on_key(ord("f"), FakeLoop())
        assert options.show_face is False
        on_key(ord("h"), FakeLoop())
        assert options.show_hands is False

    def test_unknown_key_falls_through(self):
        _, _, on_key = self._handler()
        assert on_key(ord("q"), FakeLoop()) is False
        assert on_key(ord("x"), FakeLoop()) is False


@pytest.fixture()
def client():
    blueprint, sock, _ = web.build()
    app = create_app([(blueprint, sock)], name="air_canvas_test")
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _landmark_payload(seq: int, tip: tuple[float, float]) -> dict:
    hand = draw_at(tip).hands[0]
    return {
        "seq": seq,
        "ts": seq * 0.05,
        "width": 640,
        "height": 480,
        "hands": [
            {"label": "Right", "score": 0.9, "points": [{"x": x, "y": y} for x, y in hand.points]}
        ],
    }


class TestWebApp:
    def test_page_includes_palette_and_stream_module(self, client):
        response = client.get("/air-canvas/")
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "landmark-stream.js" in body
        assert "color:cyan" in body, "palette geometry should be embedded"
        assert "slider:opacity" in body

    def test_landmarks_produce_paint_state(self, client):
        for seq in range(1, 5):
            body = client.post(
                "/air-canvas/landmarks?sid=w1", json=_landmark_payload(seq, (0.3 + seq * 0.05, 0.5))
            ).get_json()
        assert body["tool"] == "draw"
        assert body["strokeCount"] == 1
        assert body["activeStroke"] is not None
        assert body["_meta"]["accepted"] == 4

    def test_sync_command_returns_strokes(self, client):
        for seq in range(1, 4):
            client.post("/air-canvas/landmarks?sid=w1", json=_landmark_payload(seq, (0.3 + seq * 0.05, 0.5)))
        body = client.post("/air-canvas/command?sid=w1", json={"command": "sync"}).get_json()
        assert body["ok"] is True
        assert len(body["strokes"]) == 1

    def test_clear_bumps_the_revision(self, client):
        first = client.post("/air-canvas/landmarks?sid=w1", json=_landmark_payload(1, (0.4, 0.5))).get_json()
        client.post("/air-canvas/command?sid=w1", json={"command": "clear"})
        after = client.post("/air-canvas/landmarks?sid=w1", json=_landmark_payload(2, (0.4, 0.5))).get_json()
        assert after["revision"] > first["revision"]

    def test_color_command(self, client):
        client.post("/air-canvas/landmarks?sid=w1", json=_landmark_payload(1, (0.4, 0.5)))
        client.post("/air-canvas/command?sid=w1", json={"command": "color", "payload": {"name": "lime"}})
        body = client.get("/air-canvas/state?sid=w1").get_json()
        assert body["colorName"] in ("cyan", "lime")  # state is from the last frame
        follow = client.post("/air-canvas/landmarks?sid=w1", json=_landmark_payload(2, (0.45, 0.5))).get_json()
        assert follow["colorName"] == "lime"

    def test_snapshot_is_a_png_of_the_canvas(self, client):
        for seq in range(1, 6):
            client.post("/air-canvas/landmarks?sid=w1", json=_landmark_payload(seq, (0.3 + seq * 0.06, 0.5)))
        response = client.get("/air-canvas/snapshot?sid=w1")
        assert response.status_code == 200
        assert response.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_sessions_paint_independently(self, client):
        for seq in range(1, 4):
            client.post("/air-canvas/landmarks?sid=a", json=_landmark_payload(seq, (0.3 + seq * 0.05, 0.5)))
        client.post("/air-canvas/landmarks?sid=b", json=_landmark_payload(1, (0.8, 0.8)))
        a = client.get("/air-canvas/state?sid=a").get_json()
        b = client.get("/air-canvas/state?sid=b").get_json()
        assert a["strokeCount"] == 1
        assert b["strokeCount"] == 1
        assert a["activeStroke"]["points"] != b["activeStroke"]["points"]

    def test_page_context_exposes_limits(self):
        context = web._page_context()
        assert context["limits"]["sizeMax"] > context["limits"]["sizeMin"]
        assert context["base_path"] == "/air-canvas"
        assert len(context["palette"]) == 11
