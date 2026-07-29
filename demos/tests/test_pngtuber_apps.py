"""Tests for the PNGTuber desktop renderer and web app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from demos.common.webapp import create_app
from demos.pngtuber import desktop, web
from demos.pngtuber.config import PngTuberConfig
from demos.pngtuber.core import HAPPY, NEUTRAL, PngTuberCore
from demos.tests.fixtures import make_face, make_frame
from demos.tests.test_air_canvas_apps import FakeLoop
from demos.tests.test_pngtuber import FACE_ARGS
from demos.tools.make_placeholder_sprites import generate


@pytest.fixture()
def sprites_dir(tmp_path: Path) -> Path:
    generate(tmp_path, size=96)
    return tmp_path


@pytest.fixture()
def core(sprites_dir: Path) -> PngTuberCore:
    return PngTuberCore(PngTuberConfig(sprites_dir=sprites_dir, calibration_seconds=0.2))


@pytest.fixture()
def frame() -> np.ndarray:
    return np.full((360, 640, 3), 100, dtype=np.uint8)


class TestDesktopRenderer:
    def test_renders_with_a_face(self, core, frame):
        render = desktop.make_renderer(core, desktop.TuberOptions())
        out = render(frame, make_frame(face=make_face()), FakeLoop())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_renders_without_a_face(self, core, frame):
        render = desktop.make_renderer(core, desktop.TuberOptions())
        out = render(frame, make_frame(), FakeLoop())
        assert out.shape == frame.shape

    def test_solid_and_chroma_backgrounds_replace_the_camera(self, core, frame):
        landmark_frame = make_frame(face=make_face())
        loop = FakeLoop(debug=False)
        chroma = desktop.make_renderer(core, desktop.TuberOptions(background="chroma"))(
            frame.copy(), landmark_frame, loop
        )
        solid = desktop.make_renderer(core, desktop.TuberOptions(background="solid"))(
            frame.copy(), landmark_frame, loop
        )
        # Top-left corner is background only, never sprite.
        assert tuple(chroma[4, 4]) == desktop.CHROMA_GREEN
        assert tuple(solid[4, 4]) == core.config.background

    def test_expression_change_changes_the_rendered_sprite(self, core, frame):
        render = desktop.make_renderer(core, desktop.TuberOptions(background="solid"))
        loop = FakeLoop(debug=False)
        now = 0.0
        while now <= core.config.calibration_seconds:
            loop.elapsed = now
            neutral = render(frame.copy(), make_frame(face=make_face()), loop)
            now += 0.05
        for _ in range(12):
            loop.elapsed = now
            happy = render(frame.copy(), make_frame(face=make_face(**FACE_ARGS[HAPPY])), loop)
            now += 0.05
        assert core.expression == HAPPY
        assert not np.array_equal(neutral, happy)

    def test_inset_only_draws_over_non_camera_backgrounds(self, core, frame):
        landmark_frame = make_frame(face=make_face())
        loop = FakeLoop(debug=False)
        plain = desktop.make_renderer(
            core, desktop.TuberOptions(background="solid", show_inset=False)
        )(frame.copy(), landmark_frame, loop)
        with_inset = desktop.make_renderer(
            PngTuberCore(core.config), desktop.TuberOptions(background="solid", show_inset=True)
        )(frame.copy(), landmark_frame, loop)
        assert not np.array_equal(plain, with_inset)

    def test_missing_sprites_still_render(self, tmp_path, frame):
        core = PngTuberCore(PngTuberConfig(sprites_dir=tmp_path))
        out = desktop.make_renderer(core, desktop.TuberOptions())(
            frame, make_frame(face=make_face()), FakeLoop()
        )
        assert out.shape == frame.shape


class TestDesktopKeys:
    def test_recalibrate(self, core):
        on_key = desktop.make_key_handler(core, desktop.TuberOptions())
        core.update(make_frame(face=make_face()), 0.0)
        assert on_key(ord("c"), FakeLoop()) is True
        assert core.calibrating is True

    def test_background_cycles(self, core):
        options = desktop.TuberOptions()
        on_key = desktop.make_key_handler(core, options)
        assert options.background == "camera"
        on_key(ord("b"), FakeLoop())
        assert options.background == "solid"
        on_key(ord("b"), FakeLoop())
        assert options.background == "chroma"
        on_key(ord("b"), FakeLoop())
        assert options.background == "camera"

    def test_inset_toggle(self, core):
        options = desktop.TuberOptions()
        desktop.make_key_handler(core, options)(ord("v"), FakeLoop())
        assert options.show_inset is True

    def test_number_keys_preview_expressions(self, core):
        on_key = desktop.make_key_handler(core, desktop.TuberOptions())
        assert on_key(ord("2"), FakeLoop()) is True
        assert core.expression == HAPPY

    def test_unknown_key_falls_through(self, core):
        on_key = desktop.make_key_handler(core, desktop.TuberOptions())
        assert on_key(ord("q"), FakeLoop()) is False


@pytest.fixture()
def client(sprites_dir: Path):
    original = web._CONFIG.sprites_dir
    web._CONFIG.sprites_dir = sprites_dir
    web._CONFIG.calibration_seconds = 0.2
    blueprint, sock, _ = web.build()
    app = create_app([(blueprint, sock)], name="pngtuber_test")
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client
    web._CONFIG.sprites_dir = original


def _face_payload(seq: int, **face_args) -> dict:
    face = make_face(**face_args)
    return {
        "seq": seq,
        "ts": seq * 0.1,
        "width": 640,
        "height": 480,
        "face": {"points": [{"x": x, "y": y} for x, y in face.points]},
    }


class TestWebApp:
    def test_page_lists_the_sprite_set(self, client):
        response = client.get("/pngtuber/")
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "landmark-stream.js" in body
        assert "center_neutral" in body
        assert "left_angry" in body

    def test_sprites_are_served_by_id(self, client):
        response = client.get("/pngtuber/sprite/center_happy")
        assert response.status_code == 200
        assert response.mimetype == "image/png"

    def test_unknown_sprite_id_is_404(self, client):
        assert client.get("/pngtuber/sprite/center_confused").status_code == 404
        assert client.get("/pngtuber/sprite/sideways_happy").status_code == 404

    def test_landmarks_produce_a_sprite_choice(self, client):
        body = client.post("/pngtuber/landmarks?sid=t1", json=_face_payload(1)).get_json()
        assert body["sprite"] == "center_neutral"
        assert body["faceVisible"] is True
        assert body["calibrating"] is True

    def test_expression_is_tracked_after_calibration(self, client):
        seq = 0
        for _ in range(4):  # calibration window at 0.1 s per frame
            seq += 1
            client.post("/pngtuber/landmarks?sid=t1", json=_face_payload(seq))
        body = {}
        for _ in range(8):
            seq += 1
            body = client.post(
                "/pngtuber/landmarks?sid=t1", json=_face_payload(seq, **FACE_ARGS[HAPPY])
            ).get_json()
        assert body["calibrating"] is False
        assert body["expression"] == HAPPY
        assert body["sprite"] == "center_happy"

    def test_yaw_selects_a_side_sprite(self, client):
        seq = 0
        body = {}
        for _ in range(12):
            seq += 1
            body = client.post("/pngtuber/landmarks?sid=t1", json=_face_payload(seq, yaw=40.0)).get_json()
        assert body["yawBucket"] == "right"
        assert body["sprite"].startswith("right")

    def test_calibrate_command(self, client):
        client.post("/pngtuber/landmarks?sid=t1", json=_face_payload(1))
        body = client.post(
            "/pngtuber/command?sid=t1", json={"command": "calibrate", "payload": {"now": 5.0}}
        ).get_json()
        assert body["ok"] is True

    def test_sprites_command_reports_availability(self, client):
        client.post("/pngtuber/landmarks?sid=t1", json=_face_payload(1))
        body = client.post("/pngtuber/command?sid=t1", json={"command": "sprites"}).get_json()
        assert body["missing"] == []

    def test_snapshot_renders_the_avatar(self, client):
        client.post("/pngtuber/landmarks?sid=t1", json=_face_payload(1))
        response = client.get("/pngtuber/snapshot?sid=t1")
        assert response.status_code == 200
        assert response.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_sessions_calibrate_independently(self, client):
        client.post("/pngtuber/landmarks?sid=a", json=_face_payload(1, **FACE_ARGS[HAPPY]))
        client.post("/pngtuber/landmarks?sid=b", json=_face_payload(1))
        assert client.get("/pngtuber/state?sid=a").get_json()["expression"] == NEUTRAL
        assert client.get("/pngtuber/state?sid=b").get_json()["expression"] == NEUTRAL
        assert client.get("/pngtuber/state?sid=a").get_json()["baseline"]["samples"] == 1
