"""Tests for the slide presenter desktop renderer and web app."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from demos.common.webapp import create_app
from demos.slide_presenter import desktop, web
from demos.slide_presenter.config import SlideConfig
from demos.slide_presenter.core import SlidePresenterCore
from demos.tests.fixtures import make_frame, make_hand
from demos.tests.test_air_canvas_apps import FakeLoop
from demos.tests.test_slide_presenter import open_hand, pinch, point_at


@pytest.fixture()
def deck_dir(tmp_path: Path) -> Path:
    for index in range(3):
        image = np.full((90, 160, 3), 40 + index * 60, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"slide{index + 1}.png"), image)
    return tmp_path


@pytest.fixture()
def core(deck_dir: Path) -> SlidePresenterCore:
    return SlidePresenterCore(SlideConfig(slides_dir=deck_dir, advance_cooldown=0.4))


@pytest.fixture()
def frame() -> np.ndarray:
    return np.full((360, 640, 3), 80, dtype=np.uint8)


class TestDesktopRenderer:
    def test_renders_the_slide_full_frame(self, core, frame):
        render = desktop.make_renderer(core, desktop.PresenterOptions())
        out = render(frame, make_frame(), FakeLoop())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_slide_content_changes_with_the_index(self, core, frame):
        render = desktop.make_renderer(core, desktop.PresenterOptions(show_inset=False))
        first = render(frame.copy(), make_frame(), FakeLoop(debug=False))
        core.next_slide()
        second = render(frame.copy(), make_frame(), FakeLoop(debug=False))
        assert not np.array_equal(first, second)

    def test_inset_can_be_hidden(self, core, frame):
        landmark_frame = make_frame(hands=[make_hand(label="Right")])
        with_inset = desktop.make_renderer(core, desktop.PresenterOptions(show_inset=True))(
            frame.copy(), landmark_frame, FakeLoop(debug=False)
        )
        without = desktop.make_renderer(core, desktop.PresenterOptions(show_inset=False))(
            frame.copy(), landmark_frame, FakeLoop(debug=False)
        )
        assert not np.array_equal(with_inset, without)

    def test_pinch_through_the_renderer_advances(self, core, frame):
        render = desktop.make_renderer(core, desktop.PresenterOptions())
        loop = FakeLoop()
        for step in range(4):
            loop.elapsed = step * 0.05
            render(frame.copy(), pinch("Right"), loop)
        assert core.index == 1

    def test_laser_is_drawn_when_pointing(self, core, frame):
        render = desktop.make_renderer(core, desktop.PresenterOptions(show_inset=False))
        loop = FakeLoop()
        idle = render(frame.copy(), open_hand(), loop)
        for step in range(8):
            loop.elapsed = step * 0.05
            pointed = render(frame.copy(), point_at((0.5, 0.5)), loop)
        assert not np.array_equal(idle, pointed)

    def test_empty_deck_renders_guidance(self, tmp_path, frame):
        core = SlidePresenterCore(SlideConfig(slides_dir=tmp_path))
        out = desktop.make_renderer(core, desktop.PresenterOptions())(
            frame, make_frame(), FakeLoop()
        )
        assert out.max() > 0


class TestDesktopKeys:
    def test_navigation_keys(self, core):
        options = desktop.PresenterOptions()
        on_key = desktop.make_key_handler(core, options)
        loop = FakeLoop()
        assert on_key(ord("n"), loop) is True
        assert core.index == 1
        assert on_key(ord("p"), loop) is True
        assert core.index == 0
        assert on_key(desktop.KEY_RIGHT, loop) is True
        assert core.index == 1
        assert on_key(desktop.KEY_LEFT, loop) is True
        assert core.index == 0

    def test_toggles_and_reload(self, core, deck_dir):
        options = desktop.PresenterOptions()
        on_key = desktop.make_key_handler(core, options)
        assert on_key(ord("c"), FakeLoop()) is True
        assert options.show_inset is False
        assert on_key(ord("w"), FakeLoop()) is True
        assert core.config.swap_handedness is True

        cv2.imwrite(str(deck_dir / "slide4.png"), np.zeros((90, 160, 3), dtype=np.uint8))
        assert on_key(ord("r"), FakeLoop()) is True
        assert len(core.deck) == 4

    def test_unknown_key_falls_through(self, core):
        on_key = desktop.make_key_handler(core, desktop.PresenterOptions())
        assert on_key(ord("q"), FakeLoop()) is False


@pytest.fixture()
def client(deck_dir: Path):
    original = web._CONFIG.slides_dir
    web._CONFIG.slides_dir = deck_dir
    blueprint, sock, _ = web.build()
    app = create_app([(blueprint, sock)], name="slides_test")
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client
    web._CONFIG.slides_dir = original


def _pinch_payload(seq: int, label: str) -> dict:
    hand = pinch(label).hands[0]
    return {
        "seq": seq,
        "ts": seq * 0.5,
        "width": 640,
        "height": 480,
        "hands": [
            {"label": label, "score": 0.9, "points": [{"x": x, "y": y} for x, y in hand.points]}
        ],
    }


class TestWebApp:
    def test_page_lists_the_deck(self, client):
        response = client.get("/slides/")
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "landmark-stream.js" in body
        assert "slide1.png" in body

    def test_slide_images_are_served_by_index(self, client):
        response = client.get("/slides/slide/0")
        assert response.status_code == 200
        assert response.mimetype in ("image/png", "image/jpeg")

    def test_slide_index_wraps_instead_of_404(self, client):
        assert client.get("/slides/slide/99").status_code == 200

    def test_pinch_over_http_advances_one_slide(self, client):
        body = client.post("/slides/landmarks?sid=s1", json=_pinch_payload(1, "Right")).get_json()
        assert body["index"] == 1
        held = client.post("/slides/landmarks?sid=s1", json=_pinch_payload(2, "Right")).get_json()
        assert held["index"] == 1, "a held pinch must not keep advancing"

    def test_commands_navigate(self, client):
        client.post("/slides/landmarks?sid=s1", json={"seq": 1, "hands": []})
        assert client.post("/slides/command?sid=s1", json={"command": "next"}).get_json()["index"] == 1
        assert (
            client.post("/slides/command?sid=s1", json={"command": "previous"}).get_json()["index"] == 0
        )
        assert client.post("/slides/command?sid=s1", json={"command": "goto", "payload": {"index": 2}}).get_json()["index"] == 2

    def test_deck_command_returns_names(self, client):
        client.post("/slides/landmarks?sid=s1", json={"seq": 1, "hands": []})
        body = client.post("/slides/command?sid=s1", json={"command": "deck"}).get_json()
        assert body["count"] == 3

    def test_snapshot_renders_the_current_slide(self, client):
        client.post("/slides/landmarks?sid=s1", json={"seq": 1, "hands": []})
        response = client.get("/slides/snapshot?sid=s1")
        assert response.status_code == 200
        assert response.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_sessions_navigate_independently(self, client):
        client.post("/slides/command?sid=a", json={"command": "next"})
        client.post("/slides/landmarks?sid=a", json={"seq": 1, "hands": []})
        client.post("/slides/landmarks?sid=b", json={"seq": 1, "hands": []})
        assert client.get("/slides/state?sid=a").get_json()["index"] == 1
        assert client.get("/slides/state?sid=b").get_json()["index"] == 0
