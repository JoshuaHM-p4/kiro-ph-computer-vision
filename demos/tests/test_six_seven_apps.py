"""Tests for the 6-7 counter desktop renderer and web app."""

from __future__ import annotations

import numpy as np
import pytest

from demos.common.webapp import create_app
from demos.six_seven_counter import desktop, web
from demos.six_seven_counter.config import CounterConfig
from demos.six_seven_counter.core import SixSevenCore
from demos.tests.fixtures import make_frame, make_pose
from demos.tests.test_air_canvas_apps import FakeLoop
from demos.tests.test_six_seven import DOWN, UP, frame_at, tilted


@pytest.fixture()
def core() -> SixSevenCore:
    """A counter past the prepare hold, so renderer tests can count reps."""
    counter = SixSevenCore(CounterConfig())
    counter.handle_command("start", {})
    return counter


@pytest.fixture()
def frame() -> np.ndarray:
    return np.full((360, 640, 3), 70, dtype=np.uint8)


class TestDesktopRenderer:
    def test_renders_with_a_pose(self, core, frame):
        render = desktop.make_renderer(core, desktop.CounterOptions())
        out = render(frame, tilted("left"), FakeLoop())
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_renders_without_a_pose(self, core, frame):
        render = desktop.make_renderer(core, desktop.CounterOptions())
        out = render(frame, make_frame(), FakeLoop())
        assert out.shape == frame.shape

    def test_counting_works_through_the_renderer(self, core, frame):
        render = desktop.make_renderer(core, desktop.CounterOptions())
        loop = FakeLoop()
        now = 0.0
        assert core.phase == "counting", "the fixture skips the prepare hold"
        # Five tilts: the first sets the baseline, the next four are swaps.
        for index in range(5):
            higher = "left" if index % 2 == 0 else "right"
            for _ in range(4):
                loop.elapsed = now
                render(frame.copy(), tilted(higher), loop)
                now += 0.1
        assert core.count == 4

    def test_guides_can_be_hidden(self, core, frame):
        landmark_frame = tilted("left")
        loop = FakeLoop(debug=False)
        with_guides = desktop.make_renderer(core, desktop.CounterOptions(show_guides=True))(
            frame.copy(), landmark_frame, loop
        )
        without = desktop.make_renderer(
            SixSevenCore(CounterConfig()), desktop.CounterOptions(show_guides=False)
        )(frame.copy(), landmark_frame, loop)
        assert not np.array_equal(with_guides, without)

    def test_skeleton_can_be_hidden(self, core, frame):
        landmark_frame = frame_at(DOWN, DOWN)
        loop = FakeLoop(debug=False)
        shown = desktop.make_renderer(core, desktop.CounterOptions(show_skeleton=True))(
            frame.copy(), landmark_frame, loop
        )
        hidden = desktop.make_renderer(
            SixSevenCore(CounterConfig()), desktop.CounterOptions(show_skeleton=False)
        )(frame.copy(), landmark_frame, loop)
        assert not np.array_equal(shown, hidden)


class TestDesktopKeys:
    def test_reset_and_manual_add(self, core):
        on_key = desktop.make_key_handler(core, desktop.CounterOptions())
        assert on_key(ord("a"), FakeLoop()) is True
        assert core.count == 1
        assert on_key(ord("r"), FakeLoop()) is True
        assert core.count == 0

    def test_display_toggles(self, core):
        options = desktop.CounterOptions()
        on_key = desktop.make_key_handler(core, options)
        on_key(ord("g"), FakeLoop())
        assert options.show_guides is False
        on_key(ord("k"), FakeLoop())
        assert options.show_skeleton is False

    def test_unknown_key_falls_through(self, core):
        on_key = desktop.make_key_handler(core, desktop.CounterOptions())
        assert on_key(ord("q"), FakeLoop()) is False


@pytest.fixture()
def client():
    blueprint, sock, _ = web.build()
    app = create_app([(blueprint, sock)], name="six_seven_test")
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _pose_payload(seq: int, left: float, right: float) -> dict:
    pose = make_pose(left_wrist_up=left, right_wrist_up=right)
    return {
        "seq": seq,
        "ts": seq * 0.1,
        "width": 640,
        "height": 480,
        "pose": {
            "points": [{"x": x, "y": y, "visibility": 1.0} for x, y in pose.points]
        },
    }


class TestWebApp:
    def test_page_renders(self, client):
        response = client.get("/six-seven/")
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "landmark-stream.js" in body
        assert "Reps" in body

    def _prepare(self, client, sid: str, start_seq: int = 0) -> int:
        """Feed steady in-frame poses until the core leaves prepare mode."""
        seq = start_seq
        body = {}
        while seq < start_seq + 40:
            seq += 1
            body = client.post(
                f"/six-seven/landmarks?sid={sid}", json=_pose_payload(seq, DOWN, DOWN)
            ).get_json()
            if body["phase"] == "counting":
                return seq
        raise AssertionError(f"never left prepare mode: {body}")

    def test_prepare_gate_is_reported(self, client):
        body = client.post("/six-seven/landmarks?sid=p1", json=_pose_payload(1, DOWN, DOWN)).get_json()
        assert body["phase"] == "prepare"
        assert 0.0 <= body["prepareProgress"] <= 1.0
        assert body["hint"]
        seq = self._prepare(client, "p1", start_seq=1)
        assert seq > 1

    def test_swapping_hands_counts_a_rep(self, client):
        seq = self._prepare(client, "c1")
        body = {}
        for index in range(2):  # baseline tilt, then one swap
            left, right = (UP, DOWN) if index == 0 else (DOWN, UP)
            for _ in range(4):
                seq += 1
                body = client.post(
                    "/six-seven/landmarks?sid=c1", json=_pose_payload(seq, left, right)
                ).get_json()
        assert body["count"] == 1
        assert body["poseVisible"] is True
        assert body["side"] == "right"
        assert body["expected"] == "left" 

    def test_the_same_hand_twice_cannot_count(self, client):
        seq = self._prepare(client, "c1")
        body = {}
        for _ in range(3):  # raise the left hand, level off, repeat
            for heights in ((UP, DOWN), (0.0, 0.0)):
                for _ in range(4):
                    seq += 1
                    body = client.post(
                        "/six-seven/landmarks?sid=c1", json=_pose_payload(seq, *heights)
                    ).get_json()
        assert body["count"] == 0, "a swap requires both hands to take part"

    def test_reset_command(self, client):
        client.post("/six-seven/command?sid=c1", json={"command": "add"})
        assert client.post("/six-seven/command?sid=c1", json={"command": "reset"}).get_json()["count"] == 0

    def test_snapshot_is_a_scoreboard_png(self, client):
        client.post("/six-seven/landmarks?sid=c1", json=_pose_payload(1, DOWN, DOWN))
        response = client.get("/six-seven/snapshot?sid=c1")
        assert response.status_code == 200
        assert response.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_prepare_and_start_commands(self, client):
        seq = self._prepare(client, "cmd1")
        client.post("/six-seven/command?sid=cmd1", json={"command": "prepare"})
        body = client.post(
            "/six-seven/landmarks?sid=cmd1", json=_pose_payload(seq + 1, DOWN, DOWN)
        ).get_json()
        assert body["phase"] == "prepare"
        client.post("/six-seven/command?sid=cmd1", json={"command": "start"})
        body = client.post(
            "/six-seven/landmarks?sid=cmd1", json=_pose_payload(seq + 2, DOWN, DOWN)
        ).get_json()
        assert body["phase"] == "counting"

    def test_state_carries_the_tilt_signal(self, client):
        body = client.post("/six-seven/landmarks?sid=c1", json=_pose_payload(1, UP, DOWN)).get_json()
        assert body["shoulderY"] is not None
        assert body["torsoGuide"] is not None
        assert body["tiltEnter"] > 0
        assert body["tilt"] > 0, "left hand higher should read positive"
