"""Tests for the Flask/WebSocket adapter: frame guard, sessions, routes."""

from __future__ import annotations

import numpy as np
import pytest

from demos.common import landmarks as lm
from demos.common.echo import EchoCore, build
from demos.common.webapp import (
    SEQ_RESET_GAP,
    FrameGuard,
    SessionRegistry,
    _process_message,
    DemoSession,
    create_app,
)
from demos.tests.fixtures import make_hand


class TestFrameGuard:
    def test_accepts_increasing_sequence(self):
        guard = FrameGuard()
        assert [guard.accept(n) for n in (0, 1, 2, 3)] == [True, True, True, True]
        assert guard.accepted == 4
        assert guard.dropped == 0

    def test_rejects_duplicates(self):
        guard = FrameGuard()
        guard.accept(5)
        assert guard.accept(5) is False
        assert guard.dropped == 1

    def test_rejects_out_of_order_packets(self):
        guard = FrameGuard()
        guard.accept(10)
        assert guard.accept(9) is False
        assert guard.accept(4) is False
        assert guard.last_seq == 10, "state must not rewind"
        assert guard.dropped == 2

    def test_large_backwards_jump_is_treated_as_a_client_restart(self):
        guard = FrameGuard()
        guard.accept(1000)
        assert guard.accept(1000 - SEQ_RESET_GAP - 1) is True
        assert guard.resets == 1
        assert guard.last_seq == 1000 - SEQ_RESET_GAP - 1

    def test_gap_forward_is_fine(self):
        guard = FrameGuard()
        guard.accept(1)
        assert guard.accept(500) is True

    def test_stats_and_reset(self):
        guard = FrameGuard()
        guard.accept(1)
        guard.accept(1)
        stats = guard.stats()
        assert stats == {"accepted": 1, "dropped": 1, "resets": 0, "last_seq": 1}
        guard.reset()
        assert guard.stats() == {"accepted": 0, "dropped": 0, "resets": 0, "last_seq": -1}


class TestSessionRegistry:
    def test_same_id_returns_the_same_core(self):
        registry = SessionRegistry(EchoCore)
        first = registry.get("abc")
        second = registry.get("abc")
        assert first is second
        assert registry.count() == 1

    def test_different_ids_are_isolated(self):
        registry = SessionRegistry(EchoCore)
        a = registry.get("a")
        b = registry.get("b")
        assert a is not b
        assert a.core is not b.core
        assert registry.count() == 2

    def test_create_false_does_not_allocate(self):
        registry = SessionRegistry(EchoCore)
        assert registry.get("nope", create=False) is None
        assert registry.count() == 0

    def test_expired_sessions_are_pruned(self):
        registry = SessionRegistry(EchoCore, ttl=0.0)
        registry.get("old")
        registry.get("new")
        assert registry.count() <= 1

    def test_max_sessions_evicts_the_least_recent(self):
        registry = SessionRegistry(EchoCore, max_sessions=2)
        registry.get("a")
        registry.get("b")
        registry.get("c")
        assert registry.count() == 2

    def test_drop_removes_a_session(self):
        registry = SessionRegistry(EchoCore)
        registry.get("a")
        registry.drop("a")
        assert registry.count() == 0


def _payload(seq: int, *, hand=True) -> dict:
    hand_obj = make_hand(center=(0.4, 0.6))
    return {
        "seq": seq,
        "ts": float(seq),
        "width": 640,
        "height": 480,
        "hands": (
            [{"label": "Right", "score": 0.9, "points": [{"x": x, "y": y} for x, y in hand_obj.points]}]
            if hand
            else []
        ),
    }


class TestProcessMessage:
    def test_accepted_frame_updates_state(self):
        session = DemoSession(core=EchoCore())
        state = _process_message(session, _payload(1))
        assert state["frames"] == 1
        assert state["hands"] == 1
        assert state["_meta"]["skipped"] is False
        assert state["point"] is not None

    def test_stale_frame_returns_previous_state_marked_skipped(self):
        session = DemoSession(core=EchoCore())
        _process_message(session, _payload(5))
        stale = _process_message(session, _payload(2))
        assert stale["_meta"]["skipped"] is True
        assert stale["frames"] == 1, "core must not advance on a dropped frame"

    def test_missing_seq_defaults_to_zero_and_is_accepted_once(self):
        session = DemoSession(core=EchoCore())
        first = _process_message(session, {"width": 1, "height": 1})
        second = _process_message(session, {"width": 1, "height": 1})
        assert first["_meta"]["skipped"] is False
        assert second["_meta"]["skipped"] is True


@pytest.fixture()
def client():
    blueprint, sock, registry = build()
    app = create_app([(blueprint, sock)], name="test")
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        test_client.registry = registry
        yield test_client


class TestRoutes:
    def test_index_renders(self, client):
        response = client.get("/echo/")
        assert response.status_code == 200
        assert b"Echo" in response.data
        assert b"landmark-stream.js" in response.data

    def test_shared_static_is_served(self, client):
        response = client.get("/shared/static/js/landmark-stream.js")
        assert response.status_code == 200
        assert b"LandmarkStream" in response.data

    def test_health_endpoints(self, client):
        assert client.get("/health").get_json()["ok"] is True
        body = client.get("/echo/health").get_json()
        assert body == {"ok": True, "demo": "echo", "sessions": 0}

    def test_landmarks_http_fallback_returns_state(self, client):
        response = client.post("/echo/landmarks?sid=s1", json=_payload(1))
        assert response.status_code == 200
        body = response.get_json()
        assert body["frames"] == 1
        assert body["_meta"]["accepted"] == 1

    def test_state_route_reflects_the_last_frame(self, client):
        client.post("/echo/landmarks?sid=s1", json=_payload(1))
        body = client.get("/echo/state?sid=s1").get_json()
        assert body["frames"] == 1

    def test_sessions_are_isolated_by_sid(self, client):
        client.post("/echo/landmarks?sid=a", json=_payload(1))
        client.post("/echo/landmarks?sid=a", json=_payload(2))
        client.post("/echo/landmarks?sid=b", json=_payload(1))
        assert client.get("/echo/state?sid=a").get_json()["frames"] == 2
        assert client.get("/echo/state?sid=b").get_json()["frames"] == 1

    def test_session_header_is_accepted(self, client):
        client.post(
            "/echo/landmarks", json=_payload(1), headers={"X-Demo-Session": "hdr"}
        )
        assert client.get("/echo/state?sid=hdr").get_json()["frames"] == 1

    def test_command_route(self, client):
        client.post("/echo/landmarks?sid=s1", json=_payload(1))
        body = client.post("/echo/command?sid=s1", json={"command": "clear"}).get_json()
        assert body["cleared"] is True
        unknown = client.post("/echo/command?sid=s1", json={"command": "nope"}).get_json()
        assert unknown["ok"] is False

    def test_reset_route_clears_core_and_guard(self, client):
        client.post("/echo/landmarks?sid=s1", json=_payload(9))
        assert client.post("/echo/reset?sid=s1").get_json()["ok"] is True
        # After a reset the guard accepts a low sequence number again.
        body = client.post("/echo/landmarks?sid=s1", json=_payload(1)).get_json()
        assert body["frames"] == 1

    def test_snapshot_returns_a_png(self, client):
        client.post("/echo/landmarks?sid=s1", json=_payload(1))
        response = client.get("/echo/snapshot?sid=s1")
        assert response.status_code == 200
        assert response.mimetype == "image/png"
        assert response.data[:8] == b"\x89PNG\r\n\x1a\n"
        assert "attachment" in response.headers["Content-Disposition"]

    def test_snapshot_for_unknown_session_is_404(self, client):
        assert client.get("/echo/snapshot?sid=ghost").status_code == 404

    def test_malformed_json_does_not_500(self, client):
        response = client.post(
            "/echo/landmarks?sid=s1", data="not json", content_type="application/json"
        )
        assert response.status_code == 200


class TestEchoCore:
    def test_render_canvas_shape(self):
        core = EchoCore()
        canvas = core.render_canvas()
        assert canvas.shape == (720, 1280, 3)
        assert canvas.dtype == np.uint8

    def test_trail_is_bounded(self):
        core = EchoCore(trail_length=10)
        frame = lm.LandmarkFrame(hands=[make_hand()], width=640, height=480)
        for _ in range(50):
            core.update(frame, 0.0)
        assert len(core.trail) == 10

    def test_reset_clears_everything(self):
        core = EchoCore()
        core.update(lm.LandmarkFrame(hands=[make_hand()]), 0.0)
        core.reset()
        assert core.trail == []
        assert core.frames == 0
