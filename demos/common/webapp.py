"""Flask + WebSocket adapter for the browser path.

Vision runs in the browser (``@mediapipe/tasks-vision``); this module receives
the resulting landmarks over a persistent WebSocket, feeds them to the same pure
demo cores the desktop apps use, and returns state as JSON. The browser draws the
live overlay from that state, and ``/snapshot`` renders the authoritative numpy
canvas with cv2 for export.

Wire format, browser to server::

    {"seq": 41, "ts": 1699.5, "width": 640, "height": 480,
     "hands": [{"label": "Right", "score": 0.98, "points": [{"x":.., "y":..}, ...]}],
     "face": {"points": [...]}, "pose": {"points": [...]}}

Server to browser is whatever the demo core's ``update`` returns, plus a
``_meta`` block carrying the frame-guard counters.

Security note: :func:`run_standalone` binds to 127.0.0.1 and these apps have no
authentication. Anything that can reach the port can drive the demo state.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import cv2
import numpy as np
from flask import (
    Blueprint,
    Flask,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
)
from flask_sock import Sock

from . import landmarks as lm

# A client that reconnects starts counting from zero again. Treat a sequence
# number this far below the last accepted one as a restart rather than a stale
# packet, otherwise the guard would reject every frame from the new connection.
SEQ_RESET_GAP = 64


class DemoCore(Protocol):
    """What every demo's core must provide to be served over the web."""

    def update(self, frame: lm.LandmarkFrame, now: float) -> dict[str, Any]:
        """Advance the state machine and return JSON-serializable state."""

    def render_canvas(self) -> np.ndarray:
        """Render the authoritative BGR canvas for /snapshot."""

    def handle_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle a UI command (clear, undo, next slide, ...)."""

    def reset(self) -> None:
        """Return to the initial state."""


@dataclass
class FrameGuard:
    """Drops stale landmark packets so a saturated socket degrades cleanly.

    Only strictly increasing sequence numbers are accepted, so duplicates and
    out-of-order arrivals are discarded instead of rewinding the state machine.
    A sequence that jumps far backwards is treated as a client restart.
    """

    last_seq: int = -1
    accepted: int = 0
    dropped: int = 0
    resets: int = 0

    def accept(self, seq: int) -> bool:
        if seq > self.last_seq:
            self.last_seq = seq
            self.accepted += 1
            return True
        if seq < self.last_seq - SEQ_RESET_GAP:
            self.last_seq = seq
            self.accepted += 1
            self.resets += 1
            return True
        self.dropped += 1
        return False

    def stats(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "dropped": self.dropped,
            "resets": self.resets,
            "last_seq": self.last_seq,
        }

    def reset(self) -> None:
        self.last_seq = -1
        self.accepted = 0
        self.dropped = 0
        self.resets = 0


@dataclass
class DemoSession:
    """One browser client's core instance plus its frame guard."""

    core: Any
    guard: FrameGuard = field(default_factory=FrameGuard)
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    last_state: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_seen = time.monotonic()


class SessionRegistry:
    """Keeps one core per client id.

    Sessions are keyed by an id the browser generates, which is also passed to
    ``/snapshot`` so an export reflects that client's canvas. Idle sessions are
    pruned so a long-running server does not accumulate abandoned canvases.
    """

    def __init__(
        self,
        core_factory: Callable[[], Any],
        *,
        ttl: float = 900.0,
        max_sessions: int = 32,
    ):
        self.core_factory = core_factory
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._sessions: dict[str, DemoSession] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str, *, create: bool = True) -> DemoSession | None:
        with self._lock:
            self._prune_locked()
            session = self._sessions.get(session_id)
            if session is None:
                if not create:
                    return None
                if len(self._sessions) >= self.max_sessions:
                    oldest = min(self._sessions, key=lambda key: self._sessions[key].last_seen)
                    self._sessions.pop(oldest, None)
                session = DemoSession(core=self.core_factory())
                self._sessions[session_id] = session
            session.touch()
            return session

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _prune_locked(self) -> None:
        now = time.monotonic()
        stale = [key for key, s in self._sessions.items() if (now - s.last_seen) > self.ttl]
        for key in stale:
            self._sessions.pop(key, None)


def _session_id() -> str:
    """Client id from the query string, header, or a fresh one."""
    return (
        request.args.get("sid")
        or request.headers.get("X-Demo-Session")
        or uuid.uuid4().hex
    )


def create_demo_blueprint(
    *,
    name: str,
    core_factory: Callable[[], Any],
    template: str,
    url_prefix: str,
    page_context: Callable[[], dict[str, Any]] | None = None,
    import_name: str | None = None,
    static_folder: str = "static",
    template_folder: str = "templates",
) -> tuple[Blueprint, Sock, SessionRegistry]:
    """Build the standard route set for one demo.

    Routes, all relative to ``url_prefix``:

    ``GET  /``          the demo page
    ``GET  /ws``        WebSocket landmark stream (state JSON per accepted frame)
    ``POST /landmarks`` HTTP fallback for one landmark frame
    ``POST /command``   UI commands forwarded to ``core.handle_command``
    ``GET  /state``     last computed state, for polling clients
    ``GET  /snapshot``  PNG of the core's authoritative canvas
    ``GET  /health``    liveness probe
    """
    blueprint = Blueprint(
        name,
        import_name or __name__,
        template_folder=template_folder,
        static_folder=static_folder,
        url_prefix=url_prefix,
    )
    registry = SessionRegistry(core_factory)
    sock = Sock()

    @blueprint.get("/")
    def index() -> str:
        context = page_context() if page_context else {}
        return render_template(template, **context)

    @blueprint.get("/health")
    def health():
        return jsonify({"ok": True, "demo": name, "sessions": registry.count()})

    @blueprint.get("/state")
    def state():
        session = registry.get(_session_id())
        assert session is not None
        return jsonify(session.last_state)

    @blueprint.post("/landmarks")
    def landmarks():
        """HTTP fallback for environments where WebSockets are unavailable."""
        payload = request.get_json(silent=True) or {}
        session = registry.get(_session_id())
        assert session is not None
        state_json = _process_message(session, payload)
        return jsonify(state_json)

    @blueprint.post("/command")
    def command():
        payload = request.get_json(silent=True) or {}
        session = registry.get(_session_id())
        assert session is not None
        action = str(payload.get("command", ""))
        result = session.core.handle_command(action, payload.get("payload") or {})
        return jsonify(result if isinstance(result, dict) else {"ok": True})

    @blueprint.post("/reset")
    def reset():
        session = registry.get(_session_id())
        assert session is not None
        session.core.reset()
        session.guard.reset()
        return jsonify({"ok": True})

    @blueprint.get("/snapshot")
    def snapshot():
        session = registry.get(_session_id(), create=False)
        if session is None:
            return jsonify({"error": "unknown session"}), 404
        canvas = session.core.render_canvas()
        ok, buffer = cv2.imencode(".png", canvas)
        if not ok:  # pragma: no cover - imencode failure is not reproducible
            return jsonify({"error": "encode failed"}), 500
        filename = f"{name}-{int(time.time())}.png"
        return Response(
            buffer.tobytes(),
            mimetype="image/png",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @sock.route("/ws", bp=blueprint)
    def stream(ws):  # pragma: no cover - needs a live socket
        session_id = request.args.get("sid") or uuid.uuid4().hex
        session = registry.get(session_id)
        assert session is not None
        try:
            while True:
                raw = ws.receive()
                if raw is None:
                    break
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    continue

                if payload.get("type") == "command":
                    result = session.core.handle_command(
                        str(payload.get("command", "")), payload.get("payload") or {}
                    )
                    ws.send(json.dumps({"ack": payload.get("command"), "result": result}))
                    continue

                state_json = _process_message(session, payload)
                if state_json is not None:
                    ws.send(json.dumps(state_json))
        finally:
            session.touch()

    return blueprint, sock, registry


def _process_message(session: DemoSession, payload: dict[str, Any]) -> dict[str, Any]:
    """Guard, decode, and run one landmark message through the core."""
    seq = int(payload.get("seq", 0) or 0)
    if not session.guard.accept(seq):
        # Report the drop without touching demo state: the browser keeps drawing
        # the previous state, so a congested socket looks like a lower frame rate.
        skipped = dict(session.last_state)
        skipped["_meta"] = {**session.guard.stats(), "skipped": True}
        return skipped

    frame = lm.frame_from_json(payload)
    now = float(payload.get("ts") or time.monotonic())
    state_json = session.core.update(frame, now)
    state_json = dict(state_json or {})
    state_json["_meta"] = {**session.guard.stats(), "skipped": False}
    session.last_state = state_json
    session.touch()
    return state_json


def create_shared_blueprint() -> Blueprint:
    """Serves the shared JS/CSS and exposes ``base.html`` to Jinja.

    Every demo page extends ``base.html`` and loads
    ``/shared/static/js/landmark-stream.js``, so the browser-side MediaPipe
    plumbing exists in exactly one place.
    """
    return Blueprint(
        "shared",
        "demos.common",
        template_folder="templates",
        static_folder="static",
        static_url_path="/shared/static",
    )


def create_app(
    blueprints: list[tuple[Blueprint, Sock]],
    *,
    name: str = "demos",
    extra: Callable[[Flask], None] | None = None,
) -> Flask:
    """Compose one or more demo blueprints into a Flask app."""
    app = Flask(
        name,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}
    app.register_blueprint(create_shared_blueprint())
    for blueprint, sock in blueprints:
        app.register_blueprint(blueprint)
        sock.init_app(app)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "app": name})

    if extra is not None:
        extra(app)
    return app


def run_standalone(
    app: Flask,
    *,
    port: int,
    host: str = "127.0.0.1",
    debug: bool = False,
) -> None:  # pragma: no cover - starts a server
    """Run one demo on its own port.

    Defaults to loopback on purpose: the demos have no authentication, so
    binding to 0.0.0.0 would expose a webcam-driven service to the network.
    """
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"WARNING: binding to {host} exposes this unauthenticated demo "
            "beyond this machine."
        )
    print(f"  http://{host}:{port}/")
    app.run(host=host, port=port, debug=debug, threaded=True)


def add_web_arguments(parser, *, default_port: int) -> None:
    """CLI flags shared by the standalone web runners."""
    parser.add_argument("--port", type=int, default=default_port, help="Port to serve on")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (loopback by default)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
