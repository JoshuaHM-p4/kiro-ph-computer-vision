"""Scavenger hunt web app.

Run standalone::

    .venv/bin/python -m demos.scavenger_hunt.web --model ~/models/yolo26n.onnx
    SCAVENGER_MODEL=~/models/yolo26n.onnx .venv/bin/python -m demos.scavenger_hunt.web

This demo inverts the usual arrangement in this suite. The other web demos run
MediaPipe *in the browser* and stream landmarks; YOLO has no browser build here, so
the browser sends **JPEG frames** a few times a second and the server runs
inference. That is why there is no WebSocket: each frame is one POST whose response
carries the detections and the new game state.

The model is user-provided and never committed.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Any

import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, render_template, request

from ..common.webapp import (
    SessionRegistry,
    _session_id,
    add_web_arguments,
    create_app,
    run_standalone,
)
from .config import DESK_ITEMS, SETTING_BOUNDS, HuntConfig
from .core import ScavengerHuntCore
from .detector import Detector, load_detector

URL_PREFIX = "/scavenger-hunt"

_CONFIG = HuntConfig()
# One detector shared by every session: the model is read-only and loading it per
# player would multiply memory for no benefit.
_DETECTOR: Detector | None = None


def detector() -> Detector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = load_detector(_CONFIG)
    return _DETECTOR


def set_detector(instance: Detector) -> None:
    """Install a detector directly, used by the tests and ``--demo-mode``."""
    global _DETECTOR
    _DETECTOR = instance


def _decode(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def build():
    """Return ``(blueprint, sock, registry)``; ``sock`` is None for this demo."""
    blueprint = Blueprint(
        "scavenger_hunt",
        "demos.scavenger_hunt",
        template_folder="templates",
        static_folder="static",
        url_prefix=URL_PREFIX,
    )
    # Each player gets their own copy of the config: the settings panel is
    # per-session, so one person turning the timer down must not change anyone
    # else's game.
    registry = SessionRegistry(lambda: ScavengerHuntCore(replace(_CONFIG)))

    def core_for_request() -> ScavengerHuntCore:
        session = registry.get(_session_id())
        assert session is not None
        return session.core

    def _now(payload: dict[str, Any] | None = None) -> float:
        """Server clock, so a client cannot cheat the timer by lying about time."""
        return time.monotonic()

    @blueprint.get("/")
    def index() -> str:
        active = detector()
        return render_template(
            "scavenger_hunt.html",
            base_path=URL_PREFIX,
            items=list(DESK_ITEMS),
            rounds=_CONFIG.rounds,
            round_seconds=_CONFIG.round_seconds,
            bounds=SETTING_BOUNDS,
            defaults={
                "roundSeconds": _CONFIG.round_seconds,
                "confidence": _CONFIG.confidence,
                "rounds": _CONFIG.rounds,
                "holdSeconds": _CONFIG.hold_seconds,
            },
            model={"ready": active.ready, "name": active.name, "detail": active.describe()},
        )

    @blueprint.post("/settings")
    def settings():
        """Adjust this player's timer, confidence threshold, rounds and hold window."""
        payload = request.get_json(silent=True) or {}
        core = core_for_request()
        applied = core.update_settings(payload)
        return jsonify({"ok": True, "applied": applied, **core.state(_now())})

    @blueprint.get("/health")
    def health():
        active = detector()
        return jsonify(
            {
                "ok": True,
                "demo": "scavenger_hunt",
                "sessions": registry.count(),
                "model": {"ready": active.ready, "backend": active.name, "detail": active.describe()},
            }
        )

    @blueprint.get("/state")
    def state():
        return jsonify(core_for_request().state(_now()))

    @blueprint.post("/frame")
    def frame():
        """One webcam frame: detect, advance the game, return the new state."""
        core = core_for_request()
        uploaded = request.files.get("frame")
        data = uploaded.read() if uploaded is not None else request.get_data()
        image = _decode(data)
        if image is None:
            # Still advance the clock: the timer must run even if a frame is bad.
            return jsonify({"ok": False, "error": "undecodable frame", **core.state(_now())})
        detections = detector().detect(image)
        return jsonify({"ok": True, **core.update(detections, _now())})

    @blueprint.post("/upload")
    def upload():
        """A photo instead of a live frame: same detection path, one shot."""
        core = core_for_request()
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify({"ok": False, "error": "No file was sent."}), 400
        image = _decode(uploaded.read())
        if image is None:
            return jsonify({"ok": False, "error": "Could not decode that image."}), 400
        detections = detector().detect(image)
        state = core.submit_photo(detections, _now())
        return jsonify({"ok": True, "source": uploaded.filename, **state})

    @blueprint.post("/command")
    def command():
        payload = request.get_json(silent=True) or {}
        core = core_for_request()
        result = core.handle_command(
            str(payload.get("command", "")), {**(payload.get("payload") or {}), "now": _now()}
        )
        return jsonify(result)

    @blueprint.post("/reset")
    def reset():
        core = core_for_request()
        core.reset()
        return jsonify({"ok": True, **core.state(_now())})

    @blueprint.get("/snapshot")
    def snapshot():
        core = core_for_request()
        ok, buffer = cv2.imencode(".png", core.render_canvas())
        if not ok:  # pragma: no cover
            return jsonify({"error": "encode failed"}), 500
        return Response(
            buffer.tobytes(),
            mimetype="image/png",
            headers={"Content-Disposition": 'attachment; filename="scavenger-hunt.png"'},
        )

    return blueprint, None, registry


def main() -> None:  # pragma: no cover - starts a server
    parser = argparse.ArgumentParser(description="Scavenger hunt web app")
    add_web_arguments(parser, default_port=5006)
    parser.add_argument(
        "--model",
        default=None,
        help="Model name or path for ultralytics, or an .onnx file (default: yolo26n.pt, downloaded on first use)",
    )
    parser.add_argument("--rounds", type=int, default=HuntConfig.rounds)
    parser.add_argument("--seconds", type=float, default=HuntConfig.round_seconds)
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Run without a model using a scripted detector, for UI checks",
    )
    args = parser.parse_args()

    if args.model:
        _CONFIG.model = args.model
    _CONFIG.rounds = args.rounds
    _CONFIG.round_seconds = args.seconds

    if args.demo_mode:
        from .detector import ScriptedDetector

        set_detector(ScriptedDetector())
        print("Demo mode: scripted detector, nothing will ever be found.")

    blueprint, _, _ = build()
    app = create_app([], name="scavenger_hunt")
    app.register_blueprint(blueprint)
    print("Scavenger hunt:")
    print(f"  model: {detector().describe()}")
    run_standalone(app, port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
