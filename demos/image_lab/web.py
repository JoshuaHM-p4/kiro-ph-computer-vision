"""Image lab web app: upload an image, chain OpenCV operations, reveal the code.

Run standalone::

    .venv/bin/python -m demos.image_lab.web     # http://127.0.0.1:5005/image-lab/

Unlike the other demos this one is not landmark-driven, so it does not use the
WebSocket channel. It is a plain request/response app: the browser posts a
pipeline, the server runs it with OpenCV and returns the resulting PNG plus the
equivalent script.
"""

from __future__ import annotations

import argparse
from typing import Any

from flask import Blueprint, Response, abort, jsonify, render_template, request, send_file

from ..common.webapp import (
    SessionRegistry,
    _session_id,
    add_web_arguments,
    create_app,
    run_standalone,
)
from . import operations as ops
from .config import ImageLabConfig
from .core import ImageLabCore

URL_PREFIX = "/image-lab"

_CONFIG = ImageLabConfig()


def _core_factory() -> ImageLabCore:
    core = ImageLabCore(_CONFIG)
    # Open on a sample so the page has something to show before any upload.
    samples = core.samples()
    if samples:
        core.load_path(samples[0])
    return core


def build():
    """Return ``(blueprint, sock, registry)``; ``sock`` is None for this demo."""
    blueprint = Blueprint(
        "image_lab",
        "demos.image_lab",
        template_folder="templates",
        static_folder="static",
        url_prefix=URL_PREFIX,
    )
    registry = SessionRegistry(_core_factory)

    def core_for_request() -> ImageLabCore:
        session = registry.get(_session_id())
        assert session is not None
        return session.core

    @blueprint.get("/")
    def index() -> str:
        return render_template(
            "image_lab.html",
            base_path=URL_PREFIX,
            catalog=ops.catalog_json(),
            samples=[path.name for path in ImageLabCore(_CONFIG).samples()],
            max_steps=_CONFIG.max_steps,
        )

    @blueprint.get("/health")
    def health():
        return jsonify({"ok": True, "demo": "image_lab", "sessions": registry.count()})

    @blueprint.get("/state")
    def state():
        return jsonify(core_for_request().state())

    @blueprint.post("/upload")
    def upload():
        """Accept a file upload, or a webcam frame posted as a blob."""
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify({"ok": False, "error": "No file was sent."}), 400
        core = core_for_request()
        if not core.load_bytes(uploaded.read(), uploaded.filename or "upload.png"):
            return jsonify({"ok": False, "error": core.error}), 400
        return jsonify({"ok": True, **core.state()})

    @blueprint.post("/pipeline")
    def pipeline():
        """Replace the pipeline and return the new state, including the code."""
        payload = request.get_json(silent=True) or {}
        core = core_for_request()
        core.set_pipeline(payload.get("steps") or [])
        return jsonify({"ok": True, **core.state()})

    @blueprint.post("/command")
    def command():
        payload = request.get_json(silent=True) or {}
        core = core_for_request()
        result = core.handle_command(str(payload.get("command", "")), payload.get("payload") or {})
        return jsonify(result)

    @blueprint.post("/reset")
    def reset():
        core = core_for_request()
        core.reset()
        return jsonify({"ok": True, **core.state()})

    @blueprint.get("/result.png")
    def result_png():
        """The processed image. Cache-busted by the client via ?v=revision."""
        core = core_for_request()
        data = core.encode_png()
        if not data:
            abort(500)
        return Response(data, mimetype="image/png", headers={"Cache-Control": "no-store"})

    @blueprint.get("/source.png")
    def source_png():
        core = core_for_request()
        if not core.has_source:
            abort(404)
        return Response(
            core.encode_png(core.source), mimetype="image/png", headers={"Cache-Control": "no-store"}
        )

    @blueprint.get("/snapshot")
    def snapshot():
        """Download the result, matching the other demos' snapshot route."""
        core = core_for_request()
        return Response(
            core.encode_png(),
            mimetype="image/png",
            headers={"Content-Disposition": 'attachment; filename="image-lab.png"'},
        )

    @blueprint.get("/code")
    def code():
        """The script as a downloadable .py file."""
        core = core_for_request()
        return Response(
            core.code(),
            mimetype="text/x-python",
            headers={"Content-Disposition": 'attachment; filename="pipeline.py"'},
        )

    @blueprint.get("/sample/<name>")
    def sample(name: str):
        for path in ImageLabCore(_CONFIG).samples():
            if path.name == name:
                return send_file(path, max_age=3600)
        abort(404)

    return blueprint, None, registry


def main() -> None:  # pragma: no cover - starts a server
    parser = argparse.ArgumentParser(description="Interactive OpenCV image lab")
    add_web_arguments(parser, default_port=5005)
    args = parser.parse_args()

    blueprint, _, _ = build()
    app = create_app([], name="image_lab")
    app.register_blueprint(blueprint)
    print("Image lab:")
    run_standalone(app, port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
