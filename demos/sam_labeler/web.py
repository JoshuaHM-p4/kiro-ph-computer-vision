"""SAM labeler web app: capture or upload, prompt with text, style the results.

Run standalone::

    .venv/bin/python -m demos.sam_labeler.web                 # demo mode available
    HF_TOKEN=hf_... .venv/bin/python -m demos.sam_labeler.web  # real SAM 3.1

Like the image lab this is a plain request/response app, not a landmark stream: the
model runs here in Python, so the browser sends an image and gets back state.

## About the Hugging Face token

``facebook/sam3`` is gated, so loading it needs a read token. The token is:

* held in memory only, for as long as the process runs,
* never written to disk, never logged, never echoed back in any response,
* replaced by a boolean in every state payload the page receives.

Prefer the ``HF_TOKEN`` environment variable if you would rather not paste a
credential into a browser form at all. And keep this on localhost: it has no
authentication, so anyone who can reach the port can use your token's access.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import threading
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
from .backend import Backend, UnavailableBackend, load_backend
from .jobs import Job, watch_cache
from .config import EFFECTS, MODES, SETTING_BOUNDS, SamLabelerConfig
from .core import SamLabelerCore

URL_PREFIX = "/sam-labeler"

_CONFIG = SamLabelerConfig()

# A SAM checkpoint is far too large to load per visitor, so one backend is shared.
# _TOKEN_FINGERPRINT records which credential built it (a hash, never the token) so a
# different token triggers a reload rather than silently reusing someone else's access.
_BACKEND: Backend | None = None
_TOKEN_FINGERPRINT: str | None = None
# Set when a backend is injected explicitly, which pins it against the reload check
# below. Without this, the first request would helpfully "reload" a test double.
_BACKEND_FORCED = False
_DEMO_MODE = False


def _fingerprint(token: str | None) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()[:16]


def backend_for(token: str | None, *, stub: bool | None = None) -> Backend:
    """Return the shared backend, loading or reloading it if the token changed."""
    global _BACKEND, _TOKEN_FINGERPRINT
    if _BACKEND_FORCED and _BACKEND is not None:
        return _BACKEND
    use_stub = _DEMO_MODE if stub is None else stub
    fingerprint = "stub" if use_stub else _fingerprint(token)
    if _BACKEND is None or _TOKEN_FINGERPRINT != fingerprint:
        _BACKEND = load_backend(_CONFIG, token, stub=use_stub)
        _TOKEN_FINGERPRINT = fingerprint
    return _BACKEND


def set_backend(instance: Backend | None, *, fingerprint: str = "injected") -> None:
    """Install a backend directly, pinning it. Used by the tests and demo mode.

    Pass ``None`` to release the pin and go back to loading from the token.
    """
    global _BACKEND, _TOKEN_FINGERPRINT, _BACKEND_FORCED
    _BACKEND = instance
    _TOKEN_FINGERPRINT = fingerprint if instance is not None else None
    _BACKEND_FORCED = instance is not None


class Session:
    """Per-visitor state: their image, prompts, styles, and their token."""

    def __init__(self) -> None:
        self.core = SamLabelerCore(replace(_CONFIG))
        # Held in memory for this session only. Never serialised.
        self.token: str | None = os.environ.get("HF_TOKEN") or None
        self.demo_mode = _DEMO_MODE
        # Full-resolution original image, kept so resize can re-apply _fit() without
        # asking the user to re-upload.
        self._original: np.ndarray | None = None
        # Model loading and inference both run on a worker thread; the page polls
        # this for a stage and a byte count instead of waiting on a blocked request.
        self.job = Job()

    @property
    def has_token(self) -> bool:
        return bool(self.token)

    def backend(self) -> Backend:
        """The backend, loading it if necessary. May block: call from a worker."""
        return backend_for(self.token, stub=self.demo_mode)

    def peek_backend(self) -> Backend:
        """Whatever is already loaded, without triggering a download.

        Request handlers use this so rendering a page can never block on the hub.
        """
        if _BACKEND is not None:
            return _BACKEND
        if self.demo_mode:
            return backend_for(None, stub=True)
        if not self.has_token:
            return UnavailableBackend(
                f"{_CONFIG.model_id} is gated. Paste a read token or switch on demo mode."
            )
        return UnavailableBackend("Model not loaded yet. Press Use to load it.")

    def load_backend_async(self) -> bool:
        """Kick off loading on a worker thread. False if a job is already running."""
        if self.demo_mode:
            backend_for(None, stub=True)
            return True

        def work(job: Job) -> None:
            job.set_stage("resolving", f"contacting Hugging Face for {_CONFIG.model_id}")
            stop = threading.Event()
            watch_cache(job, _CONFIG.model_id, stop)
            try:
                backend = backend_for(self.token, stub=False)
            finally:
                stop.set()
            if not getattr(backend, "ready", False):
                job.fail(backend.describe())
            else:
                job.set_stage("ready", backend.describe())

        return self.job.start("load", work, stage="starting")

    def segment_async(self, prompts) -> bool:
        """Run segmentation on a worker thread."""

        def work(job: Job) -> None:
            backend = self.peek_backend()
            if not getattr(backend, "ready", False):
                job.fail(backend.describe())
                return
            job.set_stage("segmenting", f"{backend.describe()}")
            self.core.run(backend, prompts)
            if self.core.error:
                job.fail(self.core.error)
            else:
                job.set_stage(
                    "done", f"{len(self.core.instances)} object(s) found"
                )

        return self.job.start("segment", work, stage="starting")


def _decode(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def build():
    """Return ``(blueprint, sock, registry)``; ``sock`` is None for this demo."""
    blueprint = Blueprint(
        "sam_labeler",
        "demos.sam_labeler",
        template_folder="templates",
        static_folder="static",
        url_prefix=URL_PREFIX,
    )
    registry = SessionRegistry(Session)

    def session_for_request() -> Session:
        entry = registry.get(_session_id())
        assert entry is not None
        return entry.core  # SessionRegistry stores whatever the factory returned

    def payload_for(session: Session, **extra: Any) -> dict[str, Any]:
        """State plus token/backend status. The token itself never appears here."""
        backend = session.peek_backend()
        return {
            **session.core.state(),
            "job": session.job.to_json(),
            "hasToken": session.has_token,
            "demoMode": session.demo_mode,
            "backendReady": bool(getattr(backend, "ready", False)),
            "backendName": getattr(backend, "name", "unknown"),
            "backendDetail": backend.describe(),
            **extra,
        }

    @blueprint.get("/")
    def index() -> str:
        session_token_present = bool(os.environ.get("HF_TOKEN"))
        return render_template(
            "sam_labeler.html",
            base_path=URL_PREFIX,
            effects=list(EFFECTS),
            modes=list(MODES),
            bounds=SETTING_BOUNDS,
            model_id=_CONFIG.model_id,
            default_dim=_CONFIG.max_dimension,
            token_from_env=session_token_present,
            examples=["person", "laptop", "coffee mug", "chair", "plant", "phone"],
        )

    @blueprint.get("/health")
    def health():
        backend = backend_for(os.environ.get("HF_TOKEN"))
        return jsonify(
            {
                "ok": True,
                "demo": "sam_labeler",
                "sessions": registry.count(),
                "backend": {"ready": backend.ready, "name": backend.name},
            }
        )

    @blueprint.get("/state")
    def state():
        return jsonify(payload_for(session_for_request()))

    @blueprint.post("/token")
    def token():
        """Accept a Hugging Face read token for this session, or clear it."""
        body = request.get_json(silent=True) or {}
        session = session_for_request()
        raw = str(body.get("token", "")).strip()
        session.demo_mode = bool(body.get("demoMode", session.demo_mode))
        session.token = raw or None
        if session.token or session.demo_mode:
            # Loading happens on a worker; the page polls /status for progress.
            session.load_backend_async()
        else:
            set_backend(None)
        return jsonify({"ok": True, **payload_for(session)})

    @blueprint.post("/capture")
    def capture():
        """A still frame from the browser's webcam."""
        session = session_for_request()
        uploaded = request.files.get("frame")
        image = _decode(uploaded.read() if uploaded is not None else request.get_data())
        if image is None:
            return jsonify({"ok": False, "error": "Could not decode that frame."}), 400
        session._original = image.copy()
        session.core.set_source(image, "webcam.png")
        return jsonify({"ok": True, **payload_for(session)})

    @blueprint.post("/upload")
    def upload():
        session = session_for_request()
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify({"ok": False, "error": "No file was sent."}), 400
        raw = uploaded.read()
        if not session.core.load_bytes(raw, uploaded.filename or "upload.png"):
            return jsonify({"ok": False, "error": session.core.error}), 400
        # Keep the full-res bytes so resize can redecode without re-uploading.
        session._original = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return jsonify({"ok": True, **payload_for(session)})

    @blueprint.post("/run")
    def run():
        """Start segmentation on a worker thread; poll /status for the result."""
        body = request.get_json(silent=True) or {}
        session = session_for_request()
        # The explicit keys go *after* the spread: payload_for carries the core's own
        # "error" (usually None) and would otherwise overwrite the message.
        if not session.core.has_source:
            session.core.error = "Capture or upload a picture first."
            return jsonify({**payload_for(session), "ok": False, "error": session.core.error}), 400
        prompts = session.core.set_prompts(body.get("prompts", session.core.prompts))
        if not prompts:
            session.core.error = "Enter at least one thing to look for."
            return jsonify({**payload_for(session), "ok": False, "error": session.core.error}), 400
        if not session.segment_async(prompts):
            return jsonify({**payload_for(session), "ok": False, "error": "Already working."}), 409
        return jsonify({"ok": True, **payload_for(session)})

    @blueprint.get("/status")
    def status():
        """Cheap poll target: job progress plus the current state."""
        return jsonify(payload_for(session_for_request()))

    @blueprint.post("/command")
    def command():
        body = request.get_json(silent=True) or {}
        session = session_for_request()
        result = session.core.handle_command(
            str(body.get("command", "")), body.get("payload") or {}
        )
        if not result.get("ok", False) and "unknown" in result:
            return jsonify(result), 400

        # Re-apply the original image at the new resolution after a maxDim change.
        payload = body.get("payload") or {}
        if str(body.get("command")) == "settings" and "maxDim" in payload and session._original is not None:
            session.core.set_source(session._original, session.core.source_name)

        return jsonify({**result, **payload_for(session)})

    @blueprint.get("/result.png")
    def result_png():
        session = session_for_request()
        data = session.core.encode_png()
        return Response(data, mimetype="image/png", headers={"Cache-Control": "no-store"})

    @blueprint.get("/source.png")
    def source_png():
        session = session_for_request()
        if not session.core.has_source:
            return jsonify({"error": "no image"}), 404
        ok, buffer = cv2.imencode(".png", session.core.source)
        return Response(buffer.tobytes(), mimetype="image/png", headers={"Cache-Control": "no-store"})

    @blueprint.get("/snapshot")
    def snapshot():
        session = session_for_request()
        return Response(
            session.core.encode_png(),
            mimetype="image/png",
            headers={"Content-Disposition": 'attachment; filename="sam-labeler.png"'},
        )

    return blueprint, None, registry


def main() -> None:  # pragma: no cover - starts a server
    global _DEMO_MODE
    parser = argparse.ArgumentParser(description="SAM 3.1 text-prompted labeler")
    add_web_arguments(parser, default_port=5007)
    parser.add_argument("--model", default=None, help=f"Model id (default {_CONFIG.model_id})")
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Use synthetic masks instead of loading SAM, to explore the effects offline",
    )
    args = parser.parse_args()

    if args.model:
        _CONFIG.model_id = args.model
    _DEMO_MODE = args.demo_mode

    blueprint, _, _ = build()
    app = create_app([], name="sam_labeler")
    app.register_blueprint(blueprint)
    print("SAM labeler:")
    print(f"  model: {_CONFIG.model_id}")
    print(f"  token: {'from HF_TOKEN' if os.environ.get('HF_TOKEN') else 'paste one in the page'}")
    if args.demo_mode:
        print("  demo mode: synthetic masks, no model will be loaded")
    run_standalone(app, port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
