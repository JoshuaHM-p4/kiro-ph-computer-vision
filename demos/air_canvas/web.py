"""Air canvas web app.

Run standalone::

    .venv/bin/python -m demos.air_canvas.web          # http://127.0.0.1:5001/air-canvas/

Or let ``demos.home.web`` mount it alongside the other demos.

The browser runs MediaPipe Hands and streams landmarks; this process owns the
paint engine. The page draws committed strokes onto a persistent canvas and
re-syncs from the server whenever the stroke revision changes (undo, clear, or a
reconnect), which keeps the per-frame payload to just the active stroke tail.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..common.webapp import add_web_arguments, create_app, create_demo_blueprint, run_standalone
from .config import AirCanvasConfig
from .core import AirCanvasCore

URL_PREFIX = "/air-canvas"


def _page_context() -> dict[str, Any]:
    """Static data the template needs: palette geometry and brush limits."""
    core = AirCanvasCore(AirCanvasConfig())
    return {
        "palette": core.palette_json(),
        "limits": {
            "sizeMin": core.config.size_min,
            "sizeMax": core.config.size_max,
            "opacityMin": core.config.opacity_min,
            "opacityMax": core.config.opacity_max,
        },
        "base_path": URL_PREFIX,
    }


def build():
    """Return ``(blueprint, sock, registry)`` for mounting."""
    return create_demo_blueprint(
        name="air_canvas",
        core_factory=lambda: AirCanvasCore(AirCanvasConfig()),
        template="air_canvas.html",
        url_prefix=URL_PREFIX,
        page_context=_page_context,
        import_name="demos.air_canvas",
    )


def main() -> None:  # pragma: no cover - starts a server
    parser = argparse.ArgumentParser(description="Air canvas web app")
    add_web_arguments(parser, default_port=5001)
    args = parser.parse_args()

    blueprint, sock, _ = build()
    app = create_app([(blueprint, sock)], name="air_canvas")
    print("Air canvas:")
    run_standalone(app, port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
