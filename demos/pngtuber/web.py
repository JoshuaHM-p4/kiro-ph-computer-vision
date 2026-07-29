"""PNGTuber web app.

Run standalone::

    .venv/bin/python -m demos.pngtuber.web      # http://127.0.0.1:5004/pngtuber/

The browser runs MediaPipe Face Landmarker and streams landmarks; this process
computes head yaw and the expression, then reports which sprite to show. Sprite
images are served from the assets folder by name.
"""

from __future__ import annotations

import argparse
from typing import Any

from flask import abort, send_file

from ..common.webapp import add_web_arguments, create_app, create_demo_blueprint, run_standalone
from .config import EXPRESSIONS, YAW_BUCKETS, PngTuberConfig
from .core import PngTuberCore

URL_PREFIX = "/pngtuber"

_CONFIG = PngTuberConfig()


def _page_context() -> dict[str, Any]:
    core = PngTuberCore(_CONFIG)
    return {
        "base_path": URL_PREFIX,
        "sprites": [f"{bucket}_{expression}" for bucket in YAW_BUCKETS for expression in EXPRESSIONS],
        "missing": core.sprites.missing,
        "expressions": list(EXPRESSIONS),
    }


def build():
    """Return ``(blueprint, sock, registry)`` for mounting."""
    blueprint, sock, registry = create_demo_blueprint(
        name="pngtuber",
        core_factory=lambda: PngTuberCore(_CONFIG),
        template="pngtuber.html",
        url_prefix=URL_PREFIX,
        page_context=_page_context,
        import_name="demos.pngtuber",
    )

    @blueprint.get("/sprite/<name>")
    def sprite(name: str):
        """Serve one sprite by id, e.g. /pngtuber/sprite/center_happy."""
        bucket, _, expression = name.partition("_")
        if bucket not in YAW_BUCKETS or expression not in EXPRESSIONS:
            abort(404)
        path = _CONFIG.sprites_dir / _CONFIG.sprite_name(bucket, expression)
        if not path.is_file():
            abort(404)
        return send_file(path, max_age=3600)

    return blueprint, sock, registry


def main() -> None:  # pragma: no cover - starts a server
    parser = argparse.ArgumentParser(description="PNGTuber web app")
    add_web_arguments(parser, default_port=5004)
    args = parser.parse_args()

    blueprint, sock, _ = build()
    app = create_app([(blueprint, sock)], name="pngtuber")
    print("PNGTuber:")
    run_standalone(app, port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
