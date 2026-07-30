"""Home hub: every demo in one Flask app.

Run it with::

    .venv/bin/python -m demos.home.web           # http://127.0.0.1:5000/

All four demo blueprints are mounted on this single process, so the hub links
straight to them: no second server, one WebSocket endpoint per demo. Each demo
can still run alone on its own port via ``python -m demos.<demo>.web``.

Security: binds to 127.0.0.1 and has no authentication. The demos drive a webcam
in the visitor's browser, so do not expose this to a network you do not trust.
"""

from __future__ import annotations

import argparse
from typing import Any

from flask import Blueprint, Flask, jsonify, render_template

from .. import DEMOS
from ..air_canvas import web as air_canvas_web
from ..image_lab import web as image_lab_web
from ..common.echo import build as build_echo
from ..common.webapp import add_web_arguments, create_app, run_standalone
from ..pngtuber import web as pngtuber_web
from ..sam_labeler import web as sam_labeler_web
from ..scavenger_hunt import web as scavenger_web
from ..pngtuber.core import PngTuberCore
from ..six_seven_counter import web as six_seven_web
from ..slide_presenter import web as slides_web
from ..slide_presenter.core import Deck

# slug -> (module, url prefix)
BUILDERS = {
    "image-lab": image_lab_web,
    "air-canvas": air_canvas_web,
    "slide-presenter": slides_web,
    "six-seven": six_seven_web,
    "pngtuber": pngtuber_web,
    "scavenger-hunt": scavenger_web,
    "sam-labeler": sam_labeler_web,
}


def asset_status() -> dict[str, dict[str, Any]]:
    """Report missing assets so the hub can warn instead of failing at runtime.

    A demo whose assets are absent still loads; the card just carries a hint
    about the generator command to run.
    """
    deck = Deck.load(slides_web._CONFIG.slides_dir)
    tuber = PngTuberCore(pngtuber_web._CONFIG)
    return {
        "slide-presenter": {
            "ok": not deck.is_empty,
            "detail": f"{len(deck)} slides",
            "fix": ".venv/bin/python -m demos.tools.make_sample_slides",
        },
        "pngtuber": {
            "ok": not tuber.sprites.missing,
            "detail": f"{12 - len(tuber.sprites.missing)}/12 sprites",
            "fix": ".venv/bin/python -m demos.tools.make_placeholder_sprites",
        },
    }


def create_hub_blueprint() -> Blueprint:
    """The hub's own pages.

    These live on a blueprint rather than directly on the app so Jinja resolves
    ``index.html`` against ``demos/home/templates``.
    """
    blueprint = Blueprint("home", "demos.home", template_folder="templates")

    @blueprint.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            demos=DEMOS,
            prefixes={slug: module.URL_PREFIX for slug, module in BUILDERS.items()},
            assets=asset_status(),
        )

    @blueprint.get("/demos.json")
    def demos_json():
        """Machine-readable registry, handy for scripting or a custom launcher."""
        return jsonify(
            [
                {
                    "slug": demo.slug,
                    "title": demo.title,
                    "tagline": demo.tagline,
                    "description": demo.description,
                    "path": BUILDERS[demo.slug].URL_PREFIX,
                    "standalonePort": demo.port,
                    "desktop": demo.desktop_module,
                }
                for demo in DEMOS
            ]
        )

    return blueprint


def create_home_app() -> Flask:
    """Build the hub with every demo blueprint mounted."""
    blueprints = []
    plain: list = []
    for module in BUILDERS.values():
        blueprint, sock, _ = module.build()
        # The image lab has no landmark stream, so build() hands back sock=None.
        (blueprints if sock is not None else plain).append(
            (blueprint, sock) if sock is not None else blueprint
        )

    echo_bp, echo_sock, _ = build_echo()
    blueprints.append((echo_bp, echo_sock))

    app = create_app(blueprints, name="demos_home")
    for blueprint in plain:
        app.register_blueprint(blueprint)
    app.register_blueprint(create_hub_blueprint())
    return app


def main() -> None:  # pragma: no cover - starts a server
    parser = argparse.ArgumentParser(description="Demo suite home hub")
    add_web_arguments(parser, default_port=5000)
    args = parser.parse_args()

    app = create_home_app()
    print("Demo hub:")
    for demo in DEMOS:
        print(f"  {demo.title:18} {BUILDERS[demo.slug].URL_PREFIX}")
    run_standalone(app, port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
