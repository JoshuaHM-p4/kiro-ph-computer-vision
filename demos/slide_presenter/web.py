"""Slide presenter web app.

Run standalone::

    .venv/bin/python -m demos.slide_presenter.web    # http://127.0.0.1:5002/slides/

The browser streams hand landmarks; this process owns the slide index and the
laser position. Slide images are served from the deck folder by index, so the
page never needs to know the filenames.
"""

from __future__ import annotations

import argparse
from typing import Any

from flask import abort, send_file

from ..common.webapp import add_web_arguments, create_app, create_demo_blueprint, run_standalone
from .config import SlideConfig
from .core import Deck, SlidePresenterCore

URL_PREFIX = "/slides"

# One deck shared by every session: the images are identical for all viewers and
# decoding them per session would waste memory. Per-session state (index, laser)
# still lives in each session's core.
_CONFIG = SlideConfig()


def _core_factory() -> SlidePresenterCore:
    return SlidePresenterCore(_CONFIG)


def _page_context() -> dict[str, Any]:
    deck = Deck.load(_CONFIG.slides_dir)
    return {
        "base_path": URL_PREFIX,
        "deck": {"count": len(deck), "names": [p.name for p in deck.paths]},
        "hands": {"next": _CONFIG.next_hand, "previous": _CONFIG.previous_hand},
    }


def build():
    """Return ``(blueprint, sock, registry)`` for mounting."""
    blueprint, sock, registry = create_demo_blueprint(
        name="slide_presenter",
        core_factory=_core_factory,
        template="slide_presenter.html",
        url_prefix=URL_PREFIX,
        page_context=_page_context,
        import_name="demos.slide_presenter",
    )

    @blueprint.get("/slide/<int:index>")
    def slide_image(index: int):
        """Serve one slide image by position in the deck."""
        deck = Deck.load(_CONFIG.slides_dir)
        if deck.is_empty:
            abort(404)
        # Index is taken modulo the deck length so a stale client cannot 404.
        path = deck.paths[index % len(deck)]
        return send_file(path, max_age=3600)

    return blueprint, sock, registry


def main() -> None:  # pragma: no cover - starts a server
    parser = argparse.ArgumentParser(description="Slide presenter web app")
    add_web_arguments(parser, default_port=5002)
    parser.add_argument("--slides", default=None, help="Folder of slide images")
    args = parser.parse_args()
    if args.slides:
        from pathlib import Path

        _CONFIG.slides_dir = Path(args.slides)

    blueprint, sock, _ = build()
    app = create_app([(blueprint, sock)], name="slide_presenter")
    print("Slide presenter:")
    run_standalone(app, port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
