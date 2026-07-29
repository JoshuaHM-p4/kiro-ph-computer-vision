"""6-7 rep counter web app.

Run standalone::

    .venv/bin/python -m demos.six_seven_counter.web   # http://127.0.0.1:5003/six-seven/

The browser runs MediaPipe Pose and streams landmarks; this process owns the
per-wrist state machines and the alternation gate.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..common.webapp import add_web_arguments, create_app, create_demo_blueprint, run_standalone
from .config import CounterConfig
from .core import SixSevenCore

URL_PREFIX = "/six-seven"


def _page_context() -> dict[str, Any]:
    config = CounterConfig()
    return {
        "base_path": URL_PREFIX,
        "tilt": {"enter": config.tilt_enter, "prepareSeconds": config.prepare_seconds},
    }


def build():
    """Return ``(blueprint, sock, registry)`` for mounting."""
    return create_demo_blueprint(
        name="six_seven",
        core_factory=lambda: SixSevenCore(CounterConfig()),
        template="six_seven.html",
        url_prefix=URL_PREFIX,
        page_context=_page_context,
        import_name="demos.six_seven_counter",
    )


def main() -> None:  # pragma: no cover - starts a server
    parser = argparse.ArgumentParser(description="6-7 rep counter web app")
    add_web_arguments(parser, default_port=5003)
    args = parser.parse_args()

    blueprint, sock, _ = build()
    app = create_app([(blueprint, sock)], name="six_seven")
    print("6-7 counter:")
    run_standalone(app, port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
