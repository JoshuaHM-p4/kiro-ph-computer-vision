"""Minimal echo demo used to verify the landmark channel.

It is the smallest possible :class:`~demos.common.webapp.DemoCore`: it reports
the fingertip position and how many landmarks arrived, and it renders a trail of
recent fingertips onto a numpy canvas so ``/snapshot`` has something to show.
Useful for confirming the browser -> WebSocket -> Flask path works before
debugging a real demo's logic.

Run it with::

    .venv/bin/python -m demos.common.echo
"""

from __future__ import annotations

import argparse
from typing import Any

import cv2
import numpy as np

from . import hud
from . import landmarks as lm
from .geometry import EMAPoint, to_pixels
from .webapp import add_web_arguments, create_app, create_demo_blueprint, run_standalone

CANVAS_SIZE = (720, 1280)  # height, width


class EchoCore:
    """Tracks a smoothed fingertip and keeps a short trail."""

    def __init__(self, trail_length: int = 120):
        self.trail_length = trail_length
        self.trail: list[tuple[float, float]] = []
        self.smoother = EMAPoint(alpha=0.5)
        self.frames = 0

    def update(self, frame: lm.LandmarkFrame, now: float) -> dict[str, Any]:
        self.frames += 1
        hand = frame.primary_hand
        point = self.smoother.update(hand.index_tip if hand else None)
        if hand is not None and point is not None:
            self.trail.append(point)
            del self.trail[: max(0, len(self.trail) - self.trail_length)]

        return {
            "frames": self.frames,
            "hands": len(frame.hands),
            "face": frame.face is not None,
            "pose": frame.pose is not None,
            "point": {"x": point[0], "y": point[1]} if point else None,
            "trail": [{"x": x, "y": y} for x, y in self.trail[-40:]],
        }

    def render_canvas(self) -> np.ndarray:
        height, width = CANVAS_SIZE
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (14, 10, 20)
        points = [to_pixels(p, width, height) for p in self.trail]
        if len(points) > 1:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, hud.THEME.cyan, 4, cv2.LINE_AA)
        if points:
            hud.crosshair(canvas, points[-1], radius=22)
        hud.title(canvas, "ECHO", f"{self.frames} frames processed")
        return canvas

    def handle_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "clear":
            self.trail.clear()
            return {"ok": True, "cleared": True}
        return {"ok": False, "unknown": command}

    def reset(self) -> None:
        self.trail.clear()
        self.smoother.reset()
        self.frames = 0


def build() -> tuple:
    return create_demo_blueprint(
        name="echo",
        core_factory=EchoCore,
        template="echo.html",
        url_prefix="/echo",
        import_name="demos.common",
    )


def main() -> None:  # pragma: no cover - starts a server
    parser = argparse.ArgumentParser(description="Landmark channel echo test")
    add_web_arguments(parser, default_port=5009)
    args = parser.parse_args()

    blueprint, sock, _ = build()
    app = create_app([(blueprint, sock)], name="echo")
    print("Echo test:")
    run_standalone(app, port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()
