"""Air canvas desktop app.

Run it with::

    .venv/bin/python -m demos.air_canvas.desktop
    .venv/bin/python -m demos.air_canvas.desktop --camera 1 --no-face

Gestures
    index finger only        draw
    index + middle           hover: rest on a palette cell to select it
    thumb-index pinch        erase
    fist / open palm         idle

Keys
    q / ESC quit      d debug overlay   SPACE mirror   s screenshot
    c clear           z undo            [ / ] size     - / = opacity
    f face mesh       h hand skeleton   g gesture legend

The renderer and key handler are built by factories so they can be exercised
headlessly in tests; ``main`` only wires them to the camera loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..common import hud
from ..common.camera import CameraLoop, add_camera_arguments, camera_config_from_args
from ..common.geometry import to_pixels
from .config import AirCanvasConfig
from .core import DRAW, ERASE, AirCanvasCore, draw_gesture_legend, draw_palette

TOOL_COLORS = {
    DRAW: hud.THEME.lime,
    ERASE: hud.THEME.red,
    "hover": hud.THEME.amber,
}


@dataclass
class DesktopOptions:
    """Display toggles for the desktop view."""

    show_face: bool = True
    show_hands: bool = True
    # The gesture legend is on by default: the hand shapes are not discoverable.
    show_legend: bool = True
    # Darkening the webcam makes bright ink readable over a busy background.
    dim_camera: float = 0.55


def make_renderer(core: AirCanvasCore, options: DesktopOptions) -> Callable:
    """Build the ``render(frame, landmark_frame, loop)`` callback."""

    def render(frame: np.ndarray, landmark_frame, loop) -> np.ndarray:
        state = core.update(landmark_frame, loop.elapsed)
        height, width = frame.shape[:2]

        if options.dim_camera > 0:
            frame[:] = (frame.astype(np.float32) * (1.0 - options.dim_camera)).astype(np.uint8)

        if options.show_face and landmark_frame.face is not None:
            hud.draw_face(frame, landmark_frame.face)

        hud.alpha_blit(frame, core.render_layer(width, height), (0, 0))

        if options.show_hands:
            for hand in landmark_frame.hands:
                hud.draw_hand(frame, hand, color=hud.THEME.cyan)

        draw_palette(frame, core, now=loop.elapsed)
        _draw_cursor(frame, state)

        hud.title(frame, "AIR CANVAS", f"{state['strokeCount']} strokes")
        if options.show_legend:
            draw_gesture_legend(frame, state["tool"])
        if loop.debug:
            hud.text(
                frame,
                "c clear   z undo   [ ] size   - = opacity   g legend   f face   h hands",
                (26, height - 78),
                scale=0.4,
                color=hud.THEME.dim,
            )
            hud.status_strip(
                frame,
                [
                    ("TOOL", state["tool"].upper()),
                    ("COLOR", state["colorName"].upper()),
                    ("SIZE", f"{state['sizeFraction'] * 100:3.0f}%"),
                    ("OPACITY", f"{state['opacity'] * 100:3.0f}%"),
                    ("FPS", f"{loop.fps.fps:4.1f}"),
                ],
                y=height - 68,
            )
        return frame

    return render


def _draw_cursor(frame: np.ndarray, state: dict) -> None:
    """Brush preview at the fingertip, sized and coloured like the ink."""
    if state["cursor"] is None:
        return
    height, width = frame.shape[:2]
    center = to_pixels((state["cursor"]["x"], state["cursor"]["y"]), width, height)
    scale = 2.5 if state["tool"] == ERASE else 1.0
    radius = max(3, int(state["size"] * height * scale / 2))
    accent = TOOL_COLORS.get(state["tool"], hud.THEME.dim)

    if state["tool"] == ERASE:
        cv2.circle(frame, center, radius, accent, 2, cv2.LINE_AA)
        cv2.line(frame, (center[0] - radius, center[1]), (center[0] + radius, center[1]), accent, 1)
    elif state["tool"] == DRAW:
        cv2.circle(frame, center, radius, tuple(int(c) for c in state["color"]), -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius + 2, accent, 1, cv2.LINE_AA)
    else:
        hud.crosshair(frame, center, radius=16, color=accent)


def make_key_handler(core: AirCanvasCore, options: DesktopOptions) -> Callable:
    """Build the ``on_key(key, loop) -> handled`` callback."""

    def on_key(key: int, loop) -> bool:
        if key == ord("c"):
            core.clear()
        elif key == ord("z"):
            core.undo()
        elif key == ord("["):
            core.set_size(core.size * 0.8)
        elif key == ord("]"):
            core.set_size(core.size * 1.25)
        elif key == ord("-"):
            core.set_opacity(core.opacity - 0.1)
        elif key == ord("="):
            core.set_opacity(core.opacity + 0.1)
        elif key == ord("f"):
            options.show_face = not options.show_face
        elif key == ord("h"):
            options.show_hands = not options.show_hands
        elif key == ord("g"):
            options.show_legend = not options.show_legend
        else:
            return False  # fall through to the shared bindings
        return True

    return on_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Air canvas: paint with your fingertip")
    add_camera_arguments(parser)
    parser.add_argument("--no-face", action="store_true", help="Do not run or draw the face mesh")
    parser.add_argument("--no-hands", action="store_true", help="Do not draw the hand skeleton")
    parser.add_argument("--no-legend", action="store_true", help="Hide the gesture legend")
    parser.add_argument(
        "--dim-camera",
        type=float,
        default=0.55,
        help="Darken the webcam behind the paint layer, 0..1 (default 0.55)",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - interactive
    args = parse_args()
    options = DesktopOptions(
        show_face=not args.no_face,
        show_hands=not args.no_hands,
        show_legend=not args.no_legend,
        dim_camera=args.dim_camera,
    )
    core = AirCanvasCore(AirCanvasConfig())
    loop = CameraLoop(
        camera_config_from_args(args, "air canvas"),
        hands=True,
        face=options.show_face,
    )

    print(__doc__)
    loop.run(make_renderer(core, options), on_key=make_key_handler(core, options))
    if loop.last_screenshot:
        print(f"Saved {loop.last_screenshot}")


if __name__ == "__main__":  # pragma: no cover
    main()
