"""PNGTuber desktop app.

Run it with::

    .venv/bin/python -m demos.pngtuber.desktop
    .venv/bin/python -m demos.pngtuber.desktop --background chroma --no-camera

Turn your head to switch between the left/center/right sprites, and change your
expression to switch between neutral, happy, surprised and angry. The neutral
baseline is captured over the first second, so hold a relaxed face at startup (or
press ``c`` to recapture at any time).

Keys
    q / ESC quit   d debug   SPACE mirror   s screenshot
    c recalibrate neutral    b cycle background (camera / solid / chroma)
    v camera preview inset   1-4 preview an expression

A previewed expression is not sticky: it decays back to the tracked one as soon
as your face settles, which makes it handy for checking sprite art.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..common import hud
from ..common.camera import CameraLoop, add_camera_arguments, camera_config_from_args
from .config import EXPRESSIONS, PngTuberConfig
from .core import PngTuberCore, draw_debug

BACKGROUNDS = ("camera", "solid", "chroma")
CHROMA_GREEN = (0, 177, 64)  # BGR key colour for streaming software


@dataclass
class TuberOptions:
    """Display options for the PNGTuber view."""

    background: str = "camera"
    show_inset: bool = False
    inset_width: float = 0.18
    dim_camera: float = 0.55


def make_renderer(core: PngTuberCore, options: TuberOptions) -> Callable:
    """Build the ``render(frame, landmark_frame, loop)`` callback."""

    def render(frame: np.ndarray, landmark_frame, loop) -> np.ndarray:
        state = core.update(landmark_frame, loop.elapsed)
        height, width = frame.shape[:2]

        if options.background == "camera":
            output = frame
            if options.dim_camera > 0:
                output[:] = (output.astype(np.float32) * (1.0 - options.dim_camera)).astype(np.uint8)
        elif options.background == "chroma":
            output = np.full_like(frame, CHROMA_GREEN)
        else:
            output = np.full_like(frame, core.config.background)

        core.render_sprite(output, now=loop.elapsed)

        if options.show_inset and options.background != "camera":
            _draw_inset(output, frame, landmark_frame, options)

        hud.title(output, "PNGTUBER", state["sprite"].replace("_", " / "))
        if loop.debug:
            draw_debug(output, core, state)
        return output

    return render


def _draw_inset(output: np.ndarray, frame: np.ndarray, landmark_frame, options: TuberOptions) -> None:
    """Small camera preview so you can see what the tracker sees."""
    height, width = output.shape[:2]
    inset_w = max(120, int(width * options.inset_width))
    inset_h = int(inset_w * frame.shape[0] / frame.shape[1])
    inset = cv2.resize(frame, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    if landmark_frame.face is not None:
        hud.draw_face(inset, landmark_frame.face)
    x, y = width - inset_w - 18, 18
    output[y : y + inset_h, x : x + inset_w] = inset
    cv2.rectangle(output, (x - 2, y - 2), (x + inset_w + 2, y + inset_h + 2), hud.THEME.cyan, 2, cv2.LINE_AA)


def make_key_handler(core: PngTuberCore, options: TuberOptions) -> Callable:
    """Build the ``on_key(key, loop) -> handled`` callback."""

    def on_key(key: int, loop) -> bool:
        if key == ord("c"):
            core.calibrate(loop.elapsed)
        elif key == ord("b"):
            options.background = BACKGROUNDS[
                (BACKGROUNDS.index(options.background) + 1) % len(BACKGROUNDS)
            ]
        elif key == ord("v"):
            options.show_inset = not options.show_inset
        elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
            core.handle_command("expression", {"name": EXPRESSIONS[key - ord("1")]})
        else:
            return False
        return True

    return on_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PNGTuber sprite switcher")
    add_camera_arguments(parser)
    parser.add_argument(
        "--background",
        choices=BACKGROUNDS,
        default="camera",
        help="What to draw behind the sprite (chroma = green screen)",
    )
    parser.add_argument("--inset", action="store_true", help="Show the camera preview inset")
    parser.add_argument(
        "--yaw-enter",
        type=float,
        default=PngTuberConfig.yaw_enter,
        help="Degrees of yaw needed to leave the centre sprite",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - interactive
    args = parse_args()
    config = PngTuberConfig(yaw_enter=args.yaw_enter)
    core = PngTuberCore(config)
    if core.sprites.missing:
        print(f"Missing {len(core.sprites.missing)} sprites in {config.sprites_dir}")
        print("Generate placeholders:  .venv/bin/python -m demos.tools.make_placeholder_sprites")

    options = TuberOptions(background=args.background, show_inset=args.inset)
    loop = CameraLoop(camera_config_from_args(args, "pngtuber"), face=True)

    print(__doc__)
    loop.run(make_renderer(core, options), on_key=make_key_handler(core, options))


if __name__ == "__main__":  # pragma: no cover
    main()
