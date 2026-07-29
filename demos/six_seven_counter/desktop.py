"""6-7 rep counter desktop app.

Run it with::

    .venv/bin/python -m demos.six_seven_counter.desktop

Stand so your shoulders and both hands are in frame, then alternate hands like a
see-saw: as one rises the other drops. Every swap of which hand is higher counts as
one rep, so pumping the same hand twice cannot register. Your hips do not need to
be visible, so you can stand close to the camera.

Nothing is compared against a fixed height — only the difference between your two
wrists — so there is nothing to calibrate and moving up or down in frame does not
matter.

Keys
    q / ESC quit   d debug   SPACE mirror   s screenshot
    r reset count  a add one manually       p re-prepare
    g hide the see-saw overlay              k hide the skeleton
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..common import hud
from ..common.camera import CameraLoop, add_camera_arguments, camera_config_from_args
from .config import CounterConfig
from .core import SixSevenCore, draw_overlay


@dataclass
class CounterOptions:
    """Display toggles for the counter view."""

    show_skeleton: bool = True
    # The see-saw line and tilt meter: the signal the counter actually uses.
    show_guides: bool = True
    dim_camera: float = 0.25


def make_renderer(core: SixSevenCore, options: CounterOptions) -> Callable:
    """Build the ``render(frame, landmark_frame, loop)`` callback."""


    def render(frame: np.ndarray, landmark_frame, loop) -> np.ndarray:
        state = core.update(landmark_frame, loop.elapsed)

        if options.dim_camera > 0:
            frame[:] = (frame.astype(np.float32) * (1.0 - options.dim_camera)).astype(np.uint8)

        if options.show_skeleton and landmark_frame.pose is not None:
            hud.draw_pose(frame, landmark_frame.pose)

        if options.show_guides:
            draw_overlay(frame, core, state, debug=loop.debug)
        else:
            hud.text(
                frame,
                str(state["count"]),
                (frame.shape[1] - 140, 120),
                scale=3.2,
                color=hud.THEME.lime,
                thickness=6,
            )

        hud.title(frame, "6-7 COUNTER", "swap which hand is higher")
        return frame

    return render


def make_key_handler(core: SixSevenCore, options: CounterOptions) -> Callable:
    """Build the ``on_key(key, loop) -> handled`` callback."""

    def on_key(key: int, loop) -> bool:
        if key == ord("r"):
            core.reset()
        elif key == ord("a"):
            core.handle_command("add", {})
        elif key == ord("g"):
            options.show_guides = not options.show_guides
        elif key == ord("p"):
            core.prepare()
        elif key == ord("k"):
            options.show_skeleton = not options.show_skeleton
        else:
            return False
        return True

    return on_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="6-7 alternating hand rep counter")
    add_camera_arguments(parser)
    parser.add_argument(
        "--tilt",
        type=float,
        default=CounterConfig.tilt_enter,
        help=(
            "How far apart the wrists must be before a side is claimed, in torso "
            "lengths (default 0.15). Larger means bigger, more deliberate swaps."
        ),
    )
    parser.add_argument(
        "--prepare",
        type=float,
        default=CounterConfig.prepare_seconds,
        help="Seconds you must be fully in frame before counting starts",
    )
    parser.add_argument("--no-guides", action="store_true", help="Hide the see-saw overlay")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - interactive
    args = parse_args()
    core = SixSevenCore(
        CounterConfig(tilt_enter=args.tilt, prepare_seconds=args.prepare)
    )
    options = CounterOptions(show_guides=not args.no_guides)
    loop = CameraLoop(camera_config_from_args(args, "6-7 counter"), pose=True)

    print(__doc__)
    loop.run(make_renderer(core, options), on_key=make_key_handler(core, options))
    print(f"Final count: {core.count}")


if __name__ == "__main__":  # pragma: no cover
    main()
