"""Slide presenter desktop app.

Run it with::

    .venv/bin/python -m demos.slide_presenter.desktop
    .venv/bin/python -m demos.slide_presenter.desktop --slides ~/talk/png --loop

Gestures
    index finger pointing    move the laser
    pinch right hand         next slide
    pinch left hand          previous slide

Keys
    q / ESC quit    d debug    SPACE mirror    s screenshot
    right / n next  left / p previous          r reload deck
    c webcam inset  w swap hands

If next and previous feel inverted, press ``w`` (or pass ``--swap-handedness``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..common import hud
from ..common.camera import CameraLoop, add_camera_arguments, camera_config_from_args
from .config import SlideConfig
from .core import SlidePresenterCore, draw_laser, fit_into

# cv2.waitKeyEx codes for the arrow keys differ per platform; these are the
# common Linux/GTK values, and n/p work everywhere as a fallback.
KEY_LEFT = 81
KEY_RIGHT = 83


@dataclass
class PresenterOptions:
    """Display toggles for the presenter view."""

    show_inset: bool = True
    inset_width: float = 0.22
    show_hands: bool = True


def make_renderer(core: SlidePresenterCore, options: PresenterOptions) -> Callable:
    """Build the ``render(frame, landmark_frame, loop)`` callback.

    The webcam becomes a small inset; the slide fills the window, because that is
    what an audience should be looking at.
    """

    def render(frame: np.ndarray, landmark_frame, loop) -> np.ndarray:
        state = core.update(landmark_frame, loop.elapsed)
        height, width = frame.shape[:2]

        output = np.zeros_like(frame)
        slide = core.deck.image(core.index)
        if slide is None:
            hud.title(output, "NO SLIDES", str(core.config.slides_dir))
            hud.text(
                output,
                "Generate a demo deck:  .venv/bin/python -m demos.tools.make_sample_slides",
                (26, height // 2),
                scale=0.55,
                color=hud.THEME.dim,
            )
            hud.text(output, "then press r to reload", (26, height // 2 + 30), scale=0.5, color=hud.THEME.dim)
        else:
            fit_into(output, slide)

        if state["laser"] is not None:
            draw_laser(output, (state["laser"]["x"], state["laser"]["y"]), core.config.laser_radius)

        if options.show_inset:
            _draw_inset(output, frame, landmark_frame, options)

        _draw_chrome(output, core, state, loop, options)
        return output

    return render


def inset_rect(output: np.ndarray, options: PresenterOptions) -> tuple[int, int, int, int]:
    """Pixel rect of the webcam inset: bottom-left corner of the frame."""
    height, width = output.shape[:2]
    inset_w = max(120, int(width * options.inset_width))
    inset_h = int(inset_w * 9 / 16)
    x = 18
    y = height - inset_h - 18
    return x, y, inset_w, inset_h


def _draw_inset(output: np.ndarray, frame: np.ndarray, landmark_frame, options: PresenterOptions) -> None:
    """Webcam picture-in-picture in the bottom-left corner."""
    x, y, inset_w, inset_h = inset_rect(output, options)
    inset_h = min(inset_h, frame.shape[0])
    inset = cv2.resize(frame, (inset_w, int(inset_w * frame.shape[0] / frame.shape[1])),
                       interpolation=cv2.INTER_AREA)
    inset = inset[:inset_h] if inset.shape[0] > inset_h else inset
    inset_h = inset.shape[0]
    y = output.shape[0] - inset_h - 18

    if options.show_hands:
        for hand in landmark_frame.hands:
            hud.draw_hand(inset, hand, color=hud.THEME.cyan, show_label=True)

    output[y : y + inset_h, x : x + inset_w] = inset
    cv2.rectangle(output, (x - 2, y - 2), (x + inset_w + 2, y + inset_h + 2), hud.THEME.cyan, 2, cv2.LINE_AA)
    hud.text(output, "YOU", (x + 8, y + 20), scale=0.42, color=hud.THEME.cyan)


def _draw_chrome(
    output: np.ndarray,
    core: SlidePresenterCore,
    state: dict,
    loop,
    options: PresenterOptions,
) -> None:
    """Slide counter, pinch indicators and the progress bar."""
    height, width = output.shape[:2]
    count = max(state["count"], 1)

    # The counter sits to the right of the webcam inset so the two never overlap;
    # when the inset is hidden it slides back to the left edge.
    x, _, inset_w, _ = inset_rect(output, options)
    counter_x = (x + inset_w + 16) if options.show_inset else 18
    hud.panel(output, (counter_x, height - 78), (counter_x + 300, height - 18), alpha=0.6)
    hud.text(
        output,
        f"{state['index'] + 1} / {state['count']}",
        (counter_x + 16, height - 40),
        scale=0.9,
        color=hud.THEME.cyan,
        thickness=2,
    )
    hud.text(output, state["name"], (counter_x + 16, height - 22), scale=0.38, color=hud.THEME.dim)

    # Progress bar across the bottom edge.
    filled = int(width * (state["index"] + 1) / count)
    cv2.rectangle(output, (0, height - 6), (filled, height), hud.THEME.magenta, -1)

    for label, x in (("L", width - 150), ("R", width - 96)):
        hand_label = "Left" if label == "L" else "Right"
        active = state["pinch"].get(hand_label, False)
        role = "PREV" if hand_label == core.config.previous_hand else "NEXT"
        color = hud.THEME.lime if active else hud.THEME.dim
        hud.panel(output, (x, height - 78), (x + 46, height - 18), alpha=0.55, border=color)
        hud.text(output, label, (x + 16, height - 48), scale=0.6, color=color, thickness=2)
        hud.text(output, role, (x + 6, height - 26), scale=0.32, color=color)

    if loop.debug:
        hud.status_strip(
            output,
            [
                ("ACTION", (state["lastAction"] or "-").upper()),
                ("MOVES", str(state["advances"])),
                ("SWAP", "ON" if core.config.swap_handedness else "OFF"),
                ("FPS", f"{loop.fps.fps:4.1f}"),
            ],
            y=18,
        )


def make_key_handler(core: SlidePresenterCore, options: PresenterOptions) -> Callable:
    """Build the ``on_key(key, loop) -> handled`` callback."""

    def on_key(key: int, loop) -> bool:
        if key in (ord("n"), KEY_RIGHT):
            core.next_slide(loop.elapsed)
        elif key in (ord("p"), KEY_LEFT):
            core.previous_slide(loop.elapsed)
        elif key == ord("r"):
            core.reload_deck()
        elif key == ord("c"):
            options.show_inset = not options.show_inset
        elif key == ord("w"):
            core.config.swap_handedness = not core.config.swap_handedness
        else:
            return False
        return True

    return on_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture-controlled slide presenter")
    add_camera_arguments(parser)
    parser.add_argument("--slides", type=Path, default=None, help="Folder of slide images")
    parser.add_argument("--loop", action="store_true", help="Wrap around at the deck ends")
    parser.add_argument("--no-inset", action="store_true", help="Hide the webcam inset")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=SlideConfig.advance_cooldown,
        help="Seconds between accepted pinches",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - interactive
    args = parse_args()
    config = SlideConfig(
        loop_deck=args.loop,
        advance_cooldown=args.cooldown,
        swap_handedness=args.swap_handedness,
    )
    if args.slides is not None:
        config.slides_dir = args.slides

    core = SlidePresenterCore(config)
    if core.deck.is_empty:
        print(f"No slides found in {config.slides_dir}")
        print("Generate a demo deck:  .venv/bin/python -m demos.tools.make_sample_slides")

    options = PresenterOptions(show_inset=not args.no_inset)
    loop = CameraLoop(camera_config_from_args(args, "slide presenter"), hands=True)

    print(__doc__)
    loop.run(make_renderer(core, options), on_key=make_key_handler(core, options))


if __name__ == "__main__":  # pragma: no cover
    main()
