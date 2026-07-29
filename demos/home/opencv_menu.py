"""Futuristic OpenCV menu that launches the desktop demos.

Run it with::

    .venv/bin/python -m demos.home.opencv_menu
    .venv/bin/python -m demos.home.opencv_menu --no-camera   # keyboard only

Navigate with the arrow keys and press ENTER, press 1-4 directly, or hover a card
with your hand and hold still to select it. The chosen demo runs as a subprocess
using this same interpreter; when it exits, the menu comes back.

Selection logic lives in :class:`MenuModel`, separate from the drawing, so it can
be unit tested without a window.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import DEMOS, DemoInfo
from ..common import hud
from ..common.camera import CameraConfig, CameraLoop
from ..common.geometry import DwellTimer, EMAPoint, clamp

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 82, 84, 81, 83
KEY_ENTER = 13
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Card:
    """A menu card and its normalized rectangle."""

    demo: DemoInfo
    rect: tuple[float, float, float, float]

    def contains(self, point: tuple[float, float]) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.rect
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def build_cards(
    demos: tuple[DemoInfo, ...] = DEMOS,
    *,
    columns: int = 2,
    margin: float = 0.06,
    top: float = 0.30,
    bottom: float = 0.88,
    gap: float = 0.035,
) -> list[Card]:
    """Lay the demos out in a grid of normalized rectangles."""
    rows = (len(demos) + columns - 1) // columns
    width = (1.0 - 2 * margin - gap * (columns - 1)) / columns
    height = (bottom - top - gap * (rows - 1)) / rows

    cards: list[Card] = []
    for index, demo in enumerate(demos):
        row, column = divmod(index, columns)
        x1 = margin + column * (width + gap)
        y1 = top + row * (height + gap)
        cards.append(Card(demo=demo, rect=(x1, y1, x1 + width, y1 + height)))
    return cards


@dataclass
class MenuModel:
    """Keyboard and hand selection, with no drawing involved."""

    cards: list[Card] = field(default_factory=build_cards)
    columns: int = 2
    dwell_seconds: float = 0.9
    selected: int = 0
    hovered: int | None = None
    launched: DemoInfo | None = None
    _dwell: DwellTimer = field(init=False)
    _cursor: EMAPoint = field(init=False)

    def __post_init__(self) -> None:
        self._dwell = DwellTimer(threshold=self.dwell_seconds)
        self._cursor = EMAPoint(alpha=0.4)

    # -- keyboard ----------------------------------------------------------
    def move(self, dx: int, dy: int) -> int:
        """Move the selection by a grid step, clamped at the edges."""
        rows = (len(self.cards) + self.columns - 1) // self.columns
        row, column = divmod(self.selected, self.columns)
        column = int(clamp(column + dx, 0, self.columns - 1))
        row = int(clamp(row + dy, 0, rows - 1))
        index = row * self.columns + column
        # The last row can be ragged (5 demos in 2 columns leaves one empty cell).
        # Clamping to the final card keeps it reachable instead of dead-ending.
        self.selected = min(index, len(self.cards) - 1)
        return self.selected

    def handle_key(self, key: int) -> DemoInfo | None:
        """Return the demo to launch, if this key selected one."""
        if key in (KEY_LEFT, ord("h")):
            self.move(-1, 0)
        elif key in (KEY_RIGHT, ord("l")):
            self.move(1, 0)
        elif key in (KEY_UP, ord("k")):
            self.move(0, -1)
        elif key in (KEY_DOWN, ord("j")):
            self.move(0, 1)
        elif key in (KEY_ENTER, ord(" ")):
            return self.choose(self.selected)
        elif ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(self.cards):
                self.selected = index
                return self.choose(index)
        return None

    # -- hand hovering -----------------------------------------------------
    def update_pointer(
        self, point: tuple[float, float] | None, now: float
    ) -> DemoInfo | None:
        """Track a fingertip; a completed dwell selects the card under it."""
        smoothed = self._cursor.update(point) if point is not None else None
        if point is None:
            self._cursor.reset()
            self._dwell.update(None, now)
            self.hovered = None
            return None

        self.hovered = next(
            (index for index, card in enumerate(self.cards) if card.contains(smoothed)),
            None,
        )
        if self.hovered is not None:
            self.selected = self.hovered
        key = self.cards[self.hovered].demo.slug if self.hovered is not None else None
        if self._dwell.update(key, now) is not None:
            return self.choose(self.hovered)
        return None

    def dwell_progress(self, now: float) -> float:
        return self._dwell.progress(now)

    @property
    def cursor(self) -> tuple[float, float] | None:
        return self._cursor.value

    # -- launching ---------------------------------------------------------
    def choose(self, index: int | None) -> DemoInfo | None:
        if index is None or not (0 <= index < len(self.cards)):
            return None
        self.launched = self.cards[index].demo
        return self.launched

    def reset_selection(self) -> None:
        """Clear dwell state so the menu does not immediately relaunch."""
        self._dwell.reset()
        self._cursor.reset()
        self.launched = None
        self.hovered = None


def launch(demo: DemoInfo, *, python: str | None = None) -> int:
    """Run a demo's desktop module as a subprocess and wait for it.

    The same interpreter is reused so the demo inherits this virtualenv rather
    than whatever ``python`` happens to be on PATH.
    """
    command = [python or sys.executable, "-m", demo.desktop_module]
    print(f"$ {' '.join(command)}")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return result.returncode


def draw_menu(
    frame: np.ndarray,
    model: MenuModel,
    *,
    now: float,
    camera: bool,
) -> np.ndarray:
    """Render the animated menu over ``frame``."""
    height, width = frame.shape[:2]

    if camera:
        frame[:] = (frame.astype(np.float32) * 0.28).astype(np.uint8)
    else:
        frame[:] = (10, 7, 14)

    # Animated grid: horizontal lines drift upward, vertical lines stay put.
    offset = int((now * 26) % 44)
    for y in range(-44 + offset, height, 44):
        cv2.line(frame, (0, y), (width, y), hud.THEME.grid, 1)
    for x in range(0, width, 64):
        cv2.line(frame, (x, 0), (x, height), (34, 24, 44), 1)
    hud.scanlines(frame, spacing=3, strength=0.10)
    hud.vignette(frame, 0.4)

    # Title with a slow pulse.
    pulse = 0.75 + 0.25 * float(np.sin(now * 1.8))
    title_color = tuple(int(channel * pulse) for channel in hud.THEME.cyan)
    cv2.putText(frame, "VISION DEMOS", (int(width * 0.06), int(height * 0.15)),
                hud.FONT, width / 780, title_color, 3, cv2.LINE_AA)
    hud.text(
        frame,
        "arrows + enter   1-4 direct   hover a card with your hand   q quit",
        (int(width * 0.06), int(height * 0.21)),
        scale=width / 2100,
        color=hud.THEME.dim,
    )

    for index, card in enumerate(model.cards):
        x1, y1, x2, y2 = card.rect
        tl = (int(x1 * width), int(y1 * height))
        br = (int(x2 * width), int(y2 * height))
        active = index == model.selected
        border = hud.THEME.cyan if active else hud.THEME.grid

        hud.panel(frame, tl, br, alpha=0.72 if active else 0.5, border=border, corner=22)
        if active:
            # A glowing frame marks the selection without moving the layout.
            def draw(layer: np.ndarray, tl=tl, br=br) -> None:
                cv2.rectangle(layer, tl, br, hud.THEME.cyan, 2, cv2.LINE_AA)

            hud.glow(frame, draw, blur=31, intensity=0.55)

        hud.text(frame, f"{index + 1}", (tl[0] + 18, tl[1] + 40), scale=0.8,
                 color=hud.THEME.magenta, thickness=2)
        hud.text(frame, card.demo.title.upper(), (tl[0] + 56, tl[1] + 40), scale=0.78,
                 color=hud.THEME.white if active else hud.THEME.dim, thickness=2)
        hud.text(frame, card.demo.tagline, (tl[0] + 56, tl[1] + 66), scale=0.46,
                 color=hud.THEME.cyan if active else hud.THEME.dim)
        hud.text(frame, f"python -m {card.demo.desktop_module}", (tl[0] + 18, br[1] - 18),
                 scale=0.4, color=hud.THEME.dim)

        if model.hovered == index:
            progress = model.dwell_progress(now)
            hud.ring(frame, ((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2),
                     min(46, (br[1] - tl[1]) // 3), progress, thickness=5)
            hud.text(frame, "HOLD TO LAUNCH", (tl[0] + 56, br[1] - 40),
                     scale=0.44, color=hud.THEME.amber)

    if model.cursor is not None:
        hud.crosshair(
            frame,
            (int(model.cursor[0] * width), int(model.cursor[1] * height)),
            radius=18,
            color=hud.THEME.lime,
        )

    hud.status_strip(
        frame,
        [
            ("MODE", "HAND + KEYS" if camera else "KEYS ONLY"),
            ("SELECTED", model.cards[model.selected].demo.title.upper()),
            ("WEB HUB", "python -m demos.home.web"),
        ],
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Futuristic OpenCV launcher menu")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0")
    parser.add_argument("--no-camera", action="store_true", help="Keyboard-only menu")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--dwell",
        type=float,
        default=0.9,
        help="Seconds to hold a hand over a card before it launches",
    )
    return parser.parse_args()


def run_keyboard_menu(args: argparse.Namespace) -> None:  # pragma: no cover - interactive
    """Menu without a camera: a plain OpenCV window and key handling."""
    model = MenuModel(dwell_seconds=args.dwell)
    window = "vision demos"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.width, args.height)
    started = time.monotonic()

    while True:
        frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        draw_menu(frame, model, now=time.monotonic() - started, camera=False)
        cv2.imshow(window, frame)

        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), 27):
            break
        if key != 255:
            demo = model.handle_key(key)
            if demo is not None:
                cv2.destroyWindow(window)
                launch(demo)
                model.reset_selection()
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, args.width, args.height)
    cv2.destroyAllWindows()


def run_camera_menu(args: argparse.Namespace) -> None:  # pragma: no cover - interactive
    """Menu with hand hovering, driven by the shared camera loop."""
    model = MenuModel(dwell_seconds=args.dwell)
    config = CameraConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        window_name="vision demos",
        debug=True,
    )
    pending: list[DemoInfo] = []

    def render(frame: np.ndarray, landmark_frame, loop_ref) -> np.ndarray:
        from ..common import gestures as gs

        hand = next(
            (h for h in landmark_frame.hands if gs.is_pointing(h)),
            landmark_frame.primary_hand,
        )
        chosen = model.update_pointer(hand.index_tip if hand else None, loop_ref.elapsed)
        if chosen is not None:
            pending.append(chosen)
            loop_ref.stop()
        return draw_menu(frame, model, now=loop_ref.elapsed, camera=True)

    def on_key(key: int, loop_ref) -> bool:
        demo = model.handle_key(key)
        if demo is not None:
            pending.append(demo)
            loop_ref.stop()
            return True
        return key in (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_ENTER)

    while True:
        pending.clear()
        # A fresh loop per iteration: the previous one released its camera and
        # MediaPipe graphs when the chosen demo took over the webcam.
        loop = CameraLoop(config, hands=True)
        loop.run(render, on_key=on_key)
        if not pending:
            break  # the user quit
        launch(pending[0])
        model.reset_selection()


def main() -> None:  # pragma: no cover - interactive
    args = parse_args()
    print(__doc__)
    if args.no_camera:
        run_keyboard_menu(args)
    else:
        try:
            run_camera_menu(args)
        except RuntimeError as error:
            print(f"{error}\nFalling back to the keyboard-only menu.")
            run_keyboard_menu(args)


if __name__ == "__main__":  # pragma: no cover
    main()
