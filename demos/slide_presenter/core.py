"""Slide presenter logic.

Pure state machine: landmarks in, slide index and laser position out. The deck
itself is a folder of images, loaded lazily so the core can be tested with a
temporary directory (or none at all).

Gestures
    index finger pointing   move the laser
    pinch with right hand   next slide
    pinch with left hand    previous slide

Handedness is the whole point of this demo, so it is resolved explicitly rather
than falling back to "whichever hand is first".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..common import gestures as gs
from ..common import hud
from ..common import landmarks as lm
from ..common.geometry import EMAPoint, clamp, natural_key, to_pixels
from .config import IMAGE_SUFFIXES, SlideConfig


@dataclass
class Deck:
    """A folder of slide images, decoded on first use and then cached."""

    directory: Path
    paths: list[Path] = field(default_factory=list)
    _cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, directory: Path) -> "Deck":
        directory = Path(directory)
        paths: list[Path] = []
        if directory.is_dir():
            paths = sorted(
                (p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
                key=lambda p: natural_key(p.name),
            )
        return cls(directory=directory, paths=paths)

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def is_empty(self) -> bool:
        return not self.paths

    def name(self, index: int) -> str:
        if self.is_empty:
            return "no slides"
        return self.paths[index % len(self.paths)].name

    def image(self, index: int) -> np.ndarray | None:
        """Decoded BGR image for a slide, or None if it cannot be read."""
        if self.is_empty:
            return None
        index %= len(self.paths)
        if index not in self._cache:
            image = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
            if image is None:
                return None
            self._cache[index] = image
        return self._cache[index]


class SlidePresenterCore:
    """Tracks the current slide and the laser pointer."""

    def __init__(self, config: SlideConfig | None = None, deck: Deck | None = None):
        self.config = config or SlideConfig()
        self.deck = deck if deck is not None else Deck.load(self.config.slides_dir)
        self.index = 0
        self.laser: tuple[float, float] | None = None
        self.last_action = ""
        self.last_action_at = 0.0
        self.advances = 0
        self.revision = 0

        self._laser_smoother = EMAPoint(alpha=self.config.laser_alpha)
        self._laser_seen_at = -1e9
        # One detector per hand: each needs its own hysteresis and cooldown so a
        # pinch held on one hand cannot block the other.
        self._pinches = {
            "Left": gs.PinchDetector(
                start_ratio=self.config.pinch_start_ratio,
                release_ratio=self.config.pinch_release_ratio,
                cooldown=self.config.advance_cooldown,
            ),
            "Right": gs.PinchDetector(
                start_ratio=self.config.pinch_start_ratio,
                release_ratio=self.config.pinch_release_ratio,
                cooldown=self.config.advance_cooldown,
            ),
        }
        self._pinch_active = {"Left": False, "Right": False}

    # -- helpers -----------------------------------------------------------
    def _resolve(self, hand: lm.Hand) -> str:
        """Handedness label after applying the swap flag."""
        label = hand.label
        if self.config.swap_handedness:
            label = lm.flip_label(label)
        return label

    def _hand_for(self, frame: lm.LandmarkFrame, label: str) -> lm.Hand | None:
        for hand in frame.hands:
            if self._resolve(hand) == label:
                return hand
        return None

    # -- main entry point --------------------------------------------------
    def update(self, frame: lm.LandmarkFrame, now: float) -> dict[str, Any]:
        for label in ("Left", "Right"):
            hand = self._hand_for(frame, label)
            state = self._pinches[label].update(hand, now)
            self._pinch_active[label] = state.active
            if state.just_started:
                if label == self.config.next_hand:
                    self.next_slide(now)
                elif label == self.config.previous_hand:
                    self.previous_slide(now)

        self._update_laser(frame, now)
        return self.state(now)

    def _update_laser(self, frame: lm.LandmarkFrame, now: float) -> None:
        """Follow the index fingertip of whichever hand is pointing."""
        pointing = next((hand for hand in frame.hands if gs.is_pointing(hand)), None)
        if pointing is not None:
            self.laser = self._laser_smoother.update(pointing.index_tip)
            self._laser_seen_at = now
        elif (now - self._laser_seen_at) > self.config.laser_hold:
            # Hold briefly so a momentary detection dropout does not blink the dot.
            self.laser = None
            self._laser_smoother.reset()

    # -- navigation --------------------------------------------------------
    def _set_index(self, index: int, action: str, now: float) -> None:
        count = len(self.deck)
        if count == 0:
            self.index = 0
        elif self.config.loop_deck:
            self.index = index % count
        else:
            self.index = int(clamp(index, 0, count - 1))
        self.last_action = action
        self.last_action_at = now
        self.advances += 1
        self.revision += 1

    def next_slide(self, now: float = 0.0) -> None:
        self._set_index(self.index + 1, "next", now)

    def previous_slide(self, now: float = 0.0) -> None:
        self._set_index(self.index - 1, "previous", now)

    def go_to(self, index: int, now: float = 0.0) -> None:
        self._set_index(int(index), "goto", now)

    def reload_deck(self) -> None:
        self.deck = Deck.load(self.config.slides_dir)
        self.index = min(self.index, max(0, len(self.deck) - 1))
        self.revision += 1

    # -- web interface -----------------------------------------------------
    def handle_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "next":
            self.next_slide()
        elif command == "previous":
            self.previous_slide()
        elif command == "goto":
            self.go_to(payload.get("index", self.index))
        elif command == "reload":
            self.reload_deck()
        elif command == "deck":
            return {"ok": True, **self.deck_json()}
        else:
            return {"ok": False, "unknown": command}
        return {"ok": True, "index": self.index}

    def reset(self) -> None:
        self.index = 0
        self.laser = None
        self.advances = 0
        self.last_action = ""
        self._laser_smoother.reset()
        for detector in self._pinches.values():
            detector.reset()

    def deck_json(self) -> dict[str, Any]:
        return {
            "count": len(self.deck),
            "names": [path.name for path in self.deck.paths],
            "directory": str(self.deck.directory),
        }

    def state(self, now: float) -> dict[str, Any]:
        return {
            "index": self.index,
            "count": len(self.deck),
            "name": self.deck.name(self.index),
            "laser": {"x": self.laser[0], "y": self.laser[1]} if self.laser else None,
            "laserRadius": self.config.laser_radius,
            "pinch": dict(self._pinch_active),
            "lastAction": self.last_action,
            "sinceAction": round(now - self.last_action_at, 3) if self.last_action else None,
            "advances": self.advances,
            "revision": self.revision,
            "nextHand": self.config.next_hand,
            "previousHand": self.config.previous_hand,
            "atStart": self.index == 0,
            "atEnd": bool(len(self.deck)) and self.index == len(self.deck) - 1,
        }

    # -- rendering ---------------------------------------------------------
    def render_canvas(self) -> np.ndarray:
        """Current slide letterboxed onto the configured canvas, laser included."""
        width, height = self.config.canvas_width, self.config.canvas_height
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        slide = self.deck.image(self.index)
        if slide is None:
            hud.title(canvas, "NO SLIDES", str(self.deck.directory))
            hud.text(
                canvas,
                "Drop PNG or JPG files into the slides folder, then press r to reload.",
                (26, height // 2),
                scale=0.6,
                color=hud.THEME.dim,
            )
        else:
            fit_into(canvas, slide)
            hud.text(
                canvas,
                f"{self.index + 1}/{len(self.deck)}  {self.deck.name(self.index)}",
                (26, height - 26),
                scale=0.5,
                color=hud.THEME.cyan,
            )
        if self.laser is not None:
            draw_laser(canvas, self.laser, self.config.laser_radius)
        return canvas


def fit_into(canvas: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Letterbox ``image`` into ``canvas`` preserving aspect ratio."""
    ch, cw = canvas.shape[:2]
    ih, iw = image.shape[:2]
    if ih == 0 or iw == 0:
        return canvas
    scale = min(cw / iw, ch / ih)
    new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (cw - new_w) // 2
    y = (ch - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def draw_laser(
    frame: np.ndarray,
    point: tuple[float, float],
    radius: float = 0.012,
    color: tuple[int, int, int] = (80, 80, 255),
) -> np.ndarray:
    """Glowing laser dot at a normalized position."""
    height, width = frame.shape[:2]
    center = to_pixels(point, width, height)
    pixel_radius = max(3, int(radius * height))

    def draw(layer: np.ndarray) -> None:
        cv2.circle(layer, center, pixel_radius, color, -1, cv2.LINE_AA)

    hud.glow(frame, draw, blur=pixel_radius * 4 + 1, intensity=0.9)
    cv2.circle(frame, center, max(1, pixel_radius // 3), (255, 255, 255), -1, cv2.LINE_AA)
    return frame
