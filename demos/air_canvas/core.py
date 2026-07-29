"""Air canvas paint engine.

Pure logic: landmarks and a clock go in, paint state comes out. No camera, no
window, no Flask. The desktop app and the WebSocket handler both drive this, so
the gesture behaviour cannot drift between them.

Gesture map (see :mod:`demos.air_canvas.config` for thresholds):

    index only extended      DRAW    lay ink at the fingertip
    index + middle extended  HOVER   move over the palette to select
    thumb-index pinch        ERASE   rub ink out at the pinch point
    anything else            IDLE    finish the current stroke

Palette cells activate on dwell, not on entry, so a fingertip crossing the rail
does not select everything it passes. The two sliders need a dwell to "grab",
after which the value tracks the fingertip until it leaves the bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..common import gestures as gs
from ..common import hud
from ..common import landmarks as lm
from ..common.geometry import (
    DwellTimer,
    EMAPoint,
    clamp,
    distance,
    lerp,
    normalize_range,
    to_pixels,
)
from .config import AirCanvasConfig, Cell, Color

# Tool names are strings so they cross the JSON boundary unchanged.
DRAW = "draw"
ERASE = "erase"
HOVER = "hover"
IDLE = "idle"


@dataclass
class Stroke:
    """One continuous mark in normalized coordinates.

    ``size`` is a fraction of canvas height so a stroke drawn in the desktop
    window rasterizes identically in the browser and in the snapshot.
    """

    color: Color
    size: float
    opacity: float
    erase: bool = False
    points: list[tuple[float, float]] = field(default_factory=list)

    def add(self, point: tuple[float, float], min_move: float) -> bool:
        """Append a point unless it is too close to the previous one."""
        if self.points and distance(self.points[-1], point) < min_move:
            return False
        self.points.append(point)
        return True

    def to_json(self) -> dict[str, Any]:
        return {
            "color": list(self.color),
            "size": self.size,
            "opacity": self.opacity,
            "erase": self.erase,
            "points": [{"x": x, "y": y} for x, y in self.points],
        }


class _Raster:
    """Adapter giving :func:`demos.common.hud.draw_strokes_rgba` pixel points."""

    def __init__(self, stroke: Stroke, width: int, height: int):
        self.pixels = [to_pixels(p, width, height) for p in stroke.points]
        self.color = stroke.color
        self.size = max(1, int(round(stroke.size * height)))
        self.opacity = stroke.opacity
        self.erase = stroke.erase


class AirCanvasCore:
    """Gesture-driven paint state machine."""

    def __init__(self, config: AirCanvasConfig | None = None):
        self.config = config or AirCanvasConfig()
        self.strokes: list[Stroke] = []
        self.active: Stroke | None = None
        self.color_name: str = "cyan"
        self.color: Color = self.config.color_for("cyan")
        self.size: float = self.config.size_default
        self.opacity: float = self.config.opacity_default
        self.tool: str = IDLE
        self.cursor: tuple[float, float] | None = None
        self.revision: int = 0

        self._smoother = EMAPoint(alpha=self.config.cursor_alpha)
        self._dwell = DwellTimer(threshold=self.config.dwell_seconds)
        self._pinch = gs.PinchDetector(
            start_ratio=self.config.pinch_start_ratio,
            release_ratio=self.config.pinch_release_ratio,
        )
        self._grabbed_slider: str | None = None
        self._hover_key: str | None = None
        self._last_draw_at: float = 0.0
        self._last_activated: str | None = None

    # -- main entry point --------------------------------------------------
    def update(self, frame: lm.LandmarkFrame, now: float) -> dict[str, Any]:
        hand = self._pick_hand(frame)
        pinch = self._pinch.update(hand, now)

        if hand is None:
            self._end_stroke()
            self._dwell.reset()
            self._grabbed_slider = None
            self._hover_key = None
            self.tool = IDLE
            self.cursor = None
            self._smoother.reset()
            return self.state(now)

        tool = self._classify(hand, pinch)
        raw_point = pinch.point if tool == ERASE else hand.index_tip
        self.cursor = self._smoother.update(raw_point)
        self.tool = tool

        if tool == HOVER:
            self._end_stroke()
            self._handle_palette(now)
        else:
            self._dwell.reset()
            self._grabbed_slider = None
            self._hover_key = None
            if tool in (DRAW, ERASE):
                self._paint(now, erase=tool == ERASE)
            else:
                self._end_stroke()

        # A stroke also ends if drawing stops without the tool changing, which
        # happens when the hand leaves the frame mid-mark.
        if self.active is not None and (now - self._last_draw_at) > self.config.stroke_timeout:
            self._end_stroke()

        return self.state(now)

    # -- gesture handling --------------------------------------------------
    def _pick_hand(self, frame: lm.LandmarkFrame) -> lm.Hand | None:
        """Prefer a hand that is drawing or hovering over an idle one."""
        if not frame.hands:
            return None
        for hand in frame.hands:
            if gs.is_pointing(hand) or gs.is_peace(hand):
                return hand
        return frame.hands[0]

    def _classify(self, hand: lm.Hand, pinch: gs.PinchState) -> str:
        # A closed fist also brings the thumb near the index tip, so a pinch only
        # counts when the index finger is actually extended to pinch with.
        if pinch.active and gs.finger_extended(hand, "index"):
            return ERASE
        if gs.is_peace(hand):
            return HOVER
        if gs.is_pointing(hand):
            return DRAW
        return IDLE

    def _handle_palette(self, now: float) -> None:
        if self.cursor is None:
            return

        # A grabbed slider keeps tracking until the fingertip leaves its bar,
        # with padding so edge jitter does not drop the grab.
        if self._grabbed_slider is not None:
            cell = self.config.cell(self._grabbed_slider)
            if cell is not None and cell.contains(self.cursor, self.config.slider_release_pad):
                self._apply_slider(cell, self.cursor)
                self._hover_key = cell.key
                return
            self._grabbed_slider = None

        cell = self.config.cell_at(self.cursor)
        self._hover_key = cell.key if cell else None
        activated = self._dwell.update(cell.key if cell else None, now)
        if activated is None or cell is None:
            return

        self._last_activated = cell.key
        if cell.kind == "color":
            self.color_name = cell.value
            self.color = cell.color or self.color
        elif cell.kind == "tool" and cell.value == "eraser":
            # Explicit eraser select: draw strokes become erase strokes until a
            # color is chosen again.
            self.color_name = "eraser"
        elif cell.kind == "action":
            if cell.value == "undo":
                self.undo()
            elif cell.value == "clear":
                self.clear()
        elif cell.kind == "slider":
            self._grabbed_slider = cell.key
            self._apply_slider(cell, self.cursor)

    def _apply_slider(self, cell: Cell, point: tuple[float, float]) -> None:
        """Map fingertip height inside a slider bar onto its value.

        The bar reads bottom-to-top (low value at the bottom) because that is
        how people expect a vertical level control to behave.
        """
        _, y1, _, y2 = cell.rect
        fraction = 1.0 - normalize_range(point[1], y1, y2)
        if cell.value == "size":
            self.size = lerp(self.config.size_min, self.config.size_max, fraction)
        elif cell.value == "opacity":
            self.opacity = lerp(self.config.opacity_min, self.config.opacity_max, fraction)

    # -- painting ----------------------------------------------------------
    def _paint(self, now: float, *, erase: bool) -> None:
        if self.cursor is None:
            return
        erase = erase or self.color_name == "eraser"

        if self.active is None or self.active.erase != erase:
            self._end_stroke()
            self.active = Stroke(
                color=self.color,
                size=self.size * (self.config.eraser_scale if erase else 1.0),
                opacity=self.opacity if not erase else 1.0,
                erase=erase,
            )
            self.strokes.append(self.active)
            self._trim()

        if self.active.add(self.cursor, self.config.min_move):
            self._last_draw_at = now
        elif not self.active.points:
            self._last_draw_at = now

    def _end_stroke(self) -> None:
        if self.active is None:
            return
        if len(self.active.points) < 1:
            # Discard empty strokes so undo never appears to do nothing.
            if self.strokes and self.strokes[-1] is self.active:
                self.strokes.pop()
        self.active = None
        self.revision += 1

    def _trim(self) -> None:
        if len(self.strokes) > self.config.max_strokes:
            del self.strokes[: len(self.strokes) - self.config.max_strokes]
            self.revision += 1

    # -- commands ----------------------------------------------------------
    def undo(self) -> None:
        self._end_stroke()
        if self.strokes:
            self.strokes.pop()
        self.revision += 1

    def clear(self) -> None:
        self.active = None
        self.strokes.clear()
        self.revision += 1

    def set_color(self, name: str) -> None:
        self.color_name = name
        if name != "eraser":
            self.color = self.config.color_for(name)

    def set_size(self, value: float) -> None:
        self.size = clamp(float(value), self.config.size_min, self.config.size_max)

    def set_opacity(self, value: float) -> None:
        self.opacity = clamp(float(value), self.config.opacity_min, self.config.opacity_max)

    def handle_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "undo":
            self.undo()
        elif command == "clear":
            self.clear()
        elif command == "color":
            self.set_color(str(payload.get("name", self.color_name)))
        elif command == "size":
            self.set_size(payload.get("value", self.size))
        elif command == "opacity":
            self.set_opacity(payload.get("value", self.opacity))
        elif command == "sync":
            return {"ok": True, "strokes": self.strokes_json(), "revision": self.revision}
        else:
            return {"ok": False, "unknown": command}
        return {"ok": True, "revision": self.revision}

    def reset(self) -> None:
        self.clear()
        self.color_name = "cyan"
        self.color = self.config.color_for("cyan")
        self.size = self.config.size_default
        self.opacity = self.config.opacity_default
        self.tool = IDLE
        self.cursor = None
        self._smoother.reset()
        self._dwell.reset()
        self._pinch.reset()
        self._grabbed_slider = None

    # -- output ------------------------------------------------------------
    def strokes_json(self) -> list[dict[str, Any]]:
        return [stroke.to_json() for stroke in self.strokes]

    def state(self, now: float) -> dict[str, Any]:
        active_points = self.active.points[-2:] if self.active else []
        return {
            "tool": self.tool,
            "cursor": {"x": self.cursor[0], "y": self.cursor[1]} if self.cursor else None,
            "color": list(self.color),
            "colorName": self.color_name,
            "size": round(self.size, 5),
            "opacity": round(self.opacity, 3),
            "sizeFraction": round(
                normalize_range(self.size, self.config.size_min, self.config.size_max), 3
            ),
            "opacityFraction": round(
                normalize_range(self.opacity, self.config.opacity_min, self.config.opacity_max), 3
            ),
            "hover": self._hover_key,
            "dwell": round(self._dwell.progress(now), 3),
            "grabbed": self._grabbed_slider,
            "revision": self.revision,
            "strokeCount": len(self.strokes),
            "activeStroke": (
                {
                    "color": list(self.active.color),
                    "size": self.active.size,
                    "opacity": self.active.opacity,
                    "erase": self.active.erase,
                    "points": [{"x": x, "y": y} for x, y in active_points],
                }
                if self.active
                else None
            ),
        }

    def render_layer(self, width: int, height: int) -> np.ndarray:
        """Rasterize all strokes into a BGRA layer."""
        layer = np.zeros((height, width, 4), dtype=np.uint8)
        hud.draw_strokes_rgba(layer, [_Raster(s, width, height) for s in self.strokes])
        return layer

    def render_canvas(self) -> np.ndarray:
        """Flatten the strokes onto the configured background for /snapshot."""
        height = self.config.canvas_height
        width = self.config.canvas_width
        canvas = np.full((height, width, 3), self.config.background, dtype=np.uint8)
        layer = self.render_layer(width, height)
        hud.alpha_blit(canvas, layer, (0, 0))
        return canvas

    def palette_json(self) -> list[dict[str, Any]]:
        """Palette geometry for the browser to draw."""
        return [
            {
                "key": cell.key,
                "kind": cell.kind,
                "label": cell.label,
                "value": cell.value,
                "rect": list(cell.rect),
                # Browser wants CSS rgb(), so flip BGR.
                "css": (
                    f"rgb({cell.color[2]},{cell.color[1]},{cell.color[0]})" if cell.color else None
                ),
            }
            for cell in self.config.cells
        ]


def draw_palette(
    frame: np.ndarray,
    core: AirCanvasCore,
    *,
    now: float = 0.0,
) -> np.ndarray:
    """Draw the palette rails onto a desktop frame."""
    height, width = frame.shape[:2]
    config = core.config

    for cell in config.cells:
        x1, y1, x2, y2 = cell.rect
        tl = (int(x1 * width), int(y1 * height))
        br = (int(x2 * width), int(y2 * height))
        hovered = core._hover_key == cell.key
        selected = (
            (cell.kind == "color" and cell.value == core.color_name)
            or (cell.kind == "tool" and core.color_name == "eraser" and cell.value == "eraser")
            or (cell.kind == "slider" and core._grabbed_slider == cell.key)
        )

        if cell.kind == "slider":
            hud.panel(frame, tl, br, alpha=0.55, border=hud.THEME.magenta if selected else hud.THEME.grid)
            fraction = (
                core.state(now)["sizeFraction"]
                if cell.value == "size"
                else core.state(now)["opacityFraction"]
            )
            fill_top = int(lerp(br[1], tl[1], fraction))
            cv2.rectangle(frame, (tl[0] + 6, fill_top), (br[0] - 6, br[1] - 4), hud.THEME.cyan, -1)
            # Label and value sit *beneath* the bar so the fingertip and the
            # dwell ring never cover the thing you are trying to read.
            hud.text(
                frame,
                cell.label,
                (tl[0] - 2, br[1] + 18),
                scale=0.42,
                color=hud.THEME.cyan if selected else hud.THEME.dim,
            )
            hud.text(
                frame,
                f"{int(round(fraction * 100)):3d}%",
                (tl[0] - 2, br[1] + 36),
                scale=0.44,
                color=hud.THEME.white,
            )
        elif cell.color is not None:
            cv2.rectangle(frame, tl, br, cell.color, -1)
            cv2.rectangle(
                frame,
                tl,
                br,
                hud.THEME.white if selected else hud.THEME.grid,
                3 if selected else 1,
                cv2.LINE_AA,
            )
        else:
            hud.panel(frame, tl, br, alpha=0.6, border=hud.THEME.amber if selected else hud.THEME.grid)
            hud.text(
                frame,
                cell.label,
                (tl[0] + 6, (tl[1] + br[1]) // 2 + 5),
                scale=0.42,
                color=hud.THEME.amber if selected else hud.THEME.dim,
            )

        if hovered:
            center = ((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2)
            hud.ring(frame, center, min(26, (br[1] - tl[1]) // 2), core._dwell.progress(now))

    return frame


# Gesture legend: (tool this row represents, hand shape, what it does).
GESTURE_LEGEND: tuple[tuple[str, str, str], ...] = (
    (DRAW, "INDEX ONLY", "draw"),
    (HOVER, "INDEX + MIDDLE", "select on the rails"),
    (ERASE, "PINCH THUMB+INDEX", "erase"),
    (IDLE, "FIST / OPEN PALM", "idle"),
)


def draw_gesture_legend(
    frame: np.ndarray,
    active_tool: str,
    *,
    origin: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw the gesture map, highlighting the row that is currently active.

    Discoverability is the whole point: the hand shapes are not guessable, so the
    legend stays on screen and lights up the row matching what your hand is doing,
    which doubles as feedback that the gesture was recognised.
    """
    height, width = frame.shape[:2]
    x, y = origin or (int(width * 0.13), int(height * 0.10))
    row_height = 26
    panel_height = row_height * len(GESTURE_LEGEND) + 34

    hud.panel(frame, (x, y), (x + 330, y + panel_height), alpha=0.62)
    hud.text(frame, "GESTURES", (x + 14, y + 22), scale=0.44, color=hud.THEME.magenta)

    for index, (tool, shape, meaning) in enumerate(GESTURE_LEGEND):
        row_y = y + 44 + index * row_height
        active = tool == active_tool
        color = TOOL_LEGEND_COLORS.get(tool, hud.THEME.dim) if active else hud.THEME.dim
        if active:
            cv2.rectangle(
                frame, (x + 8, row_y - 15), (x + 322, row_y + 7), color, 1, cv2.LINE_AA
            )
            cv2.circle(frame, (x + 18, row_y - 4), 4, color, -1, cv2.LINE_AA)
        hud.text(frame, shape, (x + 30, row_y), scale=0.42, color=color)
        hud.text(
            frame,
            meaning,
            (x + 196, row_y),
            scale=0.4,
            color=hud.THEME.white if active else hud.THEME.dim,
        )
    return frame


TOOL_LEGEND_COLORS = {
    DRAW: hud.THEME.lime,
    HOVER: hud.THEME.amber,
    ERASE: hud.THEME.red,
    IDLE: hud.THEME.cyan,
}
