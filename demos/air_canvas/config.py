"""Tunables and palette layout for the air canvas.

Everything is expressed in normalized 0..1 coordinates so one layout serves the
desktop window, the browser canvas and the snapshot renderer at any resolution.
Brush size is a fraction of canvas height for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Color = tuple[int, int, int]  # BGR


# Swatch order matters: index 0 is the startup color.
PALETTE_COLORS: tuple[tuple[str, Color], ...] = (
    ("cyan", (255, 231, 76)),
    ("magenta", (200, 60, 255)),
    ("lime", (120, 255, 140)),
    ("amber", (60, 190, 255)),
    ("red", (80, 80, 255)),
    ("white", (245, 245, 245)),
)


@dataclass(frozen=True)
class Cell:
    """One hit target in the palette.

    ``key`` is what the dwell timer tracks, ``kind`` decides what activation does,
    and the rect is normalized ``(x1, y1, x2, y2)``.
    """

    key: str
    kind: str  # "color" | "tool" | "action" | "slider"
    rect: tuple[float, float, float, float]
    label: str = ""
    color: Color | None = None
    value: str = ""

    def contains(self, point: tuple[float, float], pad: float = 0.0) -> bool:
        """Hit test, optionally grown by ``pad`` on every side.

        The padding matters for a grabbed slider: a fingertip resting on the very
        edge of a bar jitters across the boundary, and without slack the grab
        would drop every few frames.
        """
        x1, y1, x2, y2 = self.rect
        return (x1 - pad) <= point[0] <= (x2 + pad) and (y1 - pad) <= point[1] <= (y2 + pad)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.rect
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass
class AirCanvasConfig:
    """All air canvas tunables."""

    # Canvas used by render_canvas()/snapshot, height x width.
    canvas_height: int = 720
    canvas_width: int = 1280
    background: Color = (14, 10, 20)

    # Brush size as a fraction of canvas height, and opacity range.
    size_min: float = 0.004
    size_max: float = 0.055
    size_default: float = 0.014
    opacity_min: float = 0.15
    opacity_max: float = 1.0
    opacity_default: float = 1.0
    eraser_scale: float = 2.5

    # Gesture thresholds. Pinch values are thumb-index distance over hand span.
    pinch_start_ratio: float = 0.34
    pinch_release_ratio: float = 0.52
    # How long a fingertip must rest on a cell before it activates.
    dwell_seconds: float = 0.45
    # Fingertip smoothing; lower is smoother but laggier.
    cursor_alpha: float = 0.45
    # Points closer together than this (normalized) are skipped, which keeps
    # stationary jitter from thickening a stroke into a blob.
    min_move: float = 0.004
    # Slack around a grabbed slider before the grab is released.
    slider_release_pad: float = 0.035
    # A stroke is finished if the drawing gesture disappears for this long.
    stroke_timeout: float = 0.35
    max_strokes: int = 400

    # Palette geometry.
    rail_left: tuple[float, float] = (0.015, 0.105)
    rail_right: tuple[float, float] = (0.895, 0.985)
    swatch_top: float = 0.085
    swatch_height: float = 0.072
    swatch_gap: float = 0.012
    # The sliders stop short of the bottom edge to leave room for the label and
    # value drawn beneath each bar.
    slider_top: float = 0.07
    slider_height: float = 0.32
    slider_gap: float = 0.10

    cells: tuple[Cell, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.cells:
            self.cells = build_palette(self)

    # -- lookups -----------------------------------------------------------
    def cell_at(self, point: tuple[float, float]) -> Cell | None:
        for cell in self.cells:
            if cell.contains(point):
                return cell
        return None

    def cell(self, key: str) -> Cell | None:
        for candidate in self.cells:
            if candidate.key == key:
                return candidate
        return None

    def color_for(self, name: str) -> Color:
        for candidate, color in PALETTE_COLORS:
            if candidate == name:
                return color
        return PALETTE_COLORS[0][1]

    @property
    def in_palette_x(self) -> tuple[float, float]:
        """X band that counts as "over the palette" for cursor feedback."""
        return (self.rail_left[1], self.rail_right[0])


def build_palette(config: AirCanvasConfig) -> tuple[Cell, ...]:
    """Lay out swatches and actions on the left rail, sliders on the right."""
    cells: list[Cell] = []
    lx1, lx2 = config.rail_left
    y = config.swatch_top
    step = config.swatch_height + config.swatch_gap

    for name, color in PALETTE_COLORS:
        cells.append(
            Cell(
                key=f"color:{name}",
                kind="color",
                rect=(lx1, y, lx2, y + config.swatch_height),
                label=name,
                color=color,
                value=name,
            )
        )
        y += step

    for key, label in (("tool:eraser", "ERASER"), ("action:undo", "UNDO"), ("action:clear", "CLEAR")):
        cells.append(
            Cell(
                key=key,
                kind="tool" if key.startswith("tool") else "action",
                rect=(lx1, y, lx2, y + config.swatch_height),
                label=label,
                value=key.split(":", 1)[1],
            )
        )
        y += step

    rx1, rx2 = config.rail_right
    sy = config.slider_top
    for key, label in (("slider:size", "SIZE"), ("slider:opacity", "OPACITY")):
        cells.append(
            Cell(
                key=key,
                kind="slider",
                rect=(rx1, sy, rx2, sy + config.slider_height),
                label=label,
                value=key.split(":", 1)[1],
            )
        )
        sy += config.slider_height + config.slider_gap

    return tuple(cells)
