"""Pure geometry and smoothing helpers.

No OpenCV, no MediaPipe, no camera: everything here is a plain function or a
tiny state holder so it can be unit tested without hardware.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

Point = tuple[float, float]
PointInt = tuple[int, int]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to the inclusive range [low, high]."""
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from ``a`` to ``b``."""
    return a + (b - a) * t


def lerp_point(a: Sequence[float], b: Sequence[float], t: float) -> Point:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t))


def midpoint(a: Sequence[float], b: Sequence[float]) -> Point:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def to_pixels(point: Sequence[float], width: int, height: int) -> PointInt:
    """Convert a normalized (0..1) point to integer pixel coordinates."""
    return (int(round(point[0] * width)), int(round(point[1] * height)))


def normalize_range(value: float, low: float, high: float) -> float:
    """Map ``value`` from [low, high] onto [0, 1], clamped.

    Returns 0.0 when the range is degenerate instead of dividing by zero.
    """
    span = high - low
    if abs(span) < 1e-9:
        return 0.0
    return clamp((value - low) / span, 0.0, 1.0)


def angle_degrees(origin: Sequence[float], target: Sequence[float]) -> float:
    """Angle of the origin->target vector in degrees, measured from +x."""
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0]))


def natural_key(text: str) -> tuple:
    """Sort key that orders ``slide2`` before ``slide10``."""
    parts: list[object] = []
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        else:
            if digits:
                parts.append((1, int(digits)))
                digits = ""
            parts.append((0, char.lower()))
    if digits:
        parts.append((1, int(digits)))
    return tuple(parts)


@dataclass
class EMAPoint:
    """Exponential moving average for a 2D point.

    ``alpha`` is the weight of the newest sample: higher reacts faster, lower
    smooths harder. Mirrors ``src/utils.smooth_point`` but keeps its own state.
    """

    alpha: float = 0.35
    value: Point | None = None

    def update(self, point: Sequence[float] | None) -> Point | None:
        if point is None:
            return self.value
        if self.value is None:
            self.value = (float(point[0]), float(point[1]))
        else:
            a = self.alpha
            self.value = (
                self.value[0] * (1.0 - a) + float(point[0]) * a,
                self.value[1] * (1.0 - a) + float(point[1]) * a,
            )
        return self.value

    def reset(self) -> None:
        self.value = None


@dataclass
class EMAScalar:
    """Exponential moving average for a single number."""

    alpha: float = 0.35
    value: float | None = None

    def update(self, sample: float | None) -> float | None:
        if sample is None:
            return self.value
        if self.value is None:
            self.value = float(sample)
        else:
            self.value = self.value * (1.0 - self.alpha) + float(sample) * self.alpha
        return self.value

    def reset(self) -> None:
        self.value = None


@dataclass
class HysteresisLatch:
    """Boolean latch with separate on and off thresholds.

    A raw signal that hovers near a single threshold flips constantly. Two
    thresholds fix that: the latch turns on only below ``on_below`` and off only
    above ``off_above``, so noise between them cannot toggle it.

    Set ``invert=True`` for signals that switch on when they rise (the latch
    then turns on above ``on_below`` and off below ``off_above``).
    """

    on_below: float
    off_above: float
    invert: bool = False
    state: bool = False

    def update(self, value: float | None) -> bool:
        if value is None:
            return self.state
        if self.invert:
            if not self.state and value >= self.on_below:
                self.state = True
            elif self.state and value <= self.off_above:
                self.state = False
        else:
            if not self.state and value <= self.on_below:
                self.state = True
            elif self.state and value >= self.off_above:
                self.state = False
        return self.state

    def reset(self) -> None:
        self.state = False


@dataclass
class EdgeTrigger:
    """Fires once when a boolean signal rises, with an optional cooldown.

    ``fire`` returns True only on the False->True transition and only when at
    least ``cooldown`` seconds have passed since the previous firing, which is
    what keeps one physical pinch from advancing several slides.
    """

    cooldown: float = 0.0
    previous: bool = False
    last_fired: float = field(default=-1e9)

    def fire(self, signal: bool, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        rising = signal and not self.previous
        self.previous = signal
        if rising and (now - self.last_fired) >= self.cooldown:
            self.last_fired = now
            return True
        return False

    def reset(self) -> None:
        self.previous = False
        self.last_fired = -1e9


@dataclass
class DwellTimer:
    """Tracks how long a key has been held continuously.

    Used for gesture menus: hovering a palette cell only activates it after the
    pointer stays there for ``threshold`` seconds, which prevents a fingertip
    sweeping across the rail from selecting every cell it passes.
    """

    threshold: float = 0.45
    key: object | None = None
    started_at: float = 0.0
    fired_for: object | None = None

    def update(self, key: object | None, now: float | None = None) -> object | None:
        """Return the key when its dwell completes, else None."""
        now = time.monotonic() if now is None else now
        if key != self.key:
            self.key = key
            self.started_at = now
            if key != self.fired_for:
                self.fired_for = None
            return None
        if key is None:
            return None
        if self.fired_for == key:
            return None
        if (now - self.started_at) >= self.threshold:
            self.fired_for = key
            return key
        return None

    def progress(self, now: float | None = None) -> float:
        """0..1 fraction of the dwell completed for the current key."""
        if self.key is None or self.threshold <= 0:
            return 0.0
        if self.fired_for == self.key:
            return 1.0
        now = time.monotonic() if now is None else now
        return clamp((now - self.started_at) / self.threshold, 0.0, 1.0)

    def reset(self) -> None:
        self.key = None
        self.fired_for = None
        self.started_at = 0.0


class FPSCounter:
    """Smoothed frames-per-second estimate."""

    def __init__(self, smoothing: float = 0.15):
        self.smoothing = smoothing
        self.last_time = time.monotonic()
        self.fps = 0.0

    def update(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            instant = 1.0 / dt
            self.fps = (
                self.fps * (1.0 - self.smoothing) + instant * self.smoothing
                if self.fps
                else instant
            )
        return self.fps


def polyline_length(points: Iterable[Sequence[float]]) -> float:
    """Total length of a polyline, 0 for fewer than two points."""
    total = 0.0
    previous: Sequence[float] | None = None
    for point in points:
        if previous is not None:
            total += distance(previous, point)
        previous = point
    return total
