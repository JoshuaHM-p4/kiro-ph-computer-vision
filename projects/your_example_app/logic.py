"""The decisions your app makes, as pure functions.

Nothing here opens a camera, draws a window, or talks to a network. That is what makes
it testable: feed it numbers, assert on the numbers that come back.

Replace this with your own logic — the brightness tracker is only here to show the
shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Every tunable value, in one place instead of scattered through the code."""

    # Brightness (0..1) at or above which we call the frame "bright". Two thresholds,
    # not one: a single threshold makes the state flicker when the value sits on it.
    bright_enter: float = 0.55
    bright_release: float = 0.45
    # How long the state must hold before we count it as a change.
    settle_seconds: float = 0.3


class BrightnessTracker:
    """Tracks whether the scene is bright, with hysteresis and a settle time.

    A deliberately boring example of the pattern: state in the object, decisions in
    ``update``, and a ``state`` dict that the UI layer renders without thinking.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.is_bright = False
        self.changes = 0
        self._candidate = False
        self._candidate_since: float | None = None

    def update(self, brightness: float, now: float) -> dict:
        """Feed a 0..1 brightness reading and a timestamp."""
        if self.is_bright:
            candidate = brightness > self.config.bright_release
        else:
            candidate = brightness >= self.config.bright_enter

        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = now
        elif (
            candidate != self.is_bright
            and self._candidate_since is not None
            and (now - self._candidate_since) >= self.config.settle_seconds
        ):
            self.is_bright = candidate
            self.changes += 1

        return self.state(brightness)

    def state(self, brightness: float) -> dict:
        return {
            "brightness": round(brightness, 3),
            "bright": self.is_bright,
            "changes": self.changes,
        }
