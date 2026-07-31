"""Pure game logic — no camera, no Flask, no I/O.

Feed it Eye Aspect Ratio values and timestamps, get back game state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto

from .config import GameConfig


# MediaPipe Face Mesh eye landmark indices (from the 468-point mesh).
# Left eye (viewer's perspective = subject's right eye)
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# Right eye (viewer's perspective = subject's left eye)
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


class Phase(Enum):
    """Current phase of the game."""

    WAITING = auto()       # Waiting to start
    COUNTDOWN = auto()     # 3-2-1 countdown
    PLAYING = auto()       # Game active — don't blink!
    BLINKED = auto()       # Player blinked — game over
    NO_FACE = auto()       # Face lost during play (pause/warning)


@dataclass
class Distraction:
    """A visual distraction event."""

    kind: str              # from config.distraction_types
    start_time: float      # when it triggers
    duration: float        # how long it lasts


@dataclass
class GameState:
    """The full game state returned to the UI layer."""

    phase: Phase
    survival_time: float            # seconds survived so far
    countdown_value: int            # 3, 2, 1, 0 during countdown
    ear_value: float                # current Eye Aspect Ratio
    active_distraction: Distraction | None  # currently firing distraction
    high_score: float               # best survival time this session
    message: str                    # display text
    penalty_style: str              # "bsod" or "jumpscare"
    blink_count: int                # number of near-blinks detected
    death_reason: str               # "blink" or "face_lost" or ""


def compute_ear(landmarks: list[tuple[float, float]], indices: list[int]) -> float:
    """Compute Eye Aspect Ratio from 6 landmark points.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    landmarks: list of (x, y) normalized coordinates
    indices: the 6 indices for one eye [p1, p2, p3, p4, p5, p6]
    """
    def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    p1 = landmarks[indices[0]]
    p2 = landmarks[indices[1]]
    p3 = landmarks[indices[2]]
    p4 = landmarks[indices[3]]
    p5 = landmarks[indices[4]]
    p6 = landmarks[indices[5]]

    vertical_1 = dist(p2, p6)
    vertical_2 = dist(p3, p5)
    horizontal = dist(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_ear_both_eyes(
    landmarks: list[tuple[float, float]],
) -> float:
    """Compute average EAR across both eyes."""
    left_ear = compute_ear(landmarks, LEFT_EYE_IDX)
    right_ear = compute_ear(landmarks, RIGHT_EYE_IDX)
    return (left_ear + right_ear) / 2.0


class Game:
    """Pure game engine. Call update() each frame with EAR + timestamp."""

    def __init__(self, config: GameConfig | None = None, seed: int | None = None) -> None:
        self.config = config or GameConfig()
        self._rng = random.Random(seed)

        self._phase: Phase = Phase.WAITING
        self._high_score: float = 0.0

        # Timing
        self._countdown_start: float | None = None
        self._game_start: float | None = None

        # Blink detection (hysteresis)
        self._eyes_closed_frames: int = 0
        self._eyes_open: bool = True

        # Distractions
        self._next_distraction_time: float = 0.0
        self._distraction_interval: float = self.config.distraction_interval_initial
        self._active_distraction: Distraction | None = None
        self._distraction_history: list[Distraction] = []

        # Stats
        self._survival_time: float = 0.0
        self._ear_value: float = 0.3
        self._blink_count: int = 0
        self._death_reason: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, timestamp: float) -> GameState:
        """Start the countdown. Call this when the player is ready."""
        self._phase = Phase.COUNTDOWN
        self._countdown_start = timestamp
        return self._make_state(timestamp)

    def update(self, ear: float, face_detected: bool, timestamp: float) -> GameState:
        """Process one frame. Returns the current game state."""
        self._ear_value = ear

        if self._phase == Phase.WAITING:
            return self._make_state(timestamp)

        if self._phase == Phase.BLINKED:
            return self._make_state(timestamp)

        # Handle face loss — you left the screen = instant death
        if not face_detected and self._phase == Phase.PLAYING:
            self._phase = Phase.BLINKED
            self._death_reason = "face_lost"
            self._blink_count += 1
            if self._survival_time > self._high_score:
                self._high_score = self._survival_time
            return self._make_state(timestamp)

        if face_detected and self._phase == Phase.NO_FACE:
            self._phase = Phase.PLAYING

        # Countdown
        if self._phase == Phase.COUNTDOWN:
            elapsed = timestamp - self._countdown_start
            if elapsed >= self.config.countdown_seconds:
                self._phase = Phase.PLAYING
                self._game_start = timestamp
                self._next_distraction_time = (
                    timestamp + self.config.distraction_start_delay
                )
                self._distraction_interval = self.config.distraction_interval_initial
            return self._make_state(timestamp)

        # Playing
        if self._phase == Phase.PLAYING:
            self._survival_time = timestamp - self._game_start

            # Blink detection with hysteresis
            if self._eyes_open:
                if ear < self.config.ear_blink_threshold:
                    self._eyes_closed_frames += 1
                else:
                    self._eyes_closed_frames = 0

                if self._eyes_closed_frames >= self.config.blink_frames:
                    # BLINKED!
                    self._phase = Phase.BLINKED
                    self._death_reason = "blink"
                    self._blink_count += 1
                    if self._survival_time > self._high_score:
                        self._high_score = self._survival_time
                    return self._make_state(timestamp)
            else:
                if ear > self.config.ear_open_threshold:
                    self._eyes_open = True
                    self._eyes_closed_frames = 0

            # Distractions
            self._update_distractions(timestamp)

        return self._make_state(timestamp)

    def restart(self, timestamp: float) -> GameState:
        """Reset the game for another attempt."""
        self._phase = Phase.WAITING
        self._countdown_start = None
        self._game_start = None
        self._eyes_closed_frames = 0
        self._eyes_open = True
        self._active_distraction = None
        self._distraction_history = []
        self._survival_time = 0.0
        self._blink_count = 0
        self._death_reason = ""
        self._distraction_interval = self.config.distraction_interval_initial
        return self._make_state(timestamp)

    def is_game_over(self) -> bool:
        """True if the player blinked."""
        return self._phase == Phase.BLINKED

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_distractions(self, timestamp: float) -> None:
        """Schedule and manage distractions."""
        # Clear expired distraction
        if self._active_distraction is not None:
            end_time = (
                self._active_distraction.start_time
                + self._active_distraction.duration
            )
            if timestamp >= end_time:
                self._active_distraction = None

        # Trigger next distraction
        if self._active_distraction is None and timestamp >= self._next_distraction_time:
            kind = self._rng.choice(self.config.distraction_types)
            self._active_distraction = Distraction(
                kind=kind,
                start_time=timestamp,
                duration=self.config.distraction_duration,
            )
            self._distraction_history.append(self._active_distraction)

            # Schedule next, getting faster
            self._distraction_interval = max(
                self.config.distraction_interval_min,
                self._distraction_interval * self.config.distraction_interval_decay,
            )
            self._next_distraction_time = timestamp + self._distraction_interval

    def _get_countdown_value(self, timestamp: float) -> int:
        """Get the current countdown number (3, 2, 1)."""
        if self._countdown_start is None:
            return 0
        elapsed = timestamp - self._countdown_start
        remaining = self.config.countdown_seconds - int(elapsed)
        return max(0, remaining)

    def _get_message(self, timestamp: float) -> str:
        """Generate display message based on current phase."""
        match self._phase:
            case Phase.WAITING:
                return "Press SPACE to start. DON'T BLINK."
            case Phase.COUNTDOWN:
                val = self._get_countdown_value(timestamp)
                return str(val) if val > 0 else "GO!"
            case Phase.PLAYING:
                return f"{self._survival_time:.1f}s"
            case Phase.BLINKED:
                if self._death_reason == "face_lost":
                    return f"YOU LEFT THE SCREEN! {self._survival_time:.1f}s"
                return f"YOU BLINKED! {self._survival_time:.1f}s"
            case Phase.NO_FACE:
                return "WHERE'D YOU GO? Show your face!"
            case _:
                return ""

    def _make_state(self, timestamp: float) -> GameState:
        """Build the state object for the UI."""
        return GameState(
            phase=self._phase,
            survival_time=self._survival_time,
            countdown_value=self._get_countdown_value(timestamp),
            ear_value=self._ear_value,
            active_distraction=self._active_distraction,
            high_score=self._high_score,
            message=self._get_message(timestamp),
            penalty_style=self.config.penalty_style,
            blink_count=self._blink_count,
            death_reason=self._death_reason,
        )
