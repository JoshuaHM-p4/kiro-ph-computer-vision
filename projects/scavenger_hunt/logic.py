"""Pure game logic for the Scavenger Hunt.

Nothing here opens a camera, loads a model, or draws a window.
Feed it a list of detected class labels plus a timestamp and it
returns the new game state as a plain dict — easy to test.

Round lifecycle
---------------
WAITING  ->  PLAYING  (start() called)
PLAYING  ->  FOUND    (target held for HOLD_SECONDS)
PLAYING  ->  SKIP     (skip() called or countdown hits 0)
FOUND    ->  PLAYING  (next round starts automatically after FOUND_PAUSE)
SKIP     ->  PLAYING  (next round starts automatically after SKIP_PAUSE)
PLAYING  ->  DONE     (last round finished)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Every magic number in one place."""

    # Desk-friendly COCO classes to hunt for
    targets: tuple[str, ...] = (
        "cup",
        "cell phone",
        "scissors",
        "book",
        "keyboard",
        "remote",
        "potted plant",
        "banana",
    )

    total_rounds: int = 5
    round_seconds: float = 30.0       # countdown per round
    hold_seconds: float = 1.0         # how long target must stay visible
    found_pause: float = 2.0          # seconds to show "FOUND!" before next round
    skip_pause: float = 1.5           # seconds to show "SKIPPED" before next round

    # Detection quality gate — detections below this confidence are ignored
    # 0.35 works well for typical webcams; raise it if you get false positives
    confidence: float = 0.35          # 0.0–1.0

    # Scoring
    max_points_per_round: int = 100   # full score for finding in ~0 seconds
    min_points_per_round: int = 10    # floor: still get something for a slow find
    streak_bonus_per: int = 20        # extra points added per consecutive find

    # Bounds enforced by apply_settings() so the UI cannot set nonsense values
    min_round_seconds: float = 5.0
    max_round_seconds: float = 180.0
    min_confidence: float = 0.05
    max_confidence: float = 0.95


# ---------------------------------------------------------------------------
# State enum-like constants  (plain strings so the UI can just print them)
# ---------------------------------------------------------------------------

WAITING = "WAITING"
PLAYING = "PLAYING"
FOUND   = "FOUND"
SKIP    = "SKIP"
DONE    = "DONE"


# ---------------------------------------------------------------------------
# Round result (immutable record kept in history)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoundResult:
    round_number: int     # 1-based
    target: str
    found: bool
    points: int
    time_taken: float     # seconds from round start to detection (0 if skipped)
    streak: int           # consecutive finds *including* this one (0 if skipped)


# ---------------------------------------------------------------------------
# Main game class
# ---------------------------------------------------------------------------

class ScavengerHunt:
    """Manages rounds, scoring, hold-timer, and streak across the whole game.

    Usage
    -----
    game = ScavengerHunt()
    game.start(now)
    while True:
        state = game.update(detections, now)
        # render state dict
        if state["phase"] == DONE:
            break
    """

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self._rng = random.Random()   # seeded by default; override in tests

        # Mutable state
        self.phase: str = WAITING
        self.round_number: int = 0         # 1-based, 0 before first round
        self.target: str = ""
        self.score: int = 0
        self.streak: int = 0               # consecutive successful finds
        self.history: list[RoundResult] = []

        # Per-round timestamps (monotonic seconds)
        self._round_start: float = 0.0
        self._hold_since: float | None = None   # when target was first seen this hold
        self._phase_end: float | None = None    # when FOUND/SKIP pause expires

        # Shuffle targets so we don't always start with "cup"
        self._remaining: list[str] = []

        # Settings that cannot take effect mid-round (applied at _begin_round)
        self._pending_settings: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self, now: float) -> dict:
        """Transition from WAITING to the first round."""
        if self.phase != WAITING:
            return self.state(now)
        self._begin_round(now)
        return self.state(now)

    def skip(self, now: float) -> dict:
        """Player pressed N — forfeit the current round."""
        if self.phase == PLAYING:
            self._end_round(found=False, now=now)
        return self.state(now)

    def apply_settings(
        self,
        *,
        round_seconds: float | None = None,
        confidence: float | None = None,
    ) -> dict[str, object]:
        """
        Update game settings.

        ``confidence`` takes effect immediately — it only changes what
        constitutes a valid detection, so it is safe to apply at any time.
        Lowering confidence also clears any in-progress hold, because
        progress earned under a stricter threshold should not carry over.

        ``round_seconds`` is deferred until the *next* round starts. Changing
        the countdown mid-round would retroactively shorten or extend a round
        already in play, which feels unfair.

        Values are clamped to the bounds defined in Config so the UI cannot
        set a 0 % threshold or a one-hour round.

        Returns a dict describing what was applied and what was deferred.
        """
        applied: dict[str, object] = {}
        deferred: dict[str, object] = {}

        if confidence is not None:
            clamped = max(self.cfg.min_confidence,
                          min(self.cfg.max_confidence, float(confidence)))
            self.cfg.confidence = clamped
            # Any hold progress was built under the old threshold — discard it.
            self._hold_since = None
            applied["confidence"] = clamped

        if round_seconds is not None:
            clamped = max(self.cfg.min_round_seconds,
                          min(self.cfg.max_round_seconds, float(round_seconds)))
            if self.phase in (WAITING, DONE):
                # No round is in progress; safe to apply now.
                self.cfg.round_seconds = clamped
                applied["round_seconds"] = clamped
            else:
                self._pending_settings["round_seconds"] = clamped
                deferred["round_seconds"] = clamped

        return {"applied": applied, "deferred": deferred}

    def update(self, detections: Sequence[str], now: float) -> dict:
        """
        Main tick called every frame.

        Parameters
        ----------
        detections:
            Iterable of class-name strings seen in the current frame
            (e.g. ["cup", "person"]).  May be empty.
        now:
            Monotonic timestamp in seconds.

        Returns
        -------
        A plain dict describing everything the UI needs to render.
        """
        if self.phase == WAITING:
            return self.state(now)

        if self.phase in (FOUND, SKIP):
            # Waiting for the inter-round pause to expire
            if self._phase_end is not None and now >= self._phase_end:
                if self.round_number >= self.cfg.total_rounds:
                    self.phase = DONE
                else:
                    self._begin_round(now)
            return self.state(now)

        if self.phase == DONE:
            return self.state(now)

        # ---- PLAYING ----
        elapsed = now - self._round_start
        time_left = max(0.0, self.cfg.round_seconds - elapsed)

        target_visible = self.target in detections

        if target_visible:
            if self._hold_since is None:
                self._hold_since = now
            held_for = now - self._hold_since
            if held_for >= self.cfg.hold_seconds:
                self._end_round(found=True, now=now)
                return self.state(now)
        else:
            # Reset hold timer whenever target disappears
            self._hold_since = None

        # Countdown expired
        if time_left <= 0.0:
            self._end_round(found=False, now=now)

        return self.state(now)

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def state(self, now: float) -> dict:
        """Return a fully self-contained snapshot for the UI to render."""
        elapsed = now - self._round_start if self.phase == PLAYING else 0.0
        time_left = max(0.0, self.cfg.round_seconds - elapsed) if self.phase == PLAYING else 0.0

        hold_progress = 0.0
        if self.phase == PLAYING and self._hold_since is not None:
            hold_progress = min(1.0, (now - self._hold_since) / self.cfg.hold_seconds)

        return {
            "phase": self.phase,
            "round": self.round_number,
            "total_rounds": self.cfg.total_rounds,
            "target": self.target,
            "score": self.score,
            "streak": self.streak,
            "time_left": round(time_left, 2),
            "hold_progress": round(hold_progress, 3),
            "history": list(self.history),
            # Current live settings (for the HUD)
            "round_seconds": self.cfg.round_seconds,
            "confidence": self.cfg.confidence,
            # Settings staged but not yet active (shown as "(pending)" in HUD)
            "pending_settings": dict(self._pending_settings),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_target(self) -> str:
        if not self._remaining:
            self._remaining = list(self.cfg.targets)
            self._rng.shuffle(self._remaining)
        return self._remaining.pop()

    def _begin_round(self, now: float) -> None:
        # Flush any deferred settings before the new round's clock starts.
        if "round_seconds" in self._pending_settings:
            self.cfg.round_seconds = float(self._pending_settings.pop("round_seconds"))

        self.round_number += 1
        self.target = self._pick_target()
        self.phase = PLAYING
        self._round_start = now
        self._hold_since = None
        self._phase_end = None

    def _end_round(self, found: bool, now: float) -> None:
        if found:
            elapsed = now - self._round_start
            # Linear interpolation: full points near 0 s, min points at round_seconds
            frac = min(elapsed / self.cfg.round_seconds, 1.0)
            pts_range = self.cfg.max_points_per_round - self.cfg.min_points_per_round
            base_points = round(self.cfg.max_points_per_round - frac * pts_range)
            self.streak += 1
            bonus = (self.streak - 1) * self.cfg.streak_bonus_per
            round_points = base_points + bonus
            self.score += round_points
            self.phase = FOUND
            self._phase_end = now + self.cfg.found_pause
            result = RoundResult(
                round_number=self.round_number,
                target=self.target,
                found=True,
                points=round_points,
                time_taken=round(now - self._round_start, 2),
                streak=self.streak,
            )
        else:
            self.streak = 0
            self.phase = SKIP
            self._phase_end = now + self.cfg.skip_pause
            result = RoundResult(
                round_number=self.round_number,
                target=self.target,
                found=False,
                points=0,
                time_taken=0.0,
                streak=0,
            )

        self.history.append(result)
        self._hold_since = None
