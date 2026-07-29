"""The Scavenger Hunt game logic.

"Bring me a cell phone" appears, a 30 second timer runs, and you either hold the
item up to the webcam or upload a photo of it. Detections come from a YOLO model
(COCO's 80 classes) in :mod:`demos.scavenger_hunt.detector`.

Like every demo here the core is pure: it takes **detections** plus a clock and
returns state. It never touches a camera, a model or Flask, so the whole game —
timers, scoring, streaks, near-misses — is tested with scripted detections.

Round lifecycle::

    IDLE --start--> PLAYING --target held long enough--> FOUND --pause--> next round
                       |                                                    |
                       +--timer expires--> MISSED --pause--> next round      |
                                                                            v
                                              after `rounds` rounds --> FINISHED
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from ..common import hud
from ..common.geometry import clamp, to_pixels
from .config import COCO_CLASSES, SETTING_BOUNDS, HuntConfig
from .detector import Detection

IDLE = "idle"
PLAYING = "playing"
FOUND = "found"
MISSED = "missed"
FINISHED = "finished"


@dataclass
class RoundResult:
    """What happened in one round."""

    item: str
    found: bool
    seconds: float
    points: int

    def to_json(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "found": self.found,
            "seconds": round(self.seconds, 2),
            "points": self.points,
        }


@dataclass
class ScavengerHuntCore:
    """Timer, target selection and scoring for the hunt."""

    config: HuntConfig = field(default_factory=HuntConfig)

    def __post_init__(self) -> None:
        self._random = random.Random(self.config.seed)
        self.phase: str = IDLE
        self.target: str | None = None
        self.round_index: int = 0
        self.score: int = 0
        self.streak: int = 0
        self.best_streak: int = 0
        self.results: list[RoundResult] = []
        self.detections: list[Detection] = []
        self.match: Detection | None = None
        self.revision: int = 0
        self._round_started: float = 0.0
        # The duration this round was started with. Held separately from the config
        # so changing the timer mid-round cannot retroactively end it.
        self._round_duration: float = self.config.round_seconds
        self._phase_started: float = 0.0
        self._holding_since: float | None = None
        self._recent: list[str] = []
        self.last_event: str = ""

    # -- lifecycle ---------------------------------------------------------
    def start(self, now: float = 0.0) -> None:
        """Begin a new game from round one."""
        self.phase = PLAYING
        self.round_index = 0
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.results = []
        self._recent = []
        self.last_event = "game started"
        self._next_target(now)

    def _next_target(self, now: float) -> None:
        self.target = self._pick_item()
        self.phase = PLAYING
        self._round_started = now
        self._round_duration = self.config.round_seconds
        self._phase_started = now
        self._holding_since = None
        self.match = None
        self.revision += 1

    def _pick_item(self) -> str:
        """Choose a target, avoiding immediate repeats."""
        pool = [item for item in self.config.item_pool if item in COCO_CLASSES]
        if not pool:
            pool = ["cup"]
        # Keep a short memory so a five-round game does not ask for the same thing
        # twice in a row.
        fresh = [item for item in pool if item not in self._recent[-3:]]
        chosen = self._random.choice(fresh or pool)
        self._recent.append(chosen)
        return chosen

    def submit_photo(self, detections: Sequence[Detection], now: float) -> dict[str, Any]:
        """Score an uploaded still.

        A photo cannot be "held in front of the camera", so the hold window does
        not apply: one confident detection of the target ends the round. Doing it
        with an explicit method rather than by faking a future timestamp also
        avoids float precision trouble, since ``time.monotonic()`` values are large
        enough that ``now + 0.4`` minus ``now`` is not reliably ``>= 0.4``.
        """
        self.detections = list(detections)
        self.match = self._best_match()
        if self.phase == PLAYING and self.match is not None:
            self._finish_round(found=True, now=now, event=f"found {self.target} in a photo")
        return self.state(now)

    def skip(self, now: float = 0.0) -> None:
        """Give up on the current item; counts as a miss."""
        if self.phase == PLAYING:
            self._finish_round(found=False, now=now, event="skipped")

    def reset(self) -> None:
        settings = self.config
        self.__post_init__()
        self.config = settings

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        """Apply validated settings and report what actually took effect.

        Confidence takes effect immediately, since it only changes what counts as a
        detection. The timer, round count and hold window apply from the **next**
        round: shortening the clock mid-round would otherwise retroactively end a
        round the player is still in the middle of.
        """
        applied: dict[str, Any] = {}
        for key, raw in (values or {}).items():
            if key not in SETTING_BOUNDS:
                continue
            low, high, _ = SETTING_BOUNDS[key]
            try:
                value = clamp(float(raw), low, high)
            except (TypeError, ValueError):
                continue
            if key == "roundSeconds":
                self.config.round_seconds = value
            elif key == "confidence":
                self.config.confidence = value
            elif key == "rounds":
                self.config.rounds = int(value)
            elif key == "holdSeconds":
                self.config.hold_seconds = value
            applied[key] = int(value) if key == "rounds" else round(value, 3)

        # A confidence change can make the object in view count right away, so drop
        # a stale hold rather than carrying progress earned under the old threshold.
        if "confidence" in applied:
            self._holding_since = None
        self.revision += 1
        return applied

    def settings(self) -> dict[str, Any]:
        return {
            "roundSeconds": self.config.round_seconds,
            "confidence": round(self.config.confidence, 3),
            "rounds": self.config.rounds,
            "holdSeconds": round(self.config.hold_seconds, 3),
        }

    # -- main entry point --------------------------------------------------
    def update(self, detections: Sequence[Detection], now: float) -> dict[str, Any]:
        """Feed the latest detections and advance the clock."""
        self.detections = list(detections)
        self.match = self._best_match()

        if self.phase == PLAYING:
            if self.match is not None:
                if self._holding_since is None:
                    self._holding_since = now
                if (now - self._holding_since) >= self.config.hold_seconds:
                    self._finish_round(found=True, now=now, event=f"found {self.target}")
            else:
                # Lost sight of it: the hold has to start over, which is what stops
                # a single lucky frame from winning the round.
                self._holding_since = None

            if self.phase == PLAYING and self.time_left(now) <= 0:
                self._finish_round(found=False, now=now, event=f"out of time on {self.target}")

        elif self.phase in (FOUND, MISSED):
            if (now - self._phase_started) >= self.config.celebrate_seconds:
                if self.round_index >= self.config.rounds:
                    self.phase = FINISHED
                    self.last_event = "game over"
                    self.revision += 1
                else:
                    self._next_target(now)

        return self.state(now)

    def _best_match(self) -> Detection | None:
        """Highest-confidence detection of the current target."""
        if self.target is None:
            return None
        candidates = [
            detection
            for detection in self.detections
            if detection.label == self.target and detection.confidence >= self.config.confidence
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda detection: detection.confidence)

    def _finish_round(self, *, found: bool, now: float, event: str) -> None:
        elapsed = now - self._round_started
        points = self._points_for(elapsed) if found else 0
        self.score += points
        self.streak = self.streak + 1 if found else 0
        self.best_streak = max(self.best_streak, self.streak)
        self.results.append(
            RoundResult(item=self.target or "", found=found, seconds=elapsed, points=points)
        )
        self.round_index += 1
        self.phase = FOUND if found else MISSED
        self._phase_started = now
        self._holding_since = None
        self.last_event = event
        self.revision += 1

    def _points_for(self, elapsed: float) -> int:
        """Base award, plus a bonus for the time left and the current streak."""
        remaining = clamp(
            (self._round_duration - elapsed) / max(self._round_duration, 1e-6), 0.0, 1.0
        )
        return int(
            self.config.base_points
            + self.config.speed_bonus * remaining
            + self.config.streak_bonus * self.streak
        )

    # -- derived values ----------------------------------------------------
    def time_left(self, now: float) -> float:
        if self.phase != PLAYING:
            return 0.0
        return max(0.0, self._round_duration - (now - self._round_started))

    def hold_progress(self, now: float) -> float:
        """0..1 progress of the "keep holding it there" window."""
        if self._holding_since is None or self.config.hold_seconds <= 0:
            return 0.0
        return clamp((now - self._holding_since) / self.config.hold_seconds, 0.0, 1.0)

    @property
    def prompt(self) -> str:
        if self.phase == IDLE:
            return "PRESS START"
        if self.phase == FINISHED:
            return "GAME OVER"
        if self.phase == FOUND:
            return "NICE!"
        if self.phase == MISSED:
            return "TIME'S UP"
        return f"BRING ME {(self.target or '').upper()}"

    @property
    def found_count(self) -> int:
        return sum(1 for result in self.results if result.found)

    # -- web interface -----------------------------------------------------
    def handle_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = float(payload.get("now", 0.0))
        if command == "start":
            self.start(now)
        elif command == "skip":
            self.skip(now)
        elif command == "reset":
            self.reset()
        elif command == "settings":
            applied = self.update_settings(payload)
            return {"ok": True, "applied": applied, **self.state(now)}
        else:
            return {"ok": False, "unknown": command}
        return {"ok": True, **self.state(now)}

    def state(self, now: float) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "target": self.target,
            "prompt": self.prompt,
            "hint": self.config.hint_for(self.target or ""),
            "timeLeft": round(self.time_left(now), 2),
            # The duration of the round in progress, which may differ from the
            # configured value while a settings change waits for the next round.
            "roundSeconds": self._round_duration,
            "settings": self.settings(),
            "bounds": {key: list(value) for key, value in SETTING_BOUNDS.items()},
            "roundIndex": self.round_index,
            "rounds": self.config.rounds,
            "score": self.score,
            "streak": self.streak,
            "bestStreak": self.best_streak,
            "foundCount": self.found_count,
            "holdProgress": round(self.hold_progress(now), 3),
            "match": self.match.to_json() if self.match else None,
            "detections": [detection.to_json() for detection in self.detections],
            "results": [result.to_json() for result in self.results],
            "lastEvent": self.last_event,
            "revision": self.revision,
        }

    # -- rendering ---------------------------------------------------------
    def render_canvas(self) -> np.ndarray:
        """Scoreboard for /snapshot."""
        width, height = self.config.canvas_width, self.config.canvas_height
        canvas = np.full((height, width, 3), (14, 10, 20), dtype=np.uint8)
        hud.title(canvas, "SCAVENGER HUNT", "COCO 80-class object hunt")

        text = str(self.score)
        (tw, th), _ = cv2.getTextSize(text, hud.FONT, 6.0, 14)
        origin = ((width - tw) // 2, (height + th) // 2 - 40)

        def draw(layer: np.ndarray) -> None:
            cv2.putText(layer, text, origin, hud.FONT, 6.0, hud.THEME.lime, 14, cv2.LINE_AA)

        hud.glow(canvas, draw, blur=61, intensity=0.7)
        hud.text(canvas, "POINTS", (origin[0], origin[1] + 54), scale=0.9, color=hud.THEME.dim, thickness=2)

        y = int(height * 0.72)
        for result in self.results[-6:]:
            mark = "OK  " if result.found else "MISS"
            colour = hud.THEME.lime if result.found else hud.THEME.red
            hud.text(
                canvas,
                f"{mark} {result.item:<14} {result.seconds:5.1f}s  {result.points:>4} pts",
                (int(width * 0.28), y),
                scale=0.6,
                color=colour,
            )
            y += 28

        hud.status_strip(
            canvas,
            [
                ("FOUND", f"{self.found_count}/{len(self.results)}"),
                ("BEST STREAK", str(self.best_streak)),
                ("PHASE", self.phase.upper()),
            ],
        )
        return canvas


def draw_overlay(
    frame: np.ndarray,
    core: ScavengerHuntCore,
    state: dict[str, Any],
    *,
    debug: bool = True,
) -> np.ndarray:
    """Boxes, the "bring me" prompt, the timer ring and the score."""
    height, width = frame.shape[:2]

    for detection in core.detections:
        is_target = detection.label == state["target"]
        colour = hud.THEME.lime if is_target else hud.THEME.dim
        x1, y1 = to_pixels(detection.box[:2], width, height)
        x2, y2 = to_pixels(detection.box[2:], width, height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3 if is_target else 1, cv2.LINE_AA)
        label = f"{detection.label} {detection.confidence:.0%}"
        hud.text(frame, label, (x1 + 4, max(16, y1 - 8)), scale=0.46, color=colour)

    # Prompt banner.
    banner_colour = {
        FOUND: hud.THEME.lime,
        MISSED: hud.THEME.red,
        FINISHED: hud.THEME.magenta,
    }.get(state["phase"], hud.THEME.cyan)
    hud.panel(frame, (int(width * 0.06), 60), (int(width * 0.94), 132), alpha=0.66, border=banner_colour)
    prompt = state["prompt"]
    scale = 1.5 if state["phase"] == PLAYING else 1.2
    (tw, _), _ = cv2.getTextSize(prompt, hud.FONT, scale, 3)
    hud.text(frame, prompt, ((width - tw) // 2, 116), scale=scale, color=banner_colour, thickness=3)
    if state["hint"] and state["phase"] == PLAYING:
        hud.text(frame, f"({state['hint']})", (int(width * 0.08), 156), scale=0.46, color=hud.THEME.dim)

    # Timer bar and countdown.
    if state["phase"] == PLAYING:
        fraction = state["timeLeft"] / max(state["roundSeconds"], 1e-6)
        colour = hud.THEME.lime if fraction > 0.5 else hud.THEME.amber if fraction > 0.2 else hud.THEME.red
        cv2.rectangle(frame, (0, 0), (int(width * fraction), 10), colour, -1)
        # Top-right, clear of the title panel on the left.
        countdown = f"{state['timeLeft']:4.1f}s"
        (cw, _), _ = cv2.getTextSize(countdown, hud.FONT, 0.9, 2)
        hud.text(frame, countdown, (width - cw - 30, 48), scale=0.9, color=colour, thickness=2)
        if state["holdProgress"] > 0:
            hud.text(frame, "HOLD IT THERE", (int(width * 0.36), 48), scale=0.5, color=hud.THEME.lime)
            hud.gauge(
                frame, (int(width * 0.36) + 150, 34), (140, 14), state["holdProgress"], color=hud.THEME.lime
            )

    # Score block.
    score_text = str(state["score"])
    (sw, _), _ = cv2.getTextSize(score_text, hud.FONT, 2.0, 4)
    origin = (width - sw - 40, height - 40)

    def draw_score(layer: np.ndarray) -> None:
        cv2.putText(layer, score_text, origin, hud.FONT, 2.0, hud.THEME.lime, 4, cv2.LINE_AA)

    hud.glow(frame, draw_score, blur=31, intensity=0.7)
    hud.text(frame, "POINTS", (origin[0], height - 16), scale=0.44, color=hud.THEME.dim)

    if debug:
        hud.status_strip(
            frame,
            [
                ("ROUND", f"{min(state['roundIndex'] + 1, state['rounds'])}/{state['rounds']}"),
                ("FOUND", str(state["foundCount"])),
                ("STREAK", str(state["streak"])),
                ("SEEN", str(len(state["detections"]))),
                ("PASS AT", f"{state['settings']['confidence']:.0%}"),
                ("TIMER", f"{state['settings']['roundSeconds']:.0f}s"),
            ],
            y=height - 96,
        )
    return frame
