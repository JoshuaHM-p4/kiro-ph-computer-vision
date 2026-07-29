"""6-7 rep counter.

The motion being counted is a see-saw: one hand rises while the other drops, over
and over. So instead of asking "is this wrist above a line?", the detector watches
the *difference* between the two wrist heights and counts every time that
difference changes sign — i.e. every time the hands swap which one is higher.

    tilt = (right_wrist_y - left_wrist_y) / torso_length      (positive: left higher)

    tilt >= +deadband  ->  side = "left"
    tilt <= -deadband  ->  side = "right"
    in between         ->  hold the previous side

    count += 1 whenever side changes

Why this beats absolute thresholds:

* Nothing has to be calibrated. There is no "high enough" to tune, so it works
  for tall and short users, seated or standing.
* Moving the whole body up or down in frame cancels out, because both wrists move
  together and only their difference matters.
* Alternation is intrinsic. A sign flip cannot happen unless both hands take part,
  so pumping one hand twice simply cannot register.

Everything is pure: pose landmarks and a clock in, counter state out.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..common import gestures as gs
from ..common import hud
from ..common import landmarks as lm
from ..common.geometry import EMAScalar, clamp, distance, normalize_range, to_pixels
from .config import CounterConfig

SIDES = ("left", "right")

# Counting is gated behind a prepare phase so reps are not lost (or invented)
# while the user is still walking into frame.
PREPARE = "prepare"
COUNTING = "counting"

# Landmarks that must be in frame before counting starts. Deliberately just the
# shoulders and the hands: those are what the measurement needs. Hips would force
# the user to stand much further back for no benefit, since the body scale falls
# back to shoulder width when they are out of frame.
REQUIRED_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_wrist",
    "right_wrist",
)

# Used for the scale reference when available, but never required.
OPTIONAL_LANDMARKS = ("left_hip", "right_hip")


def body_scale(pose: lm.Pose, config: CounterConfig | None = None) -> float:
    """A distance-invariant length reference for this body.

    Prefers shoulder-to-hip length, but falls back to shoulder width scaled by
    ``shoulder_to_torso`` when the hips are missing or outside the frame — which is
    the common case when someone stands close enough to fill the frame. Both give
    the deadband the same practical meaning, so counting behaves identically
    whether or not the hips are in shot.
    """
    config = config or CounterConfig()
    margin = config.frame_margin
    hips_usable = all(
        pose.named_visibility(name) >= config.min_visibility
        and -margin <= pose.named(name)[1] <= 1 + margin
        for name in OPTIONAL_LANDMARKS
    )
    if hips_usable:
        scale = gs.torso_scale(pose)
        if scale > 1e-6:
            return scale

    shoulder_width = distance(pose.named("left_shoulder"), pose.named("right_shoulder"))
    return max(shoulder_width * config.shoulder_to_torso, 1e-6)


def wrist_tilt(pose: lm.Pose, config: CounterConfig | None = None) -> float:
    """Height difference between the wrists, in body-scale units.

    Positive means the **left** wrist is higher on screen. Image y grows downward,
    hence right minus left. Dividing by :func:`body_scale` keeps the deadband
    meaningful at any distance from the camera.
    """
    left = pose.named("left_wrist")
    right = pose.named("right_wrist")
    return (right[1] - left[1]) / body_scale(pose, config)


class TiltTracker:
    """Ternary latch over the tilt signal: left / right / hold.

    Outside the deadband the higher wrist claims the side; inside it the previous
    side is held, which is what stops level hands from chattering.
    """

    def __init__(self, config: CounterConfig):
        self.config = config
        self.side: str | None = None
        self.tilt: float | None = None
        self._smoother = EMAScalar(alpha=config.tilt_alpha)
        self._last_swap_at: float | None = None

    def update(self, tilt: float | None, now: float) -> str | None:
        """Feed a new tilt. Returns the new side when it flips, else None."""
        if tilt is None:
            return None
        self.tilt = self._smoother.update(tilt)
        assert self.tilt is not None

        if self.tilt >= self.config.tilt_enter:
            candidate = "left"
        elif self.tilt <= -self.config.tilt_enter:
            candidate = "right"
        else:
            return None  # deadband: hold the current side

        if candidate == self.side:
            return None
        if self.side is None:
            # First side seen only establishes a baseline; there was nothing to
            # flip from, so it is not a rep.
            self.side = candidate
            self._last_swap_at = now
            return None
        if (
            self._last_swap_at is not None
            and (now - self._last_swap_at) < self.config.min_swap_seconds
        ):
            return None

        self.side = candidate
        self._last_swap_at = now
        return candidate

    @property
    def magnitude(self) -> float:
        """How far past the deadband the current tilt is, 0..1."""
        if self.tilt is None or self.config.tilt_enter <= 0:
            return 0.0
        return clamp(abs(self.tilt) / (self.config.tilt_enter * 2.0), 0.0, 1.0)

    def reset(self) -> None:
        self.side = None
        self.tilt = None
        self._smoother.reset()
        self._last_swap_at = None


class SixSevenCore:
    """Counts hand swaps from pose landmarks."""

    def __init__(self, config: CounterConfig | None = None):
        self.config = config or CounterConfig()
        self.tracker = TiltTracker(self.config)
        self.count = 0
        self.last_swap_seconds: float | None = None
        self.last_swap_at: float | None = None
        self.pose_visible = False
        self.phase = PREPARE
        self.missing: list[str] = list(REQUIRED_LANDMARKS)
        self._ready_since: float | None = None
        self._lost_since: float | None = None
        self.shoulder_y: float | None = None
        self.torso: float | None = None
        self.heights: dict[str, float | None] = {"left": None, "right": None}
        # Normalized wrist positions, kept so the overlay can draw the see-saw
        # line between the actual hands rather than guessing where they are.
        self.wrists: dict[str, tuple[float, float] | None] = {"left": None, "right": None}
        self.revision = 0
        self.history: list[float] = []  # timestamps of counted swaps

    # -- main entry point --------------------------------------------------
    def update(self, frame: lm.LandmarkFrame, now: float) -> dict[str, Any]:
        pose = frame.pose
        self.missing = self._missing_landmarks(pose)
        self.pose_visible = pose is not None and not self.missing

        if not self.pose_visible:
            # Never reset the count: stepping out of frame should not cost reps.
            if self._lost_since is None:
                self._lost_since = now
            if (now - self._lost_since) >= self.config.lost_grace_seconds:
                self.phase = PREPARE
                self._ready_since = None
            return self.state(now)

        self._lost_since = None
        assert pose is not None
        self.shoulder_y = gs.shoulder_line_y(pose)
        self.torso = body_scale(pose, self.config)
        self.heights = {side: gs.wrist_height(pose, side) for side in SIDES}
        self.wrists = {side: pose.named(f"{side}_wrist") for side in SIDES}

        if self._ready_since is None:
            self._ready_since = now
        if self.phase == PREPARE and (now - self._ready_since) >= self.config.prepare_seconds:
            self.phase = COUNTING

        flipped = self.tracker.update(wrist_tilt(pose, self.config), now)

        # During prepare the tracker still runs so its side and smoother are
        # settled the instant counting begins; the flip itself is discarded.
        if flipped is not None and self.phase == COUNTING:
            self.count += 1
            if self.last_swap_at is not None:
                self.last_swap_seconds = round(now - self.last_swap_at, 3)
            self.last_swap_at = now
            self.history.append(now)
            del self.history[: max(0, len(self.history) - 30)]
            self.revision += 1

        return self.state(now)

    def _missing_landmarks(self, pose: lm.Pose | None) -> list[str]:
        """Required landmarks that are absent, occluded, or outside the frame.

        Reported by name so the UI can say *what* to fix ("show both hands")
        instead of a bare "no pose".
        """
        if pose is None:
            return list(REQUIRED_LANDMARKS)

        margin = self.config.frame_margin
        missing: list[str] = []
        for name in REQUIRED_LANDMARKS:
            if pose.named_visibility(name) < self.config.min_visibility:
                missing.append(name)
                continue
            x, y = pose.named(name)
            if not (-margin <= x <= 1 + margin and -margin <= y <= 1 + margin):
                missing.append(name)
        return missing

    # -- phases ------------------------------------------------------------
    def prepare(self) -> None:
        """Return to prepare mode: counting pauses until the pose is steady again."""
        self.phase = PREPARE
        self._ready_since = None
        self._lost_since = None
        self.tracker.reset()

    def prepare_progress(self, now: float) -> float:
        """0..1 fraction of the prepare hold completed."""
        if self.phase == COUNTING:
            return 1.0
        if self._ready_since is None or self.config.prepare_seconds <= 0:
            return 0.0
        return clamp((now - self._ready_since) / self.config.prepare_seconds, 0.0, 1.0)

    # -- derived values ----------------------------------------------------
    @property
    def side(self) -> str | None:
        """Which hand is currently the higher one."""
        return self.tracker.side

    @property
    def expected_hand(self) -> str | None:
        """Which hand must rise next to score the following rep."""
        if self.tracker.side is None:
            return None
        return "right" if self.tracker.side == "left" else "left"

    def reps_per_minute(self) -> float | None:
        """Tempo over the recent swaps, or None before the second one."""
        if len(self.history) < 2:
            return None
        span = self.history[-1] - self.history[0]
        if span <= 0:
            return None
        return round((len(self.history) - 1) / span * 60.0, 1)

    def hint(self) -> str:
        """One short instruction for the user, driven by what is missing."""
        if not self.missing:
            if self.phase == PREPARE:
                return "HOLD STILL"
            if self.expected_hand:
                return f"NEXT: RAISE {self.expected_hand.upper()} HAND"
            return "RAISE ONE HAND ABOVE THE OTHER"

        if len(self.missing) >= len(REQUIRED_LANDMARKS):
            return "STEP INTO FRAME"
        groups = {name.split("_", 1)[1] for name in self.missing}
        if "wrist" in groups:
            return "SHOW BOTH HANDS"
        if "shoulder" in groups:
            return "STEP BACK - SHOULDERS NOT VISIBLE"
        return "ADJUST YOUR POSITION"

    # -- web interface -----------------------------------------------------
    def handle_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command in ("reset", "zero"):
            self.reset()
        elif command == "add":
            self.count += 1
            self.revision += 1
        elif command == "prepare":
            self.prepare()
        elif command == "start":
            # Skip the hold and count immediately.
            self.phase = COUNTING
            self._ready_since = 0.0
        else:
            return {"ok": False, "unknown": command}
        return {"ok": True, "count": self.count}

    def reset(self) -> None:
        self.count = 0
        self.last_swap_at = None
        self.last_swap_seconds = None
        self.history.clear()
        self.tracker.reset()
        self.phase = PREPARE
        self._ready_since = None
        self._lost_since = None
        self.revision += 1

    def state(self, now: float) -> dict[str, Any]:
        return {
            "count": self.count,
            "side": self.tracker.side,
            "tilt": round(self.tracker.tilt, 4) if self.tracker.tilt is not None else None,
            "tiltEnter": self.config.tilt_enter,
            "tiltMagnitude": round(self.tracker.magnitude, 3),
            "hands": {
                side: {
                    "up": self.tracker.side == side,
                    "height": (
                        round(self.heights[side], 4) if self.heights[side] is not None else None
                    ),
                    "point": (
                        {"x": round(self.wrists[side][0], 4), "y": round(self.wrists[side][1], 4)}
                        if self.wrists[side] is not None
                        else None
                    ),
                }
                for side in SIDES
            },
            "expected": self.expected_hand,
            "lastRepSeconds": self.last_swap_seconds,
            "repsPerMinute": self.reps_per_minute(),
            "poseVisible": self.pose_visible,
            "phase": self.phase,
            "prepareProgress": round(self.prepare_progress(now), 3),
            "missing": list(self.missing),
            "hint": self.hint(),
            "shoulderY": round(self.shoulder_y, 4) if self.shoulder_y is not None else None,
            "torsoGuide": round(self.torso, 4) if self.torso else None,
            "revision": self.revision,
        }

    # -- rendering ---------------------------------------------------------
    def render_canvas(self) -> np.ndarray:
        """Scoreboard image for /snapshot."""
        width, height = self.config.canvas_width, self.config.canvas_height
        canvas = np.full((height, width, 3), (14, 10, 20), dtype=np.uint8)
        hud.title(canvas, "6-7 COUNTER", "hand swaps")

        text = str(self.count)
        scale = 9.0
        (tw, th), _ = cv2.getTextSize(text, hud.FONT, scale, 18)
        origin = ((width - tw) // 2, (height + th) // 2)

        def draw(layer: np.ndarray) -> None:
            cv2.putText(layer, text, origin, hud.FONT, scale, hud.THEME.lime, 18, cv2.LINE_AA)

        hud.glow(canvas, draw, blur=61, intensity=0.7)
        hud.text(canvas, "REPS", (origin[0], origin[1] + 60), scale=1.0, color=hud.THEME.dim, thickness=2)

        tempo = self.reps_per_minute()
        hud.status_strip(
            canvas,
            [
                ("TEMPO", f"{tempo:.1f}/min" if tempo else "-"),
                ("LAST", f"{self.last_swap_seconds:.2f}s" if self.last_swap_seconds else "-"),
            ],
        )
        return canvas


def draw_overlay(frame: np.ndarray, core: SixSevenCore, state: dict, *, debug: bool = True) -> np.ndarray:
    """Wrist see-saw line, tilt meter, per-hand pills and the big count.

    There are deliberately no horizontal threshold lines any more: nothing is
    compared against a fixed height, so drawing one would misrepresent how the
    counter works.
    """
    height, width = frame.shape[:2]

    _draw_seesaw(frame, core, state)
    _draw_tilt_meter(frame, state)

    for index, side in enumerate(SIDES):
        info = state["hands"][side]
        color = hud.THEME.lime if info["up"] else hud.THEME.dim
        x = 20 + index * 132
        y = height - 150
        hud.panel(frame, (x, y), (x + 120, y + 58), alpha=0.6, border=color)
        hud.text(frame, side.upper(), (x + 10, y + 24), scale=0.5, color=color)
        hud.text(frame, "HIGHER" if info["up"] else "lower", (x + 10, y + 46), scale=0.4, color=color)

    text = str(state["count"])
    (tw, _), _ = cv2.getTextSize(text, hud.FONT, 3.2, 6)
    origin = (width - tw - 40, 110)

    def draw(layer: np.ndarray) -> None:
        cv2.putText(layer, text, origin, hud.FONT, 3.2, hud.THEME.lime, 6, cv2.LINE_AA)

    hud.glow(frame, draw, blur=41, intensity=0.8)
    hud.text(frame, "REPS", (origin[0], 132), scale=0.5, color=hud.THEME.dim)

    preparing = state["phase"] != COUNTING
    hud.text(
        frame,
        state["hint"],
        (26, 86),
        scale=0.6,
        color=hud.THEME.amber if preparing else hud.THEME.cyan,
        thickness=2,
    )
    if preparing:
        hud.text(frame, "PREPARE", (26, 116), scale=0.44, color=hud.THEME.amber)
        hud.gauge(frame, (110, 104), (220, 14), state["prepareProgress"], color=hud.THEME.amber)
        hud.text(
            frame,
            "counting starts once your shoulders and both hands are visible",
            (26, 140),
            scale=0.4,
            color=hud.THEME.dim,
        )

    if debug:
        hud.status_strip(
            frame,
            [
                ("TILT", f"{state['tilt']:+.2f}" if state["tilt"] is not None else "-"),
                ("HIGHER", (state["side"] or "-").upper()),
                ("TEMPO", f"{state['repsPerMinute']:.1f}/min" if state["repsPerMinute"] else "-"),
                ("LAST", f"{state['lastRepSeconds']:.2f}s" if state["lastRepSeconds"] else "-"),
                ("PHASE", state["phase"].upper()),
            ],
            y=height - 78,
        )
    return frame


def _draw_seesaw(frame: np.ndarray, core: SixSevenCore, state: dict) -> None:
    """Line between the two wrists, tinted by which one is higher.

    This draws the signal literally: the line tips as the hands trade places, and
    the count advances each time it tips past the deadband.
    """
    left = state["hands"]["left"]["point"]
    right = state["hands"]["right"]["point"]
    if not state["poseVisible"] or left is None or right is None:
        return

    height, width = frame.shape[:2]
    left_px = to_pixels((left["x"], left["y"]), width, height)
    right_px = to_pixels((right["x"], right["y"]), width, height)
    active = state["side"] is not None and state["tiltMagnitude"] >= 0.5
    color = hud.THEME.lime if active else hud.THEME.dim

    def draw(layer: np.ndarray) -> None:
        cv2.line(layer, left_px, right_px, color, 4, cv2.LINE_AA)

    hud.glow(frame, draw, blur=25, intensity=0.6)
    for point, side in ((left_px, "left"), (right_px, "right")):
        higher = state["side"] == side
        cv2.circle(frame, point, 13 if higher else 8, color if higher else hud.THEME.dim, -1, cv2.LINE_AA)
        cv2.circle(frame, point, 17, hud.THEME.white if higher else hud.THEME.grid, 1, cv2.LINE_AA)
        if higher:
            hud.text(frame, "HIGHER", (point[0] - 26, point[1] - 24), scale=0.38, color=color)


def _draw_tilt_meter(frame: np.ndarray, state: dict) -> None:
    """Horizontal meter showing the tilt, with the deadband marked."""
    height, width = frame.shape[:2]
    bar_w, bar_h = 280, 16
    x = (width - bar_w) // 2
    y = height - 116

    hud.panel(frame, (x - 12, y - 26), (x + bar_w + 12, y + bar_h + 22), alpha=0.55)
    hud.text(frame, "RIGHT HIGHER", (x - 8, y - 10), scale=0.34, color=hud.THEME.dim)
    hud.text(frame, "LEFT HIGHER", (x + bar_w - 96, y - 10), scale=0.34, color=hud.THEME.dim)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), hud.THEME.grid, 1, cv2.LINE_AA)

    # Deadband shading: inside this band the previous side is held.
    tilt_enter = state["tiltEnter"] or 0.15
    span = tilt_enter * 3.0
    dead_half = int((tilt_enter / span) * (bar_w / 2))
    centre = x + bar_w // 2
    cv2.rectangle(
        frame, (centre - dead_half, y + 1), (centre + dead_half, y + bar_h - 1), (40, 30, 52), -1
    )
    cv2.line(frame, (centre, y - 4), (centre, y + bar_h + 4), hud.THEME.dim, 1, cv2.LINE_AA)

    if state["tilt"] is None:
        return
    position = int(centre + clamp(state["tilt"] / span, -1.0, 1.0) * (bar_w / 2))
    color = hud.THEME.lime if state["tiltMagnitude"] >= 0.5 else hud.THEME.amber
    cv2.circle(frame, (position, y + bar_h // 2), 9, color, -1, cv2.LINE_AA)
    hud.text(
        frame,
        f"{state['tilt']:+.2f} torso",
        (centre - 44, y + bar_h + 18),
        scale=0.36,
        color=hud.THEME.white,
    )
