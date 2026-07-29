"""PNGTuber: head yaw and facial expression drive which sprite is shown.

Pipeline for each frame:

    face landmarks
      -> head_pose()          yaw in degrees (solvePnP, ratio fallback)
      -> yaw bucket           left / center / right, with hysteresis
      -> landmark ratios      MAR, mouth corner lift, brow raise, EAR
      -> deltas vs baseline   captured during calibration
      -> expression           neutral / happy / surprised / angry
      -> sprite id            "{bucket}_{expression}"

The legacy MediaPipe solutions API exposes no blendshapes, and the browser's
blendshapes are deliberately ignored, so this classifier is the single source of
truth for both front ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..common import gestures as gs
from ..common import hud
from ..common import landmarks as lm
from ..common.geometry import EMAScalar, clamp
from .config import EXPRESSIONS, YAW_BUCKETS, PngTuberConfig

NEUTRAL, HAPPY, SURPRISED, ANGRY = EXPRESSIONS


@dataclass
class Ratios:
    """The four landmark measurements the classifier uses."""

    mouth_open: float
    smile: float
    brow: float
    eye: float

    @classmethod
    def measure(cls, face: lm.Face) -> "Ratios":
        return cls(
            mouth_open=gs.mouth_aspect_ratio(face),
            smile=gs.mouth_corner_lift(face),
            brow=gs.brow_raise(face),
            eye=gs.average_ear(face),
        )

    def scaled(self, factor: float) -> "Ratios":
        """All four ratios multiplied by ``factor`` (see yaw compensation)."""
        return Ratios(
            mouth_open=self.mouth_open * factor,
            smile=self.smile * factor,
            brow=self.brow * factor,
            eye=self.eye * factor,
        )

    def to_json(self) -> dict[str, float]:
        return {
            "mouthOpen": round(self.mouth_open, 4),
            "smile": round(self.smile, 4),
            "brow": round(self.brow, 4),
            "eye": round(self.eye, 4),
        }


@dataclass
class Baseline:
    """Neutral-face averages, collected over the calibration window."""

    samples: int = 0
    mouth_open: float = 0.0
    smile: float = 0.0
    brow: float = 0.0
    eye: float = 0.0
    ready: bool = False

    def add(self, ratios: Ratios) -> None:
        self.samples += 1
        weight = 1.0 / self.samples
        self.mouth_open += (ratios.mouth_open - self.mouth_open) * weight
        self.smile += (ratios.smile - self.smile) * weight
        self.brow += (ratios.brow - self.brow) * weight
        self.eye += (ratios.eye - self.eye) * weight

    def deltas(self, ratios: Ratios) -> Ratios:
        return Ratios(
            mouth_open=ratios.mouth_open - self.mouth_open,
            smile=ratios.smile - self.smile,
            brow=ratios.brow - self.brow,
            eye=ratios.eye - self.eye,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "samples": self.samples,
            "mouthOpen": round(self.mouth_open, 4),
            "smile": round(self.smile, 4),
            "brow": round(self.brow, 4),
            "eye": round(self.eye, 4),
        }


def classify_expression(deltas: Ratios, config: PngTuberConfig) -> str:
    """Pick an expression from baseline-relative deltas.

    Order matters: surprise (wide eyes, raised brows, open mouth) is checked
    before happy, because a broad smile also opens the mouth. Angry is last so a
    smile with slightly lowered brows still reads as happy.
    """
    mouth_open = deltas.mouth_open >= config.mouth_open_delta
    smiling = deltas.smile >= config.smile_delta
    brows_up = deltas.brow >= config.brow_raise_delta
    brows_down = deltas.brow <= -config.brow_lower_delta
    squinting = deltas.eye <= -config.squint_delta

    if brows_up and mouth_open:
        return SURPRISED
    if smiling:
        return HAPPY
    if brows_down or (squinting and not mouth_open):
        return ANGRY
    if brows_up:
        return SURPRISED
    return NEUTRAL


class YawBucketer:
    """Maps a yaw angle to left/center/right with hysteresis.

    Leaving centre needs ``yaw_enter`` degrees, returning needs the angle to fall
    back inside ``yaw_release``. The gap is what keeps a head held near the
    boundary from oscillating between two sprites.
    """

    def __init__(self, config: PngTuberConfig):
        self.config = config
        self.bucket = "center"

    def update(self, yaw: float) -> str:
        enter, release = self.config.yaw_enter, self.config.yaw_release
        if self.bucket == "center":
            if yaw >= enter:
                self.bucket = "right"
            elif yaw <= -enter:
                self.bucket = "left"
        elif self.bucket == "right":
            if yaw < release:
                self.bucket = "center" if yaw > -enter else "left"
        elif self.bucket == "left":
            if yaw > -release:
                self.bucket = "center" if yaw < enter else "right"
        return self.bucket

    def reset(self) -> None:
        self.bucket = "center"


class SpriteSet:
    """Loads ``{yaw}_{expression}.png`` sprites, with graceful fallbacks."""

    def __init__(self, directory: Path, config: PngTuberConfig):
        self.directory = Path(directory)
        self.config = config
        self._cache: dict[str, np.ndarray | None] = {}

    def available(self) -> dict[str, bool]:
        return {
            f"{bucket}_{expression}": (self.directory / config_name).is_file()
            for bucket in YAW_BUCKETS
            for expression in EXPRESSIONS
            for config_name in (self.config.sprite_name(bucket, expression),)
        }

    @property
    def missing(self) -> list[str]:
        return [key for key, present in self.available().items() if not present]

    def get(self, bucket: str, expression: str) -> np.ndarray | None:
        """Sprite as BGRA, falling back to neutral then to centre-neutral."""
        for candidate_bucket, candidate_expression in (
            (bucket, expression),
            (bucket, NEUTRAL),
            ("center", NEUTRAL),
        ):
            key = f"{candidate_bucket}_{candidate_expression}"
            if key not in self._cache:
                path = self.directory / self.config.sprite_name(candidate_bucket, candidate_expression)
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
                if image is not None and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                self._cache[key] = image
            if self._cache[key] is not None:
                return self._cache[key]
        return None

    def reload(self) -> None:
        self._cache.clear()


class PngTuberCore:
    """Chooses and renders the sprite that matches the current face."""

    def __init__(self, config: PngTuberConfig | None = None):
        self.config = config or PngTuberConfig()
        self.sprites = SpriteSet(self.config.sprites_dir, self.config)
        self.bucketer = YawBucketer(self.config)

        self.baseline = Baseline()
        self.ratios: Ratios | None = None
        self.deltas: Ratios | None = None
        self.yaw: float | None = None
        self.method = "none"
        self.expression = NEUTRAL
        self.face_visible = False
        self.frames = 0
        self.switches = 0
        self.revision = 0

        self._yaw_smoother = EMAScalar(alpha=self.config.yaw_alpha)
        self._ratio_smoothers = {
            name: EMAScalar(alpha=self.config.ratio_alpha)
            for name in ("mouth_open", "smile", "brow", "eye")
        }
        self._calibrating_until: float | None = None
        self._candidate = NEUTRAL
        self._candidate_since = 0.0
        self._started_at: float | None = None

    # -- main entry point --------------------------------------------------
    def update(self, frame: lm.LandmarkFrame, now: float) -> dict[str, Any]:
        self.frames += 1
        face = frame.face
        self.face_visible = face is not None

        if self._started_at is None:
            self._started_at = now
            self._calibrating_until = now + self.config.calibration_seconds

        if face is None:
            return self.state(now)

        # Head pose comes first: the yaw it reports is needed to undo the
        # foreshortening baked into the ratios below.
        pose = gs.head_pose(face, frame.width or 640, frame.height or 480)
        self.method = pose.method
        self.yaw = self._yaw_smoother.update(pose.yaw) or 0.0
        self.bucketer.update(self.yaw)

        # Compensate with THIS frame's yaw, not the smoothed one: the ratios come
        # from this frame too, and during a fast turn the smoothed yaw lags far
        # enough behind to under-correct and flash a false expression.
        raw = Ratios.measure(face).scaled(self._yaw_factor(pose.yaw))
        self.ratios = Ratios(
            mouth_open=self._ratio_smoothers["mouth_open"].update(raw.mouth_open) or 0.0,
            smile=self._ratio_smoothers["smile"].update(raw.smile) or 0.0,
            brow=self._ratio_smoothers["brow"].update(raw.brow) or 0.0,
            eye=self._ratio_smoothers["eye"].update(raw.eye) or 0.0,
        )

        if self._calibrating_until is not None and now <= self._calibrating_until:
            # Whatever the user is doing during the window is defined as neutral.
            self.baseline.add(self.ratios)
            self.expression = NEUTRAL
            self._candidate = NEUTRAL
            return self.state(now)

        if self._calibrating_until is not None:
            self._calibrating_until = None
            self.baseline.ready = self.baseline.samples > 0

        self.deltas = self.baseline.deltas(self.ratios)
        self._settle_expression(classify_expression(self.deltas, self.config), now)
        return self.state(now)

    def _yaw_factor(self, yaw: float | None = None) -> float:
        """cos(yaw), floored, or 1.0 when compensation is disabled."""
        yaw = self.yaw if yaw is None else yaw
        if not self.config.yaw_compensation or yaw is None:
            return 1.0
        return max(math.cos(math.radians(yaw)), self.config.min_yaw_cosine)

    def _settle_expression(self, candidate: str, now: float) -> None:
        """Require an expression to persist before it takes effect."""
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_since = now
            return
        if candidate != self.expression and (now - self._candidate_since) >= self.config.expression_hold:
            self.expression = candidate
            self.switches += 1
            self.revision += 1

    # -- commands ----------------------------------------------------------
    def calibrate(self, now: float = 0.0) -> None:
        """Restart the neutral-baseline capture."""
        self.baseline = Baseline()
        self._calibrating_until = now + self.config.calibration_seconds
        self.expression = NEUTRAL
        self._candidate = NEUTRAL

    def handle_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "calibrate":
            self.calibrate(float(payload.get("now", 0.0)))
        elif command == "reload":
            self.sprites.reload()
        elif command == "sprites":
            return {"ok": True, "available": self.sprites.available(), "missing": self.sprites.missing}
        elif command == "expression":
            # Manual override, handy for testing sprite art without pulling faces.
            wanted = str(payload.get("name", NEUTRAL))
            if wanted not in EXPRESSIONS:
                return {"ok": False, "unknown": wanted}
            self.expression = wanted
            self._candidate = wanted
            self.revision += 1
        else:
            return {"ok": False, "unknown": command}
        return {"ok": True, "sprite": self.sprite_id}

    def reset(self) -> None:
        self.baseline = Baseline()
        self.ratios = None
        self.deltas = None
        self.yaw = None
        self.expression = NEUTRAL
        self._candidate = NEUTRAL
        self.switches = 0
        self.frames = 0
        self._started_at = None
        self._calibrating_until = None
        self._yaw_smoother.reset()
        for smoother in self._ratio_smoothers.values():
            smoother.reset()
        self.bucketer.reset()
        self.revision += 1

    # -- output ------------------------------------------------------------
    @property
    def yaw_bucket(self) -> str:
        return self.bucketer.bucket

    @property
    def sprite_id(self) -> str:
        return f"{self.yaw_bucket}_{self.expression}"

    @property
    def calibrating(self) -> bool:
        return self._calibrating_until is not None

    def state(self, now: float) -> dict[str, Any]:
        return {
            "sprite": self.sprite_id,
            "yawBucket": self.yaw_bucket,
            "expression": self.expression,
            "yaw": round(self.yaw, 2) if self.yaw is not None else None,
            "yawMethod": self.method,
            "faceVisible": self.face_visible,
            "calibrating": self.calibrating,
            "baseline": self.baseline.to_json(),
            "ratios": self.ratios.to_json() if self.ratios else None,
            "deltas": self.deltas.to_json() if self.deltas else None,
            "switches": self.switches,
            "frames": self.frames,
            "revision": self.revision,
            "missingSprites": self.sprites.missing,
        }

    # -- rendering ---------------------------------------------------------
    def render_sprite(self, frame: np.ndarray, now: float = 0.0) -> np.ndarray:
        """Composite the current sprite onto ``frame``, with an idle bob."""
        sprite = self.sprites.get(self.yaw_bucket, self.expression)
        height, width = frame.shape[:2]
        if sprite is None:
            hud.text(
                frame,
                "No sprites found. Run:",
                (40, height // 2 - 20),
                scale=0.7,
                color=hud.THEME.amber,
            )
            hud.text(
                frame,
                ".venv/bin/python -m demos.tools.make_placeholder_sprites",
                (40, height // 2 + 16),
                scale=0.6,
                color=hud.THEME.dim,
            )
            return frame

        target_h = max(1, int(height * self.config.sprite_scale))
        scale = target_h / sprite.shape[0]
        target_w = max(1, int(sprite.shape[1] * scale))
        resized = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Bob faster while talking so the avatar feels alive.
        speed = self.config.bob_speed
        if self.expression in (SURPRISED, HAPPY):
            speed *= self.config.talk_bob_boost
        bob = int(np.sin(now * speed * 2 * np.pi) * self.config.bob_amplitude * height)

        x = (width - target_w) // 2
        y = height - target_h + bob
        hud.alpha_blit(frame, resized, (x, y))
        return frame

    def render_canvas(self) -> np.ndarray:
        """Sprite over the configured background, for /snapshot."""
        canvas = np.full(
            (self.config.canvas_height, self.config.canvas_width, 3),
            self.config.background,
            dtype=np.uint8,
        )
        self.render_sprite(canvas, now=0.0)
        hud.title(canvas, "PNGTUBER", self.sprite_id.replace("_", " / "))
        return canvas


def draw_debug(frame: np.ndarray, core: PngTuberCore, state: dict) -> np.ndarray:
    """Yaw dial, ratio gauges and the chosen sprite id."""
    height, width = frame.shape[:2]
    hud.panel(frame, (18, height - 150), (330, height - 24), alpha=0.6)

    yaw = state["yaw"]
    hud.text(
        frame,
        f"YAW {yaw:+.1f}deg ({state['yawMethod']})" if yaw is not None else "YAW  --",
        (32, height - 124),
        scale=0.5,
        color=hud.THEME.cyan,
    )
    # Yaw dial: a marker sliding along a track from -45 to +45 degrees.
    track_y = height - 108
    cv2.line(frame, (32, track_y), (312, track_y), hud.THEME.grid, 2, cv2.LINE_AA)
    if yaw is not None:
        marker = int(172 + clamp(yaw / 45.0, -1.0, 1.0) * 140)
        cv2.circle(frame, (marker, track_y), 6, hud.THEME.magenta, -1, cv2.LINE_AA)
    for label, x in (("L", 32), ("C", 172), ("R", 312)):
        hud.text(frame, label, (x - 4, track_y + 18), scale=0.36, color=hud.THEME.dim)

    deltas = state["deltas"]
    if deltas:
        for index, (label, key, span) in enumerate(
            (
                ("MOUTH", "mouthOpen", core.config.mouth_open_delta * 2),
                ("SMILE", "smile", core.config.smile_delta * 2),
                ("BROW", "brow", core.config.brow_raise_delta * 2),
                ("EYE", "eye", core.config.squint_delta * 2),
            )
        ):
            y = height - 84 + index * 15
            hud.text(frame, label, (32, y + 9), scale=0.34, color=hud.THEME.dim)
            value = (deltas[key] / span + 0.5) if span else 0.5
            hud.gauge(frame, (92, y), (150, 10), value, color=hud.THEME.lime)
            hud.text(frame, f"{deltas[key]:+.3f}", (250, y + 9), scale=0.34, color=hud.THEME.white)

    hud.text(
        frame,
        state["sprite"].upper().replace("_", "  "),
        (26, 86),
        scale=0.7,
        color=hud.THEME.lime,
        thickness=2,
    )
    if state["calibrating"]:
        hud.text(frame, "CALIBRATING NEUTRAL - hold a relaxed face", (26, 112), scale=0.5, color=hud.THEME.amber)
    elif not state["faceVisible"]:
        hud.text(frame, "NO FACE DETECTED", (26, 112), scale=0.5, color=hud.THEME.red)
    return frame
