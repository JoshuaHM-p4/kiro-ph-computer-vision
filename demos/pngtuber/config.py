"""Tunables for the PNGTuber sprite switcher.

Expressions are inferred from landmark ratios measured *against a calibrated
neutral baseline*, not against absolute numbers. Faces differ enough that a fixed
"mouth open above 0.35" rule misfires on some people; a per-user baseline plus
deltas behaves consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Sprite filenames are "{yaw}_{expression}.png" so the set is easy to browse and
# to replace with real art.
YAW_BUCKETS: tuple[str, ...] = ("left", "center", "right")
EXPRESSIONS: tuple[str, ...] = ("neutral", "happy", "surprised", "angry")


@dataclass
class PngTuberConfig:
    """Sprite source, yaw bucketing, and expression thresholds."""

    sprites_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "assets" / "sprites"
    )

    # Yaw bucketing, in degrees. Positive yaw means the head turned toward the
    # right of the image. ``yaw_release`` is deliberately smaller than
    # ``yaw_enter``: a head parked near a boundary would otherwise flip sprites
    # every frame.
    yaw_enter: float = 18.0
    yaw_release: float = 11.0
    yaw_alpha: float = 0.35

    # Expression deltas relative to the neutral baseline, in eye widths.
    mouth_open_delta: float = 0.14      # mouth aspect ratio rise -> surprised/happy
    smile_delta: float = 0.045          # mouth corner lift rise -> happy
    brow_raise_delta: float = 0.055     # brow-eye distance rise -> surprised
    brow_lower_delta: float = 0.035     # brow-eye distance drop -> angry
    squint_delta: float = 0.055         # eye aspect ratio drop -> angry
    # An expression must persist this long before the sprite changes, which stops
    # a single noisy frame from flickering the avatar.
    expression_hold: float = 0.18

    # Every expression ratio divides a vertical distance by a horizontal one
    # (interocular distance, eye width, mouth width). Turning the head
    # foreshortens the horizontal denominator by cos(yaw), which inflates all of
    # them: at 40 degrees a relaxed face reads as surprised. Multiplying the
    # ratios back by cos(yaw) removes that coupling.
    yaw_compensation: bool = True
    # Floor on the cosine so a near-profile face cannot amplify noise (0.45 is
    # about 63 degrees, past which the far eye is barely visible anyway).
    min_yaw_cosine: float = 0.45

    # Calibration: the neutral baseline is averaged over this many seconds at
    # startup, and can be recaptured with the "c" key or the calibrate command.
    calibration_seconds: float = 1.0
    # Ratio smoothing; lower is steadier but laggier.
    ratio_alpha: float = 0.4

    # Presentation.
    sprite_scale: float = 0.85          # fraction of frame height
    bob_amplitude: float = 0.012        # idle bob, fraction of frame height
    bob_speed: float = 1.6              # cycles per second
    talk_bob_boost: float = 2.0         # bob faster while the mouth is open
    background: tuple[int, int, int] = (18, 12, 24)
    canvas_width: int = 1280
    canvas_height: int = 720

    def sprite_name(self, yaw_bucket: str, expression: str) -> str:
        return f"{yaw_bucket}_{expression}.png"
