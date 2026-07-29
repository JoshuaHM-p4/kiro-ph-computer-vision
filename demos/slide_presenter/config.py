"""Tunables for the slide presenter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class SlideConfig:
    """Deck source, gesture thresholds and laser behaviour."""

    slides_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "assets" / "slides"
    )
    # Pinch thresholds are thumb-index distance over hand span.
    pinch_start_ratio: float = 0.34
    pinch_release_ratio: float = 0.52
    # One physical pinch must not fire twice; this also rate-limits fast pinching.
    advance_cooldown: float = 0.7
    # Laser smoothing: lower is steadier but lags the fingertip.
    laser_alpha: float = 0.45
    # The laser fades out when the pointing gesture stops.
    laser_hold: float = 0.35
    laser_radius: float = 0.012

    # Which hand does what. MediaPipe labels assume a mirrored (selfie) image;
    # set swap_handedness if next/previous feel inverted on your setup.
    next_hand: str = "Right"
    previous_hand: str = "Left"
    swap_handedness: bool = False

    # Wrap from the last slide back to the first.
    loop_deck: bool = False
    # Rendering size for the web /snapshot canvas.
    canvas_width: int = 1280
    canvas_height: int = 720
