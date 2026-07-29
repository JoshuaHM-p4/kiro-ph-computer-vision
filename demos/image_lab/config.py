"""Tunables for the image lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImageLabConfig:
    """Sizes and limits for the interactive OpenCV playground."""

    samples_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "assets" / "samples"
    )
    # Uploads are downscaled to this longest edge before anything runs. Keeps the
    # preview responsive and stops a 40 MP phone photo from stalling the server.
    max_dimension: int = 1280
    # Guard against absurd uploads before decoding.
    max_upload_bytes: int = 12 * 1024 * 1024
    # A pipeline this long is already past the point of being a teaching aid.
    max_steps: int = 8
    # Fallback canvas when there is no image at all.
    canvas_width: int = 1024
    canvas_height: int = 640
    # Pipeline the demo opens with, so there is something to look at immediately.
    default_pipeline: tuple[str, ...] = ("gaussian_blur", "canny")
