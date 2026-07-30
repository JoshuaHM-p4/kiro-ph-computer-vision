"""Tunables for the SAM 3.1 text-prompted labeler."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Segmentation draws the mask itself; detection reduces each mask to its bounding
# box, which is what you would export to a YOLO detection dataset.
SEGMENTATION = "segmentation"
DETECTION = "detection"
MODES: tuple[str, ...] = (SEGMENTATION, DETECTION)

# Per-label effects, all implemented with plain OpenCV in core.py.
EFFECTS: tuple[str, ...] = (
    "fill",       # semi-transparent colour over the mask
    "outline",    # contour only, nothing filled
    "blur",       # blur the pixels inside the mask (anonymise a face, a screen)
    "pixelate",   # coarse mosaic inside the mask
    "spotlight",  # darken everything outside the mask
    "cutout",     # keep the mask, replace the background with a flat colour
    "hide",       # draw nothing at all, keep it in the list
)
DEFAULT_EFFECT = "fill"

# Distinct, readable defaults assigned to labels in order. BGR, as OpenCV wants.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 231, 76),   # cyan
    (200, 60, 255),   # magenta
    (120, 255, 140),  # lime
    (60, 190, 255),   # amber
    (80, 80, 255),    # red
    (245, 245, 245),  # white
    (255, 160, 80),   # sky
    (140, 255, 240),  # mint
)

# Bounds for the settings the UI exposes, shared so sliders and server clamping
# cannot disagree.
SETTING_BOUNDS: dict[str, tuple[float, float, float]] = {
    "maxDim": (256.0, 1280.0, 128.0),
    "confidence": (0.05, 0.95, 0.05),
    "maskThreshold": (0.1, 0.9, 0.05),
    "alpha": (0.1, 1.0, 0.05),
    "outlineThickness": (1, 10, 1),
    "blurStrength": (3, 99, 2),
    "pixelSize": (4, 60, 2),
}


@dataclass
class SamLabelerConfig:
    """Model, thresholds and drawing defaults.

    The Hugging Face token is deliberately **not** stored here: it lives in the
    session for the life of the request only (see ``web.py``), or comes from the
    ``HF_TOKEN`` environment variable. Never write it to disk or a log.
    """

    model_id: str = field(default_factory=lambda: os.environ.get("SAM_MODEL_ID", "facebook/sam3"))
    # Score below which a candidate mask is dropped entirely.
    confidence: float = 0.4
    # Probability at which a soft mask becomes a binary one.
    mask_threshold: float = 0.5
    # Fill opacity for the "fill" effect.
    alpha: float = 0.5
    outline_thickness: int = 3
    blur_strength: int = 31
    pixel_size: int = 16
    mode: str = SEGMENTATION
    # Uploads and captures are scaled to this longest edge before inference: SAM on
    # a CPU is slow enough that a 12 MP phone photo is not worth the wait.
    max_dimension: int = 1024
    max_upload_bytes: int = 16 * 1024 * 1024
    # Guard against someone pasting an essay as prompts.
    max_prompts: int = 8
    max_prompt_length: int = 60
    canvas_width: int = 1024
    canvas_height: int = 640

    def colour_for(self, index: int) -> tuple[int, int, int]:
        return PALETTE[index % len(PALETTE)]
