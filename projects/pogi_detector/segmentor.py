"""Segmentation backend for the Pogi Detector.

Follows the same pattern as demos/sam_labeler/backend.py:
- A Backend protocol with name/ready/segment/describe
- StubBackend for offline/demo use (deterministic synthetic masks)
- SamBackend for real SAM segmentation via transformers
- load_backend() that never raises — returns StubBackend on failure

The key integration point with the translator: prompts arrive as Tagalog slang,
get translated to English descriptions, then fed to the segmentation model.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

from translator import translate_or_passthrough

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    """One segmented object: a binary mask plus metadata."""

    label: str          # The original slang term used
    english: str        # The English translation used as the actual prompt
    score: float        # Confidence score (0..1)
    mask: np.ndarray    # bool, same height/width as the source image

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Tight bounding box (x, y, w, h) of the mask."""
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

    @property
    def area(self) -> int:
        """Number of True pixels in the mask."""
        return int(self.mask.sum())


@dataclass
class SegmentorConfig:
    """Configuration for the segmentation backend."""

    model_id: str = "facebook/sam-vit-base"
    confidence: float = 0.4
    mask_threshold: float = 0.5
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class Backend(Protocol):
    """Interface that all segmentation backends implement."""

    name: str
    ready: bool

    def segment(self, image: np.ndarray, prompts: Sequence[str]) -> list[Instance]:
        """Segment the image using the given text prompts.

        Args:
            image: BGR numpy array from the camera.
            prompts: Tagalog slang terms (will be translated internally).

        Returns:
            List of Instance objects with masks.
        """
        ...

    def describe(self) -> str:
        """Human-readable description of the backend state."""
        ...


# ---------------------------------------------------------------------------
# Stub backend (offline / demo mode)
# ---------------------------------------------------------------------------

class StubBackend:
    """Deterministic synthetic masks for offline use.

    Generates ellipse masks based on a hash of the prompt text, so the app
    can demonstrate the full pipeline without downloading SAM weights.
    Position, size, and score are stable across runs but vary per prompt.
    """

    name = "stub"
    ready = True

    def describe(self) -> str:
        return "Demo mode: synthetic masks (no SAM model loaded)"

    def segment(self, image: np.ndarray, prompts: Sequence[str]) -> list[Instance]:
        height, width = image.shape[:2]
        instances: list[Instance] = []

        for prompt in prompts:
            english = translate_or_passthrough(prompt)

            # Use hash of english translation for deterministic placement
            digest = hashlib.sha1(english.encode("utf-8")).digest()
            cx = 0.30 + 0.40 * (digest[0] / 255)
            cy = 0.25 + 0.45 * (digest[1] / 255)
            rx = 0.12 + 0.15 * (digest[2] / 255)
            ry = 0.15 + 0.20 * (digest[3] / 255)
            score = 0.60 + 0.39 * (digest[4] / 255)

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                mask,
                center=(int(cx * width), int(cy * height)),
                axes=(max(4, int(rx * width)), max(4, int(ry * height))),
                angle=float(digest[5]) / 255 * 180,
                startAngle=0,
                endAngle=360,
                color=255,
                thickness=-1,
            )

            instances.append(Instance(
                label=prompt,
                english=english,
                score=score,
                mask=mask.astype(bool),
            ))

        return instances


# ---------------------------------------------------------------------------
# SAM backend (real model)
# ---------------------------------------------------------------------------

class SamBackend:
    """SAM segmentation via transformers (text-prompted).

    Attempts to load the SAM model. If successful, uses text prompts
    (translated from Tagalog) to generate real segmentation masks.
    """

    name = "sam"

    def __init__(self, config: SegmentorConfig, token: str | None = None):
        import torch
        from transformers import SamModel, SamProcessor

        self._torch = torch
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else config.device

        kwargs: dict[str, Any] = {"token": token} if token else {}
        self.model = SamModel.from_pretrained(config.model_id, **kwargs).to(self.device)
        self.processor = SamProcessor.from_pretrained(config.model_id, **kwargs)
        self.ready = True

    def describe(self) -> str:
        return f"SAM: {self.config.model_id} on {self.device.upper()}"

    def segment(self, image: np.ndarray, prompts: Sequence[str]) -> list[Instance]:
        from PIL import Image

        torch = self._torch
        height, width = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        instances: list[Instance] = []

        for prompt in prompts:
            english = translate_or_passthrough(prompt)

            # SAM with text prompt
            inputs = self.processor(
                images=pil, text=[english], return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            if hasattr(self.processor, "post_process_instance_segmentation"):
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.config.confidence,
                    mask_threshold=self.config.mask_threshold,
                    target_sizes=[[height, width]],
                )
                if results and len(results) > 0 and "masks" in results[0] and len(results[0]["masks"]) > 0:
                    mask_raw = np.asarray(results[0]["masks"][0].cpu())
                    while mask_raw.ndim > 2:
                        mask_raw = mask_raw[0]
                    mask = mask_raw.astype(bool)

                    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
                        continue

                    scores = results[0].get("scores")
                    score = float(scores[0].cpu()) if scores is not None and len(scores) > 0 else 0.85
                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    instances.append(Instance(
                        label=prompt,
                        english=english,
                        score=score,
                        mask=mask,
                    ))
            else:
                orig_sizes = inputs.get("original_sizes", torch.tensor([[height, width]]))
                reshaped_sizes = inputs.get("reshaped_input_sizes", orig_sizes)
                masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    orig_sizes.cpu(),
                    reshaped_sizes.cpu(),
                )

                if masks and len(masks[0]) > 0:
                    scores = outputs.iou_scores.cpu().squeeze()
                    best_idx = scores.argmax().item() if scores.dim() > 0 else 0
                    mask_raw = masks[0][best_idx].numpy()

                    while mask_raw.ndim > 2:
                        mask_raw = mask_raw[0]
                    mask = mask_raw.astype(bool)

                    # Skip if mask ended up degenerate (0-d or 1-d)
                    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
                        continue

                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                    score = float(scores[best_idx]) if scores.dim() > 0 else float(scores)
                    instances.append(Instance(
                        label=prompt,
                        english=english,
                        score=score,
                        mask=mask,
                    ))

            # Cleanup
            del inputs, outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return instances


# ---------------------------------------------------------------------------
# Unavailable backend (explains why it failed)
# ---------------------------------------------------------------------------

class UnavailableBackend:
    """Placeholder when the real model cannot load — carries the reason."""

    name = "unavailable"
    ready = False

    def __init__(self, reason: str):
        self.reason = reason

    def segment(self, image: np.ndarray, prompts: Sequence[str]) -> list[Instance]:
        return []

    def describe(self) -> str:
        return self.reason


# ---------------------------------------------------------------------------
# Factory function (never raises)
# ---------------------------------------------------------------------------

def load_backend(
    config: SegmentorConfig | None = None,
    token: str | None = None,
    *,
    stub: bool = False,
) -> StubBackend | SamBackend | UnavailableBackend:
    """Load the best available backend. Never raises.

    Args:
        config: Segmentor configuration. Uses defaults if None.
        token: Hugging Face token for gated models.
        stub: Force stub mode regardless of model availability.

    Returns:
        A backend instance (Stub, Sam, or Unavailable with reason).
    """
    if config is None:
        config = SegmentorConfig()

    if stub:
        return StubBackend()

    try:
        from transformers import SamModel, SamProcessor  # noqa: F401
    except ImportError:
        logger.info("transformers not available or missing SamModel, using stub")
        return StubBackend()

    if not token:
        logger.info("No HF token provided, using stub backend")
        return StubBackend()

    try:
        return SamBackend(config, token)
    except Exception as error:
        reason = f"Failed to load SAM: {type(error).__name__}: {error}"
        logger.warning(reason)
        return StubBackend()  # Fall back to stub so the app still works
