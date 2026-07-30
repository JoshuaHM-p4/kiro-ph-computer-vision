"""Segmentation backends for the labeler.

``Sam3Backend``
    The real thing: ``facebook/sam3`` through transformers, prompted with text.
    Weights are gated on Hugging Face, so it needs a read token, and transformers
    must be new enough to expose ``Sam3Processor`` / ``Sam3Model``.

``StubBackend``
    Synthetic ellipse masks derived from the prompt text. No model, no token, no
    download. It exists so the effects pipeline, the UI and the tests all work
    offline — and so a workshop participant can see the demo before deciding
    whether to pull a multi-gigabyte model.

``UnavailableBackend``
    Nothing loadable. Carries the reason so the page can explain itself.

Never log the token. It arrives per request, is used to authenticate the model
download, and is not written anywhere.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

from .config import SamLabelerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Instance:
    """One detected object: a binary mask plus the prompt that found it."""

    label: str
    score: float
    mask: np.ndarray  # bool, same height/width as the source image

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Tight bounding box (x, y, w, h) of the mask, or zeros if empty."""
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def to_json(self, width: int, height: int) -> dict[str, Any]:
        x, y, w, h = self.box
        return {
            "label": self.label,
            "score": round(float(self.score), 3),
            "area": self.area,
            # Normalized so the browser can draw on any canvas size.
            "box": [
                round(x / max(width, 1), 4),
                round(y / max(height, 1), 4),
                round(w / max(width, 1), 4),
                round(h / max(height, 1), 4),
            ],
        }


class Backend(Protocol):
    name: str
    ready: bool

    def segment(
        self, image: np.ndarray, prompts: Sequence[str], config: SamLabelerConfig
    ) -> list[Instance]:
        ...

    def describe(self) -> str:
        ...


class UnavailableBackend:
    """Placeholder that explains why segmentation cannot run."""

    name = "unavailable"
    ready = False

    def __init__(self, reason: str):
        self.reason = reason

    def segment(self, image, prompts, config) -> list[Instance]:
        return []

    def describe(self) -> str:
        return self.reason


class StubBackend:
    """Deterministic synthetic masks, so everything downstream can be exercised.

    Each prompt gets an ellipse whose position, size and score are derived from a
    hash of the prompt text: stable across runs, different per prompt, and entirely
    offline. Obviously not segmentation — it is a stand-in for the shape of the
    data.
    """

    name = "stub"
    ready = True

    def describe(self) -> str:
        return "Demo mode: synthetic masks, no model loaded."

    def segment(
        self, image: np.ndarray, prompts: Sequence[str], config: SamLabelerConfig
    ) -> list[Instance]:
        height, width = image.shape[:2]
        instances: list[Instance] = []
        for index, prompt in enumerate(prompts):
            digest = hashlib.sha1(prompt.encode("utf-8")).digest()
            cx = 0.25 + 0.5 * (digest[0] / 255)
            cy = 0.30 + 0.4 * (digest[1] / 255)
            rx = 0.10 + 0.12 * (digest[2] / 255)
            ry = 0.10 + 0.14 * (digest[3] / 255)
            score = 0.55 + 0.44 * (digest[4] / 255)

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                mask,
                (int(cx * width), int(cy * height)),
                (max(4, int(rx * width)), max(4, int(ry * height))),
                float(digest[5]) / 255 * 180,
                0,
                360,
                255,
                -1,
            )
            if score < config.confidence:
                continue
            instances.append(Instance(label=prompt, score=score, mask=mask.astype(bool)))
        return instances


class Sam3Backend:
    """SAM 3.1 (``facebook/sam3``) prompted with text, via transformers.

    Mirrors the batched call shape used in the reference pipeline: the same image
    repeated once per prompt, then ``post_process_instance_segmentation`` with the
    original image size so masks come back at full resolution.
    """

    name = "sam3"

    def __init__(self, config: SamLabelerConfig, token: str | None = None):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._torch = torch
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # token= is passed straight through to the hub call rather than being stored
        # or exported into the environment.
        kwargs: dict[str, Any] = {"token": token} if token else {}
        self.model = Sam3Model.from_pretrained(config.model_id, **kwargs).to(self.device)
        self.processor = Sam3Processor.from_pretrained(config.model_id, **kwargs)
        self.ready = True

    def describe(self) -> str:
        return f"SAM 3.1: {self.config.model_id} on {self.device.upper()}"

    def segment(
        self, image: np.ndarray, prompts: Sequence[str], config: SamLabelerConfig
    ) -> list[Instance]:
        from PIL import Image

        torch = self._torch
        height, width = image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        prompts = list(prompts)
        if not prompts:
            return []

        inputs = self.processor(
            images=[pil] * len(prompts), text=prompts, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=config.confidence,
            mask_threshold=config.mask_threshold,
            target_sizes=[[height, width]] * len(prompts),
        )

        instances: list[Instance] = []
        for prompt, result in zip(prompts, results):
            masks = result.get("masks")
            if masks is None or len(masks) == 0:
                continue
            scores = result["scores"].cpu().tolist()
            for mask_tensor, score in zip(masks.cpu(), scores):
                mask = np.asarray(mask_tensor).astype(bool)
                if mask.shape != (height, width):
                    mask = (
                        cv2.resize(
                            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    )
                if not mask.any():
                    continue
                instances.append(Instance(label=prompt, score=float(score), mask=mask))

        # Free the graph promptly; SAM masks at full resolution are large.
        del inputs, outputs, results
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return instances


def explain_load_failure(error: Exception, model_id: str) -> str:
    """Turn a hub or torch exception into something a human can act on.

    Hugging Face errors bury the useful sentence in a long multi-line body, and the
    first line is often just the HTTP status. This keeps the whole message but leads
    with a hint for the failures people actually hit.
    """
    text = " ".join(str(error).split())
    name = type(error).__name__
    lowered = text.lower()

    hint = ""
    if "401" in text or "unauthorized" in lowered or "invalid credentials" in lowered:
        hint = (
            "The token was rejected. Check it is a valid *read* token and has not "
            "been revoked."
        )
    elif "403" in text or "restricted" in lowered or "gated" in lowered or "awaiting" in lowered:
        hint = (
            f"Your account does not have access to {model_id} yet. Open its model page "
            "on Hugging Face, accept the licence, and wait for approval."
        )
    elif "404" in text or "not found" in lowered or "repository not found" in lowered:
        hint = f"No such model: {model_id}. Check the id, or pass --model."
    elif isinstance(error, (ConnectionError, TimeoutError)) or "connection" in lowered or "resolve" in lowered:
        hint = "Could not reach huggingface.co. Check your network or proxy."
    elif "out of memory" in lowered or "cannot allocate" in lowered:
        hint = "Ran out of memory loading the weights. Close other programs, or use a smaller model."
    elif "sam3" in lowered and ("attribute" in lowered or "config" in lowered):
        hint = (
            "This transformers build cannot read the model config. Install a newer one: "
            "pip install 'git+https://github.com/huggingface/transformers.git'"
        )

    # Log the full traceback for the terminal, where there is room for it. The token
    # is never part of the exception text we construct, and is not logged here.
    logger.exception("Loading %s failed", model_id)

    detail = text[:600] if text else name
    return f"{name}: {detail}" + (f"\n\nWhat to try: {hint}" if hint else "")


def load_backend(
    config: SamLabelerConfig, token: str | None = None, *, stub: bool = False
) -> Backend:
    """Pick a backend. Never raises: failures become :class:`UnavailableBackend`."""
    if stub:
        return StubBackend()
    try:
        from transformers import Sam3Model, Sam3Processor  # noqa: F401
    except ImportError:
        return UnavailableBackend(
            "transformers does not expose Sam3Model yet. Install a build that has it:\n"
            "  pip install 'git+https://github.com/huggingface/transformers.git'\n"
            "Or switch on demo mode to explore the effects with synthetic masks."
        )
    if not token:
        return UnavailableBackend(
            f"{config.model_id} is gated on Hugging Face. Paste a read token, or set "
            "HF_TOKEN before starting the server."
        )
    try:
        return Sam3Backend(config, token)
    except Exception as error:  # gated repo, no network, out of memory...
        return UnavailableBackend(explain_load_failure(error, config.model_id))
