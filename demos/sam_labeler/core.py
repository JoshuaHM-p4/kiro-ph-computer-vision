"""Label state and the OpenCV rendering pipeline for the SAM labeler.

The model is only half the demo; the other half is what you *do* with a mask, and
that half is plain OpenCV:

    fill       addWeighted a colour over the masked pixels
    outline    findContours + drawContours
    blur       GaussianBlur the whole frame, then copy it back inside the mask
    pixelate   resize down, resize up nearest-neighbour, copy back inside the mask
    spotlight  darken everything outside the mask
    cutout     replace everything outside the mask with a flat colour

All of it operates on a BGR frame plus boolean masks, so it is testable with
synthetic masks and needs no model. Keeping the render pure is what lets the desktop
window and the web page produce identical images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from ..common import hud
from ..common.geometry import clamp
from .backend import Instance
from .config import (
    DEFAULT_EFFECT,
    DETECTION,
    EFFECTS,
    MODES,
    SEGMENTATION,
    SETTING_BOUNDS,
    SamLabelerConfig,
)


@dataclass
class LabelStyle:
    """How one label should be drawn."""

    label: str
    colour: tuple[int, int, int]
    effect: str = DEFAULT_EFFECT
    visible: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "colour": list(self.colour),
            # CSS wants RGB; OpenCV colours are BGR.
            "css": f"rgb({self.colour[2]},{self.colour[1]},{self.colour[0]})",
            "effect": self.effect,
            "visible": self.visible,
        }


def _binary(mask: np.ndarray) -> np.ndarray:
    """Mask as uint8 0/255, whatever it arrived as."""
    if mask.dtype == np.uint8 and mask.max() > 1:
        return mask
    return (mask.astype(bool).astype(np.uint8)) * 255


def apply_fill(frame: np.ndarray, mask: np.ndarray, colour, alpha: float) -> None:
    """Blend a flat colour over the masked pixels, in place."""
    region = mask.astype(bool)
    if not region.any():
        return
    overlay = np.empty_like(frame)
    overlay[:] = colour
    blended = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)
    frame[region] = blended[region]


def apply_outline(frame: np.ndarray, mask: np.ndarray, colour, thickness: int) -> None:
    contours, _ = cv2.findContours(_binary(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, colour, max(1, thickness), cv2.LINE_AA)


def apply_blur(frame: np.ndarray, mask: np.ndarray, strength: int) -> None:
    """Blur only inside the mask: the anonymise-a-face effect."""
    region = mask.astype(bool)
    if not region.any():
        return
    kernel = max(3, int(strength) | 1)  # GaussianBlur needs an odd kernel
    frame[region] = cv2.GaussianBlur(frame, (kernel, kernel), 0)[region]


def apply_pixelate(frame: np.ndarray, mask: np.ndarray, size: int) -> None:
    """Mosaic inside the mask, by downscaling and scaling back up."""
    region = mask.astype(bool)
    if not region.any():
        return
    height, width = frame.shape[:2]
    block = max(2, int(size))
    small = cv2.resize(
        frame, (max(1, width // block), max(1, height // block)), interpolation=cv2.INTER_AREA
    )
    frame[region] = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)[region]


def apply_spotlight(frame: np.ndarray, keep: np.ndarray, darkness: float = 0.72) -> None:
    """Darken everything outside ``keep``."""
    outside = ~keep.astype(bool)
    if not outside.any():
        return
    frame[outside] = (frame[outside].astype(np.float32) * (1.0 - darkness)).astype(np.uint8)


def apply_cutout(frame: np.ndarray, keep: np.ndarray, colour=(12, 10, 16)) -> None:
    """Replace everything outside ``keep`` with a flat colour."""
    frame[~keep.astype(bool)] = colour


class SamLabelerCore:
    """Holds the source image, the detected instances, and each label's style."""

    def __init__(self, config: SamLabelerConfig | None = None):
        self.config = config or SamLabelerConfig()
        self.source: np.ndarray | None = None
        self.source_name = ""
        self.prompts: list[str] = []
        self.instances: list[Instance] = []
        self.styles: dict[str, LabelStyle] = {}
        self.error: str | None = None
        self.last_backend = ""
        self.revision = 0

    # -- source ------------------------------------------------------------
    def set_source(self, image: np.ndarray, name: str = "capture") -> None:
        """Install a new image, scaled down to the configured longest edge."""
        self.source = self._fit(image)
        self.source_name = name
        self.instances = []
        self.error = None
        self.revision += 1

    def _fit(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)
        limit = self.config.max_dimension
        if longest <= limit:
            return image.copy()
        scale = limit / longest
        return cv2.resize(
            image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )

    def load_bytes(self, data: bytes, name: str = "upload") -> bool:
        if not data or len(data) > self.config.max_upload_bytes:
            self.error = "That file is empty or larger than the upload limit."
            return False
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.error = "Could not decode that file as an image."
            return False
        self.set_source(image, name)
        return True

    @property
    def has_source(self) -> bool:
        return self.source is not None

    # -- prompts -----------------------------------------------------------
    def parse_prompts(self, raw: str | Sequence[str]) -> list[str]:
        """Split a comma or newline separated prompt string into clean labels."""
        if isinstance(raw, str):
            parts: Iterable[str] = raw.replace("\n", ",").split(",")
        else:
            parts = raw
        seen: list[str] = []
        for part in parts:
            text = " ".join(str(part).split()).lower()[: self.config.max_prompt_length]
            if text and text not in seen:
                seen.append(text)
        return seen[: self.config.max_prompts]

    def set_prompts(self, raw: str | Sequence[str]) -> list[str]:
        self.prompts = self.parse_prompts(raw)
        # Give every new label a colour from the palette, keeping existing choices.
        for index, label in enumerate(self.prompts):
            if label not in self.styles:
                self.styles[label] = LabelStyle(
                    label=label, colour=self.config.colour_for(index), effect=DEFAULT_EFFECT
                )
        self.revision += 1
        return self.prompts

    # -- detection ---------------------------------------------------------
    def run(self, backend, prompts: str | Sequence[str] | None = None) -> dict[str, Any]:
        """Segment the current image with ``backend`` and keep the instances."""
        if prompts is not None:
            self.set_prompts(prompts)
        if self.source is None:
            self.error = "Load an image first."
            return self.state()
        if not self.prompts:
            self.error = "Enter at least one thing to look for."
            return self.state()

        self.last_backend = backend.describe()
        if not getattr(backend, "ready", False):
            self.error = backend.describe()
            self.instances = []
            return self.state()

        try:
            self.instances = backend.segment(self.source, self.prompts, self.config)
            self.error = None if self.instances else "Nothing matched those prompts."
        except Exception as failure:  # inference is the least predictable step here
            self.instances = []
            self.error = f"Segmentation failed: {str(failure).splitlines()[0][:180]}"
        self.revision += 1
        return self.state()

    # -- styles and settings ----------------------------------------------
    def style_for(self, label: str) -> LabelStyle:
        if label not in self.styles:
            self.styles[label] = LabelStyle(
                label=label, colour=self.config.colour_for(len(self.styles))
            )
        return self.styles[label]

    def set_style(
        self,
        label: str,
        *,
        colour: Sequence[int] | None = None,
        effect: str | None = None,
        visible: bool | None = None,
    ) -> bool:
        style = self.style_for(label)
        if colour is not None:
            try:
                b, g, r = (int(channel) for channel in colour)
            except (TypeError, ValueError):
                return False
            style.colour = (
                int(clamp(b, 0, 255)),
                int(clamp(g, 0, 255)),
                int(clamp(r, 0, 255)),
            )
        if effect is not None:
            if effect not in EFFECTS:
                return False
            style.effect = effect
        if visible is not None:
            style.visible = bool(visible)
        self.revision += 1
        return True

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        """Apply clamped settings; returns what actually took effect."""
        applied: dict[str, Any] = {}
        for key, raw in (values or {}).items():
            if key == "mode":
                if raw in MODES:
                    self.config.mode = raw
                    applied["mode"] = raw
                continue
            if key not in SETTING_BOUNDS:
                continue
            low, high, _ = SETTING_BOUNDS[key]
            try:
                value = clamp(float(raw), low, high)
            except (TypeError, ValueError):
                continue
            if key == "confidence":
                self.config.confidence = value
            elif key == "maskThreshold":
                self.config.mask_threshold = value
            elif key == "alpha":
                self.config.alpha = value
            elif key == "maxDim":
                self.config.max_dimension = int(value)
                self.revision += 1
            elif key == "outlineThickness":
                self.config.outline_thickness = int(value)
            elif key == "blurStrength":
                self.config.blur_strength = int(value) | 1
            elif key == "pixelSize":
                self.config.pixel_size = int(value)
            applied[key] = round(value, 3)
        self.revision += 1
        return applied

    def settings(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "maxDim": self.config.max_dimension,
            "confidence": round(self.config.confidence, 3),
            "maskThreshold": round(self.config.mask_threshold, 3),
            "alpha": round(self.config.alpha, 3),
            "outlineThickness": self.config.outline_thickness,
            "blurStrength": self.config.blur_strength,
            "pixelSize": self.config.pixel_size,
        }

    # -- rendering ---------------------------------------------------------
    def visible_instances(self) -> list[Instance]:
        return [
            instance
            for instance in self.instances
            if self.style_for(instance.label).visible
            and self.style_for(instance.label).effect != "hide"
        ]

    def render(self, *, labels: bool = True) -> np.ndarray:
        """Draw the current instances over the source image."""
        if self.source is None:
            return self.placeholder()

        frame = self.source.copy()
        instances = self.visible_instances()

        # Whole-image effects come first: they act on everything *outside* the union
        # of the masks that asked for them, so they cannot be done per instance.
        union = self._union(instance for instance in instances
                            if self.style_for(instance.label).effect == "spotlight")
        if union is not None:
            apply_spotlight(frame, union)
        union = self._union(instance for instance in instances
                            if self.style_for(instance.label).effect == "cutout")
        if union is not None:
            apply_cutout(frame, union)

        # Largest first, so a small object drawn inside a big one stays visible.
        for instance in sorted(instances, key=lambda item: -item.area):
            style = self.style_for(instance.label)
            mask = instance.mask
            if self.config.mode == DETECTION:
                mask = self._box_mask(instance)

            if style.effect == "fill":
                apply_fill(frame, mask, style.colour, self.config.alpha)
                apply_outline(frame, mask, style.colour, max(1, self.config.outline_thickness - 1))
            elif style.effect == "outline":
                apply_outline(frame, mask, style.colour, self.config.outline_thickness)
            elif style.effect == "blur":
                apply_blur(frame, mask, self.config.blur_strength)
                apply_outline(frame, mask, style.colour, 1)
            elif style.effect == "pixelate":
                apply_pixelate(frame, mask, self.config.pixel_size)
                apply_outline(frame, mask, style.colour, 1)
            else:  # spotlight and cutout already applied; just outline them
                apply_outline(frame, mask, style.colour, self.config.outline_thickness)

        if labels:
            for instance in sorted(instances, key=lambda item: -item.area):
                self._draw_chip(frame, instance)
        return frame

    def _union(self, instances: Iterable[Instance]) -> np.ndarray | None:
        combined: np.ndarray | None = None
        for instance in instances:
            mask = instance.mask.astype(bool)
            combined = mask if combined is None else (combined | mask)
        return combined

    def _box_mask(self, instance: Instance) -> np.ndarray:
        """Detection mode: replace the mask with its filled bounding box."""
        mask = np.zeros(instance.mask.shape, dtype=bool)
        x, y, w, h = instance.box
        if w and h:
            mask[y : y + h, x : x + w] = True
        return mask

    def _draw_chip(self, frame: np.ndarray, instance: Instance) -> None:
        """Label text on a filled chip, with contrast-aware text colour."""
        style = self.style_for(instance.label)
        x, y, _, _ = instance.box
        text = f"{instance.label} {instance.score:.0%}"
        scale = 0.5
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(0, y - th - base - 6)
        cv2.rectangle(frame, (x, top), (x + tw + 10, top + th + base + 6), style.colour, -1)

        # Dark text on light chips, light text on dark ones.
        b, g, r = style.colour
        luminance = 0.114 * b + 0.587 * g + 0.299 * r
        colour = (0, 0, 0) if luminance > 150 else (255, 255, 255)
        cv2.putText(
            frame,
            text,
            (x + 5, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            colour,
            1,
            cv2.LINE_AA,
        )

    def placeholder(self) -> np.ndarray:
        canvas = np.full(
            (self.config.canvas_height, self.config.canvas_width, 3), (18, 13, 24), dtype=np.uint8
        )
        hud.title(canvas, "SAM LABELER", "no image yet")
        hud.text(
            canvas,
            "Capture from the webcam or upload a picture, then say what to look for.",
            (40, self.config.canvas_height // 2),
            scale=0.62,
            color=hud.THEME.dim,
        )
        return canvas

    def render_canvas(self) -> np.ndarray:
        return self.render()

    def encode_png(self) -> bytes:
        ok, buffer = cv2.imencode(".png", self.render())
        return buffer.tobytes() if ok else b""

    # -- output ------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        height, width = (self.source.shape[:2] if self.source is not None else (0, 0))
        return {
            "hasSource": self.has_source,
            "sourceName": self.source_name,
            "sourceSize": {"width": width, "height": height},
            "prompts": list(self.prompts),
            "labels": [self.style_for(label).to_json() for label in self._label_order()],
            "instances": [
                instance.to_json(width, height)
                for instance in sorted(self.instances, key=lambda item: -item.area)
            ],
            "counts": self._counts(),
            "settings": self.settings(),
            "bounds": {key: list(value) for key, value in SETTING_BOUNDS.items()},
            "effects": list(EFFECTS),
            "modes": list(MODES),
            "backend": self.last_backend,
            "error": self.error,
            "revision": self.revision,
        }

    def _label_order(self) -> list[str]:
        """Prompts first, then any label the model returned that we did not ask for."""
        order = list(self.prompts)
        for instance in self.instances:
            if instance.label not in order:
                order.append(instance.label)
        return order

    def _counts(self) -> dict[str, int]:
        counts: dict[str, int] = {label: 0 for label in self.prompts}
        for instance in self.instances:
            counts[instance.label] = counts.get(instance.label, 0) + 1
        return counts

    def handle_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "style":
            ok = self.set_style(
                str(payload.get("label", "")),
                colour=payload.get("colour"),
                effect=payload.get("effect"),
                visible=payload.get("visible"),
            )
            if not ok:
                return {"ok": False, "error": "Unknown effect or bad colour."}
        elif command == "settings":
            return {"ok": True, "applied": self.update_settings(payload), **self.state()}
        elif command == "prompts":
            self.set_prompts(payload.get("prompts", ""))
        elif command == "clear":
            self.instances = []
            self.revision += 1
        elif command == "reset":
            self.reset()
        else:
            return {"ok": False, "unknown": command}
        return {"ok": True, **self.state()}

    def reset(self) -> None:
        self.source = None
        self.source_name = ""
        self.prompts = []
        self.instances = []
        self.styles = {}
        self.error = None
        self.revision += 1
