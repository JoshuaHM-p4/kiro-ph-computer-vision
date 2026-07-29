"""Interactive OpenCV playground.

The core holds a source image and an ordered **pipeline** of operations. Applying
the pipeline produces the preview; asking it for code produces a runnable script
that performs exactly the same steps with the same parameters. Those two come from
one place — :mod:`demos.image_lab.operations` — so the code shown can never drift
from the image displayed, which is the entire point of the "reveal code" button.

Pure module: no camera, no window, no Flask. Images arrive as numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..common import hud
from . import operations as ops
from .config import ImageLabConfig


@dataclass
class Step:
    """One operation in the pipeline, with its parameter values."""

    operation: ops.Operation
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def create(cls, key: str, params: dict[str, Any] | None = None, enabled: bool = True) -> "Step | None":
        operation = ops.get(key)
        if operation is None:
            return None
        return cls(operation=operation, params=operation.coerce(params), enabled=enabled)

    def set_params(self, params: dict[str, Any]) -> None:
        merged = {**self.params, **(params or {})}
        self.params = self.operation.coerce(merged)

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.operation.key,
            "label": self.operation.label,
            "category": self.operation.category,
            "summary": self.operation.summary,
            "docs": self.operation.docs,
            "enabled": self.enabled,
            "params": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.params.items()
            },
            "schema": [param.to_json() for param in self.operation.params],
        }


class ImageLabCore:
    """Source image plus a pipeline of OpenCV operations."""

    def __init__(self, config: ImageLabConfig | None = None):
        self.config = config or ImageLabConfig()
        self.source: np.ndarray | None = None
        self.source_name: str = ""
        self.steps: list[Step] = []
        self.error: str | None = None
        self.revision = 0
        self.reset_pipeline()

    # -- source image ------------------------------------------------------
    def load_bytes(self, data: bytes, name: str = "upload") -> bool:
        """Decode an uploaded file. Returns False if it is not a usable image."""
        if not data or len(data) > self.config.max_upload_bytes:
            self.error = "That file is empty or larger than the upload limit."
            return False
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            self.error = "Could not decode that file as an image."
            return False
        self.set_source(image, name)
        return True

    def load_path(self, path: Path) -> bool:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self.error = f"Could not read {path.name}."
            return False
        self.set_source(image, path.name)
        return True

    def set_source(self, image: np.ndarray, name: str = "frame") -> None:
        """Install a new source image, downscaled to the configured limit."""
        self.source = self._fit(ops.to_bgr(image))
        self.source_name = name
        self.error = None
        self.revision += 1

    def _fit(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)
        limit = self.config.max_dimension
        if longest <= limit:
            return image
        scale = limit / longest
        return cv2.resize(
            image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
        )

    @property
    def has_source(self) -> bool:
        return self.source is not None

    def samples(self) -> list[Path]:
        directory = self.config.samples_dir
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

    def load_sample(self, name: str) -> bool:
        for path in self.samples():
            if path.name == name:
                return self.load_path(path)
        self.error = f"No sample named {name}."
        return False

    # -- pipeline ----------------------------------------------------------
    def reset_pipeline(self) -> None:
        self.steps = []
        for key in self.config.default_pipeline:
            self.add_step(key)
        self.revision += 1

    def add_step(self, key: str, params: dict[str, Any] | None = None) -> Step | None:
        if len(self.steps) >= self.config.max_steps:
            self.error = f"Pipelines are capped at {self.config.max_steps} steps."
            return None
        step = Step.create(key, params)
        if step is None:
            self.error = f"Unknown operation: {key}"
            return None
        self.steps.append(step)
        self.revision += 1
        return step

    def remove_step(self, index: int) -> bool:
        if 0 <= index < len(self.steps):
            del self.steps[index]
            self.revision += 1
            return True
        return False

    def move_step(self, index: int, delta: int) -> bool:
        target = index + delta
        if 0 <= index < len(self.steps) and 0 <= target < len(self.steps):
            self.steps[index], self.steps[target] = self.steps[target], self.steps[index]
            self.revision += 1
            return True
        return False

    def update_step(self, index: int, params: dict[str, Any] | None = None, enabled: bool | None = None) -> bool:
        if not (0 <= index < len(self.steps)):
            return False
        if params:
            self.steps[index].set_params(params)
        if enabled is not None:
            self.steps[index].enabled = bool(enabled)
        self.revision += 1
        return True

    def set_pipeline(self, entries: list[dict[str, Any]]) -> None:
        """Replace the whole pipeline, e.g. from a browser payload."""
        self.steps = []
        for entry in entries[: self.config.max_steps]:
            step = Step.create(
                str(entry.get("key", "")),
                entry.get("params"),
                bool(entry.get("enabled", True)),
            )
            if step is not None:
                self.steps.append(step)
        self.revision += 1

    @property
    def active_steps(self) -> list[Step]:
        return [step for step in self.steps if step.enabled]

    # -- running -----------------------------------------------------------
    def apply(self, image: np.ndarray | None = None) -> np.ndarray:
        """Run the enabled steps in order and return the result.

        A failing step is skipped rather than aborting the whole preview: a slider
        can easily land on a combination OpenCV rejects, and losing the image on
        every such frame would make the demo feel broken.
        """
        base = image if image is not None else self.source
        if base is None:
            return self.placeholder()

        result = base.copy()
        self.error = None
        for step in self.active_steps:
            try:
                candidate = step.operation.run(result, step.params)
            except cv2.error as exc:  # invalid parameter combination
                self.error = f"{step.operation.label}: {str(exc).splitlines()[-1][:120]}"
                continue
            if candidate is None or candidate.size == 0:
                continue
            result = ops.to_bgr(candidate)
        return result

    def placeholder(self) -> np.ndarray:
        """Shown until an image is loaded."""
        canvas = np.full(
            (self.config.canvas_height, self.config.canvas_width, 3), (18, 13, 24), dtype=np.uint8
        )
        hud.title(canvas, "IMAGE LAB", "no image loaded")
        hud.text(
            canvas,
            "Upload an image, pick a sample, or grab a webcam frame.",
            (40, self.config.canvas_height // 2),
            scale=0.7,
            color=hud.THEME.dim,
        )
        return canvas

    def render_canvas(self) -> np.ndarray:
        """Result image, for /snapshot and screenshots."""
        return self.apply()

    # -- code generation ---------------------------------------------------
    def code(self, *, source_name: str | None = None) -> str:
        """A runnable script reproducing the current pipeline.

        Generated from the same parameter values the preview used, so what you read
        is what you saw.
        """
        name = source_name or self.source_name or "input.jpg"
        lines: list[str] = [
            "import cv2",
            "import numpy as np",
            "",
            f'img = cv2.imread("{name}")',
        ]
        if not self.active_steps:
            lines += ["", "# No operations enabled yet."]
        for index, step in enumerate(self.active_steps, start=1):
            lines.append("")
            lines.append(f"# {index}. {step.operation.label} - {step.operation.summary}")
            lines.extend(step.operation.code(step.params))
        lines += [
            "",
            'cv2.imwrite("output.png", img)',
            'cv2.imshow("result", img)',
            "cv2.waitKey(0)",
        ]
        return "\n".join(lines)

    # -- web interface -----------------------------------------------------
    def handle_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "add":
            added = self.add_step(str(payload.get("key", "")), payload.get("params"))
            if added is None:
                return {"ok": False, "error": self.error}
        elif command == "remove":
            self.remove_step(int(payload.get("index", -1)))
        elif command == "move":
            self.move_step(int(payload.get("index", -1)), int(payload.get("delta", 0)))
        elif command == "update":
            self.update_step(
                int(payload.get("index", -1)), payload.get("params"), payload.get("enabled")
            )
        elif command == "pipeline":
            self.set_pipeline(payload.get("steps") or [])
        elif command == "reset":
            self.reset_pipeline()
        elif command == "clear":
            self.steps = []
            self.revision += 1
        elif command == "sample":
            if not self.load_sample(str(payload.get("name", ""))):
                return {"ok": False, "error": self.error}
        else:
            return {"ok": False, "unknown": command}
        return {"ok": True, **self.state()}

    def reset(self) -> None:
        self.source = None
        self.source_name = ""
        self.error = None
        self.reset_pipeline()

    def state(self) -> dict[str, Any]:
        height, width = (self.source.shape[:2] if self.source is not None else (0, 0))
        return {
            "hasSource": self.has_source,
            "sourceName": self.source_name,
            "sourceSize": {"width": width, "height": height},
            "steps": [step.to_json() for step in self.steps],
            "maxSteps": self.config.max_steps,
            "samples": [path.name for path in self.samples()],
            "code": self.code(),
            "error": self.error,
            "revision": self.revision,
        }

    def encode_png(self, image: np.ndarray | None = None) -> bytes:
        """PNG bytes of the result, for serving over HTTP."""
        target = self.apply() if image is None else image
        ok, buffer = cv2.imencode(".png", target)
        return buffer.tobytes() if ok else b""
