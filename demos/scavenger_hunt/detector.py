"""Object detection backends for the scavenger hunt.

Ultralytics ships COCO-pretrained weights, so the demo needs no manual download:
naming ``yolo26n.pt`` fetches and caches it on first use. Weights themselves are
never committed (see AGENTS.md); ``models/`` is git-ignored.

Backends, picked automatically by :func:`load_detector`:

``UltralyticsDetector``
    The default. Any name or path ultralytics understands, including ones it
    downloads for you.

``OnnxDetector``
    A YOLO ``.onnx`` export run through ``cv2.dnn``, needing nothing beyond the
    OpenCV already in this project. Useful for a torch-free deployment, and
    YOLO26n was built for fast CPU ONNX inference.

``NullDetector``
    Nothing loadable. The game still runs and explains what went wrong, and the
    tests use a scripted stub instead of real weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from .config import COCO_CLASSES, MODEL_DIR, HuntConfig


@dataclass(frozen=True)
class Detection:
    """One detected object in normalized image coordinates."""

    label: str
    confidence: float
    # (x1, y1, x2, y2) as fractions of the image, so the browser can draw it on
    # any canvas size without knowing the inference resolution.
    box: tuple[float, float, float, float]

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "box": [round(value, 4) for value in self.box],
        }


class Detector(Protocol):
    """Anything that can turn a BGR frame into detections."""

    name: str
    ready: bool

    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...

    def describe(self) -> str:
        ...


class NullDetector:
    """Stands in when no model is configured."""

    name = "none"
    ready = False

    def __init__(self, reason: str = "No model configured."):
        self.reason = reason

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return []

    def describe(self) -> str:
        return self.reason


class ScriptedDetector:
    """Returns canned detections. Used by the tests, and by ``--demo-mode``."""

    name = "scripted"
    ready = True

    def __init__(self, detections: Sequence[Detection] | None = None):
        self.queue: list[list[Detection]] = []
        self.fixed = list(detections or [])

    def push(self, detections: Sequence[Detection]) -> None:
        self.queue.append(list(detections))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.queue:
            return self.queue.pop(0)
        return list(self.fixed)

    def describe(self) -> str:
        return "Scripted detector (no model)."


class OnnxDetector:
    """YOLO ONNX model through ``cv2.dnn``.

    Output layouts differ between YOLO generations, so the decoder sniffs the
    shape rather than assuming one:

    * ``(1, 4 + classes, anchors)`` - YOLOv8/v11 style, needs decoding and NMS.
    * ``(1, anchors, 4 + classes)`` - the same, transposed.
    * ``(1, n, 6)`` - already decoded ``[x1, y1, x2, y2, score, class]``, which is
      what end-to-end (NMS-free) exports such as YOLO26 produce.
    """

    name = "onnx"

    def __init__(self, path: Path, config: HuntConfig):
        self.path = Path(path)
        self.config = config
        self.net = cv2.dnn.readNetFromONNX(str(self.path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.ready = True

    def describe(self) -> str:
        return f"ONNX via cv2.dnn: {self.path.name}"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        size = self.config.input_size
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1 / 255.0, size=(size, size), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        raw = self.net.forward()
        output = np.squeeze(raw)
        if output.ndim != 2:
            return []
        # Put anchors on axis 0 so both (channels, anchors) and (anchors, channels)
        # layouts are handled: class counts are far smaller than anchor counts.
        if output.shape[0] < output.shape[1]:
            output = output.T
        if output.shape[1] == 6:
            return self._decode_end_to_end(output)
        return self._decode_yolo(output)

    def _decode_end_to_end(self, output: np.ndarray) -> list[Detection]:
        """Rows of [x1, y1, x2, y2, score, class]: already NMS-ed."""
        size = float(self.config.input_size)
        detections: list[Detection] = []
        for x1, y1, x2, y2, score, class_id in output:
            if score < self.config.confidence:
                continue
            label = self._label(int(class_id))
            if label is None:
                continue
            detections.append(
                Detection(
                    label=label,
                    confidence=float(score),
                    box=(
                        float(np.clip(x1 / size, 0, 1)),
                        float(np.clip(y1 / size, 0, 1)),
                        float(np.clip(x2 / size, 0, 1)),
                        float(np.clip(y2 / size, 0, 1)),
                    ),
                )
            )
        return detections

    def _decode_yolo(self, output: np.ndarray) -> list[Detection]:
        """Rows of [cx, cy, w, h, class scores...]: needs NMS."""
        if output.shape[1] <= 4:
            return []
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        keep = confidences >= self.config.confidence
        if not np.any(keep):
            return []

        boxes_xywh = output[keep, :4]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # cv2.dnn.NMSBoxes wants integer x, y, w, h in pixels.
        rects = []
        for cx, cy, w, h in boxes_xywh:
            rects.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
        indices = cv2.dnn.NMSBoxes(
            rects, confidences.astype(float).tolist(), self.config.confidence, self.config.nms_threshold
        )
        if len(indices) == 0:
            return []

        size = float(self.config.input_size)
        detections: list[Detection] = []
        for index in np.array(indices).flatten():
            x, y, w, h = rects[int(index)]
            label = self._label(int(class_ids[int(index)]))
            if label is None:
                continue
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidences[int(index)]),
                    box=(
                        float(np.clip(x / size, 0, 1)),
                        float(np.clip(y / size, 0, 1)),
                        float(np.clip((x + w) / size, 0, 1)),
                        float(np.clip((y + h) / size, 0, 1)),
                    ),
                )
            )
        return detections

    @staticmethod
    def _label(class_id: int) -> str | None:
        if 0 <= class_id < len(COCO_CLASSES):
            return COCO_CLASSES[class_id]
        return None


class UltralyticsDetector:
    """A checkpoint loaded through ultralytics, downloading it if necessary."""

    name = "ultralytics"

    def __init__(self, model: str, config: HuntConfig):
        # Imported lazily so the rest of the suite does not pay torch's import
        # cost, and so a missing install degrades instead of breaking collection.
        from ultralytics import YOLO

        self.config = config
        self.reference = str(model)
        self.model = YOLO(self._resolve(self.reference))
        self.labels = dict(getattr(self.model, "names", {}) or dict(enumerate(COCO_CLASSES)))
        self.ready = True

    @staticmethod
    def _resolve(model: str) -> str:
        """Prefer a cached copy in ``models/`` over re-downloading to the cwd.

        Ultralytics downloads a bare name into the current working directory, which
        would litter the repo root. Pointing at ``models/`` keeps weights in the one
        place .gitignore already excludes.
        """
        candidate = Path(model)
        if candidate.exists():
            return str(candidate)
        cached = MODEL_DIR / candidate.name
        if cached.exists():
            return str(cached)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return str(MODEL_DIR / candidate.name)

    def describe(self) -> str:
        return f"ultralytics: {Path(self.reference).name} ({len(self.labels)} classes)"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        results = self.model.predict(
            frame, conf=self.config.confidence, imgsz=self.config.input_size, verbose=False
        )
        names = self.labels
        detections: list[Detection] = []
        for result in results:
            for box in getattr(result, "boxes", []):
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
                detections.append(
                    Detection(
                        label=str(names.get(class_id, class_id)),
                        confidence=float(box.conf[0]),
                        box=(
                            max(0.0, x1 / width),
                            max(0.0, y1 / height),
                            min(1.0, x2 / width),
                            min(1.0, y2 / height),
                        ),
                    )
                )
        return detections


def load_detector(config: HuntConfig) -> Detector:
    """Pick a backend for the configured model.

    Never raises: anything unloadable degrades to :class:`NullDetector` so the page
    explains itself instead of returning a 500. A bare name such as ``yolo26n.pt``
    is handed to ultralytics, which downloads it on first use.
    """
    model = str(config.model or "").strip()
    if not model:
        return NullDetector("No model configured. Pass --model or set SCAVENGER_MODEL.")

    suffix = Path(model).suffix.lower()
    try:
        if suffix == ".onnx":
            path = Path(model)
            if not path.is_file():
                path = MODEL_DIR / path.name
            if not path.is_file():
                return NullDetector(f"ONNX model not found: {model}")
            return OnnxDetector(path, config)
        return UltralyticsDetector(model, config)
    except ImportError:
        return NullDetector(
            "ultralytics is not installed. Run 'pip install -r requirements.txt', "
            "or export your model to ONNX and pass the .onnx path instead."
        )
    except Exception as error:  # download failure, corrupt file, bad opset...
        return NullDetector(f"Could not load {model}: {str(error).splitlines()[0][:160]}")
