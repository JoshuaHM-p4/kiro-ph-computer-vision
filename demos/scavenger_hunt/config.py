"""Tunables, COCO class names and the item pool for the scavenger hunt."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The 80 COCO classes, in the exact order YOLO models output them. The index is
# the class id, so this list doubles as the id -> label mapping.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Things a workshop participant can plausibly find at a desk in 30 seconds. Kept
# to COCO classes a small model detects reliably at webcam distance.
DESK_ITEMS: tuple[str, ...] = (
    "cell phone", "cup", "bottle", "book", "scissors", "keyboard", "mouse",
    "remote", "laptop", "chair", "potted plant", "banana", "apple", "orange",
    "clock", "vase", "teddy bear", "toothbrush", "bowl", "backpack", "tie",
    "wine glass", "fork", "knife", "spoon",
)

# A friendlier hint for the less obvious ones.
ITEM_HINTS: dict[str, str] = {
    "remote": "a TV remote works, so does a game controller",
    "mouse": "a computer mouse, not the animal",
    "tie": "a necktie",
    "vase": "any vase or tall jar",
    "potted plant": "any houseplant in a pot",
    "sports ball": "any ball will do",
    "teddy bear": "any plush toy usually reads as one",
    "toothbrush": "hold it close to the camera",
    "wine glass": "any stemmed glass",
    "book": "hold the cover to the camera",
    "keyboard": "point the camera at your keyboard",
}


# Ultralytics fetches COCO-pretrained weights on first use, so the demo needs no
# manual download. They land in MODEL_DIR, which is git-ignored: weights are never
# committed (see AGENTS.md).
DEFAULT_MODEL = "yolo26n.pt"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


# Bounds for the in-game settings panel. Shared so the UI sliders and the server
# side clamping cannot disagree.
SETTING_BOUNDS: dict[str, tuple[float, float, float]] = {
    # key: (minimum, maximum, step)
    "roundSeconds": (5.0, 180.0, 5.0),
    "confidence": (0.05, 0.95, 0.05),
    "rounds": (1, 20, 1),
    "holdSeconds": (0.0, 3.0, 0.1),
}


@dataclass
class HuntConfig:
    """Game rules and detector settings.

    ``model`` is whatever ultralytics accepts: a bare name like ``yolo26n.pt``
    (downloaded on first use and cached), a path to your own checkpoint, or a
    ``.onnx`` export, which runs on OpenCV's DNN backend instead of torch.
    Override with ``--model`` or the ``SCAVENGER_MODEL`` environment variable.
    """

    model: str = field(
        default_factory=lambda: os.environ.get("SCAVENGER_MODEL", DEFAULT_MODEL)
    )
    # Square input the network expects; YOLO detect models are trained at 640.
    input_size: int = 640
    confidence: float = 0.35
    nms_threshold: float = 0.45

    # Game rules.
    round_seconds: float = 30.0
    rounds: int = 5
    # A target must be seen for this long before it counts, so one flukey frame
    # cannot win a round.
    hold_seconds: float = 0.4
    # Pause on the "found it" card before the next target appears.
    celebrate_seconds: float = 2.0
    # Score: a base award plus a bonus for the time you had left.
    base_points: int = 100
    speed_bonus: int = 100
    streak_bonus: int = 25
    item_pool: tuple[str, ...] = DESK_ITEMS
    # Seed makes a run reproducible; None picks fresh targets each game.
    seed: int | None = None

    canvas_width: int = 1280
    canvas_height: int = 720

    def hint_for(self, item: str) -> str:
        return ITEM_HINTS.get(item, "")
