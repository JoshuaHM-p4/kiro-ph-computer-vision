"""Image lab desktop app: OpenCV trackbars over an image, with the code on demand.

Run it with::

    .venv/bin/python -m demos.image_lab.desktop
    .venv/bin/python -m demos.image_lab.desktop --image photo.jpg
    .venv/bin/python -m demos.image_lab.desktop --webcam --op canny

Trackbars are the native OpenCV way to explore parameters, so this is the desktop
counterpart to the web sliders: one operation at a time, every parameter on a
trackbar, and ``r`` prints the equivalent code for the current settings.

Keys
    q / ESC quit          n / p   next / previous operation
    r reveal the code     s       save the result next to the source
    c compare (side by side original and result)
    SPACE re-grab a webcam frame (with --webcam)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..common import hud
from . import operations as ops
from .config import ImageLabConfig
from .core import ImageLabCore

WINDOW = "image lab"
CONTROLS = "controls"


def _trackbar_range(param: ops.Param) -> tuple[int, int, int]:
    """Map a parameter onto integer trackbar units.

    Trackbars are integer-only and start at zero, so floats are scaled by 100 and
    negative minimums are shifted; :func:`_from_trackbar` undoes both.
    """
    if param.kind == "bool":
        return 0, 1, int(bool(param.default))
    if param.kind == "choice":
        return 0, max(0, len(param.choices) - 1), max(0, list(param.choices).index(param.default))
    if param.kind == "float":
        low = int(param.minimum * 100)
        high = int(param.maximum * 100)
        return 0, high - low, int(param.default * 100) - low
    return 0, int(param.maximum - param.minimum), int(param.default - param.minimum)


def _from_trackbar(param: ops.Param, raw: int):
    if param.kind == "bool":
        return bool(raw)
    if param.kind == "choice":
        return param.choices[min(raw, len(param.choices) - 1)]
    if param.kind == "float":
        return round(raw / 100 + param.minimum, 4)
    return int(raw + param.minimum)


class DesktopLab:
    """One operation at a time, its parameters bound to trackbars."""

    def __init__(self, core: ImageLabCore, start_key: str | None = None):
        self.core = core
        self.index = 0
        if start_key:
            keys = [operation.key for operation in ops.OPERATIONS]
            if start_key in keys:
                self.index = keys.index(start_key)
        self.compare = True
        self.result: np.ndarray | None = None

    @property
    def operation(self) -> ops.Operation:
        return ops.OPERATIONS[self.index]

    def select(self, delta: int) -> None:
        self.index = (self.index + delta) % len(ops.OPERATIONS)
        self.core.set_pipeline([{"key": self.operation.key}])
        self.build_controls()

    def build_controls(self) -> None:
        """Recreate the trackbar window for the current operation."""
        cv2.destroyWindow(CONTROLS)
        cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROLS, 460, 60 + 52 * max(1, len(self.operation.params)))
        for param in self.operation.params:
            low, high, initial = _trackbar_range(param)
            label = param.label[:38]
            cv2.createTrackbar(label, CONTROLS, initial, max(1, high), lambda _v: None)
            cv2.setTrackbarMin(label, CONTROLS, low)

    def read_params(self) -> dict:
        values = {}
        for param in self.operation.params:
            label = param.label[:38]
            try:
                raw = cv2.getTrackbarPos(label, CONTROLS)
            except cv2.error:
                raw = 0
            values[param.key] = _from_trackbar(param, raw)
        return self.operation.coerce(values)

    def step_params(self) -> dict:
        return self.core.steps[0].params if self.core.steps else {}

    def render(self) -> np.ndarray:
        params = self.read_params()
        if self.core.steps:
            self.core.steps[0].set_params(params)
        self.result = self.core.apply()
        return self.compose(self.result)

    def compose(self, result: np.ndarray) -> np.ndarray:
        """Result alone, or original and result side by side."""
        source = self.core.source
        if not self.compare or source is None:
            canvas = result.copy()
        else:
            height = max(source.shape[0], result.shape[0])
            panels = []
            for image, caption in ((source, "SOURCE"), (result, "RESULT")):
                scaled = image
                if image.shape[0] != height:
                    scale = height / image.shape[0]
                    scaled = cv2.resize(image, (int(image.shape[1] * scale), height))
                labelled = scaled.copy()
                hud.text(labelled, caption, (14, 26), scale=0.5, color=hud.THEME.cyan)
                panels.append(labelled)
            canvas = np.hstack(panels)

        hud.title(canvas, "IMAGE LAB", f"{self.operation.category} / {self.operation.label}")
        hud.status_strip(
            canvas,
            [
                ("OP", f"{self.index + 1}/{len(ops.OPERATIONS)}"),
                ("CV2", self.operation.docs),
                ("KEYS", "n/p op   r code   c compare   s save   q quit"),
            ],
        )
        if self.core.error:
            hud.text(canvas, self.core.error, (26, 116), scale=0.5, color=hud.THEME.red)
        return canvas

    def print_code(self) -> str:
        code = self.core.code()
        print("\n" + "=" * 70)
        print(f"# {self.operation.label} - parameters: {self.step_params()}")
        print("=" * 70)
        print(code)
        print("=" * 70 + "\n")
        return code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive OpenCV image lab")
    parser.add_argument("--image", type=Path, default=None, help="Image file to load")
    parser.add_argument("--webcam", action="store_true", help="Grab the source from the webcam")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for --webcam")
    parser.add_argument("--op", default=None, help="Operation to start on, e.g. canny")
    parser.add_argument("--list", action="store_true", help="List the available operations and exit")
    return parser.parse_args()


def _grab_frame(camera_index: int) -> np.ndarray | None:  # pragma: no cover - needs a camera
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f"Could not open camera {camera_index}.")
        return None
    for _ in range(5):  # discard the first frames while exposure settles
        ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def main() -> None:  # pragma: no cover - interactive
    args = parse_args()

    if args.list:
        for category in ops.CATEGORIES:
            members = [op for op in ops.OPERATIONS if op.category == category]
            print(f"{category}:")
            for operation in members:
                print(f"  {operation.key:22} cv2.{operation.docs}")
        return

    core = ImageLabCore(ImageLabConfig())
    if args.image is not None:
        if not core.load_path(args.image):
            print(core.error)
            return
    elif args.webcam:
        frame = _grab_frame(args.camera)
        if frame is None:
            return
        core.set_source(frame, "webcam.png")
    else:
        samples = core.samples()
        if not samples:
            print("No samples found. Run: .venv/bin/python -m demos.tools.make_sample_images")
            return
        core.load_path(samples[0])
        print(f"Loaded sample {samples[0].name}. Use --image to load your own.")

    lab = DesktopLab(core, args.op)
    core.set_pipeline([{"key": lab.operation.key}])
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    lab.build_controls()
    print(__doc__)

    while True:
        cv2.imshow(WINDOW, lab.render())
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            lab.select(1)
        elif key == ord("p"):
            lab.select(-1)
        elif key == ord("r"):
            lab.print_code()
        elif key == ord("c"):
            lab.compare = not lab.compare
        elif key == ord("s") and lab.result is not None:
            out = Path("demos/screenshots")
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"image-lab-{lab.operation.key}.png"
            cv2.imwrite(str(path), lab.result)
            print(f"Saved {path}")
        elif key == ord(" ") and args.webcam:
            frame = _grab_frame(args.camera)
            if frame is not None:
                core.set_source(frame, "webcam.png")

    cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
    main()
