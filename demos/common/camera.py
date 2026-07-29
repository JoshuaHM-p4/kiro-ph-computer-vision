"""Camera loop shared by the desktop demos.

Owns the boring parts: opening the capture device, mirroring, FPS, screenshots,
the common key bindings, and guaranteed teardown. A demo supplies a
``render(frame, landmark_frame, loop) -> frame`` callback and stays focused on
its own logic.

Key bindings match the existing app in ``src/app.py``:

    q / ESC   quit
    d         toggle debug overlay
    SPACE     toggle mirroring
    s         save a screenshot

Extra per-demo keys go through ``on_key``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import hud
from . import landmarks as lm
from .detectors import DetectorConfig, VisionPipeline
from .geometry import FPSCounter

RenderFn = Callable[[np.ndarray, lm.LandmarkFrame, "CameraLoop"], np.ndarray]
KeyFn = Callable[[int, "CameraLoop"], bool]


@dataclass
class CameraConfig:
    """Capture and window settings."""

    camera_index: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True
    debug: bool = True
    window_name: str = "demo"
    screenshot_dir: Path = field(default_factory=lambda: Path("demos/screenshots"))
    # Mirroring flips which physical hand MediaPipe calls "Left"; when we mirror
    # for display we also mirror before detection, which is the selfie
    # orientation MediaPipe assumes, so labels need no swap in that case.
    swap_handedness: bool = False


class CameraLoop:
    """Drives capture -> detect -> render -> show."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        hands: bool = False,
        face: bool = False,
        pose: bool = False,
        detector_config: DetectorConfig | None = None,
    ):
        self.config = config
        self.fps = FPSCounter()
        self.frame_index = 0
        self.running = False
        self.debug = config.debug
        self.mirror = config.mirror
        self.last_screenshot: Path | None = None
        self.started_at = time.monotonic()

        detector_config = detector_config or DetectorConfig()
        detector_config.swap_handedness = config.swap_handedness
        self.pipeline = VisionPipeline(
            hands=hands, face=face, pose=pose, config=detector_config
        )
        self.capture: cv2.VideoCapture | None = None

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> None:
        capture = cv2.VideoCapture(self.config.camera_index)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.config.camera_index}. "
                "Try a different --camera index."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.capture = capture

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.pipeline.close()
        cv2.destroyAllWindows()

    # -- helpers -----------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def save_screenshot(self, frame: np.ndarray, prefix: str | None = None) -> Path:
        directory = self.config.screenshot_dir
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{prefix or self.config.window_name}-{int(time.time())}.png"
        path = directory / name
        cv2.imwrite(str(path), frame)
        self.last_screenshot = path
        return path

    def stop(self) -> None:
        self.running = False

    # -- main loop ---------------------------------------------------------
    def run(self, render: RenderFn, *, on_key: KeyFn | None = None) -> None:
        """Block until the user quits.

        ``render`` returns the frame to display, which may be a different array
        than the one passed in (the PNGTuber replaces it entirely).
        """
        if self.capture is None:
            self.open()
        assert self.capture is not None

        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.window_name, self.config.width, self.config.height)
        self.running = True

        try:
            while self.running:
                ok, frame = self.capture.read()
                if not ok:
                    # A dropped frame is normal on some webcams; back off briefly.
                    time.sleep(0.01)
                    continue

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                self.frame_index += 1
                now = time.monotonic()
                landmark_frame = self.pipeline.detect(frame, timestamp=now)
                self.fps.update(now)

                output = render(frame, landmark_frame, self)
                if output is None:
                    output = frame

                if self.debug:
                    hud.status_strip(
                        output,
                        [
                            ("FPS", f"{self.fps.fps:5.1f}"),
                            ("MIRROR", "ON" if self.mirror else "OFF"),
                            ("KEYS", "q quit  d debug  space mirror  s save"),
                        ],
                    )

                cv2.imshow(self.config.window_name, output)

                key = cv2.waitKey(1) & 0xFF
                if key == 255:
                    continue
                if on_key is not None and on_key(key, self):
                    continue
                self._handle_key(key, output)
        finally:
            self.close()

    def _handle_key(self, key: int, frame: np.ndarray) -> None:
        if key in (ord("q"), 27):
            self.running = False
        elif key == ord("d"):
            self.debug = not self.debug
        elif key == ord(" "):
            self.mirror = not self.mirror
        elif key == ord("s"):
            self.save_screenshot(frame)


def add_camera_arguments(parser) -> None:
    """Register the CLI flags every desktop demo shares."""
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the webcam")
    parser.add_argument("--no-debug", action="store_true", help="Hide the debug overlay at start")
    parser.add_argument(
        "--swap-handedness",
        action="store_true",
        help="Swap Left/Right hand labels if they feel inverted",
    )


def camera_config_from_args(args, window_name: str) -> CameraConfig:
    return CameraConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
        debug=not args.no_debug,
        window_name=window_name,
        swap_handedness=args.swap_handedness,
    )


def _preview() -> None:  # pragma: no cover - interactive
    """``python -m demos.common.camera --preview`` sanity check."""
    import argparse

    parser = argparse.ArgumentParser(description="Preview all three MediaPipe streams")
    add_camera_arguments(parser)
    parser.add_argument("--preview", action="store_true", help="Run the preview")
    parser.add_argument("--dense-face", action="store_true", help="Draw every face point")
    args = parser.parse_args()

    config = camera_config_from_args(args, "mediapipe preview")
    loop = CameraLoop(config, hands=True, face=True, pose=True)

    def render(frame, landmark_frame, loop_ref):
        hud.vignette(frame, 0.25)
        hud.draw_frame_landmarks(frame, landmark_frame)
        if args.dense_face and landmark_frame.face is not None:
            hud.draw_face(frame, landmark_frame.face, dense=True)
        hud.title(frame, "VISION PREVIEW", "hands / face mesh / pose")
        counts = (
            f"hands={len(landmark_frame.hands)} "
            f"face={'yes' if landmark_frame.face else 'no'} "
            f"pose={'yes' if landmark_frame.pose else 'no'}"
        )
        hud.text(frame, counts, (26, 86), scale=0.5, color=hud.THEME.lime)
        return frame

    loop.run(render)


if __name__ == "__main__":  # pragma: no cover
    _preview()
