"""Scavenger hunt desktop app.

Run it with::

    .venv/bin/python -m demos.scavenger_hunt.desktop --model ~/models/yolo26n.onnx
    SCAVENGER_MODEL=~/models/yolo26n.onnx .venv/bin/python -m demos.scavenger_hunt.desktop

"Bring me a cell phone" appears with a 30 second timer; hold the item up to the
webcam. Detections come from a COCO-pretrained YOLO model you supply, either an
ONNX export (runs on OpenCV's DNN backend, no extra packages) or a ``.pt`` if you
have ultralytics installed.

Keys
    q / ESC quit    SPACE start / restart    n skip the current item
    d debug         s screenshot             m model info
    + / -           round timer by 5s        [ / ] confidence by 5%

Timer changes take effect from the next round; confidence applies immediately.
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from ..common import hud
from ..common.camera import CameraLoop, add_camera_arguments, camera_config_from_args
from .config import HuntConfig
from .core import IDLE, ScavengerHuntCore, draw_overlay
from .detector import Detector, load_detector


def make_renderer(core: ScavengerHuntCore, detector: Detector, *, every: int = 3) -> Callable:
    """Build the ``render(frame, landmark_frame, loop)`` callback.

    Detection runs on every ``every``-th frame and the last result is reused in
    between: a small YOLO on a CPU cannot keep up with 30 fps, and re-drawing the
    previous boxes looks far better than a stuttering preview.
    """
    cached: list = []

    def render(frame: np.ndarray, _landmarks, loop) -> np.ndarray:
        nonlocal cached
        if detector.ready and loop.frame_index % every == 0:
            cached = detector.detect(frame)
        state = core.update(cached, loop.elapsed)

        draw_overlay(frame, core, state, debug=loop.debug)
        hud.title(frame, "SCAVENGER HUNT", detector.describe())

        if not detector.ready:
            hud.text(
                frame,
                "NO MODEL LOADED - nothing can be detected",
                (26, frame.shape[0] // 2),
                scale=0.7,
                color=hud.THEME.amber,
                thickness=2,
            )
            hud.text(
                frame,
                "pass --model path/to/yolo26n.onnx",
                (26, frame.shape[0] // 2 + 32),
                scale=0.5,
                color=hud.THEME.dim,
            )
        elif core.phase == IDLE:
            hud.text(
                frame,
                "PRESS SPACE TO START",
                (26, frame.shape[0] // 2),
                scale=0.9,
                color=hud.THEME.cyan,
                thickness=2,
            )
        return frame

    return render


def make_key_handler(core: ScavengerHuntCore, detector: Detector) -> Callable:
    """Build the ``on_key(key, loop) -> handled`` callback."""

    def on_key(key: int, loop) -> bool:
        if key == ord(" "):
            core.start(loop.elapsed)
        elif key == ord("n"):
            core.skip(loop.elapsed)
        elif key == ord("m"):
            print(f"Detector: {detector.describe()} (ready={detector.ready})")
        elif key in (ord("+"), ord("=")):
            _tune(core, roundSeconds=core.config.round_seconds + 5)
        elif key == ord("-"):
            _tune(core, roundSeconds=core.config.round_seconds - 5)
        elif key == ord("]"):
            _tune(core, confidence=core.config.confidence + 0.05)
        elif key == ord("["):
            _tune(core, confidence=core.config.confidence - 0.05)
        else:
            return False
        return True

    return on_key


def _tune(core: ScavengerHuntCore, **values) -> None:
    """Apply a settings change and echo the result, the desktop counterpart to the
    web settings panel. Clamping and the "timer applies next round" rule live in the
    core, so both front ends behave identically."""
    core.update_settings(values)
    settings = core.settings()
    print(
        f"settings: timer {settings['roundSeconds']:.0f}s  "
        f"confidence {settings['confidence']:.0%}  hold {settings['holdSeconds']:.1f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO scavenger hunt game")
    add_camera_arguments(parser)
    parser.add_argument(
        "--model",
        default=None,
        help="Model name or path for ultralytics, or an .onnx file (default: yolo26n.pt)",
    )
    parser.add_argument("--rounds", type=int, default=HuntConfig.rounds)
    parser.add_argument("--seconds", type=float, default=HuntConfig.round_seconds)
    parser.add_argument(
        "--confidence", type=float, default=HuntConfig.confidence, help="Detection threshold"
    )
    parser.add_argument(
        "--every", type=int, default=3, help="Run detection every Nth frame (default 3)"
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - interactive
    args = parse_args()
    config = HuntConfig(rounds=args.rounds, round_seconds=args.seconds, confidence=args.confidence)
    if args.model:
        config.model = args.model

    detector = load_detector(config)
    print(__doc__)
    print(f"Detector: {detector.describe()}")
    if not detector.ready:
        print("The game will run, but nothing will be detected until the model loads.")

    core = ScavengerHuntCore(config)
    loop = CameraLoop(camera_config_from_args(args, "scavenger hunt"))
    loop.run(
        make_renderer(core, detector, every=max(1, args.every)),
        on_key=make_key_handler(core, detector),
    )
    print(f"Final score: {core.score} ({core.found_count}/{len(core.results)} found)")


if __name__ == "__main__":  # pragma: no cover
    main()
