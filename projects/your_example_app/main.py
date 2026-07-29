"""Entry point: argument parsing, the camera loop, and the window.

All the messy I/O lives here so that logic.py can stay pure. Run it from the
repository root with the venv active:

    python projects/your_example_app/main.py
    python projects/your_example_app/main.py --camera 1
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from logic import BrightnessTracker, Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build-night starter app")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the webcam")
    return parser.parse_args()


def measure(frame: np.ndarray) -> float:
    """Mean brightness of a BGR frame, as 0..1. Replace with your own measurement."""
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() / 255.0)


def draw(frame: np.ndarray, state: dict) -> np.ndarray:
    """Everything on-screen. Keep decisions out of here — just render the state."""
    colour = (140, 255, 140) if state["bright"] else (135, 146, 168)
    label = "BRIGHT" if state["bright"] else "DARK"
    cv2.putText(frame, label, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"brightness {state['brightness']:.2f}   changes {state['changes']}   q to quit",
        (24, frame.shape[0] - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")

    tracker = BrightnessTracker(Config())
    started = time.monotonic()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            state = tracker.update(measure(frame), time.monotonic() - started)
            cv2.imshow("starter app", draw(frame, state))

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
