"""SAM labeler desktop app.

Run it with::

    # explore the effects with synthetic masks, no model, no token
    .venv/bin/python -m demos.sam_labeler.desktop --demo-mode --prompts "cat,laptop"

    # the real thing (needs a gated model and a read token)
    HF_TOKEN=hf_... .venv/bin/python -m demos.sam_labeler.desktop \
        --image photo.jpg --prompts "person,mug"

    # grab the picture from the webcam instead of a file
    .venv/bin/python -m demos.sam_labeler.desktop --webcam --prompts "person"

Keys
    q / ESC quit          SPACE   re-grab a webcam frame
    r run segmentation    TAB     select the next label
    e cycle the selected label's effect       c cycle its colour
    h hide / show the selected label          m segmentation <-> detection
    s save the result     l     list what was found

The token is read from HF_TOKEN and never printed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from ..common import hud
from .backend import load_backend
from .config import EFFECTS, MODES, PALETTE, SEGMENTATION, SamLabelerConfig
from .core import SamLabelerCore

WINDOW = "sam labeler"


def draw_chrome(frame: np.ndarray, core: SamLabelerCore, selected: int, backend_note: str) -> np.ndarray:
    """Title, the label list with its selection, and the key hints."""
    height, width = frame.shape[:2]
    hud.title(frame, "SAM LABELER", backend_note[:60])

    labels = [core.style_for(name) for name in core.prompts]
    if labels:
        panel_height = 26 * len(labels) + 30
        hud.panel(frame, (16, 70), (330, 70 + panel_height), alpha=0.62)
        hud.text(frame, "LABELS", (30, 92), scale=0.44, color=hud.THEME.magenta)
        counts = core._counts()
        for index, style in enumerate(labels):
            y = 116 + index * 26
            active = index == selected
            colour = style.colour if style.visible else hud.THEME.dim
            if active:
                cv2.rectangle(frame, (24, y - 14), (322, y + 8), hud.THEME.cyan, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (32, y - 10), (46, y + 4), colour, -1)
            text = f"{style.label}  x{counts.get(style.label, 0)}  {style.effect}"
            hud.text(frame, text, (54, y), scale=0.42,
                     color=hud.THEME.white if style.visible else hud.THEME.dim)

    if core.error:
        hud.text(frame, core.error[:90], (26, height - 74), scale=0.46, color=hud.THEME.amber)

    hud.status_strip(
        frame,
        [
            ("MODE", core.config.mode.upper()),
            ("FOUND", str(len(core.instances))),
            ("KEYS", "r run  TAB label  e effect  c colour  h hide  m mode  s save  q quit"),
        ],
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM 3.1 text-prompted labeler")
    parser.add_argument("--image", type=Path, default=None, help="Picture to label")
    parser.add_argument("--webcam", action="store_true", help="Capture the picture from a webcam")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for --webcam")
    parser.add_argument("--prompts", default="person", help="Comma separated things to find")
    parser.add_argument("--model", default=None, help="Override the model id")
    parser.add_argument(
        "--demo-mode", action="store_true", help="Synthetic masks: no model, no token"
    )
    parser.add_argument("--mode", choices=MODES, default=SEGMENTATION)
    return parser.parse_args()


def _grab(camera_index: int) -> np.ndarray | None:  # pragma: no cover - needs a camera
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f"Could not open camera {camera_index}.")
        return None
    frame = None
    for _ in range(5):  # let exposure settle
        ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def main() -> None:  # pragma: no cover - interactive
    args = parse_args()
    config = SamLabelerConfig(mode=args.mode)
    if args.model:
        config.model_id = args.model

    core = SamLabelerCore(config)
    if args.image is not None:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not read {args.image}")
            return
        core.set_source(image, args.image.name)
    elif args.webcam:
        frame = _grab(args.camera)
        if frame is None:
            return
        core.set_source(frame, "webcam.png")
    else:
        print("Pass --image or --webcam. Showing the placeholder for now.")

    core.set_prompts(args.prompts)
    backend = load_backend(config, os.environ.get("HF_TOKEN"), stub=args.demo_mode)
    print(__doc__)
    print(f"Backend: {backend.describe()}")

    if core.has_source and backend.ready:
        print("Running segmentation...")
        core.run(backend)
        print(f"Found {len(core.instances)} object(s).")

    selected = 0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    while True:
        frame = core.render()
        cv2.imshow(WINDOW, draw_chrome(frame, core, selected, backend.describe()))
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break
        if not core.prompts:
            continue
        selected = min(selected, len(core.prompts) - 1)
        label = core.prompts[selected]

        if key == ord("r"):
            core.run(backend)
            print(f"Found {len(core.instances)} object(s). {core.error or ''}")
        elif key == 9:  # TAB
            selected = (selected + 1) % len(core.prompts)
        elif key == ord("e"):
            style = core.style_for(label)
            core.set_style(label, effect=EFFECTS[(EFFECTS.index(style.effect) + 1) % len(EFFECTS)])
        elif key == ord("c"):
            style = core.style_for(label)
            current = PALETTE.index(style.colour) if style.colour in PALETTE else -1
            core.set_style(label, colour=PALETTE[(current + 1) % len(PALETTE)])
        elif key == ord("h"):
            core.set_style(label, visible=not core.style_for(label).visible)
        elif key == ord("m"):
            core.update_settings(
                {"mode": MODES[(MODES.index(core.config.mode) + 1) % len(MODES)]}
            )
        elif key == ord("l"):
            for instance in sorted(core.instances, key=lambda item: -item.area):
                print(f"  {instance.label:16} {instance.score:.2f}  box={instance.box}")
        elif key == ord("s"):
            out = Path("demos/screenshots")
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"sam-labeler-{core.source_name or 'result'}.png"
            cv2.imwrite(str(path), core.render())
            print(f"Saved {path}")
        elif key == ord(" ") and args.webcam:
            frame = _grab(args.camera)
            if frame is not None:
                core.set_source(frame, "webcam.png")

    cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover
    main()
