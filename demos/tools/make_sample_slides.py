"""Generate the project presentation deck for the slide presenter demo.

    .venv/bin/python -m demos.tools.make_sample_slides
    .venv/bin/python -m demos.tools.make_sample_slides --out /tmp/deck --width 1920

Six slides about computer vision and how this project was built: a title card,
four content slides with two points each, and a closing card. Swap in your own
deck any time — the presenter simply reads image files in natural filename order.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..common import hud

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "slide_presenter" / "assets" / "slides"

TITLE = "title"
CONTENT = "content"
CLOSING = "closing"


@dataclass
class Slide:
    """One slide's copy. ``kind`` selects the layout."""

    kind: str
    title: str
    subtitle: str = ""
    kicker: str = ""
    bullets: tuple[str, ...] = field(default_factory=tuple)
    footer: str = ""


DECK: tuple[Slide, ...] = (
    Slide(
        kind=TITLE,
        kicker="KIROVERSE WEEK 7  //  BUILD NIGHTS",
        title="COMPUTER VISION",
        subtitle="Building a real-time gesture suite with OpenCV, MediaPipe and Kiro",
        footer="JoshuaHM-p4  //  kiro-computer-vision",
    ),
    Slide(
        kind=CONTENT,
        kicker="01  //  THE RAW MATERIAL",
        title="WHAT THE CAMERA GIVES US",
        bullets=(
            "MediaPipe turns every frame into normalized landmark coordinates: "
            "21 points per hand, 478 for a face, 33 for the whole body.",
            "Nothing is trained. Pretrained graphs run on the CPU, so all of the "
            "real work is deciding what those numbers mean.",
        ),
    ),
    Slide(
        kind=CONTENT,
        kicker="02  //  THE ARCHITECTURE",
        title="ONE BRAIN, TWO FRONT ENDS",
        bullets=(
            "Each demo's logic is a pure function: landmarks and a clock in, state "
            "out. No camera, no window, no web framework inside it.",
            "So the OpenCV window and the browser share one implementation, and "
            "440+ tests run on synthetic landmarks without a webcam.",
        ),
    ),
    Slide(
        kind=CONTENT,
        kicker="03  //  THE HARD PART",
        title="MAKING GESTURES FEEL RELIABLE",
        bullets=(
            "Hysteresis everywhere: separate enter and release thresholds, so a "
            "hand resting near a boundary cannot flicker the state.",
            "Measure ratios, not pixels. Dividing by hand span, eye width or body "
            "scale makes distance from the camera stop mattering.",
        ),
    ),
    Slide(
        kind=CONTENT,
        kicker="04  //  THE RESULT",
        title="FOUR DEMOS, ONE SUITE",
        bullets=(
            "Paint in the air, present slides by pinching, count 6-7 reps from "
            "pose, and drive a PNGTuber with head yaw and expression.",
            "In the browser MediaPipe streams landmarks to Flask over a WebSocket, "
            "where a frame guard drops stale packets instead of corrupting state.",
        ),
    ),
    Slide(
        kind=CLOSING,
        kicker="THANK YOU",
        title="TRY IT",
        subtitle="python -m demos.home.web      |      python -m demos.home.opencv_menu",
        footer="Pinch right for the next slide, left to go back.",
    ),
)


# OpenCV's Hershey fonts are ASCII-only: anything else is drawn as "?". Map the
# typography that creeps into copy, so a stray en dash cannot silently corrupt a
# slide.
ASCII_REPLACEMENTS = {
    "\u00b7": "-",   # middle dot
    "\u2022": "-",   # bullet
    "\u2013": "-",   # en dash
    "\u2014": "--",  # em dash
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u00a0": " ",
    "\u00b0": " deg",
}


def ascii_only(text: str) -> str:
    """Make ``text`` safe for the Hershey fonts."""
    for source, target in ASCII_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text.encode("ascii", "replace").decode("ascii")


def wrap(text: str, scale: float, thickness: int, max_width: int) -> list[str]:
    """Greedy word wrap measured with the real font metrics."""
    words = ascii_only(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        (width, _), _ = cv2.getTextSize(candidate, hud.FONT, scale, thickness)
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _background(width: int, height: int, accent: float) -> np.ndarray:
    """Dark gradient with a faint grid, tinted per slide."""
    slide = np.zeros((height, width, 3), dtype=np.uint8)
    top = np.array([38 + accent * 26, 16, 30 + accent * 34], dtype=np.float32)
    bottom = np.array([10, 7, 14], dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    slide[:] = (top[None, :] * (1 - ramp) + bottom[None, :] * ramp)[:, None, :].astype(np.uint8)

    step = max(24, width // 28)
    for x in range(0, width, step):
        cv2.line(slide, (x, 0), (x, height), (30, 22, 40), 1)
    for y in range(0, height, step):
        cv2.line(slide, (0, y), (width, y), (30, 22, 40), 1)
    return slide


def _frame_corners(slide: np.ndarray, color: tuple[int, int, int]) -> None:
    """Bracketed corners, matching the HUD used across the demos."""
    height, width = slide.shape[:2]
    margin = int(width * 0.045)
    length = int(width * 0.05)
    for cx, cy, dx, dy in (
        (margin, margin, 1, 1),
        (width - margin, margin, -1, 1),
        (margin, height - margin, 1, -1),
        (width - margin, height - margin, -1, -1),
    ):
        cv2.line(slide, (cx, cy), (cx + dx * length, cy), color, 3, cv2.LINE_AA)
        cv2.line(slide, (cx, cy), (cx, cy + dy * length), color, 3, cv2.LINE_AA)


def render(slide_data: Slide, index: int, total: int, width: int, height: int) -> np.ndarray:
    """Render one slide to a BGR image."""
    accent = index / max(total - 1, 1)
    slide = _background(width, height, accent)
    _frame_corners(slide, hud.THEME.cyan)

    left = int(width * 0.10)
    text_width = int(width * 0.80)
    unit = width / 1600  # all sizes scale with the output resolution

    if slide_data.kicker:
        cv2.putText(
            slide,
            ascii_only(slide_data.kicker),
            (left, int(height * 0.20)),
            hud.FONT,
            0.62 * unit,
            hud.THEME.magenta,
            2,
            cv2.LINE_AA,
        )

    if slide_data.kind == TITLE:
        title_scale, title_y = 2.5 * unit, 0.42
    elif slide_data.kind == CLOSING:
        title_scale, title_y = 2.2 * unit, 0.45
    else:
        title_scale, title_y = 1.35 * unit, 0.31

    y = int(height * title_y)
    for line in wrap(slide_data.title, title_scale, 4, text_width):
        cv2.putText(slide, line, (left, y), hud.FONT, title_scale, hud.THEME.cyan, 4, cv2.LINE_AA)
        y += int(title_scale * 46)

    # Accent rule under the title.
    cv2.line(slide, (left, y + int(14 * unit)), (left + int(width * 0.14), y + int(14 * unit)),
             hud.THEME.magenta, max(2, int(4 * unit)), cv2.LINE_AA)

    if slide_data.subtitle:
        y += int(64 * unit)
        for line in wrap(slide_data.subtitle, 0.78 * unit, 2, text_width):
            cv2.putText(slide, line, (left, y), hud.FONT, 0.78 * unit, hud.THEME.white, 2, cv2.LINE_AA)
            y += int(38 * unit)

    if slide_data.bullets:
        y += int(64 * unit)
        for bullet in slide_data.bullets:
            marker_y = y - int(9 * unit)
            cv2.rectangle(
                slide,
                (left, marker_y),
                (left + int(14 * unit), marker_y + int(14 * unit)),
                hud.THEME.lime,
                -1,
                cv2.LINE_AA,
            )
            for offset, line in enumerate(wrap(bullet, 0.82 * unit, 2, text_width - int(46 * unit))):
                cv2.putText(
                    slide,
                    line,
                    (left + int(46 * unit), y),
                    hud.FONT,
                    0.82 * unit,
                    hud.THEME.white if offset == 0 else (198, 198, 198),
                    2,
                    cv2.LINE_AA,
                )
                y += int(44 * unit)
            y += int(46 * unit)

    if slide_data.footer:
        cv2.putText(
            slide,
            ascii_only(slide_data.footer),
            (left, int(height * 0.88)),
            hud.FONT,
            0.6 * unit,
            hud.THEME.dim,
            2,
            cv2.LINE_AA,
        )

    # Slide number and progress bar.
    label = f"{index + 1:02d} / {total:02d}"
    (label_w, _), _ = cv2.getTextSize(label, hud.FONT, 0.6 * unit, 2)
    cv2.putText(
        slide,
        label,
        (width - left - label_w, int(height * 0.88)),
        hud.FONT,
        0.6 * unit,
        hud.THEME.dim,
        2,
        cv2.LINE_AA,
    )
    filled = int(width * (index + 1) / total)
    cv2.rectangle(slide, (0, height - max(4, int(6 * unit))), (filled, height), hud.THEME.magenta, -1)
    return slide


def generate(out_dir: Path, width: int = 1600, height: int = 900) -> list[Path]:
    """Write the deck as ``slide01.png`` ... ``slide06.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, slide_data in enumerate(DECK):
        path = out_dir / f"slide{index + 1:02d}.png"
        cv2.imwrite(str(path), render(slide_data, index, len(DECK), width, height))
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the project slide deck")
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR, help="Output directory")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing slide*.png in the target directory first",
    )
    args = parser.parse_args()

    if args.clean:
        for stale in sorted(args.out.glob("slide*.png")):
            stale.unlink()

    written = generate(args.out, args.width, args.height)
    print(f"Wrote {len(written)} slides to {args.out}")
    for path, slide_data in zip(written, DECK):
        print(f"  {path.name}  {slide_data.title}")


if __name__ == "__main__":
    main()
