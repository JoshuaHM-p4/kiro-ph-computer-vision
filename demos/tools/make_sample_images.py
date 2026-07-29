"""Generate sample images for the image lab demo.

    .venv/bin/python -m demos.tools.make_sample_images

Four images chosen to make different operations interesting: shapes for contours,
a noisy gradient for the blurs, a document scan with uneven lighting for adaptive
threshold, and colour blocks for HSV masking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "image_lab" / "assets" / "samples"

WHITE = (245, 245, 245)


def shapes(width: int = 900, height: int = 600) -> np.ndarray:
    """Solid shapes on a dark ground: ideal for contours, boxes and centroids."""
    image = np.full((height, width, 3), (24, 20, 30), dtype=np.uint8)
    cv2.rectangle(image, (70, 90), (300, 300), (120, 255, 140), -1)
    cv2.circle(image, (520, 200), 105, (255, 231, 76), -1)
    triangle = np.array([[760, 120], [870, 330], [650, 330]], dtype=np.int32)
    cv2.fillPoly(image, [triangle], (200, 60, 255))
    cv2.ellipse(image, (250, 470), (150, 80), 25, 0, 360, (60, 190, 255), -1)
    pentagon = np.array(
        [
            [600 + int(90 * np.cos(a)), 460 + int(90 * np.sin(a))]
            for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [pentagon], (80, 80, 255))
    # A couple of specks, so the contour area filter has something to remove.
    for x, y in ((420, 90), (830, 520), (120, 560)):
        cv2.circle(image, (x, y), 4, WHITE, -1)
    return image


def gradient_noise(width: int = 900, height: int = 600) -> np.ndarray:
    """Smooth gradient plus salt-and-pepper noise: shows what each blur does."""
    # meshgrid so all three channels are full (height, width) arrays; the 1-D
    # forms broadcast against each other but not into a stack.
    xx, yy = np.meshgrid(
        np.linspace(0, 1, width, dtype=np.float32),
        np.linspace(0, 1, height, dtype=np.float32),
    )
    base = np.stack(
        [(0.35 + 0.5 * xx) * 255, (0.25 + 0.55 * yy) * 255, (0.5 + 0.4 * xx * yy) * 255], axis=2
    ).astype(np.uint8)

    rng = np.random.default_rng(7)
    speckle = rng.random((height, width))
    base[speckle < 0.02] = 0
    base[speckle > 0.98] = 255
    # Some structure to sharpen or edge-detect.
    for index in range(6):
        cv2.line(base, (120 + index * 130, 60), (60 + index * 130, height - 60), WHITE, 2, cv2.LINE_AA)
    return base


def document(width: int = 900, height: int = 620) -> np.ndarray:
    """Text with a lighting gradient: global threshold fails, adaptive works."""
    page = np.full((height, width, 3), 235, dtype=np.uint8)
    lines = [
        "IMAGE PROCESSING",
        "cv2.threshold picks one level for the",
        "whole image, so a page lit from one",
        "side loses either the bright text or",
        "the dark text.",
        "cv2.adaptiveThreshold chooses a level",
        "per neighbourhood instead, which is",
        "why scanned documents use it.",
    ]
    y = 90
    for index, line in enumerate(lines):
        scale = 1.1 if index == 0 else 0.7
        thickness = 3 if index == 0 else 2
        cv2.putText(page, line, (70, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (40, 40, 40), thickness, cv2.LINE_AA)
        y += 70 if index == 0 else 52

    # Vignette-style lighting gradient across the page.
    ramp = np.linspace(1.0, 0.42, width, dtype=np.float32)[None, :, None]
    return np.clip(page.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def colour_blocks(width: int = 900, height: int = 600) -> np.ndarray:
    """Saturated colour patches for inRange and hue shifting."""
    image = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
    hues = [0, 15, 30, 45, 60, 75, 90, 120, 150, 170]
    block_w = width // 5
    block_h = height // 2
    for index, hue in enumerate(hues):
        row, col = divmod(index, 5)
        patch = np.full((block_h, block_w, 3), (hue, 220, 230), dtype=np.uint8)
        image[row * block_h : (row + 1) * block_h, col * block_w : (col + 1) * block_w] = (
            cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)
        )
        cv2.putText(
            image,
            f"H{hue}",
            (col * block_w + 12, row * block_h + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return image


SAMPLES = {
    "shapes.png": shapes,
    "gradient-noise.png": gradient_noise,
    "document.png": document,
    "colour-blocks.png": colour_blocks,
}


def generate(out_dir: Path = DEFAULT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, builder in SAMPLES.items():
        path = out_dir / name
        cv2.imwrite(str(path), builder())
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image lab samples")
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    written = generate(args.out)
    print(f"Wrote {len(written)} sample images to {args.out}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
