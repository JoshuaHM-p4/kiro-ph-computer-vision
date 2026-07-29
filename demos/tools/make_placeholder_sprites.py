"""Generate the 12 placeholder PNGTuber sprites.

    .venv/bin/python -m demos.tools.make_placeholder_sprites
    .venv/bin/python -m demos.tools.make_placeholder_sprites --size 768 --out /tmp/sprites

Files are named ``{yaw}_{expression}.png`` for the three yaw buckets
(left, center, right) crossed with the four expressions (neutral, happy,
surprised, angry), all BGRA with a transparent background. Swap in real art by
keeping the filenames.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..pngtuber.config import EXPRESSIONS, YAW_BUCKETS, PngTuberConfig

DEFAULT_DIR = PngTuberConfig().sprites_dir

# Flat BGRA fills, one hue per expression so switching is obvious at a glance.
FACE_COLORS: dict[str, tuple[int, int, int]] = {
    "neutral": (196, 214, 232),
    "happy": (150, 235, 255),
    "surprised": (235, 205, 150),
    "angry": (150, 160, 250),
}
# Horizontal shift of the features per yaw bucket, as a fraction of face width.
YAW_SHIFT: dict[str, float] = {"left": -0.14, "center": 0.0, "right": 0.14}
INK = (40, 26, 48, 255)


def make_sprite(bucket: str, expression: str, size: int = 512) -> np.ndarray:
    """Draw one labelled placeholder head as a BGRA image."""
    sprite = np.zeros((size, size, 4), dtype=np.uint8)
    cx, cy = size // 2, int(size * 0.46)
    radius = int(size * 0.34)
    shift = int(YAW_SHIFT[bucket] * radius * 2)

    color = FACE_COLORS[expression]
    # Head and a simple shoulder wedge, so the sprite reads as a character.
    cv2.ellipse(sprite, (cx, int(size * 0.92)), (int(radius * 1.5), int(radius * 0.8)),
                0, 180, 360, (*color, 255), -1, cv2.LINE_AA)
    cv2.circle(sprite, (cx, cy), radius, (*color, 255), -1, cv2.LINE_AA)
    cv2.circle(sprite, (cx, cy), radius, INK, max(2, size // 160), cv2.LINE_AA)

    eye_dx = int(radius * 0.36)
    eye_y = cy - int(radius * 0.14)
    eye_r = max(3, int(radius * 0.12))
    brow_y = eye_y - int(radius * 0.30)

    for side in (-1, 1):
        ex = cx + side * eye_dx + shift
        if expression == "surprised":
            cv2.circle(sprite, (ex, eye_y), int(eye_r * 1.5), INK, 2, cv2.LINE_AA)
            cv2.circle(sprite, (ex, eye_y), int(eye_r * 0.8), INK, -1, cv2.LINE_AA)
        elif expression == "angry":
            # Squinted eye: a thick short line rather than a circle.
            cv2.line(sprite, (ex - eye_r, eye_y), (ex + eye_r, eye_y), INK, max(3, eye_r), cv2.LINE_AA)
        elif expression == "happy":
            cv2.ellipse(sprite, (ex, eye_y), (eye_r, eye_r), 0, 180, 360, INK, max(3, eye_r // 2), cv2.LINE_AA)
        else:
            cv2.circle(sprite, (ex, eye_y), eye_r, INK, -1, cv2.LINE_AA)

        # Brows carry most of the expression.
        brow_half = int(radius * 0.22)
        if expression == "angry":
            inner = (ex + side * -brow_half, brow_y + int(radius * 0.12))
            outer = (ex + side * brow_half, brow_y - int(radius * 0.02))
        elif expression == "surprised":
            inner = (ex - brow_half, brow_y - int(radius * 0.14))
            outer = (ex + brow_half, brow_y - int(radius * 0.14))
        else:
            inner = (ex - brow_half, brow_y)
            outer = (ex + brow_half, brow_y)
        cv2.line(sprite, inner, outer, INK, max(3, size // 110), cv2.LINE_AA)

    mouth_y = cy + int(radius * 0.36)
    mouth_w = int(radius * 0.34)
    if expression == "happy":
        cv2.ellipse(sprite, (cx + shift, mouth_y), (mouth_w, int(radius * 0.22)),
                    0, 0, 180, INK, max(3, size // 120), cv2.LINE_AA)
    elif expression == "surprised":
        cv2.circle(sprite, (cx + shift, mouth_y), int(radius * 0.17), INK, -1, cv2.LINE_AA)
    elif expression == "angry":
        cv2.ellipse(sprite, (cx + shift, mouth_y + int(radius * 0.12)), (mouth_w, int(radius * 0.16)),
                    0, 180, 360, INK, max(3, size // 120), cv2.LINE_AA)
    else:
        cv2.line(sprite, (cx + shift - mouth_w, mouth_y), (cx + shift + mouth_w, mouth_y),
                 INK, max(3, size // 130), cv2.LINE_AA)

    # Nose gives the yaw shift something to read against.
    cv2.circle(sprite, (cx + shift, cy + int(radius * 0.12)), max(2, int(radius * 0.05)), INK, -1, cv2.LINE_AA)

    label = f"{bucket} / {expression}"
    cv2.putText(sprite, label, (int(size * 0.06), int(size * 0.07)),
                cv2.FONT_HERSHEY_SIMPLEX, size / 900, INK, max(1, size // 320), cv2.LINE_AA)
    return sprite


def generate(out_dir: Path, size: int = 512) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    config = PngTuberConfig()
    for bucket in YAW_BUCKETS:
        for expression in EXPRESSIONS:
            path = out_dir / config.sprite_name(bucket, expression)
            cv2.imwrite(str(path), make_sprite(bucket, expression, size))
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate placeholder PNGTuber sprites")
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR, help="Output directory")
    parser.add_argument("--size", type=int, default=512, help="Sprite edge length in pixels")
    args = parser.parse_args()

    written = generate(args.out, args.size)
    print(f"Wrote {len(written)} sprites to {args.out}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
