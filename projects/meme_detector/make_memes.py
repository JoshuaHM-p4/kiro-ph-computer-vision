"""Generate placeholder meme images for the Emotion Analyzer.

Run once to populate projects/MacXenix/memes/ with simple colored images:
    python projects/MacXenix/make_memes.py

The app also generates these on-the-fly if the files are missing, but having
them on disk lets you swap in real memes later.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MEME_DIR = Path(__file__).resolve().parent / "memes"
WIDTH = 300
HEIGHT = 400


def make_meme(emotion: str, color: tuple, emoji: str) -> None:
    """Create a single placeholder meme image."""
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # Colored background rectangle
    margin = 15
    cv2.rectangle(canvas, (margin, margin), (WIDTH - margin, HEIGHT - margin),
                  color, thickness=cv2.FILLED)

    # Large emoji text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(emoji, font, 4.0, 5)[0]
    text_x = (WIDTH - text_size[0]) // 2
    text_y = (HEIGHT + text_size[1]) // 2 - 40
    cv2.putText(canvas, emoji, (text_x, text_y), font, 4.0,
                (255, 255, 255), 5, cv2.LINE_AA)

    # Emotion label below
    label_size = cv2.getTextSize(emotion, font, 1.0, 2)[0]
    label_x = (WIDTH - label_size[0]) // 2
    label_y = text_y + 60
    cv2.putText(canvas, emotion, (label_x, label_y), font, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Decorative border
    cv2.rectangle(canvas, (margin, margin), (WIDTH - margin, HEIGHT - margin),
                  (255, 255, 255), thickness=2)

    # Save
    path = MEME_DIR / f"{emotion.lower()}.jpg"
    cv2.imwrite(str(path), canvas)
    print(f"  Created: {path}")


def main() -> None:
    """Generate all placeholder memes."""
    MEME_DIR.mkdir(parents=True, exist_ok=True)

    emotions = [
        ("Happy", (80, 200, 80), ":D"),
        ("Surprised", (50, 180, 255), ":O"),
        ("Sad", (200, 100, 50), ":("),
        ("Neutral", (150, 150, 150), ":|"),
    ]

    print(f"Generating placeholder memes in {MEME_DIR}/")
    for emotion, color, emoji in emotions:
        make_meme(emotion, color, emoji)

    print("Done! Replace these with real meme images whenever you like.")


if __name__ == "__main__":
    main()
