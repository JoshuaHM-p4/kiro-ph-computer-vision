"""Emotion Analyzer & Meme Popup — camera loop and display.

Captures webcam frames, runs MediaPipe Face Mesh to get landmarks, feeds them
to the EmotionAnalyzer (pure logic), and renders the detected emotion along with
a matching meme image in a side panel.

Run from the repository root:
    python projects/MacXenix/app.py
    python projects/MacXenix/app.py --camera 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Ensure this module can find emotion.py when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emotion import Config, Emotion, EmotionAnalyzer  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_NAME = "Emotion Analyzer & Meme Popup"
MEME_DIR = Path(__file__).resolve().parent / "memes"
MEME_PANEL_WIDTH = 300  # pixels for the side panel


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Emotion Analyzer & Meme Popup")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror mode")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Meme loading
# ---------------------------------------------------------------------------

def load_meme(emotion: str, panel_height: int) -> np.ndarray:
    """Load the meme image for a given emotion, or generate a placeholder.

    Looks for memes/<emotion>.jpg (case-insensitive). If not found, creates a
    colored placeholder with the emotion name drawn on it.
    """
    # Try common extensions
    for ext in (".jpg", ".png", ".jpeg"):
        path = MEME_DIR / f"{emotion.lower()}{ext}"
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                return _resize_to_panel(img, panel_height)

    # Generate a placeholder
    return _make_placeholder(emotion, panel_height)


def _resize_to_panel(img: np.ndarray, panel_height: int) -> np.ndarray:
    """Resize image to fit the side panel while maintaining aspect ratio."""
    h, w = img.shape[:2]
    scale = min(MEME_PANEL_WIDTH / w, panel_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center on a black canvas of panel dimensions
    canvas = np.zeros((panel_height, MEME_PANEL_WIDTH, 3), dtype=np.uint8)
    y_offset = (panel_height - new_h) // 2
    x_offset = (MEME_PANEL_WIDTH - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def _make_placeholder(emotion: str, panel_height: int) -> np.ndarray:
    """Generate a colored placeholder meme with the emotion emoji/text."""
    # Color scheme per emotion
    colors = {
        "happy": (80, 200, 80),      # green
        "surprised": (50, 180, 255),  # orange-ish (BGR)
        "sad": (200, 100, 50),       # blue-ish
        "neutral": (150, 150, 150),  # gray
    }
    emojis = {
        "happy": ":D",
        "surprised": ":O",
        "sad": ":(",
        "neutral": ":|",
    }

    color = colors.get(emotion.lower(), (150, 150, 150))
    emoji = emojis.get(emotion.lower(), "?")

    canvas = np.zeros((panel_height, MEME_PANEL_WIDTH, 3), dtype=np.uint8)
    # Fill with a tinted background
    canvas[:] = (30, 30, 30)

    # Draw a big colored rectangle
    margin = 20
    cv2.rectangle(
        canvas,
        (margin, margin),
        (MEME_PANEL_WIDTH - margin, panel_height - margin),
        color,
        thickness=cv2.FILLED,
    )

    # Draw the emoji large and centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(emoji, font, 3.0, 4)[0]
    text_x = (MEME_PANEL_WIDTH - text_size[0]) // 2
    text_y = (panel_height + text_size[1]) // 2 - 30
    cv2.putText(canvas, emoji, (text_x, text_y), font, 3.0, (255, 255, 255), 4, cv2.LINE_AA)

    # Draw the emotion name below
    label_size = cv2.getTextSize(emotion, font, 0.8, 2)[0]
    label_x = (MEME_PANEL_WIDTH - label_size[0]) // 2
    label_y = text_y + 50
    cv2.putText(canvas, emotion, (label_x, label_y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Measurement (extract landmarks from MediaPipe result)
# ---------------------------------------------------------------------------

def extract_landmarks(result) -> list | None:
    """Extract the first face's landmarks as a list of (x, y, z) tuples.

    Returns None if no face is detected.
    """
    if not result.multi_face_landmarks:
        return None
    face = result.multi_face_landmarks[0]
    return [(lm.x, lm.y, lm.z) for lm in face.landmark]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_emotion_overlay(frame: np.ndarray, state: dict) -> np.ndarray:
    """Draw the detected emotion and metrics on the camera frame."""
    emotion = state.get("emotion", "Neutral")

    # Color per emotion for the label
    label_colors = {
        "Happy": (80, 200, 80),
        "Surprised": (50, 180, 255),
        "Sad": (200, 100, 50),
        "Neutral": (200, 200, 200),
    }
    color = label_colors.get(emotion, (200, 200, 200))

    # Main emotion label (top-left)
    cv2.putText(
        frame, f"Emotion: {emotion}",
        (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA,
    )

    # Metrics (smaller, below)
    metrics_text = (
        f"smile: {state.get('smile_ratio', 0):.3f}  "
        f"mouth: {state.get('mouth_ratio', 0):.3f}  "
        f"eye: {state.get('eye_ratio', 0):.3f}"
    )
    cv2.putText(
        frame, metrics_text,
        (24, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
    )

    # Instructions at bottom
    cv2.putText(
        frame, "Press 'q' or ESC to quit",
        (24, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (180, 180, 180), 1, cv2.LINE_AA,
    )

    return frame


def compose_with_meme(frame: np.ndarray, meme: np.ndarray) -> np.ndarray:
    """Place the camera frame and meme panel side by side."""
    h_frame, w_frame = frame.shape[:2]
    h_meme, w_meme = meme.shape[:2]

    # Resize meme panel to match frame height if needed
    if h_meme != h_frame:
        meme = cv2.resize(meme, (w_meme, h_frame), interpolation=cv2.INTER_AREA)

    return np.hstack([frame, meme])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: camera loop with emotion detection and meme overlay."""
    args = parse_args()

    # Open camera
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize emotion analyzer
    analyzer = EmotionAnalyzer(Config())
    started = time.monotonic()

    # Meme cache (avoids reloading every frame)
    current_meme_emotion: str = "Neutral"
    meme_image = load_meme("Neutral", args.height)

    print(f"[Emotion Analyzer] Camera {args.camera} opened. Press 'q' or ESC to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            # Extract landmarks and analyze
            landmarks = extract_landmarks(result)
            now = time.monotonic() - started

            if landmarks is not None:
                state = analyzer.update(landmarks, now)
            else:
                # No face detected — keep last state, show "No face"
                state = analyzer.state()
                state["emotion"] = "No face"

            # Update meme if emotion changed
            detected = state.get("emotion", "Neutral")
            if detected != current_meme_emotion and detected != "No face":
                current_meme_emotion = detected
                meme_image = load_meme(detected, frame.shape[0])

            # Draw overlays
            frame = draw_emotion_overlay(frame, state)
            composite = compose_with_meme(frame, meme_image)

            cv2.imshow(WINDOW_NAME, composite)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    finally:
        capture.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
