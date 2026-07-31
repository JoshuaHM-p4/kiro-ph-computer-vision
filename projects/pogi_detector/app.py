"""Pogi Detector — OpenCV webcam application.

Real-time webcam app that uses SAM segmentation (or a stub fallback) to
detect and highlight subjects based on Tagalog slang prompts. Once a scan
term is active, segmentation runs continuously every few frames, keeping
the highlight locked onto the subject in real time.

Controls:
    p - Toggle continuous Pogi scan
    g - Toggle continuous Ganda scan
    c - Toggle continuous Chibog scan
    t - Toggle continuous Tsismis scan
    x - Stop scanning (clear highlights)
    s - Save screenshot
    q / ESC - Quit

Run from the repository root:
    python projects/pogi_detector/app.py
    python projects/pogi_detector/app.py --camera 1
    python projects/pogi_detector/app.py --stub    (force demo mode)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure imports work when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmentor import Instance, SegmentorConfig, load_backend  # noqa: E402
from translator import get_all_slang, translate  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_NAME = "Pogi Detector"
SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"

# HUD banner messages per slang term
BANNERS: dict[str, str] = {
    "pogi": "POGI DETECTED!",
    "ganda": "GANDA DETECTED!",
    "chibog": "CHIBOG DETECTED!",
    "tsismis": "TSISMIS DETECTED!",
}

# Glow colors per slang (BGR)
GLOW_COLORS: dict[str, tuple[int, int, int]] = {
    "pogi": (0, 200, 255),      # gold
    "ganda": (255, 100, 200),   # pink
    "chibog": (50, 220, 50),    # green
    "tsismis": (255, 180, 0),   # cyan-blue
}

# Banner display duration (seconds)
BANNER_DURATION = 3.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Pogi Detector — SAM + Tagalog Slang")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror mode")
    parser.add_argument("--stub", action="store_true", help="Force stub/demo backend")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--scan-interval", type=float, default=0.3,
                        help="Seconds between segmentation updates in real-time mode (default 0.3)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

def draw_glowing_outline(
    frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 3
) -> np.ndarray:
    """Draw a stylish glowing outline around the segmented mask.

    Creates a multi-layer glow effect by drawing contours at decreasing
    thickness with increasing transparency.
    """
    # Find contours of the mask
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return frame

    # Draw outer glow layers (wider, more transparent)
    overlay = frame.copy()
    for layer in range(4, 0, -1):
        layer_thickness = thickness + layer * 3
        alpha = 0.15 + 0.1 * (4 - layer)  # inner layers more opaque
        cv2.drawContours(overlay, contours, -1, color, layer_thickness, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        overlay = frame.copy()

    # Draw the sharp inner contour
    cv2.drawContours(frame, contours, -1, color, thickness, cv2.LINE_AA)

    # Subtle mask tint
    tint_overlay = frame.copy()
    tint_overlay[mask] = (
        np.clip(tint_overlay[mask].astype(np.int16) + np.array(color, dtype=np.int16) // 4, 0, 255)
    ).astype(np.uint8)
    frame = cv2.addWeighted(tint_overlay, 0.3, frame, 0.7, 0)

    return frame


def draw_banner(
    frame: np.ndarray, text: str, color: tuple[int, int, int], progress: float
) -> np.ndarray:
    """Draw a big, fun HUD banner across the top of the frame.

    Args:
        frame: The video frame to draw on.
        text: Banner text (e.g., "POGI DETECTED!").
        color: BGR color for the banner.
        progress: 0..1, how much of the banner duration has elapsed (for fade).
    """
    h, w = frame.shape[:2]

    # Banner background with fade-out
    alpha = max(0.0, 1.0 - progress * 0.5)  # fade gently
    banner_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), cv2.FILLED)
    frame = cv2.addWeighted(overlay, alpha * 0.85, frame, 1.0 - alpha * 0.85, 0)

    # Pulsing emoji border
    pulse = int(5 * abs(np.sin(progress * np.pi * 4)))
    border_color = tuple(min(255, c + 50 + pulse * 10) for c in color)

    cv2.rectangle(frame, (0, 0), (w, banner_h), border_color, 3)

    # Main text centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thickness = 3
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (banner_h + text_size[1]) // 2

    # Drop shadow
    cv2.putText(frame, text, (text_x + 2, text_y + 2), font, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, text, (text_x, text_y), font, scale,
                color, thickness, cv2.LINE_AA)

    return frame


def draw_hud(frame: np.ndarray, backend_name: str, active_term: str | None) -> np.ndarray:
    """Draw the persistent HUD info (controls, backend status, active scan)."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Bottom info bar
    controls = "[P]ogi  [G]anda  [C]hibog  [T]sismis  [X]stop  [S]creenshot  [Q]uit"
    cv2.putText(frame, controls, (10, h - 15), font, 0.4,
                (180, 180, 180), 1, cv2.LINE_AA)

    # Backend status (top-right)
    status = f"Backend: {backend_name}"
    status_size = cv2.getTextSize(status, font, 0.4, 1)[0]
    cv2.putText(frame, status, (w - status_size[0] - 10, 20), font, 0.4,
                (140, 140, 140), 1, cv2.LINE_AA)

    # Active scan indicator
    if active_term:
        english = translate(active_term) or active_term
        scan_text = f"SCANNING: \"{active_term}\" -> \"{english}\" [LIVE]"
        # Pulsing green dot to show it's active
        pulse = int(127 + 127 * np.sin(time.monotonic() * 4))
        cv2.circle(frame, (18, h - 47), 6, (0, pulse, 0), cv2.FILLED)
        cv2.putText(frame, scan_text, (30, h - 40), font, 0.45,
                    (100, 255, 100), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Press a key to start scanning...", (10, h - 40), font, 0.45,
                    (150, 150, 150), 1, cv2.LINE_AA)

    return frame


def draw_score(frame: np.ndarray, instance: Instance) -> np.ndarray:
    """Draw the confidence score near the bounding box."""
    x, y, w_box, h_box = instance.box
    if w_box == 0:
        return frame

    label = f"{instance.label} ({instance.score:.0%})"
    cv2.putText(frame, label, (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------

def save_screenshot(frame: np.ndarray) -> str:
    """Save the current frame as a screenshot."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = SCREENSHOT_DIR / f"pogi_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: camera loop with SAM segmentation and Pogi HUD."""
    args = parse_args()

    # Load .env file if present
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    # Resolve HF token
    token = args.token or os.environ.get("HF_TOKEN")

    # Load backend (never raises)
    config = SegmentorConfig()
    backend = load_backend(config, token=token, stub=args.stub)

    print(f"[Pogi Detector] Backend: {backend.describe()}")
    print(f"[Pogi Detector] Slang dictionary: {get_all_slang()}")

    # Open camera
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # State
    current_instances: list[Instance] = []
    banner_text: str | None = None
    banner_color: tuple[int, int, int] = (0, 200, 255)
    banner_start: float = 0.0
    active_term: str | None = None       # Currently scanning term (None = idle)
    last_segment_time: float = 0.0       # When we last ran segmentation

    print(f"[Pogi Detector] Camera {args.camera} ready. Press 'p' to start scanning!")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()

            # --- Real-time segmentation (throttled) ---
            if active_term and (now - last_segment_time) >= args.scan_interval:
                current_instances = backend.segment(frame, [active_term])
                last_segment_time = now

                # Show banner on first detection
                if current_instances and banner_text is None:
                    banner_text = f"😎 {BANNERS.get(active_term, active_term.upper() + ' DETECTED!')} 😎"
                    banner_color = GLOW_COLORS.get(active_term, (0, 200, 255))
                    banner_start = now

            # --- Draw existing segmentation results ---
            for inst in current_instances:
                color = GLOW_COLORS.get(inst.label, (0, 200, 255))
                frame = draw_glowing_outline(frame, inst.mask, color)
                frame = draw_score(frame, inst)

            # --- Draw banner if active ---
            if banner_text and (now - banner_start) < BANNER_DURATION:
                progress = (now - banner_start) / BANNER_DURATION
                frame = draw_banner(frame, banner_text, banner_color, progress)
            elif banner_text:
                banner_text = None

            # --- Persistent HUD ---
            frame = draw_hud(frame, backend.name, active_term)

            cv2.imshow(WINDOW_NAME, frame)

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord("p"):
                active_term = "pogi" if active_term != "pogi" else None
                current_instances = []
                banner_text = None
                if active_term:
                    print(f"[Pogi Detector] Now scanning for 'pogi' (real-time)")
                else:
                    print(f"[Pogi Detector] Scanning stopped.")
            elif key == ord("g"):
                active_term = "ganda" if active_term != "ganda" else None
                current_instances = []
                banner_text = None
                if active_term:
                    print(f"[Pogi Detector] Now scanning for 'ganda' (real-time)")
                else:
                    print(f"[Pogi Detector] Scanning stopped.")
            elif key == ord("c"):
                active_term = "chibog" if active_term != "chibog" else None
                current_instances = []
                banner_text = None
                if active_term:
                    print(f"[Pogi Detector] Now scanning for 'chibog' (real-time)")
                else:
                    print(f"[Pogi Detector] Scanning stopped.")
            elif key == ord("t"):
                active_term = "tsismis" if active_term != "tsismis" else None
                current_instances = []
                banner_text = None
                if active_term:
                    print(f"[Pogi Detector] Now scanning for 'tsismis' (real-time)")
                else:
                    print(f"[Pogi Detector] Scanning stopped.")
            elif key == ord("x"):
                active_term = None
                current_instances = []
                banner_text = None
                print(f"[Pogi Detector] Scanning stopped.")
            elif key == ord("s"):
                path = save_screenshot(frame)
                print(f"[Pogi Detector] Screenshot saved: {path}")
            elif key in (ord("q"), 27):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
