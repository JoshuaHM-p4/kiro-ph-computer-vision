"""AI Posture & Slouch Coach — OpenCV webcam application.

Real-time posture monitoring using MediaPipe Pose. Works from both frontal
and side/profile camera angles. Draws a full anatomical skeleton colored
by posture quality, with joint nodes and a spine line.

Controls:
    c - Calibrate baseline (sit up straight first!)
    s - Save screenshot
    q / ESC - Quit

Run from the repository root:
    python projects/posture_checker/app.py
    python projects/posture_checker/app.py --camera 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Ensure imports work when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracker import Config, PostureState, PostureTracker  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_NAME = "Posture Coach"
SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"

# Full skeleton connections grouped by body region
# (index1, index2, region_name)
SKELETON_CONNECTIONS = [
    # Head & neck
    (0, 7, "head"),     # nose to left ear
    (0, 8, "head"),     # nose to right ear
    (7, 11, "neck"),    # left ear to left shoulder
    (8, 12, "neck"),    # right ear to right shoulder

    # Spine (virtual: shoulder_mid to hip_mid, drawn separately)

    # Shoulders
    (11, 12, "shoulders"),

    # Arms
    (11, 13, "left_arm"),   # left shoulder to left elbow
    (13, 15, "left_arm"),   # left elbow to left wrist
    (12, 14, "right_arm"),  # right shoulder to right elbow
    (14, 16, "right_arm"),  # right elbow to right wrist

    # Torso
    (11, 23, "torso"),  # left shoulder to left hip
    (12, 24, "torso"),  # right shoulder to right hip

    # Hips
    (23, 24, "hips"),

    # Legs (upper only for seated detection)
    (23, 25, "left_leg"),   # left hip to left knee
    (24, 26, "right_leg"),  # right hip to right knee
]

# Joint importance levels (bigger = more important for display)
JOINT_SIZES = {
    0: 6,   # nose
    7: 4, 8: 4,    # ears
    11: 7, 12: 7,  # shoulders (key joints)
    13: 5, 14: 5,  # elbows
    15: 4, 16: 4,  # wrists
    23: 7, 24: 7,  # hips (key joints)
    25: 5, 26: 5,  # knees
}

# Colors
COLOR_GOOD = (80, 220, 80)       # green
COLOR_SLOUCH = (60, 60, 255)     # red
COLOR_TILT = (50, 180, 255)      # orange
COLOR_SPINE_GOOD = (100, 255, 100)
COLOR_SPINE_BAD = (80, 80, 255)
COLOR_ALERT_BG = (0, 0, 180)
COLOR_HUD_BG = (30, 30, 30)
COLOR_JOINT_OUTLINE = (255, 255, 255)

# Region-specific color adjustments (subtle shade variations)
REGION_TINTS = {
    "head": (0, 0, 20),
    "neck": (0, 10, 0),
    "shoulders": (20, 0, 0),
    "left_arm": (0, 0, 0),
    "right_arm": (0, 0, 0),
    "torso": (0, 0, 0),
    "hips": (10, 0, 10),
    "left_leg": (0, 0, 0),
    "right_leg": (0, 0, 0),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="AI Posture & Slouch Coach")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror mode")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------

def extract_landmarks(result) -> list | None:
    """Extract pose landmarks as list of (x, y, z) tuples."""
    if not result.pose_landmarks:
        return None
    return [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark]


# ---------------------------------------------------------------------------
# Skeleton drawing
# ---------------------------------------------------------------------------

def _to_px(landmark: tuple, w: int, h: int) -> tuple[int, int]:
    """Convert normalized landmark to pixel coordinates."""
    return (int(landmark[0] * w), int(landmark[1] * h))


def draw_pose_skeleton(
    frame: np.ndarray,
    landmarks: list,
    state: dict,
) -> np.ndarray:
    """Draw full anatomical skeleton with glow, joint nodes, and spine line."""
    h, w = frame.shape[:2]
    posture = state["posture"]

    # Pick base color and style
    if posture == PostureState.GOOD_POSTURE.value:
        base_color = COLOR_GOOD
        glow = False
    elif posture == PostureState.SLOUCHING.value:
        base_color = COLOR_SLOUCH
        glow = True
    elif posture == PostureState.HEAD_TILTED.value:
        base_color = COLOR_TILT
        glow = True
    else:
        return frame

    # --- Glow layer (for bad posture) ---
    if glow:
        glow_overlay = frame.copy()
        for i1, i2, region in SKELETON_CONNECTIONS:
            if i1 < len(landmarks) and i2 < len(landmarks):
                pt1 = _to_px(landmarks[i1], w, h)
                pt2 = _to_px(landmarks[i2], w, h)
                cv2.line(glow_overlay, pt1, pt2, base_color, 10, cv2.LINE_AA)
        frame = cv2.addWeighted(glow_overlay, 0.25, frame, 0.75, 0)

    # --- Draw bone connections ---
    for i1, i2, region in SKELETON_CONNECTIONS:
        if i1 < len(landmarks) and i2 < len(landmarks):
            pt1 = _to_px(landmarks[i1], w, h)
            pt2 = _to_px(landmarks[i2], w, h)

            # Tinted color per region
            tint = REGION_TINTS.get(region, (0, 0, 0))
            color = tuple(min(255, c + t) for c, t in zip(base_color, tint))

            # Thicker for core structure
            thickness = 3 if region in ("shoulders", "torso", "hips", "neck") else 2
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # --- Draw spine line (virtual: shoulder_mid → hip_mid) ---
    if 11 < len(landmarks) and 12 < len(landmarks) and 23 < len(landmarks) and 24 < len(landmarks):
        shoulder_mid = (
            (landmarks[11][0] + landmarks[12][0]) / 2.0,
            (landmarks[11][1] + landmarks[12][1]) / 2.0,
        )
        hip_mid = (
            (landmarks[23][0] + landmarks[24][0]) / 2.0,
            (landmarks[23][1] + landmarks[24][1]) / 2.0,
        )
        pt_sh = (int(shoulder_mid[0] * w), int(shoulder_mid[1] * h))
        pt_hp = (int(hip_mid[0] * w), int(hip_mid[1] * h))

        spine_color = COLOR_SPINE_GOOD if not glow else COLOR_SPINE_BAD
        # Dashed spine (draw segments)
        _draw_dashed_line(frame, pt_sh, pt_hp, spine_color, thickness=3, dash_length=10)

        # Spine midpoint marker
        spine_mid = ((pt_sh[0] + pt_hp[0]) // 2, (pt_sh[1] + pt_hp[1]) // 2)
        cv2.circle(frame, spine_mid, 4, spine_color, cv2.FILLED)

    # --- Draw joint nodes ---
    for idx, size in JOINT_SIZES.items():
        if idx < len(landmarks):
            pt = _to_px(landmarks[idx], w, h)
            # Filled circle with white outline
            cv2.circle(frame, pt, size, base_color, cv2.FILLED)
            cv2.circle(frame, pt, size + 1, COLOR_JOINT_OUTLINE, 1, cv2.LINE_AA)

    # --- Draw neck line (ear_mid → nose) for posture visualization ---
    if 7 < len(landmarks) and 8 < len(landmarks):
        ear_mid_px = (
            int((landmarks[7][0] + landmarks[8][0]) / 2.0 * w),
            int((landmarks[7][1] + landmarks[8][1]) / 2.0 * h),
        )
        nose_px = _to_px(landmarks[0], w, h)
        neck_color = COLOR_SPINE_GOOD if not glow else COLOR_SPINE_BAD
        cv2.arrowedLine(frame, ear_mid_px, nose_px, neck_color, 2, cv2.LINE_AA, tipLength=0.2)

    return frame


def _draw_dashed_line(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 10,
) -> None:
    """Draw a dashed line between two points."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = max(1, int((dx * dx + dy * dy) ** 0.5))
    num_dashes = dist // (dash_length * 2)

    for i in range(num_dashes + 1):
        t_start = (i * 2 * dash_length) / dist
        t_end = min(1.0, ((i * 2 + 1) * dash_length) / dist)
        start = (int(pt1[0] + dx * t_start), int(pt1[1] + dy * t_start))
        end = (int(pt1[0] + dx * t_end), int(pt1[1] + dy * t_end))
        cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------

def draw_hud(frame: np.ndarray, state: dict) -> np.ndarray:
    """Draw the posture HUD panel with score, timer, metrics, and view indicator."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Semi-transparent HUD background
    hud_h, hud_w = 165, 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), COLOR_HUD_BG, cv2.FILLED)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (80, 80, 80), 1)

    # Status line
    posture = state["posture"]
    if posture == PostureState.GOOD_POSTURE.value:
        status_color = COLOR_GOOD
        icon = "OK"
    elif posture == PostureState.SLOUCHING.value:
        status_color = COLOR_SLOUCH
        icon = "!!"
    elif posture == PostureState.HEAD_TILTED.value:
        status_color = COLOR_TILT
        icon = "~"
    else:
        status_color = (150, 150, 150)
        icon = "?"

    cv2.putText(frame, f"[{icon}] {posture}", (20, 35),
                font, 0.55, status_color, 2, cv2.LINE_AA)

    # View indicator
    view_text = "SIDE VIEW" if state.get("side_view") else "FRONT VIEW"
    view_color = (200, 200, 100) if state.get("side_view") else (150, 150, 150)
    cv2.putText(frame, view_text, (220, 35), font, 0.35, view_color, 1, cv2.LINE_AA)

    # Score with bar
    score = state["score"]
    score_color = COLOR_GOOD if score >= 70 else COLOR_TILT if score >= 40 else COLOR_SLOUCH
    cv2.putText(frame, f"Score: {score:.0f}%", (20, 60),
                font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 130, 47, 160, 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), cv2.FILLED)
    fill_w = int(bar_w * score / 100.0)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), score_color, cv2.FILLED)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

    # Slouch timer
    slouch_sec = state["slouch_seconds"]
    if slouch_sec > 0:
        timer_color = COLOR_SLOUCH if slouch_sec >= 3.0 else COLOR_TILT
        cv2.putText(frame, f"Slouch Timer: {slouch_sec:.1f}s", (20, 85),
                    font, 0.45, timer_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Slouch Timer: --", (20, 85),
                    font, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

    # Metrics
    cv2.putText(
        frame,
        f"Neck: {state['neck_angle']:.0f}deg  Spine: {state['spine_angle']:.0f}deg",
        (20, 110), font, 0.38, (150, 150, 150), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Fwd Head: {state['forward_head']:.2f}  Tilt: {state['shoulder_tilt']:.2f}",
        (20, 130), font, 0.38, (150, 150, 150), 1, cv2.LINE_AA,
    )

    # Calibration
    cal_text = "Calibrated" if state["calibrated"] else "Not calibrated [C]"
    cal_color = COLOR_GOOD if state["calibrated"] else (150, 150, 100)
    cv2.putText(frame, cal_text, (20, 155), font, 0.35, cal_color, 1, cv2.LINE_AA)

    # Bottom controls
    controls = "[C]alibrate  [S]creenshot  [Q]uit"
    cv2.putText(frame, controls, (10, h - 15), font, 0.4,
                (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def draw_alert_banner(frame: np.ndarray, slouch_seconds: float) -> np.ndarray:
    """Draw pulsing warning banner for sustained slouching."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    pulse = 0.6 + 0.4 * abs(np.sin(time.monotonic() * 3))

    # Banner background
    banner_h = 70
    y1 = h // 2 - banner_h // 2
    y2 = h // 2 + banner_h // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), COLOR_ALERT_BG, cv2.FILLED)
    frame = cv2.addWeighted(overlay, pulse * 0.85, frame, 1.0 - pulse * 0.85, 0)

    cv2.rectangle(frame, (0, y1), (w, y2), (0, 0, 255), 3)

    # Text
    text = "SLOUCH ALERT! SIT UP STRAIGHT!"
    text_size = cv2.getTextSize(text, font, 0.9, 3)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h // 2 + text_size[1] // 2

    cv2.putText(frame, text, (text_x + 2, text_y + 2), font, 0.9,
                (0, 0, 0), 4, cv2.LINE_AA)
    text_color = (int(255 * pulse), int(255 * pulse), 255)
    cv2.putText(frame, text, (text_x, text_y), font, 0.9,
                text_color, 3, cv2.LINE_AA)

    dur_text = f"({slouch_seconds:.0f}s)"
    dur_size = cv2.getTextSize(dur_text, font, 0.45, 1)[0]
    cv2.putText(frame, dur_text, ((w - dur_size[0]) // 2, text_y + 22),
                font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------

def save_screenshot(frame: np.ndarray) -> str:
    """Save the current frame as a screenshot."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = SCREENSHOT_DIR / f"posture_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: camera loop with posture tracking and HUD."""
    args = parse_args()

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    tracker = PostureTracker(Config())
    started = time.monotonic()

    print(f"[Posture Coach] Camera {args.camera} ready.")
    print(f"[Posture Coach] Works from FRONT and SIDE views!")
    print(f"[Posture Coach] Press 'c' to calibrate your good posture baseline.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            landmarks = extract_landmarks(result)
            now = time.monotonic() - started
            state = tracker.update(landmarks, now)

            # Draw skeleton
            if landmarks is not None:
                frame = draw_pose_skeleton(frame, landmarks, state)

            # Draw HUD
            frame = draw_hud(frame, state)

            # Alert banner
            if state["alert"]:
                frame = draw_alert_banner(frame, state["slouch_seconds"])

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                success = tracker.calibrate_baseline(landmarks)
                if success:
                    print("[Posture Coach] Baseline calibrated! This is your 'good posture' reference.")
                else:
                    print("[Posture Coach] Calibration failed — no pose detected.")
            elif key == ord("s"):
                path = save_screenshot(frame)
                print(f"[Posture Coach] Screenshot saved: {path}")
            elif key in (ord("q"), 27):
                break

    finally:
        capture.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
