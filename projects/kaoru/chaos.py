"""Psychological warfare, escalation, and meta bullshit.

All the effects that make the game actively hostile to the player.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------
# Leaderboard of shame
# ------------------------------------------------------------------

SHAME_FILE = Path(__file__).parent / ".leaderboard_of_shame.json"


def load_leaderboard() -> list[dict]:
    """Load the leaderboard from disk."""
    if SHAME_FILE.exists():
        try:
            return json.loads(SHAME_FILE.read_text())
        except Exception:
            return []
    return []


def save_to_leaderboard(survival_time: float, death_reason: str) -> list[dict]:
    """Save a death to the leaderboard and return the full list."""
    board = load_leaderboard()
    board.append({
        "time": round(survival_time, 3),
        "reason": death_reason,
        "when": time.strftime("%H:%M:%S"),
    })
    # Keep last 20
    board = board[-20:]
    try:
        SHAME_FILE.write_text(json.dumps(board, indent=2))
    except Exception:
        pass
    return board


# ------------------------------------------------------------------
# Fake cursor
# ------------------------------------------------------------------

class FakeCursor:
    """A fake mouse cursor that moves on its own to distract."""

    def __init__(self, w: int, h: int):
        self.x = w // 2
        self.y = h // 2
        self.target_x = w // 2
        self.target_y = h // 2
        self.w = w
        self.h = h
        self._next_move = 0.0

    def update(self, timestamp: float, rng: random.Random) -> None:
        """Move the cursor toward a random target."""
        if timestamp > self._next_move:
            self.target_x = rng.randint(50, self.w - 50)
            self.target_y = rng.randint(50, self.h - 50)
            self._next_move = timestamp + rng.uniform(0.5, 2.0)

        # Smooth movement
        self.x += int((self.target_x - self.x) * 0.1)
        self.y += int((self.target_y - self.y) * 0.1)

    def draw(self, frame: np.ndarray) -> None:
        """Draw a fake cursor on the frame."""
        # Arrow shape
        pts = np.array([
            [self.x, self.y],
            [self.x, self.y + 18],
            [self.x + 5, self.y + 14],
            [self.x + 9, self.y + 20],
            [self.x + 12, self.y + 18],
            [self.x + 8, self.y + 12],
            [self.x + 13, self.y + 12],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (255, 255, 255))
        cv2.polylines(frame, [pts], True, (0, 0, 0), 1)


# ------------------------------------------------------------------
# Fake system notifications
# ------------------------------------------------------------------

FAKE_NOTIFICATIONS = [
    "Windows Update: Restart required",
    "Low battery: 3% remaining",
    "Your webcam is being accessed by another application",
    "Discord: someone is typing...",
    "Virus detected: blink_detector.exe",
    "Storage almost full (99.7%)",
    "New email: 'We need to talk'",
    "Mom is calling...",
    "Screenshot saved to Desktop",
    "Your screen is being recorded",
    "WiFi disconnected",
    "System error: eyelid_driver.sys has crashed",
    "Task Manager: main.py (Not Responding)",
    "Reminder: blink in 3 seconds",
    "Windows Defender: Threat detected",
    "Bluetooth: Unknown device connected",
]


def draw_notification(frame: np.ndarray, text: str, progress: float) -> np.ndarray:
    """Draw a fake Windows-style notification in the bottom-right.

    progress: 0.0 (just appeared) to 1.0 (about to disappear)
    """
    h, w = frame.shape[:2]

    # Slide in/out animation
    notif_w = min(350, w - 20)
    notif_h = 60
    x_base = w - notif_w - 10

    if progress < 0.1:
        # Sliding in
        y_offset = int((1 - progress / 0.1) * notif_h)
    elif progress > 0.9:
        # Sliding out
        y_offset = int((progress - 0.9) / 0.1 * notif_h)
    else:
        y_offset = 0

    y_base = h - notif_h - 10 + y_offset

    # Draw notification box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_base, y_base), (x_base + notif_w, y_base + notif_h),
                  (50, 50, 50), -1)
    cv2.rectangle(overlay, (x_base, y_base), (x_base + notif_w, y_base + notif_h),
                  (100, 100, 100), 1)
    frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

    # Icon (fake Windows logo - just a blue square)
    cv2.rectangle(frame, (x_base + 8, y_base + 15), (x_base + 28, y_base + 35),
                  (255, 150, 0), -1)

    # Text
    cv2.putText(frame, text[:40], (x_base + 35, y_base + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, "just now", (x_base + 35, y_base + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    return frame


# ------------------------------------------------------------------
# Creepy face transformation
# ------------------------------------------------------------------

def apply_creepy_face(frame: np.ndarray, intensity: float) -> np.ndarray:
    """Gradually make the player's face look creepy.

    intensity: 0.0 to 1.0 (increases over time)
    """
    if intensity <= 0:
        return frame

    # Desaturate
    if intensity > 0.1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        desat = min(1.0, intensity * 0.8)
        frame = cv2.addWeighted(frame, 1 - desat, gray_bgr, desat, 0)

    # Darken
    if intensity > 0.2:
        darkness = min(0.5, intensity * 0.4)
        frame = (frame.astype(np.float32) * (1 - darkness)).astype(np.uint8)

    # Green tint (horror movie look)
    if intensity > 0.3:
        green_overlay = np.zeros_like(frame)
        green_overlay[:, :, 1] = int(min(60, intensity * 50))
        frame = cv2.add(frame, green_overlay)

    # Increase contrast (makes shadows darker, harsh look)
    if intensity > 0.5:
        alpha = 1.0 + intensity * 0.5
        frame = np.clip(alpha * (frame.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)

    return frame


# ------------------------------------------------------------------
# Heartbeat effect
# ------------------------------------------------------------------

def apply_heartbeat(frame: np.ndarray, survival_time: float) -> np.ndarray:
    """Pulse the frame (zoom slightly) at an increasing heartbeat rate."""
    # BPM increases from 60 to 180 over 60 seconds
    bpm = min(180, 60 + survival_time * 2)
    beat_period = 60.0 / bpm
    phase = (survival_time % beat_period) / beat_period

    # Pulse on the "beat" (first 20% of the cycle)
    if phase < 0.2:
        pulse = math.sin(phase / 0.2 * math.pi) * 0.02  # 2% zoom
        h, w = frame.shape[:2]
        zoom = 1.0 + pulse
        new_h, new_w = int(h * zoom), int(w * zoom)
        resized = cv2.resize(frame, (new_w, new_h))
        y_off = (new_h - h) // 2
        x_off = (new_w - w) // 2
        frame = resized[y_off:y_off + h, x_off:x_off + w]

    return frame


# ------------------------------------------------------------------
# Gravity tilt
# ------------------------------------------------------------------

def apply_gravity(frame: np.ndarray, survival_time: float) -> np.ndarray:
    """The frame slowly tilts more to one side over time."""
    if survival_time < 10:
        return frame

    # Slowly increasing tilt (max 8 degrees)
    angle = min(8.0, (survival_time - 10) * 0.15)
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    frame = cv2.warpAffine(frame, rot, (w, h))
    return frame


# ------------------------------------------------------------------
# Shrinking feed
# ------------------------------------------------------------------

def apply_shrink(frame: np.ndarray, survival_time: float) -> np.ndarray:
    """The camera feed gets smaller over time, surrounded by darkness."""
    if survival_time < 20:
        return frame

    # Shrink from 100% to 40% over the next 40 seconds
    shrink_progress = min(0.6, (survival_time - 20) * 0.015)
    scale = 1.0 - shrink_progress

    h, w = frame.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w < 100 or new_h < 100:
        return frame

    small = cv2.resize(frame, (new_w, new_h))
    result = np.zeros_like(frame)
    x_off = (w - new_w) // 2
    y_off = (h - new_h) // 2
    result[y_off:y_off + new_h, x_off:x_off + new_w] = small
    return result


# ------------------------------------------------------------------
# Ghost face
# ------------------------------------------------------------------

def apply_ghost_face(frame: np.ndarray, survival_time: float, rng: random.Random) -> np.ndarray:
    """Occasionally overlay a faint second 'face' in the background."""
    if survival_time < 25:
        return frame

    # Only show occasionally (30% of the time after 25s)
    if rng.random() > 0.03:  # ~3% chance per frame = frequent enough to be creepy
        return frame

    h, w = frame.shape[:2]

    # Create a ghostly oval
    ghost = np.zeros_like(frame)
    center_x = rng.randint(w // 4, 3 * w // 4)
    center_y = rng.randint(h // 4, 3 * h // 4)

    # Pale oval face shape
    cv2.ellipse(ghost, (center_x, center_y), (40, 55), 0, 0, 360, (180, 180, 180), -1)
    # Dark eye sockets
    cv2.circle(ghost, (center_x - 15, center_y - 10), 8, (30, 30, 30), -1)
    cv2.circle(ghost, (center_x + 15, center_y - 10), 8, (30, 30, 30), -1)
    # Mouth
    cv2.ellipse(ghost, (center_x, center_y + 20), (12, 8), 0, 0, 360, (30, 30, 30), -1)

    # Blend very faintly
    alpha = rng.uniform(0.03, 0.08)
    frame = cv2.addWeighted(frame, 1 - alpha, ghost, alpha, 0)
    return frame


# ------------------------------------------------------------------
# Subliminal frames
# ------------------------------------------------------------------

SUBLIMINAL_TEXTS = [
    "BLINK", "SLEEP", "CLOSE YOUR EYES", "GIVE UP",
    "YOU CAN'T WIN", "I SEE YOU", "TIRED?", "SUBMIT",
]


def apply_subliminal(frame: np.ndarray, survival_time: float, rng: random.Random) -> np.ndarray:
    """Very brief subliminal text flashes (1-2 frames worth)."""
    if survival_time < 5:
        return frame

    # ~1% chance per frame
    if rng.random() > 0.01:
        return frame

    h, w = frame.shape[:2]
    text = rng.choice(SUBLIMINAL_TEXTS)
    # Big, faint, centered
    scale = rng.uniform(2.0, 4.0)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0]
    x = (w - size[0]) // 2
    y = (h + size[1]) // 2
    # Very low opacity
    overlay = frame.copy()
    cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 100), 3)
    frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
    return frame


# ------------------------------------------------------------------
# Timer lies
# ------------------------------------------------------------------

def get_lying_timer(real_time: float, rng: random.Random) -> str:
    """Occasionally show a wrong time to mess with the player."""
    if real_time < 5:
        return f"{real_time:.1f}s"

    # 5% chance to lie
    if rng.random() < 0.05:
        # Jump forward or backward
        fake = real_time + rng.choice([-2.0, -1.5, 1.5, 2.0, 3.0])
        return f"{max(0, fake):.1f}s"

    return f"{real_time:.1f}s"


# ------------------------------------------------------------------
# Fake crash
# ------------------------------------------------------------------

class FakeCrash:
    """Simulates a brief fake BSOD that recovers — just to scare you."""

    def __init__(self):
        self._active = False
        self._start_time = 0.0
        self._duration = 0.6  # seconds
        self._next_crash: float | None = None
        self._triggered_count = 0

    def maybe_trigger(self, survival_time: float, rng: random.Random) -> None:
        """Decide whether to trigger a fake crash."""
        if self._active:
            return
        if survival_time < 12:
            return
        if self._next_crash is None:
            self._next_crash = survival_time + rng.uniform(8, 20)
            return
        if survival_time >= self._next_crash and self._triggered_count < 3:
            self._active = True
            self._start_time = survival_time
            self._triggered_count += 1
            self._next_crash = survival_time + rng.uniform(15, 30)

    def is_active(self, survival_time: float) -> bool:
        """Check if fake crash is currently showing."""
        if not self._active:
            return False
        if survival_time - self._start_time > self._duration:
            self._active = False
            return False
        return True

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the fake BSOD (brief)."""
        h, w = frame.shape[:2]
        frame[:] = (200, 50, 0)
        cv2.putText(frame, "INVOLUNTARY_BLINK_EXCEPTION", (40, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Collecting error data...", (40, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame


# ------------------------------------------------------------------
# Window title trolling
# ------------------------------------------------------------------

WINDOW_TITLES = [
    "Don't Blink",
    "Don't Blink - ARE YOU STILL THERE?",
    "Don't Blink - I can see you struggling",
    "Don't Blink - just give up already",
    "Don't Blink - YOUR EYES LOOK DRY",
    "Don't Blink - (Not Responding)",
    "Don't Blink - VIRUS DETECTED",
    "Don't Blink - recording...",
    "Don't Blink - uploading to youtube...",
    "Don't Blink - sending to mom...",
    "Task Manager",
    "system32/delete_homework.exe",
    "Don't Blink - you will lose",
    "Don't Blink - lol",
]


def get_window_title(survival_time: float, rng: random.Random) -> str | None:
    """Get a new window title to set (or None to keep current)."""
    if survival_time < 5:
        return None
    # Change every ~5 seconds
    if rng.random() < 0.005:  # ~0.5% per frame at 30fps = every ~6s
        return rng.choice(WINDOW_TITLES)
    return None


# ------------------------------------------------------------------
# Death screen with blink face screenshot and fake upload
# ------------------------------------------------------------------

def capture_blink_face(frame: np.ndarray) -> np.ndarray:
    """Capture and store the exact frame when the player blinked."""
    return frame.copy()


def draw_death_screen_enhanced(
    frame: np.ndarray,
    state,
    config,
    blink_face: np.ndarray | None,
    leaderboard: list[dict],
    death_time: float,
    current_time: float,
) -> np.ndarray:
    """Draw the enhanced death screen with screenshot, leaderboard, fake upload."""
    h, w = frame.shape[:2]
    scale = max(0.5, min(h / 1080, w / 1920))
    elapsed_since_death = current_time - death_time

    # Compute rank
    t = state.survival_time
    if t < 1:
        rank = "LITERALLY A REFLEX"
    elif t < 3:
        rank = "PATHETIC"
    elif t < 7:
        rank = "BARELY HUMAN"
    elif t < 15:
        rank = "ACCEPTABLE"
    elif t < 30:
        rank = "IMPRESSIVE"
    elif t < 60:
        rank = "MACHINE"
    else:
        rank = "ARE YOU EVEN HUMAN?!"

    frame[:] = (200, 50, 0)  # BSOD blue

    # Main error text
    death_msg = "INVOLUNTARY_BLINK_EXCEPTION" if state.death_reason == "blink" else "FACE_ABANDONMENT_EXCEPTION"
    cause = "eyelid_control.sys" if state.death_reason == "blink" else "cowardice.dll"

    lines = [
        "A problem has been detected and your eyes have",
        "shut down to prevent damage to your retinas.",
        "",
        death_msg,
        "",
        f"*** STOP: 0x0000BLINK (0x{int(t * 100):08X})",
        f"    ({cause}, willpower.dll, human_weakness.exe)",
        "",
        "If this is the first time you've seen this screen,",
        "maybe try not having human reflexes." if state.death_reason == "blink"
        else "maybe try not being a coward who runs from the screen.",
        "",
        f"*** survival_time:     {t:.3f} seconds",
        f"*** high_score:        {state.high_score:.3f} seconds",
        f"*** dignity_remaining: 0%",
        f"*** rank:              {rank}",
        "",
    ]

    font_scale = 0.5 * scale
    thickness = max(1, int(1.5 * scale))
    line_height = int(26 * scale)
    y = int(40 * scale)
    x = int(40 * scale)

    for line in lines:
        cv2.putText(frame, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        y += line_height

    # Blink face screenshot (small, in corner)
    if blink_face is not None:
        thumb_h = int(100 * scale)
        thumb_w = int(130 * scale)
        thumb = cv2.resize(blink_face, (thumb_w, thumb_h))
        tx = w - thumb_w - int(20 * scale)
        ty = int(20 * scale)
        frame[ty:ty + thumb_h, tx:tx + thumb_w] = thumb
        cv2.rectangle(frame, (tx - 1, ty - 1), (tx + thumb_w + 1, ty + thumb_h + 1),
                      (255, 255, 255), 1)
        cv2.putText(frame, "EVIDENCE:", (tx, ty - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 255), 1)

    # Fake upload progress bar
    if elapsed_since_death < 5.0:
        progress = min(1.0, elapsed_since_death / 4.0)
        bar_w = int(250 * scale)
        bar_h = int(18 * scale)
        bar_x = x
        bar_y = y + int(10 * scale)

        upload_text = "Uploading blink_face.jpg to Facebook..."
        if progress >= 1.0:
            upload_text = "Upload complete! Your friends will see this."

        cv2.putText(frame, upload_text, (bar_x, bar_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (255, 255, 255), 1)
        bar_y += int(15 * scale)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (100, 100, 100), -1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      (0, 200, 0), -1)
        cv2.putText(frame, f"{int(progress * 100)}%",
                    (bar_x + bar_w + 5, bar_y + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, (255, 255, 255), 1)
        y = bar_y + bar_h + int(20 * scale)
    else:
        y += int(10 * scale)

    # Leaderboard of shame
    if leaderboard:
        y += int(10 * scale)
        cv2.putText(frame, "=== LEADERBOARD OF SHAME ===", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (255, 255, 0), 1)
        y += line_height
        for entry in leaderboard[-5:]:  # Last 5 attempts
            reason_icon = "x_x" if entry["reason"] == "blink" else "->|"
            line = f"  [{entry['when']}] {entry['time']:.2f}s  ({reason_icon})"
            cv2.putText(frame, line, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (200, 200, 200), 1)
            y += int(20 * scale)

    # Controls at the bottom
    cv2.putText(frame, "SPACE = humiliate yourself again | Q = quit in shame",
                (x, h - int(30 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (200, 200, 200), 1)

    return frame
