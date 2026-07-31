"""Desktop posture checker with OpenCV window.

Opens the webcam, runs MediaPipe Pose, analyzes posture via core.py,
draws annotations, shows streak timers, and saves replay clips on slouch events.

Usage:
    python main.py [--camera 0] [--no-replay]
"""

from __future__ import annotations

import argparse
import collections
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from config import PostureConfig
from core import PostureChecker, PostureState, ViewType


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-time posture checker")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-replay", action="store_true",
                        help="Disable slouch replay recording")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    return parser.parse_args()


class ReplayBuffer:
    """Captures a clip around the moment bad posture is detected.

    Keeps a rolling buffer of the last few seconds. When a slouch event fires,
    saves the buffer (before) + a few more seconds (after) as one clip.
    Only records once per bad posture streak.
    """

    def __init__(self, config: PostureConfig, fps: int = 30) -> None:
        self.config = config
        self.fps = fps
        # Rolling buffer: keeps last 4 seconds before the alarm
        self._pre_seconds = 4
        self._post_seconds = 4  # Record 4 seconds after alarm
        self._buffer_size = self._pre_seconds * fps
        self._buffer: collections.deque = collections.deque(maxlen=self._buffer_size)
        self._post_frames: list = []
        self._post_remaining: int = 0
        self._saved_this_streak: bool = False  # Only one clip per streak
        self._replay_dir = Path(config.replay_dir)
        self._replay_dir.mkdir(parents=True, exist_ok=True)

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the rolling buffer or post-event recording."""
        if self._post_remaining > 0:
            # Currently recording post-event frames
            self._post_frames.append(frame.copy())
            self._post_remaining -= 1
            if self._post_remaining == 0:
                self._save_clip()
        else:
            # Normal: keep rolling buffer
            self._buffer.append(frame.copy())

    def trigger_save(self) -> None:
        """Start capturing post-event frames (called when alarm fires)."""
        if self._saved_this_streak:
            return  # Already saved for this streak
        self._saved_this_streak = True
        self._post_remaining = self._post_seconds * self.fps
        self._post_frames = []

    def reset_streak(self) -> None:
        """Call when posture goes back to good — allows recording on next streak."""
        self._saved_this_streak = False

    def _save_clip(self) -> None:
        """Save pre-buffer + post-frames as one clip."""
        frames = list(self._buffer) + self._post_frames
        if not frames:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._replay_dir / f"slouch_{timestamp}.mp4"

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(filename), fourcc, self.fps, (w, h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(filename), fourcc, self.fps, (w, h))

        for frame in frames:
            writer.write(frame)
        writer.release()

        self._post_frames = []
        self._cleanup_old_clips()
        print(f"[Replay] Saved: {filename} ({self._pre_seconds}s before + {self._post_seconds}s after)")

    def _cleanup_old_clips(self) -> None:
        """Remove oldest clips if over the limit."""
        clips = sorted(self._replay_dir.glob("slouch_*.mp4"))
        while len(clips) > self.config.max_replay_clips:
            clips[0].unlink()
            clips.pop(0)


def draw_overlay(frame: np.ndarray, state: PostureState,
                 pose_landmarks, mp_drawing, mp_pose) -> np.ndarray:
    """Draw posture annotations on the frame."""
    h, w = frame.shape[:2]

    # Draw pose skeleton
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
        )

    # Status bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Score with color coding
    score_color = _score_color(state.score)
    cv2.putText(frame, f"Score: {state.score:.0f}/100", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)

    # View type
    cv2.putText(frame, f"View: {state.view.value}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Posture status
    if state.is_slouching:
        status_text = "BAD POSTURE"
        status_color = (0, 0, 255)
    else:
        status_text = "GOOD POSTURE"
        status_color = (0, 255, 0)
    cv2.putText(frame, status_text, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Streak timer
    if state.good_streak_seconds > 0:
        streak_text = f"Good streak: {_format_time(state.good_streak_seconds)}"
        cv2.putText(frame, streak_text, (w - 300, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif state.bad_streak_seconds > 0:
        streak_text = f"Bad streak: {_format_time(state.bad_streak_seconds)}"
        cv2.putText(frame, streak_text, (w - 300, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Issue messages
    y_offset = 150
    for msg in state.messages:
        if msg == "Good posture!":
            color = (0, 255, 0)
        else:
            color = (0, 100, 255)
        cv2.putText(frame, f"• {msg}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 30

    # Slouch event flash
    if state.slouch_event_triggered:
        cv2.putText(frame, "SLOUCH RECORDED!", (w // 2 - 150, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return frame


def _score_color(score: float) -> tuple[int, int, int]:
    """Map score 0-100 to a BGR color (red -> yellow -> green)."""
    if score >= 80:
        return (0, 255, 0)
    elif score >= 50:
        ratio = (score - 50) / 30.0
        return (0, int(255 * ratio), int(255 * (1 - ratio)))
    else:
        return (0, 0, 255)


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def main() -> None:
    """Run the desktop posture checker."""
    args = parse_args()
    config = PostureConfig()
    checker = PostureChecker(config)

    # Change to script directory so replay_dir is relative to project
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Setup replay buffer
    replay = None if args.no_replay else ReplayBuffer(config, config.fps_target)

    # Setup MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        print("Try: python main.py --camera 1")
        return

    print("Posture Checker running. Press 'q' to quit.")
    start_time = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Run MediaPipe Pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # Extract landmarks
            landmarks = []
            pose_landmarks = None
            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks
                for lm in results.pose_landmarks.landmark:
                    landmarks.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    })

            # Analyze posture
            timestamp = time.monotonic() - start_time
            state = checker.update(landmarks, timestamp)

            # Draw overlay
            frame = draw_overlay(frame, state, pose_landmarks, mp_drawing, mp_pose)

            # Replay buffer
            if replay:
                replay.add_frame(frame)
                if state.slouch_event_triggered:
                    replay.trigger_save()

            # Display
            cv2.imshow(config.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("Posture Checker stopped.")


if __name__ == "__main__":
    main()
