"""Flask web dashboard for the posture checker.

Shows live annotated video feed, real-time posture score, streak timers,
a session score graph, and a gallery of slouch replay clips.

Usage:
    python web.py [--camera 0] [--port 5000] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template, jsonify, send_from_directory, request
from flask_sock import Sock

from config import PostureConfig
from core import PostureChecker, PostureState


# --- App Setup ---

app = Flask(__name__)
sock = Sock(app)

# Global state shared between camera thread and Flask routes
_state_lock = threading.Lock()
_current_state: dict = {}
_score_history: list[dict] = []  # [{timestamp, score}]
_latest_frame: bytes = b""
_recording_enabled: bool = False  # Recording starts disabled; user toggles it
_active_filter: str = "none"  # Current face filter
_alarm_until: float = 0.0  # Alarm stays active until this timestamp
_capture_until: float = 0.0  # Capture sound stays active until this timestamp
_active_viewers: int = 0  # Number of connected clients (WS + video feed)
_camera_thread = None  # Reference to the active CameraThread


# --- Camera Thread ---

class CameraThread:
    """Background thread that captures, processes, and streams frames."""

    def __init__(self, camera_index: int = 0, config: PostureConfig = None) -> None:
        self.camera_index = camera_index
        self.config = config or PostureConfig()
        self.checker = PostureChecker(self.config)
        self._running = False
        self._thread: threading.Thread | None = None

        # MediaPipe setup
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Face mesh for face filters
        self._mp_face = mp.solutions.face_mesh
        self._face_mesh = self._mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Replay buffer (reuse from main.py concept)
        from main import ReplayBuffer
        self._replay = ReplayBuffer(self.config, self.config.fps_target)

    def start(self) -> None:
        """Start the camera processing thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the camera processing thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """Main camera loop running in background."""
        global _latest_frame, _current_state, _score_history, _alarm_until, _capture_until

        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return

        start_time = time.monotonic()
        score_sample_interval = 1.0  # Record score every second
        last_score_time = 0.0

        try:
            while self._running:
                # Pause camera processing when no one is viewing
                if _active_viewers <= 0:
                    time.sleep(0.5)
                    continue

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)

                # Run MediaPipe Pose
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._pose.process(rgb)

                # Run face mesh for face filters
                face_results = self._face_mesh.process(rgb)

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

                # Analyze
                timestamp = time.monotonic() - start_time
                state = self.checker.update(landmarks, timestamp)

                # Draw skeleton and overlay
                frame = self._draw_frame(frame, state, pose_landmarks)

                # Apply face filter
                if _active_filter != "none" and face_results.multi_face_landmarks:
                    frame = self._draw_face_filter(frame, face_results.multi_face_landmarks[0])

                # Replay buffer (only record when enabled)
                if _recording_enabled:
                    self._replay.add_frame(frame)
                    if state.slouch_event_triggered:
                        self._replay.trigger_save()
                        # Hold capture signal for 1.5s so WebSocket reliably catches it
                        _capture_until = time.monotonic() + 1.5
                    # Reset streak flag when posture returns to good
                    if not state.is_slouching and not state.paused:
                        self._replay.reset_streak()

                # Alarm: hold signal for 1.5s when slouch event fires
                if state.slouch_event_triggered:
                    _alarm_until = time.monotonic() + 1.5

                # Encode frame as JPEG for streaming
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # Update shared state
                now = time.monotonic()
                with _state_lock:
                    _latest_frame = jpeg.tobytes()
                    _current_state = {
                        "score": state.score,
                        "view": state.view.value,
                        "paused": state.paused,
                        "is_slouching": state.is_slouching,
                        "head_forward": state.head_forward,
                        "shoulders_uneven": state.shoulders_uneven,
                        "torso_leaning": state.torso_leaning,
                        "head_forward_degrees": state.head_forward_degrees,
                        "shoulder_tilt_degrees": state.shoulder_tilt_degrees,
                        "shoulder_higher_side": state.shoulder_higher_side,
                        "torso_lean_degrees": state.torso_lean_degrees,
                        "good_streak": state.good_streak_seconds,
                        "bad_streak": state.bad_streak_seconds,
                        "messages": state.messages,
                        "slouch_event": now < _alarm_until,
                        "capture_event": now < _capture_until,
                    }

                    # Sample score for the graph
                    if timestamp - last_score_time >= score_sample_interval:
                        _score_history.append({
                            "time": round(timestamp, 1),
                            "score": state.score,
                        })
                        # Keep max 600 points (10 min at 1/sec)
                        if len(_score_history) > 600:
                            _score_history.pop(0)
                        last_score_time = timestamp

                # Frame rate control
                time.sleep(1.0 / self.config.fps_target)

        finally:
            cap.release()
            self._pose.close()

    def _draw_frame(self, frame: np.ndarray, state: PostureState,
                    pose_landmarks) -> np.ndarray:
        """Draw overlay with red highlighting on bad body parts."""
        h, w = frame.shape[:2]

        if pose_landmarks:
            lm = pose_landmarks.landmark

            # Define landmark groups for each metric
            # Head: nose, ears
            head_indices = [0, 7, 8]
            # Shoulders
            shoulder_indices = [11, 12]
            # Torso: shoulders + hips
            torso_indices = [11, 12, 23, 24]

            # Draw all connections in default green first
            self._mp_drawing.draw_landmarks(
                frame, pose_landmarks, self._mp_pose.POSE_CONNECTIONS,
                self._mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2,
                                            circle_radius=3),
                self._mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
            )

            # Overlay red circles on problem areas
            if state.head_forward:
                for idx in head_indices:
                    px = int(lm[idx].x * w)
                    py = int(lm[idx].y * h)
                    cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)
                    cv2.circle(frame, (px, py), 12, (0, 0, 200), 2)

            if state.shoulders_uneven:
                for idx in shoulder_indices:
                    px = int(lm[idx].x * w)
                    py = int(lm[idx].y * h)
                    cv2.circle(frame, (px, py), 12, (0, 0, 255), -1)
                    cv2.circle(frame, (px, py), 14, (0, 0, 200), 2)
                # Draw red line between shoulders
                ls = (int(lm[11].x * w), int(lm[11].y * h))
                rs = (int(lm[12].x * w), int(lm[12].y * h))
                cv2.line(frame, ls, rs, (0, 0, 255), 3)

            if state.torso_leaning:
                for idx in torso_indices:
                    px = int(lm[idx].x * w)
                    py = int(lm[idx].y * h)
                    cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)
                    cv2.circle(frame, (px, py), 12, (0, 0, 200), 2)
                # Draw red lines for torso
                ls = (int(lm[11].x * w), int(lm[11].y * h))
                lh = (int(lm[23].x * w), int(lm[23].y * h))
                rs = (int(lm[12].x * w), int(lm[12].y * h))
                rh = (int(lm[24].x * w), int(lm[24].y * h))
                cv2.line(frame, ls, lh, (0, 0, 255), 3)
                cv2.line(frame, rs, rh, (0, 0, 255), 3)

        # Posture status bar at top
        color = (0, 0, 255) if state.is_slouching else (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (w, 5), color, -1)

        return frame

    def _draw_face_filter(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        """Draw fun face filter overlays using face mesh landmarks."""
        h, w = frame.shape[:2]
        lm = face_landmarks.landmark
        active = _active_filter

        # Key landmark indices for face mesh (468 landmarks)
        # Nose tip: 1, Forehead top: 10, Chin: 152
        # Left eye outer: 33, Right eye outer: 263
        # Left eye inner: 133, Right eye inner: 362
        # Upper lip: 13, Lower lip: 14
        # Left ear: 234, Right ear: 454

        def pt(idx):
            """Convert landmark index to pixel coordinates."""
            return int(lm[idx].x * w), int(lm[idx].y * h)

        if active == "glasses":
            # Draw sunglasses across both eyes
            left_eye = pt(33)
            right_eye = pt(263)
            center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            eye_width = int(abs(right_eye[0] - left_eye[0]) * 0.6)
            eye_height = int(eye_width * 0.45)

            # Left lens
            cv2.ellipse(frame, left_eye, (eye_width // 2, eye_height // 2), 0, 0, 360, (20, 20, 20), -1)
            cv2.ellipse(frame, left_eye, (eye_width // 2, eye_height // 2), 0, 0, 360, (50, 50, 50), 3)
            # Right lens
            cv2.ellipse(frame, right_eye, (eye_width // 2, eye_height // 2), 0, 0, 360, (20, 20, 20), -1)
            cv2.ellipse(frame, right_eye, (eye_width // 2, eye_height // 2), 0, 0, 360, (50, 50, 50), 3)
            # Bridge
            cv2.line(frame, left_eye, right_eye, (50, 50, 50), 3)

        elif active == "hat":
            # Top hat above forehead
            forehead = pt(10)
            left_temple = pt(54)
            right_temple = pt(284)
            hat_width = int(abs(right_temple[0] - left_temple[0]) * 1.3)
            hat_height = int(hat_width * 0.8)
            brim_width = int(hat_width * 1.4)

            hat_top = (forehead[0], forehead[1] - hat_height)
            # Hat body
            cv2.rectangle(frame,
                          (forehead[0] - hat_width // 2, forehead[1] - hat_height),
                          (forehead[0] + hat_width // 2, forehead[1]),
                          (30, 30, 30), -1)
            # Brim
            cv2.ellipse(frame, forehead, (brim_width // 2, int(hat_height * 0.15)), 0, 0, 360, (30, 30, 30), -1)
            # Hat band
            cv2.rectangle(frame,
                          (forehead[0] - hat_width // 2, forehead[1] - int(hat_height * 0.25)),
                          (forehead[0] + hat_width // 2, forehead[1] - int(hat_height * 0.15)),
                          (0, 0, 180), -1)

        elif active == "mustache":
            # Curly mustache below nose
            nose_tip = pt(1)
            upper_lip = pt(13)
            mid_y = (nose_tip[1] + upper_lip[1]) // 2
            face_width = abs(pt(454)[0] - pt(234)[0])
            moustache_w = int(face_width * 0.45)

            # Left curl
            cv2.ellipse(frame, (nose_tip[0] - moustache_w // 3, mid_y),
                        (moustache_w // 3, int(moustache_w * 0.15)), 10, 0, 180, (30, 30, 30), -1)
            # Right curl
            cv2.ellipse(frame, (nose_tip[0] + moustache_w // 3, mid_y),
                        (moustache_w // 3, int(moustache_w * 0.15)), -10, 0, 180, (30, 30, 30), -1)
            # Center
            cv2.ellipse(frame, (nose_tip[0], mid_y),
                        (int(moustache_w * 0.15), int(moustache_w * 0.1)), 0, 0, 360, (30, 30, 30), -1)

        elif active == "crown":
            # Golden crown above head
            forehead = pt(10)
            left_temple = pt(54)
            right_temple = pt(284)
            crown_width = int(abs(right_temple[0] - left_temple[0]) * 1.2)
            crown_height = int(crown_width * 0.5)
            base_y = forehead[1] - int(crown_height * 0.2)

            # Crown body
            pts = np.array([
                [forehead[0] - crown_width // 2, base_y],
                [forehead[0] - crown_width // 2, base_y - crown_height // 2],
                [forehead[0] - crown_width // 4, base_y - crown_height // 4],
                [forehead[0] - crown_width // 8, base_y - crown_height],
                [forehead[0], base_y - crown_height // 3],
                [forehead[0] + crown_width // 8, base_y - crown_height],
                [forehead[0] + crown_width // 4, base_y - crown_height // 4],
                [forehead[0] + crown_width // 2, base_y - crown_height // 2],
                [forehead[0] + crown_width // 2, base_y],
            ], np.int32)
            cv2.fillPoly(frame, [pts], (0, 200, 255))
            cv2.polylines(frame, [pts], True, (0, 150, 200), 2)
            # Jewels
            for dx in [-crown_width // 8, 0, crown_width // 8]:
                cv2.circle(frame, (forehead[0] + dx, base_y - crown_height + int(crown_height * 0.35)),
                           int(crown_width * 0.04), (0, 0, 255), -1)

        elif active == "clown":
            # Big red nose
            nose = pt(1)
            face_width = abs(pt(454)[0] - pt(234)[0])
            radius = int(face_width * 0.12)
            cv2.circle(frame, nose, radius, (0, 0, 230), -1)
            cv2.circle(frame, nose, radius, (0, 0, 180), 2)
            # Shiny spot
            cv2.circle(frame, (nose[0] - radius // 3, nose[1] - radius // 3),
                       radius // 4, (100, 100, 255), -1)

        elif active == "cat":
            # Cat ears above head
            forehead = pt(10)
            left_temple = pt(54)
            right_temple = pt(284)
            face_width = abs(right_temple[0] - left_temple[0])
            ear_size = int(face_width * 0.35)

            # Left ear
            left_base = (forehead[0] - int(face_width * 0.35), forehead[1] - int(ear_size * 0.2))
            left_tip = (left_base[0], left_base[1] - ear_size)
            left_pts = np.array([
                [left_base[0] - ear_size // 3, left_base[1]],
                [left_tip[0], left_tip[1]],
                [left_base[0] + ear_size // 3, left_base[1]],
            ], np.int32)
            cv2.fillPoly(frame, [left_pts], (60, 60, 60))
            # Inner ear
            inner_left = np.array([
                [left_base[0] - ear_size // 5, left_base[1] - ear_size // 8],
                [left_tip[0], left_tip[1] + ear_size // 4],
                [left_base[0] + ear_size // 5, left_base[1] - ear_size // 8],
            ], np.int32)
            cv2.fillPoly(frame, [inner_left], (180, 140, 180))

            # Right ear
            right_base = (forehead[0] + int(face_width * 0.35), forehead[1] - int(ear_size * 0.2))
            right_tip = (right_base[0], right_base[1] - ear_size)
            right_pts = np.array([
                [right_base[0] - ear_size // 3, right_base[1]],
                [right_tip[0], right_tip[1]],
                [right_base[0] + ear_size // 3, right_base[1]],
            ], np.int32)
            cv2.fillPoly(frame, [right_pts], (60, 60, 60))
            # Inner ear
            inner_right = np.array([
                [right_base[0] - ear_size // 5, right_base[1] - ear_size // 8],
                [right_tip[0], right_tip[1] + ear_size // 4],
                [right_base[0] + ear_size // 5, right_base[1] - ear_size // 8],
            ], np.int32)
            cv2.fillPoly(frame, [inner_right], (180, 140, 180))

        return frame


# --- Flask Routes ---

@app.route("/")
def index():
    """Serve the dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the annotated camera feed."""
    def generate():
        while True:
            with _state_lock:
                frame = _latest_frame
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1.0 / 30)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def get_state():
    """Get current posture state as JSON."""
    with _state_lock:
        return jsonify(_current_state)


@app.route("/api/history")
def get_history():
    """Get score history for the graph."""
    with _state_lock:
        return jsonify(_score_history)


@app.route("/api/replays")
def get_replays():
    """List available slouch replay clips."""
    replay_dir = Path("replays")
    if not replay_dir.exists():
        return jsonify([])

    clips = sorted(replay_dir.glob("slouch_*.mp4"), reverse=True)
    return jsonify([
        {
            "filename": c.name,
            "timestamp": c.stem.replace("slouch_", ""),
            "size_kb": round(c.stat().st_size / 1024, 1),
        }
        for c in clips[:20]  # Last 20 clips
    ])


@app.route("/api/recording", methods=["GET", "POST"])
def recording_toggle():
    """Get or toggle recording state."""
    global _recording_enabled
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "enabled" in data:
            _recording_enabled = bool(data["enabled"])
        else:
            _recording_enabled = not _recording_enabled
    return jsonify({"recording": _recording_enabled})


@app.route("/api/filter", methods=["GET", "POST"])
def filter_control():
    """Get or set the active face filter."""
    global _active_filter
    valid_filters = {"none", "glasses", "hat", "mustache", "crown", "clown", "cat"}
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        new_filter = data.get("filter", "none")
        if new_filter in valid_filters:
            _active_filter = new_filter
    return jsonify({"filter": _active_filter})


@app.route("/api/cameras")
def list_cameras():
    """List available camera indices."""
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({"index": i, "name": f"Camera {i}"})
            cap.release()
    return jsonify(cameras)


@app.route("/api/camera", methods=["GET", "POST"])
def camera_control():
    """Get or switch the active camera."""
    global _camera_thread
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        new_index = data.get("camera")
        if new_index is not None and isinstance(new_index, int):
            # Restart camera thread with new index
            _camera_thread.stop()
            _camera_thread = CameraThread(new_index, _camera_thread.config)
            _camera_thread.start()
            return jsonify({"camera": new_index, "status": "switched"})
    return jsonify({"camera": _camera_thread.camera_index})


@app.route("/replays/<path:filename>")
def serve_replay(filename: str):
    """Serve a replay video file."""
    return send_from_directory("replays", filename)


@sock.route("/ws")
def ws_state(ws):
    """WebSocket endpoint streaming posture state at ~10 Hz."""
    global _active_viewers
    _active_viewers += 1
    try:
        while True:
            with _state_lock:
                data = _current_state.copy()
            ws.send(json.dumps(data))
            time.sleep(0.1)
    except Exception:
        pass  # Client disconnected
    finally:
        _active_viewers -= 1


# --- Entry Point ---

def main() -> None:
    """Start the Flask web dashboard."""
    parser = argparse.ArgumentParser(description="Posture checker web dashboard")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    args = parser.parse_args()

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    config = PostureConfig()

    # Start camera thread
    global _camera_thread
    _camera_thread = CameraThread(args.camera, config)
    _camera_thread.start()
    print(f"Camera thread started on camera {args.camera}")

    # Start Flask
    print(f"Dashboard: http://{args.host}:{args.port}/")
    if args.host == "0.0.0.0":
        print("WARNING: Binding to 0.0.0.0 exposes your webcam feed to the network!")

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    finally:
        _camera_thread.stop()


if __name__ == "__main__":
    main()
