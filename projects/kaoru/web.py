"""Flask web version — Don't Blink in the browser.

Run from the repo root:
    python -m projects.kaoru.web

Opens browser at http://127.0.0.1:5001
"""

from __future__ import annotations

import base64
import json
import time
import threading

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template
from flask_sock import Sock

from .config import GameConfig
from .core import Game, Phase, compute_ear_both_eyes


app = Flask(__name__)
sock = Sock(app)

_config = GameConfig()


@app.route("/")
def index():
    """Serve the game dashboard."""
    return render_template("index.html")


@sock.route("/ws")
def game_ws(ws):
    """WebSocket: streams JSON with base64 frame + game state."""
    config = _config
    game = Game(config)

    # MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )

    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    if not cap.isOpened():
        ws.send(json.dumps({"error": "Could not open camera"}))
        return

    start_time = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            timestamp = time.monotonic() - start_time
            frame = cv2.flip(frame, 1)

            # Face Mesh
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            face_detected = False
            ear = 0.3

            if results.multi_face_landmarks:
                face_detected = True
                landmarks = [
                    (lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark
                ]
                ear = compute_ear_both_eyes(landmarks)

            # Update game
            state = game.update(ear, face_detected, timestamp)

            # Encode frame
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            # Build payload
            payload = {
                "frame": frame_b64,
                "phase": state.phase.name,
                "survival_time": round(state.survival_time, 1),
                "countdown_value": state.countdown_value,
                "ear": round(state.ear_value, 3),
                "high_score": round(state.high_score, 1),
                "message": state.message,
                "penalty_style": state.penalty_style,
                "distraction": state.active_distraction.kind if state.active_distraction else None,
                "is_game_over": game.is_game_over(),
            }

            ws.send(json.dumps(payload))

            # Check for client messages
            try:
                msg = ws.receive(timeout=0.01)
                if msg:
                    data = json.loads(msg)
                    cmd = data.get("command")
                    if cmd == "start":
                        game.start(time.monotonic() - start_time)
                    elif cmd == "restart":
                        game.restart(time.monotonic() - start_time)
                        game.start(time.monotonic() - start_time)
            except Exception:
                pass

    except Exception as e:
        print(f"WebSocket closed: {e}")
    finally:
        cap.release()
        face_mesh.close()


def run(host: str = "127.0.0.1", port: int = 5001) -> None:
    """Start the Flask app."""
    print(f"\n  Don't Blink")
    print(f"  Open http://{host}:{port}/ in your browser\n")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run()
