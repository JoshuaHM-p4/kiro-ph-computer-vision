"""Desktop version — fullscreen Don't Blink with MediaPipe Face Mesh.

Run from the repo root:
    python -m projects.kaoru.main
"""

from __future__ import annotations

import random
import time

import cv2
import mediapipe as mp
import numpy as np

from .config import GameConfig
from .core import Game, Phase, compute_ear_both_eyes
from . import sfx
from . import chaos


def run(config: GameConfig | None = None) -> None:
    """Main loop: open camera, detect face, run game, draw distractions."""
    config = config or GameConfig()
    game = Game(config)

    # MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh

    def _create_face_mesh():
        return mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    face_mesh = _create_face_mesh()
    frame_count = 0
    FACE_MESH_RESET_INTERVAL = 500  # Recreate every 500 frames to avoid TFLite segfault

    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {config.camera_index}")
        return

    # Fullscreen window
    cv2.namedWindow("Don't Blink", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Don't Blink", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Get the actual screen size for penalty frames
    # We detect it by checking the window size after fullscreen
    screen_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        # Try to get actual screen resolution
        import subprocess
        out = subprocess.check_output(
            ["xdpyinfo"], stderr=subprocess.DEVNULL
        ).decode()
        for line in out.split("\n"):
            if "dimensions:" in line:
                dims = line.split()[1]  # e.g. "1920x1080"
                screen_w, screen_h = map(int, dims.split("x"))
                break
    except Exception:
        # Fallback: just use a large frame
        screen_w, screen_h = 1920, 1080

    print("Don't Blink — press SPACE to start, Q to quit, R to restart")
    start_time = time.monotonic()
    rng = random.Random()

    # Track state for sound triggers
    prev_phase = Phase.WAITING
    prev_countdown = 0
    prev_distraction = None
    last_drone_time = 0.0

    # Chaos systems
    fake_cursor = chaos.FakeCursor(config.frame_width, config.frame_height)
    fake_crash = chaos.FakeCrash()
    notification_text: str | None = None
    notification_start: float = 0.0
    notification_duration: float = 3.0
    next_notification: float = 8.0
    blink_face: np.ndarray | None = None
    leaderboard: list[dict] = chaos.load_leaderboard()
    death_time: float = 0.0
    last_frame_before_death: np.ndarray | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.monotonic() - start_time

        # Mirror the frame for natural interaction
        frame = cv2.flip(frame, 1)

        # Periodically recreate face mesh to prevent TFLite memory corruption
        frame_count += 1
        if frame_count % FACE_MESH_RESET_INTERVAL == 0:
            face_mesh.close()
            face_mesh = _create_face_mesh()

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = face_mesh.process(rgb)
        except Exception:
            # If MediaPipe crashes, recreate it
            face_mesh.close()
            face_mesh = _create_face_mesh()
            results = None

        # Extract EAR
        face_detected = False
        ear = 0.3  # default (open)

        if results and results.multi_face_landmarks:
            face_detected = True
            landmarks = [
                (lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark
            ]
            ear = compute_ear_both_eyes(landmarks)

        # Save frame before death (for screenshot)
        if state_phase_is_playing := (prev_phase == Phase.PLAYING):
            last_frame_before_death = frame.copy()

        # Update game
        state = game.update(ear, face_detected, timestamp)

        # --- Sound effects ---
        if state.phase == Phase.COUNTDOWN and state.countdown_value != prev_countdown:
            sfx.countdown_beep(state.countdown_value)
            prev_countdown = state.countdown_value

        if state.phase == Phase.PLAYING and prev_phase == Phase.COUNTDOWN:
            sfx.game_start()

        current_distraction = state.active_distraction
        if current_distraction is not None and current_distraction != prev_distraction:
            sfx.distraction_hit()
        prev_distraction = current_distraction

        if state.phase == Phase.BLINKED and prev_phase != Phase.BLINKED:
            sfx.blink_death()
            blink_face = last_frame_before_death
            death_time = timestamp
            leaderboard = chaos.save_to_leaderboard(state.survival_time, state.death_reason)

        if state.phase == Phase.PLAYING and timestamp - last_drone_time > 2.0:
            intensity = min(1.0, state.survival_time / 30.0)
            sfx.tension_drone(intensity)
            last_drone_time = timestamp

        prev_phase = state.phase

        # === CHAOS EFFECTS (only during PLAYING) ===
        if state.phase == Phase.PLAYING:
            t = state.survival_time
            frame_start = time.monotonic()

            try:
                # Apply distraction effects
                if state.active_distraction is not None:
                    frame = _apply_distraction(frame, state.active_distraction.kind, rng)

                # Permanent screen corruption
                if t > config.corruption_start_time:
                    corruption = min(1.0, (t - config.corruption_start_time)
                                     * config.corruption_intensity_per_second)
                    frame = _apply_corruption(frame, corruption, rng)

                # Near-blink panic indicator
                frame = _draw_panic_indicator(frame, ear, config)

                # Only apply expensive chaos if we have frame budget left
                frame_budget_ok = (time.monotonic() - frame_start) < 0.020

                # Psychological warfare: creepy face (after 10s)
                if t > 10 and frame_budget_ok:
                    creepy_intensity = min(1.0, (t - 10) / 50.0)
                    frame = chaos.apply_creepy_face(frame, creepy_intensity)

                # Heartbeat pulse (after 5s)
                if t > 5:
                    frame = chaos.apply_heartbeat(frame, t)

                # Gravity tilt (after 10s)
                if t > 10:
                    frame = chaos.apply_gravity(frame, t)

                # Shrinking feed (after 20s)
                if t > 20 and frame_budget_ok:
                    frame = chaos.apply_shrink(frame, t)

                # Ghost face (after 25s)
                if t > 25:
                    frame = chaos.apply_ghost_face(frame, t, rng)

                # Subliminal frames
                frame = chaos.apply_subliminal(frame, t, rng)

                # Ensure frame is contiguous (np.roll and slicing can make it non-contiguous)
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)

            except Exception:
                # If any effect crashes, just continue with whatever frame we have
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)

            # Fake cursor
            fake_cursor.update(timestamp, rng)
            fake_cursor.draw(frame)

            # Fake crash
            fake_crash.maybe_trigger(t, rng)
            if fake_crash.is_active(t):
                frame = fake_crash.draw(frame)

            # Fake notifications
            if t > next_notification:
                notification_text = rng.choice(chaos.FAKE_NOTIFICATIONS)
                notification_start = t
                next_notification = t + rng.uniform(5, 12)

            if notification_text is not None:
                progress = (t - notification_start) / notification_duration
                if progress <= 1.0:
                    frame = chaos.draw_notification(frame, notification_text, progress)
                else:
                    notification_text = None

            # Window title trolling
            new_title = chaos.get_window_title(t, rng)
            if new_title:
                cv2.setWindowTitle("Don't Blink", new_title)

        # Draw HUD
        frame = _draw_hud(frame, state, config, rng)

        # Draw penalty on game over OR resize for display
        if state.phase == Phase.BLINKED:
            # Death animation: quick glitch frames (keep short so GNOME doesn't kill us)
            if prev_phase == Phase.PLAYING or (prev_phase == Phase.BLINKED and death_time == timestamp):
                death_frame = cv2.resize(frame, (screen_w, screen_h))
                for i in range(5):
                    glitch = death_frame.copy()
                    noise = np.random.randint(0, 255, glitch.shape, dtype=np.uint8)
                    alpha = i / 5.0
                    glitch = cv2.addWeighted(noise, alpha, glitch, 1 - alpha, 0)
                    num_glitch_lines = i * 5
                    for _ in range(num_glitch_lines):
                        y_line = rng.randint(0, screen_h - 1)
                        shift = rng.randint(-150, 150)
                        glitch[y_line] = np.roll(glitch[y_line], shift, axis=0)
                    cv2.imshow("Don't Blink", glitch)
                    cv2.waitKey(25)

            # Full-screen enhanced death screen
            frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            frame = chaos.draw_death_screen_enhanced(
                frame, state, config, blink_face, leaderboard,
                death_time, timestamp,
            )
            # Reset window title
            cv2.setWindowTitle("Don't Blink", "Don't Blink - YOU DIED")
        else:
            # Resize camera frame to screen size
            frame = cv2.resize(frame, (screen_w, screen_h))

        cv2.imshow("Don't Blink", frame)

        # waitKey processes X11 events — GNOME kills the window if these
        # aren't processed frequently enough. Use at least 1ms.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if state.phase == Phase.WAITING:
                game.start(timestamp)
            elif state.phase == Phase.BLINKED:
                game.restart(timestamp)
                game.start(timestamp)
                # Reset chaos state
                fake_crash = chaos.FakeCrash()
                notification_text = None
                next_notification = timestamp + 8.0
                blink_face = None
                cv2.setWindowTitle("Don't Blink", "Don't Blink")
        elif key == ord("r"):
            game.restart(timestamp)
            fake_crash = chaos.FakeCrash()
            notification_text = None
            next_notification = timestamp + 8.0
            blink_face = None
            cv2.setWindowTitle("Don't Blink", "Don't Blink")

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()


def _apply_distraction(frame: np.ndarray, kind: str, rng: random.Random) -> np.ndarray:
    """Apply a visual distraction effect to the frame. Maximum chaos."""
    h, w = frame.shape[:2]

    match kind:
        case "flash_white":
            overlay = np.ones_like(frame) * 255
            frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

        case "flash_red":
            overlay = np.zeros_like(frame)
            overlay[:, :, 2] = 255
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        case "flash_green":
            overlay = np.zeros_like(frame)
            overlay[:, :, 1] = 255
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        case "shake":
            dx = rng.randint(-40, 40)
            dy = rng.randint(-40, 40)
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, m, (w, h))

        case "jumpscare_text":
            texts = ["BOO!", "BLINK!", "DON'T!", "BEHIND YOU",
                     "LOOK OUT!", "RUN", "IT SEES YOU", "WAKE UP",
                     ":)", "ERROR", "404", "GAME OVER", "YOU DIED",
                     "ALT+F4", "VIRUS DETECTED", "DELETING...",
                     "I CAN SEE YOU", "HELP", "NO ESCAPE"]
            text = rng.choice(texts)
            scale = rng.uniform(2.5, 5.0)
            x = rng.randint(0, max(1, w // 2))
            y = rng.randint(h // 4, 3 * h // 4)
            color = (rng.randint(0, 50), rng.randint(0, 50), rng.randint(200, 255))
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, rng.randint(3, 6))

        case "static":
            noise = np.random.randint(0, 255, frame.shape, dtype=np.uint8)
            frame = cv2.addWeighted(noise, 0.6, frame, 0.4, 0)

        case "invert":
            frame = cv2.bitwise_not(frame)

        case "zoom":
            center_x, center_y = w // 2, h // 2
            crop_size = int(min(w, h) * 0.2)
            x1 = max(0, center_x - crop_size)
            y1 = max(0, center_y - crop_size)
            x2 = min(w, center_x + crop_size)
            y2 = min(h, center_y + crop_size)
            if x2 - x1 > 10 and y2 - y1 > 10:
                cropped = frame[y1:y2, x1:x2].copy()
                frame = cv2.resize(cropped, (w, h))

        case "strobe":
            color = [rng.randint(0, 255) for _ in range(3)]
            overlay = np.full_like(frame, color, dtype=np.uint8)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        case "mirror":
            frame = cv2.flip(frame, 1)

        case "pixelate":
            small = cv2.resize(frame, (max(1, w // 20), max(1, h // 20)),
                               interpolation=cv2.INTER_LINEAR)
            frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        case "blackout":
            frame[:] = 0

        case "face_warp":
            k1 = rng.uniform(0.4, 1.0) * rng.choice([-1, 1])
            cam_matrix = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.array([k1, 0, 0, 0], dtype=np.float32)
            frame = cv2.undistort(frame, cam_matrix, dist_coeffs)

        case "split":
            mid = h // 2
            shift = rng.randint(30, 80) * rng.choice([-1, 1])
            top = np.roll(frame[:mid], shift, axis=1)
            frame[:mid] = top
            # Also shift bottom the other way
            bottom = np.roll(frame[mid:], -shift, axis=1)
            frame[mid:] = bottom

        case "red_eye":
            frame[:, :, 0] = 0
            frame[:, :, 1] = 0
            frame[:, :, 2] = np.clip(frame[:, :, 2].astype(int) + 100, 0, 255).astype(np.uint8)

        case "tilt":
            angle = rng.uniform(-25, 25)
            center = (w // 2, h // 2)
            rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            frame = cv2.warpAffine(frame, rot, (w, h))

        case "double_vision":
            shift_x = rng.randint(10, 30)
            shift_y = rng.randint(-10, 10)
            ghost = np.roll(np.roll(frame, shift_x, axis=1), shift_y, axis=0)
            frame = cv2.addWeighted(frame, 0.6, ghost, 0.4, 0)

        case "scanlines":
            for y in range(0, h, 3):
                frame[y, :] = frame[y, :] // 3

        case "glitch_color":
            # Randomly swap channels
            channels = list(cv2.split(frame))
            rng.shuffle(channels)
            frame = cv2.merge(channels)

        case "shrink":
            small_w, small_h = w // 3, h // 3
            small = cv2.resize(frame, (small_w, small_h))
            frame[:] = 0
            x_off = (w - small_w) // 2
            y_off = (h - small_h) // 2
            frame[y_off:y_off + small_h, x_off:x_off + small_w] = small

        case "pulse":
            intensity = rng.uniform(1.5, 2.5)
            frame = np.clip(frame.astype(np.float32) * intensity, 0, 255).astype(np.uint8)

    return frame


def _apply_corruption(frame: np.ndarray, intensity: float, rng: random.Random) -> np.ndarray:
    """Apply permanent screen corruption that builds over time.

    intensity: 0.0 to 1.0 (grows as the game goes on)
    Uses vectorized numpy ops to stay fast.
    """
    h, w = frame.shape[:2]
    if intensity <= 0:
        return frame

    # Random dead pixels (vectorized)
    num_pixels = int(intensity * 300)
    if num_pixels > 0:
        ys = np.random.randint(0, h, num_pixels)
        xs = np.random.randint(0, w, num_pixels)
        colors = np.random.randint(0, 255, (num_pixels, 3), dtype=np.uint8)
        frame[ys, xs] = colors

    # Horizontal line glitches
    num_lines = int(intensity * 3)
    for _ in range(num_lines):
        y = rng.randint(0, h - 1)
        shift = rng.randint(-15, 15)
        frame[y] = np.roll(frame[y], shift, axis=0)

    return frame


def _draw_panic_indicator(frame: np.ndarray, ear: float, config: GameConfig) -> np.ndarray:
    """Draw a panic indicator when eyes are getting close to blinking."""
    h, w = frame.shape[:2]

    if ear < config.ear_panic_threshold and ear >= config.ear_blink_threshold:
        # Eyes getting tired — red border vignette
        closeness = 1.0 - (ear - config.ear_blink_threshold) / (
            config.ear_panic_threshold - config.ear_blink_threshold
        )
        border_size = int(20 * closeness)
        color_intensity = int(200 * closeness)

        # Red border
        frame[:border_size, :, 2] = color_intensity
        frame[-border_size:, :, 2] = color_intensity
        frame[:, :border_size, 2] = color_intensity
        frame[:, -border_size:, 2] = color_intensity

        # Warning text if really close
        if closeness > 0.6:
            _draw_centered_text(frame, "YOUR EYES...", (w // 2, 30),
                                0.6, (0, 0, int(255 * closeness)), 2)

    return frame


def _draw_hud(frame: np.ndarray, state, config: GameConfig, rng: random.Random) -> np.ndarray:
    """Draw the HUD overlay — increasingly unhinged."""
    h, w = frame.shape[:2]

    if state.phase == Phase.WAITING:
        # Big centered message
        _draw_centered_text(frame, "DON'T BLINK", (w // 2, h // 3),
                            2.0, (255, 255, 255), 3)
        _draw_centered_text(frame, "Press SPACE to start", (w // 2, h // 2),
                            0.8, (200, 200, 200), 2)
        _draw_centered_text(frame, "...if you dare", (w // 2, h // 2 + 40),
                            0.6, (100, 100, 100), 1)
        # Epilepsy warning
        _draw_centered_text(frame, "WARNING: FLASHING LIGHTS / STROBES", (w // 2, h - 50),
                            0.5, (0, 0, 255), 2)
        _draw_centered_text(frame, "Not suitable for photosensitive epilepsy", (w // 2, h - 25),
                            0.4, (0, 0, 200), 1)

    elif state.phase == Phase.COUNTDOWN:
        # Big countdown number
        _draw_centered_text(frame, state.message, (w // 2, h // 2),
                            5.0, (0, 255, 255), 6)

    elif state.phase == Phase.PLAYING:
        # Timer in top-right — grows more red as time goes on
        t = state.survival_time
        green = max(0, int(255 - t * 8))
        red = min(255, int(t * 15))
        timer_color = (0, green, red) if t < 10 else (0, 0, 255)

        timer_text = chaos.get_lying_timer(t, rng)
        cv2.putText(frame, timer_text, (w - 180, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, timer_color, 3)

        # High score
        if state.high_score > 0:
            cv2.putText(frame, f"Best: {state.high_score:.1f}s", (w - 180, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Taunts — show one every few seconds, getting more frequent
        taunt_interval = max(2.0, 6.0 - t * 0.3)
        if int(t * 10) % int(taunt_interval * 10) < 2:
            taunt = rng.choice(config.taunts)
            taunt_x = rng.randint(10, max(11, w - 300))
            taunt_y = rng.randint(h // 2, h - 30)
            # Semi-transparent by drawing dark behind it
            cv2.putText(frame, taunt, (taunt_x + 1, taunt_y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, taunt, (taunt_x, taunt_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 2)

        # Vignette effect that gets darker over time
        if t > 5:
            darkness = min(0.4, (t - 5) * 0.02)
            overlay = np.zeros_like(frame)
            frame = cv2.addWeighted(frame, 1 - darkness * 0.3, overlay, darkness * 0.3, 0)

    elif state.phase == Phase.NO_FACE:
        _draw_centered_text(frame, "WHERE'D YOU GO?!", (w // 2, h // 2 - 20),
                            1.5, (0, 0, 255), 3)
        _draw_centered_text(frame, "COWARD", (w // 2, h // 2 + 30),
                            1.0, (0, 0, 200), 2)

    return frame


def _draw_centered_text(
    frame: np.ndarray,
    text: str,
    center: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw text centered at a point."""
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = center[0] - size[0] // 2
    y = center[1] + size[1] // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


if __name__ == "__main__":
    run()
