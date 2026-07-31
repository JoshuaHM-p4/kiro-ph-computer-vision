"""Webcam Brick Breaker — a hand-tracked Breakout-style game.

Control a paddle with your index finger to bounce a ball and destroy bricks.

Usage (from the repo root, venv active):
    python projects/francis-anciro/webcam_brick_breaker.py

Controls:
    - Move your index finger (landmark 8) left/right to position the paddle.
    - Press SPACE to launch the ball (or move your finger when ball is on paddle).
    - Press 'r' to restart after game over or victory.
    - Press 'q' to quit.

Dependencies:
    pip install opencv-python mediapipe pygame numpy
"""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np

try:
    import pygame
except ImportError:
    raise SystemExit(
        "pygame is required for audio. Install it with:\n"
        "    pip install pygame"
    )

import mediapipe as mp

# ==========================================================================
# CONFIGURATION — tweak these to change gameplay feel
# ==========================================================================

# Brick grid
BRICK_ROWS = 4
BRICK_COLS = 8
BRICK_TOP_OFFSET_RATIO = 0.08  # top margin before bricks start (fraction of height)
BRICK_HEIGHT_RATIO = 0.04  # brick height as fraction of frame height
BRICK_GAP = 4  # pixels between bricks
BRICK_SIDE_MARGIN_RATIO = 0.03  # horizontal margin on each side

# Paddle
PADDLE_WIDTH_RATIO = 0.12  # paddle width as fraction of frame width
PADDLE_HEIGHT_RATIO = 0.02  # paddle height as fraction of frame height
PADDLE_Y_RATIO = 0.90  # vertical position of paddle (fraction of frame height)

# Ball
BALL_RADIUS_RATIO = 0.012  # ball radius as fraction of frame width
BALL_SPEED_RATIO = 0.01  # base ball speed as fraction of frame height per frame
BALL_MAX_ANGLE_FACTOR = 0.7  # how much paddle offset affects dx (0 = straight, 1 = extreme)

# Ball speed progression
BALL_SPEED_INCREMENT = 0.2  # extra speed per level (+0.2 px/frame per 10 points)
BALL_SCORE_PER_LEVEL = 10  # points before speed increases
BALL_MAX_SPEED = 20.0  # absolute speed cap in pixels per frame

# Lives and scoring
STARTING_LIVES = 3
POINTS_PER_BRICK = 10

# Audio
SAMPLE_RATE = 44100

# Colours (BGR)
BRICK_COLOURS = [
    (0, 0, 220),     # red row
    (0, 140, 255),   # orange row
    (0, 220, 220),   # yellow row
    (0, 200, 0),     # green row
    (220, 100, 0),   # teal row
    (200, 0, 200),   # magenta row
    (255, 100, 100), # light blue row
    (100, 100, 255), # pink row
]
COLOUR_PADDLE = (255, 255, 255)
COLOUR_BALL = (0, 255, 255)
COLOUR_HUD = (255, 255, 255)
COLOUR_GAME_OVER = (0, 0, 255)
COLOUR_VICTORY = (0, 255, 0)


# ==========================================================================
# AUDIO — synthesize sound effects with pygame
# ==========================================================================


def init_audio() -> None:
    """Initialize pygame mixer for low-latency playback."""
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()


def make_bounce_sound() -> pygame.mixer.Sound:
    """Generate a clean short tone for wall/paddle bounces."""
    duration_ms = 60
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    envelope = np.linspace(1.0, 0.0, num_samples)
    wave = (np.sin(2 * math.pi * 440 * t) * envelope * 32767 * 0.4).astype(np.int16)
    stereo = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo)


def make_brick_sound() -> pygame.mixer.Sound:
    """Generate a distinct higher-pitched tone for brick breaking."""
    duration_ms = 100
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    # Two-tone chirp for a more satisfying break sound
    freq = np.linspace(880, 1200, num_samples)
    phase = np.cumsum(2 * math.pi * freq / SAMPLE_RATE)
    envelope = np.linspace(1.0, 0.0, num_samples) ** 1.5
    wave = (np.sin(phase) * envelope * 32767 * 0.4).astype(np.int16)
    stereo = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo)


# ==========================================================================
# GAME ENTITIES
# ==========================================================================


class Brick:
    """A single destructible brick."""

    def __init__(self, x: int, y: int, w: int, h: int, colour: tuple):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.colour = colour
        self.alive = True

    def draw(self, frame: np.ndarray) -> None:
        if not self.alive:
            return
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), self.colour, -1)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (40, 40, 40), 1)


class Ball:
    """The bouncing game ball."""

    def __init__(self, x: float, y: float, radius: int, speed: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        self.dx = 0.0
        self.dy = 0.0
        self.launched = False

    def launch(self) -> None:
        """Launch the ball upward with a slight random angle."""
        self.dx = self.speed * 0.3
        self.dy = -self.speed
        self.launched = True

    def update(self, frame_w: int, frame_h: int) -> str:
        """Move the ball and bounce off walls. Returns 'lost' if it falls off bottom."""
        if not self.launched:
            return "on_paddle"

        self.x += self.dx
        self.y += self.dy

        # Bounce off left/right walls
        if self.x - self.radius <= 0:
            self.x = self.radius
            self.dx = abs(self.dx)
            return "wall"
        if self.x + self.radius >= frame_w:
            self.x = frame_w - self.radius
            self.dx = -abs(self.dx)
            return "wall"

        # Bounce off top wall
        if self.y - self.radius <= 0:
            self.y = self.radius
            self.dy = abs(self.dy)
            return "wall"

        # Lost off bottom
        if self.y - self.radius > frame_h:
            return "lost"

        return "ok"

    def check_paddle(self, paddle_x: int, paddle_y: int, paddle_w: int, paddle_h: int) -> bool:
        """Check and handle paddle collision. Returns True if bounced."""
        if self.dy <= 0:
            return False  # only check when ball is moving downward

        # Ball bottom edge vs paddle top
        ball_bottom = self.y + self.radius
        if ball_bottom >= paddle_y and self.y < paddle_y + paddle_h:
            if paddle_x <= self.x <= paddle_x + paddle_w:
                # Bounce upward
                self.y = paddle_y - self.radius
                self.dy = -abs(self.dy)

                # Alter dx based on where the ball hit the paddle
                # Center of paddle = 0, edges = +/-1
                paddle_center = paddle_x + paddle_w / 2
                offset = (self.x - paddle_center) / (paddle_w / 2)
                self.dx = self.speed * offset * BALL_MAX_ANGLE_FACTOR

                return True
        return False

    def check_brick(self, brick: Brick) -> bool:
        """Check collision between ball and a brick. Returns True if hit."""
        if not brick.alive:
            return False

        # Find closest point on brick rectangle to ball center
        closest_x = max(brick.x, min(self.x, brick.x + brick.w))
        closest_y = max(brick.y, min(self.y, brick.y + brick.h))

        dist_x = self.x - closest_x
        dist_y = self.y - closest_y
        dist_sq = dist_x * dist_x + dist_y * dist_y

        if dist_sq <= self.radius * self.radius:
            brick.alive = False

            # Determine bounce direction based on which side was hit
            # Overlap from each side
            overlap_left = (self.x + self.radius) - brick.x
            overlap_right = (brick.x + brick.w) - (self.x - self.radius)
            overlap_top = (self.y + self.radius) - brick.y
            overlap_bottom = (brick.y + brick.h) - (self.y - self.radius)

            min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

            if min_overlap == overlap_left or min_overlap == overlap_right:
                self.dx = -self.dx
            else:
                self.dy = -self.dy

            return True
        return False

    def draw(self, frame: np.ndarray) -> None:
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, COLOUR_BALL, -1)


# ==========================================================================
# GAME STATE
# ==========================================================================


class Game:
    """Manages bricks, ball, paddle, scoring, and state."""

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.score = 0
        self.lives = STARTING_LIVES
        self.game_over = False
        self.victory = False

        # Paddle dimensions
        self.paddle_w = int(frame_w * PADDLE_WIDTH_RATIO)
        self.paddle_h = int(frame_h * PADDLE_HEIGHT_RATIO)
        self.paddle_y = int(frame_h * PADDLE_Y_RATIO)
        self.paddle_x = frame_w // 2 - self.paddle_w // 2

        # Ball
        self.ball_radius = max(5, int(frame_w * BALL_RADIUS_RATIO))
        self.ball_speed = max(3, int(frame_h * BALL_SPEED_RATIO))
        self.ball = self._make_ball()

        # Bricks
        self.bricks = self._make_bricks()

    def _make_ball(self) -> Ball:
        """Create a ball sitting on top of the paddle."""
        bx = self.paddle_x + self.paddle_w // 2
        by = self.paddle_y - self.ball_radius - 2
        return Ball(bx, by, self.ball_radius, self.ball_speed)

    def _make_bricks(self) -> list[Brick]:
        """Generate the brick grid."""
        bricks: list[Brick] = []
        margin = int(self.frame_w * BRICK_SIDE_MARGIN_RATIO)
        top_offset = int(self.frame_h * BRICK_TOP_OFFSET_RATIO)
        brick_h = int(self.frame_h * BRICK_HEIGHT_RATIO)

        usable_w = self.frame_w - 2 * margin - (BRICK_COLS - 1) * BRICK_GAP
        brick_w = usable_w // BRICK_COLS

        for row in range(BRICK_ROWS):
            colour = BRICK_COLOURS[row % len(BRICK_COLOURS)]
            for col in range(BRICK_COLS):
                x = margin + col * (brick_w + BRICK_GAP)
                y = top_offset + row * (brick_h + BRICK_GAP)
                bricks.append(Brick(x, y, brick_w, brick_h, colour))

        return bricks

    def reset(self) -> None:
        """Full game restart."""
        self.score = 0
        self.lives = STARTING_LIVES
        self.game_over = False
        self.victory = False
        self.ball = self._make_ball()
        self.bricks = self._make_bricks()

    def reset_ball(self) -> None:
        """Reset ball onto paddle after losing a life."""
        self.ball = self._make_ball()

    @property
    def current_ball_speed(self) -> float:
        """Dynamic ball speed: increases every BALL_SCORE_PER_LEVEL points, capped."""
        speed = self.ball_speed + (self.score // BALL_SCORE_PER_LEVEL) * BALL_SPEED_INCREMENT
        return min(speed, BALL_MAX_SPEED)

    @property
    def speed_level(self) -> int:
        """Current speed level for display (1-based)."""
        return 1 + (self.score // BALL_SCORE_PER_LEVEL)

    def _rescale_ball_velocity(self) -> None:
        """Rescale dx/dy to match current_ball_speed without changing direction."""
        if not self.ball.launched:
            return
        current_magnitude = math.sqrt(self.ball.dx ** 2 + self.ball.dy ** 2)
        if current_magnitude < 0.01:
            return
        target = self.current_ball_speed
        scale = target / current_magnitude
        self.ball.dx *= scale
        self.ball.dy *= scale
        self.ball.speed = target

    def set_paddle_x(self, finger_x: int) -> None:
        """Position paddle centered on finger x, clamped to frame bounds."""
        self.paddle_x = max(0, min(self.frame_w - self.paddle_w, finger_x - self.paddle_w // 2))

        # If ball is on paddle, keep it centered
        if not self.ball.launched:
            self.ball.x = self.paddle_x + self.paddle_w // 2
            self.ball.y = self.paddle_y - self.ball_radius - 2

    def update(self, bounce_sound: pygame.mixer.Sound, brick_sound: pygame.mixer.Sound) -> None:
        """Advance game state one frame."""
        if self.game_over or self.victory:
            return

        result = self.ball.update(self.frame_w, self.frame_h)

        if result == "wall":
            bounce_sound.play()
        elif result == "lost":
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                self.reset_ball()
            return

        # Paddle collision
        if self.ball.check_paddle(self.paddle_x, self.paddle_y, self.paddle_w, self.paddle_h):
            bounce_sound.play()

        # Brick collisions
        for brick in self.bricks:
            if self.ball.check_brick(brick):
                self.score += POINTS_PER_BRICK
                brick_sound.play()
                # Rescale ball velocity to match new speed level
                self._rescale_ball_velocity()
                break  # only one brick per frame to avoid tunneling

        # Check victory
        if all(not b.alive for b in self.bricks):
            self.victory = True


# ==========================================================================
# DRAWING
# ==========================================================================


def draw_bricks(frame: np.ndarray, bricks: list[Brick]) -> None:
    for brick in bricks:
        brick.draw(frame)


def draw_paddle(frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOUR_PADDLE, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)


def draw_hud(frame: np.ndarray, score: int, lives: int, speed_level: int, frame_w: int, frame_h: int) -> None:
    font_scale = frame_h / 500.0
    thickness = max(1, int(font_scale * 2))

    # Score — top left
    cv2.putText(
        frame, f"Score: {score}", (15, int(frame_h * 0.06)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOUR_HUD, thickness, cv2.LINE_AA,
    )

    # Speed level — below score
    cv2.putText(
        frame, f"Ball Speed: Level {speed_level}", (15, int(frame_h * 0.11)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (180, 180, 255),
        max(1, thickness - 1), cv2.LINE_AA,
    )

    # Lives — top right
    lives_text = f"Lives: {lives}"
    text_size = cv2.getTextSize(lives_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness)[0]
    cv2.putText(
        frame, lives_text, (frame_w - text_size[0] - 15, int(frame_h * 0.06)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 100, 255), thickness, cv2.LINE_AA,
    )

    # Launch hint if ball is on paddle
    cv2.putText(
        frame, "SPACE to launch", (frame_w // 2 - 100, frame_h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (150, 150, 150),
        max(1, thickness - 1), cv2.LINE_AA,
    )


def draw_game_over(frame: np.ndarray, score: int, frame_w: int, frame_h: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font_scale = frame_h / 300.0
    thickness = max(2, int(font_scale * 3))

    text = "GAME OVER"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    tx = (frame_w - text_size[0]) // 2
    ty = (frame_h - text_size[1]) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOUR_GAME_OVER, thickness, cv2.LINE_AA)

    score_text = f"Final Score: {score}"
    st_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, thickness - 1)[0]
    cv2.putText(
        frame, score_text, ((frame_w - st_size[0]) // 2, ty + int(frame_h * 0.08)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, COLOUR_HUD, thickness - 1, cv2.LINE_AA,
    )

    prompt = "Press 'R' to restart  |  'Q' to quit"
    pt_size = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.4, thickness - 1)[0]
    cv2.putText(
        frame, prompt, ((frame_w - pt_size[0]) // 2, ty + int(frame_h * 0.15)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.4, (180, 180, 180), max(1, thickness - 1), cv2.LINE_AA,
    )


def draw_victory(frame: np.ndarray, score: int, frame_w: int, frame_h: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    font_scale = frame_h / 300.0
    thickness = max(2, int(font_scale * 3))

    text = "YOU WIN!"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    tx = (frame_w - text_size[0]) // 2
    ty = (frame_h - text_size[1]) // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOUR_VICTORY, thickness, cv2.LINE_AA)

    score_text = f"Score: {score}"
    st_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, thickness - 1)[0]
    cv2.putText(
        frame, score_text, ((frame_w - st_size[0]) // 2, ty + int(frame_h * 0.08)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, COLOUR_HUD, thickness - 1, cv2.LINE_AA,
    )

    prompt = "Press 'R' to play again  |  'Q' to quit"
    pt_size = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.4, thickness - 1)[0]
    cv2.putText(
        frame, prompt, ((frame_w - pt_size[0]) // 2, ty + int(frame_h * 0.15)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.4, (180, 180, 180), max(1, thickness - 1), cv2.LINE_AA,
    )


def draw_fingertip(frame: np.ndarray, x: int, y: int, frame_w: int) -> None:
    """Draw a small indicator at the tracked fingertip."""
    radius = max(4, int(frame_w * 0.006))
    cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)


# ==========================================================================
# HAND TRACKING
# ==========================================================================


def get_index_finger_x(results: Any, frame_w: int, frame_h: int) -> tuple[int, int] | None:
    """Get (x, y) pixel position of index finger tip (landmark 8). Returns None if not detected."""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            return (px, py)
    return None


# ==========================================================================
# MAIN LOOP
# ==========================================================================

WINDOW_NAME = "Webcam Brick Breaker"


def main() -> None:
    # Initialise audio
    init_audio()
    bounce_sound = make_bounce_sound()
    brick_sound = make_brick_sound()

    # Initialise MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # Open webcam and request HD resolution
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera index or permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create a resizable window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    game: Game | None = None

    print("Webcam Brick Breaker started! Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Mirror for intuitive interaction
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # Initialize game on first frame (need actual dimensions)
            if game is None:
                game = Game(frame_w, frame_h)

            # Hand tracking
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            finger_pos = get_index_finger_x(results, frame_w, frame_h)

            # Update paddle position from finger
            if finger_pos is not None:
                game.set_paddle_x(finger_pos[0])

            # Update game physics
            game.update(bounce_sound, brick_sound)

            # Draw everything
            draw_bricks(frame, game.bricks)
            draw_paddle(frame, game.paddle_x, game.paddle_y, game.paddle_w, game.paddle_h)
            game.ball.draw(frame)
            draw_hud(frame, game.score, game.lives, game.speed_level, frame_w, frame_h)

            # Draw finger indicator
            if finger_pos is not None:
                draw_fingertip(frame, finger_pos[0], finger_pos[1], frame_w)

            # Overlays
            if game.game_over:
                draw_game_over(frame, game.score, frame_w, frame_h)
            elif game.victory:
                draw_victory(frame, game.score, frame_w, frame_h)

            cv2.imshow(WINDOW_NAME, frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r") and (game.game_over or game.victory):
                game.reset()
            elif key == ord(" ") and not game.ball.launched:
                game.ball.launch()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        pygame.mixer.quit()
        print(f"\nSession ended. Final score: {game.score if game else 0}")


if __name__ == "__main__":
    main()
