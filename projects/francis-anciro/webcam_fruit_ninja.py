"""Webcam Fruit Ninja — a contactless slicing game controlled by your fingertips.

Virtual fruits launch upward from the bottom of the screen and arc back down under
gravity. Slice them with your index fingers to score points. Miss too many and it's
game over.

Usage (from the repo root, venv active):
    python projects/francis-anciro/webcam_fruit_ninja.py

Controls:
    - Move your index fingers (landmark 8) to slice fruits.
    - Press 'r' to restart after game over.
    - Press 'q' to quit.

Dependencies:
    pip install opencv-python mediapipe pygame numpy
"""

from __future__ import annotations

import math
import random
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

# Gravity and physics (expressed as fractions of frame height per frame^2)
GRAVITY = 0.28  # base pixels/frame^2 downward acceleration (lower = floatier arcs)
LAUNCH_VY_MIN = -26.0  # base minimum upward velocity (negative = up, stronger launch)
LAUNCH_VY_MAX = -21.0  # base maximum upward velocity
LAUNCH_VX_RANGE = 4.0  # horizontal velocity spread (+/-)

# Difficulty progression — scales with score
SCORE_PER_LEVEL = 5  # points per difficulty level (each slice = 10pts, so every ~half slice)
VELOCITY_INCREMENT = 0.8  # extra upward speed added per level (makes launches faster)
GRAVITY_INCREMENT = 0.03  # extra gravity per level (makes falls quicker)
SPAWN_INTERVAL_REDUCTION = 0.04  # seconds shaved off spawn cooldown per level
FRUITS_PER_WAVE_BASE = 1  # starting number of fruits per spawn wave
FRUITS_PER_WAVE_INCREMENT = 1  # extra fruit in wave every N levels
WAVE_LEVEL_STEP = 3  # how many levels before an extra fruit is added to waves

# Caps to keep the game stable and playable at high scores
MAX_LAUNCH_VY = -40.0  # fastest upward launch allowed (more negative = faster up)
MAX_GRAVITY = 0.7  # maximum gravity so physics stays stable
MIN_SPAWN_INTERVAL = 0.2  # fastest allowed spawn cooldown (seconds)
MAX_FRUITS_PER_WAVE = 4  # never spawn more than this at once

# Fruit properties
FRUIT_RADIUS_MIN_RATIO = 0.03  # min radius as fraction of frame width
FRUIT_RADIUS_MAX_RATIO = 0.05  # max radius as fraction of frame width
SPAWN_INTERVAL_BASE = 0.6  # base seconds between fruit spawns
MAX_FRUITS_ON_SCREEN = 5  # absolute cap on active fruits

# Lives and scoring
STARTING_LIVES = 3
POINTS_PER_SLICE = 10

# Slash trail
TRAIL_LENGTH = 8  # how many past positions to keep for the trail line
SLASH_SPEED_THRESHOLD = 15.0  # minimum finger movement (px) to count as a slash

# Particle effects
PARTICLE_COUNT = 12  # particles per slice
PARTICLE_LIFETIME = 0.4  # seconds
PARTICLE_SPEED = 8.0  # initial outward speed

# Split halves
HALF_LIFETIME = 0.5  # seconds the halves stay visible
HALF_HORIZONTAL_SPEED = 6.0  # how fast halves fly apart

# Audio
SAMPLE_RATE = 44100

# Colours (BGR for OpenCV)
FRUIT_COLOURS = [
    (0, 100, 255),   # orange
    (0, 200, 0),     # green
    (0, 0, 220),     # red
    (200, 0, 200),   # magenta
    (0, 220, 220),   # yellow
    (220, 100, 0),   # teal
]
COLOUR_TRAIL = (200, 255, 200)  # slash trail
COLOUR_HUD = (255, 255, 255)  # HUD text
COLOUR_GAME_OVER = (0, 0, 255)  # game over text


# ==========================================================================
# AUDIO — synthesize sound effects with pygame
# ==========================================================================


def init_audio() -> None:
    """Initialize pygame mixer for low-latency playback."""
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()


def make_whoosh_sound() -> pygame.mixer.Sound:
    """Generate a short whoosh (filtered white noise with fast fade)."""
    duration_ms = 120
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    noise = np.random.uniform(-1, 1, num_samples)
    # Apply a fast exponential decay envelope
    envelope = np.exp(-np.linspace(0, 6, num_samples))
    wave = (noise * envelope * 32767 * 0.3).astype(np.int16)
    stereo = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo)


def make_slice_sound() -> pygame.mixer.Sound:
    """Generate a splat/slice sound (short burst with descending tone)."""
    duration_ms = 180
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    # Descending frequency chirp mixed with noise
    freq = np.linspace(800, 200, num_samples)
    phase = np.cumsum(2 * math.pi * freq / SAMPLE_RATE)
    tone = np.sin(phase)
    noise = np.random.uniform(-0.3, 0.3, num_samples)
    envelope = np.linspace(1.0, 0.0, num_samples) ** 2
    wave = ((tone + noise) * envelope * 32767 * 0.4).astype(np.int16)
    stereo = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo)


# ==========================================================================
# GAME ENTITIES
# ==========================================================================


class Fruit:
    """A launchable fruit that arcs through the screen under gravity."""

    def __init__(self, frame_w: int, frame_h: int, launch_vy_min: float, launch_vy_max: float, gravity: float):
        self.radius = random.randint(
            int(frame_w * FRUIT_RADIUS_MIN_RATIO),
            int(frame_w * FRUIT_RADIUS_MAX_RATIO),
        )
        # Spawn along the bottom edge at a random x
        margin = self.radius + int(frame_w * 0.05)
        self.x = float(random.randint(margin, frame_w - margin))
        self.y = float(frame_h + self.radius)  # just below the visible area

        # Velocity: upward with some horizontal spread (scaled by difficulty)
        self.vx = random.uniform(-LAUNCH_VX_RANGE, LAUNCH_VX_RANGE)
        self.vy = random.uniform(launch_vy_min, launch_vy_max)
        self.gravity = gravity

        self.colour = random.choice(FRUIT_COLOURS)
        self.alive = True
        self.sliced = False

    def update(self) -> None:
        """Apply velocity and per-fruit gravity."""
        self.vy += self.gravity
        self.x += self.vx
        self.y += self.vy

    def is_offscreen(self, frame_h: int) -> bool:
        """True if the fruit has fallen below the screen after going up."""
        return self.y > frame_h + self.radius * 2 and self.vy > 0

    def draw(self, frame: np.ndarray) -> None:
        """Draw the fruit as a filled circle."""
        if not self.alive:
            return
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.colour, -1)
        # Highlight
        cv2.circle(
            frame, (int(self.x - self.radius * 0.3), int(self.y - self.radius * 0.3)),
            max(2, self.radius // 4), (255, 255, 255), -1,
        )


class HalfFruit:
    """A split half that flies away after slicing."""

    def __init__(self, x: float, y: float, radius: int, colour: tuple, direction: int):
        self.x = x
        self.y = y
        self.radius = radius
        self.colour = colour
        self.direction = direction  # -1 = left, +1 = right
        self.vx = HALF_HORIZONTAL_SPEED * direction
        self.vy = random.uniform(-3, -1)
        self.spawn_time = time.time()
        self.alive = True

    def update(self) -> None:
        """Move and apply gravity, expire after lifetime."""
        self.vy += GRAVITY * 0.5
        self.x += self.vx
        self.y += self.vy
        if time.time() - self.spawn_time > HALF_LIFETIME:
            self.alive = False

    def draw(self, frame: np.ndarray) -> None:
        """Draw as a half-circle (using ellipse with limited angle)."""
        if not self.alive:
            return
        angle = 0 if self.direction > 0 else 180
        cv2.ellipse(
            frame, (int(self.x), int(self.y)), (self.radius, self.radius),
            0, angle, angle + 180, self.colour, -1,
        )


class Particle:
    """A small splash particle emitted on slice."""

    def __init__(self, x: float, y: float, colour: tuple):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(PARTICLE_SPEED * 0.5, PARTICLE_SPEED)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.colour = colour
        self.spawn_time = time.time()
        self.alive = True

    def update(self) -> None:
        """Move outward, expire after lifetime."""
        self.x += self.vx
        self.y += self.vy
        self.vy += GRAVITY * 0.3  # slight gravity on particles
        if time.time() - self.spawn_time > PARTICLE_LIFETIME:
            self.alive = False

    def draw(self, frame: np.ndarray) -> None:
        """Draw as a small filled circle."""
        if not self.alive:
            return
        alpha = 1.0 - (time.time() - self.spawn_time) / PARTICLE_LIFETIME
        radius = max(1, int(4 * alpha))
        cv2.circle(frame, (int(self.x), int(self.y)), radius, self.colour, -1)


# ==========================================================================
# GAME STATE
# ==========================================================================


class Game:
    """Manages all game entities, scoring, state, and difficulty progression."""

    def __init__(self):
        self.fruits: list[Fruit] = []
        self.halves: list[HalfFruit] = []
        self.particles: list[Particle] = []
        self.score: int = 0
        self.lives: int = STARTING_LIVES
        self.last_spawn_time: float = time.time()
        self.game_over: bool = False

    # ------------------------------------------------------------------
    # Difficulty progression properties
    # ------------------------------------------------------------------

    @property
    def difficulty_level(self) -> int:
        """Current difficulty level (1-based), increases every SCORE_PER_LEVEL points."""
        return 1 + (self.score // SCORE_PER_LEVEL)

    @property
    def current_gravity(self) -> float:
        """Gravity increases with level, capped at MAX_GRAVITY."""
        g = GRAVITY + (self.difficulty_level - 1) * GRAVITY_INCREMENT
        return min(g, MAX_GRAVITY)

    @property
    def current_launch_vy_min(self) -> float:
        """Min upward velocity gets stronger (more negative) with level, capped."""
        v = LAUNCH_VY_MIN - (self.difficulty_level - 1) * VELOCITY_INCREMENT
        return max(v, MAX_LAUNCH_VY)  # max() because more negative = faster

    @property
    def current_launch_vy_max(self) -> float:
        """Max upward velocity gets stronger with level, capped."""
        v = LAUNCH_VY_MAX - (self.difficulty_level - 1) * VELOCITY_INCREMENT
        return max(v, MAX_LAUNCH_VY)

    @property
    def current_spawn_interval(self) -> float:
        """Spawn cooldown decreases with level, capped at MIN_SPAWN_INTERVAL."""
        interval = SPAWN_INTERVAL_BASE - (self.difficulty_level - 1) * SPAWN_INTERVAL_REDUCTION
        return max(interval, MIN_SPAWN_INTERVAL)

    @property
    def current_wave_size(self) -> int:
        """Number of fruits spawned per wave, increases every WAVE_LEVEL_STEP levels."""
        size = FRUITS_PER_WAVE_BASE + ((self.difficulty_level - 1) // WAVE_LEVEL_STEP) * FRUITS_PER_WAVE_INCREMENT
        return min(size, MAX_FRUITS_PER_WAVE)

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restart the game."""
        self.fruits.clear()
        self.halves.clear()
        self.particles.clear()
        self.score = 0
        self.lives = STARTING_LIVES
        self.last_spawn_time = time.time()
        self.game_over = False

    def maybe_spawn(self, frame_w: int, frame_h: int) -> None:
        """Spawn a wave of fruits at dynamic intervals based on difficulty."""
        if self.game_over:
            return
        now = time.time()
        if now - self.last_spawn_time >= self.current_spawn_interval:
            wave_count = self.current_wave_size
            for _ in range(wave_count):
                if len(self.fruits) >= MAX_FRUITS_ON_SCREEN:
                    break
                self.fruits.append(
                    Fruit(
                        frame_w, frame_h,
                        self.current_launch_vy_min,
                        self.current_launch_vy_max,
                        self.current_gravity,
                    )
                )
            self.last_spawn_time = now

    def update(self, frame_h: int) -> None:
        """Advance all entities, remove dead ones, check for missed fruits."""
        if self.game_over:
            return

        for fruit in self.fruits:
            fruit.update()

        # Check for missed fruits (fell off bottom without being sliced)
        for fruit in self.fruits:
            if fruit.alive and fruit.is_offscreen(frame_h):
                fruit.alive = False
                self.lives -= 1
                if self.lives <= 0:
                    self.game_over = True

        self.fruits = [f for f in self.fruits if f.alive]

        for half in self.halves:
            half.update()
        self.halves = [h for h in self.halves if h.alive]

        for particle in self.particles:
            particle.update()
        self.particles = [p for p in self.particles if p.alive]

    def try_slice(
        self, prev: tuple[int, int], curr: tuple[int, int],
        slice_sound: pygame.mixer.Sound,
    ) -> bool:
        """Check if the finger movement line slices any fruit. Returns True if sliced."""
        if self.game_over:
            return False

        sliced_any = False
        for fruit in self.fruits:
            if not fruit.alive:
                continue
            if line_intersects_circle(prev, curr, (fruit.x, fruit.y), fruit.radius):
                fruit.alive = False
                fruit.sliced = True
                self.score += POINTS_PER_SLICE
                sliced_any = True

                # Spawn halves
                self.halves.append(
                    HalfFruit(fruit.x, fruit.y, fruit.radius, fruit.colour, -1)
                )
                self.halves.append(
                    HalfFruit(fruit.x, fruit.y, fruit.radius, fruit.colour, +1)
                )

                # Spawn particles
                for _ in range(PARTICLE_COUNT):
                    self.particles.append(Particle(fruit.x, fruit.y, fruit.colour))

                slice_sound.play()

        return sliced_any


# ==========================================================================
# GEOMETRY — line-circle intersection for slicing detection
# ==========================================================================


def line_intersects_circle(
    p1: tuple[float, float], p2: tuple[float, float],
    center: tuple[float, float], radius: float,
) -> bool:
    """Check if the line segment p1->p2 intersects or passes through a circle."""
    # Vector math: find closest point on segment to circle center
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    fx = p1[0] - center[0]
    fy = p1[1] - center[1]

    a = dx * dx + dy * dy
    if a < 1e-6:
        # Segment has zero length, just check point distance
        dist_sq = fx * fx + fy * fy
        return dist_sq <= radius * radius

    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False

    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)

    # Check if either intersection point is within the segment [0, 1]
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    # Check if the segment is entirely inside the circle
    if t1 < 0 and t2 > 1:
        return True

    return False


# ==========================================================================
# DRAWING — HUD and effects
# ==========================================================================


def draw_hud(frame: np.ndarray, score: int, lives: int, difficulty_level: int, frame_w: int, frame_h: int) -> None:
    """Draw score, difficulty level (top-left) and lives (top-right), scaled to frame size."""
    font_scale = frame_h / 500.0
    thickness = max(1, int(font_scale * 2))

    # Score — top left
    cv2.putText(
        frame, f"Score: {score}", (15, int(frame_h * 0.06)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOUR_HUD, thickness, cv2.LINE_AA,
    )

    # Difficulty level — below score
    cv2.putText(
        frame, f"Speed: Level {difficulty_level}", (15, int(frame_h * 0.12)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (180, 180, 255), max(1, thickness - 1), cv2.LINE_AA,
    )

    # Lives — top right (draw heart symbols)
    lives_text = f"Lives: {'<3 ' * lives}".strip()
    text_size = cv2.getTextSize(lives_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, thickness)[0]
    cv2.putText(
        frame, lives_text, (frame_w - text_size[0] - 15, int(frame_h * 0.06)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 100, 255), thickness, cv2.LINE_AA,
    )


def draw_trail(frame: np.ndarray, trail: list[tuple[int, int]]) -> None:
    """Draw a fading slash trail from finger history."""
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        thickness = max(1, int(alpha * 6))
        colour = (
            int(COLOUR_TRAIL[0] * alpha),
            int(COLOUR_TRAIL[1] * alpha),
            int(COLOUR_TRAIL[2] * alpha),
        )
        cv2.line(frame, trail[i - 1], trail[i], colour, thickness, cv2.LINE_AA)


def draw_game_over(frame: np.ndarray, score: int, frame_w: int, frame_h: int) -> None:
    """Draw a semi-transparent game over overlay."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font_scale = frame_h / 300.0
    thickness = max(2, int(font_scale * 3))

    # GAME OVER
    text = "GAME OVER"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    tx = (frame_w - text_size[0]) // 2
    ty = (frame_h - text_size[1]) // 2
    cv2.putText(
        frame, text, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOUR_GAME_OVER, thickness, cv2.LINE_AA,
    )

    # Final score
    score_text = f"Final Score: {score}"
    st_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, thickness - 1)[0]
    cv2.putText(
        frame, score_text, ((frame_w - st_size[0]) // 2, ty + int(frame_h * 0.08)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, COLOUR_HUD, thickness - 1, cv2.LINE_AA,
    )

    # Restart prompt
    prompt = "Press 'R' to restart  |  'Q' to quit"
    pt_size = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.4, thickness - 1)[0]
    cv2.putText(
        frame, prompt, ((frame_w - pt_size[0]) // 2, ty + int(frame_h * 0.15)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.4, (180, 180, 180), max(1, thickness - 1), cv2.LINE_AA,
    )


# ==========================================================================
# HAND TRACKING
# ==========================================================================


def get_index_fingertips(results: Any, frame_w: int, frame_h: int) -> list[tuple[int, int]]:
    """Extract index finger tip (landmark 8) pixel coords from all detected hands."""
    points: list[tuple[int, int]] = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            points.append((px, py))
    return points


# ==========================================================================
# MAIN LOOP
# ==========================================================================

WINDOW_NAME = "Webcam Fruit Ninja"


def main() -> None:
    # Initialise audio
    init_audio()
    whoosh_sound = make_whoosh_sound()
    slice_sound = make_slice_sound()

    # Initialise MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
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

    game = Game()

    # Per-hand trailing positions (keyed by hand index for simplicity, max 2 hands)
    trails: list[list[tuple[int, int]]] = [[], []]
    prev_positions: list[tuple[int, int] | None] = [None, None]
    last_whoosh_time: float = 0.0

    print("Webcam Fruit Ninja started! Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Mirror for intuitive interaction
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # Hand tracking
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            fingertips = get_index_fingertips(results, frame_w, frame_h)

            # Update trails and check slicing
            for i in range(2):  # up to 2 hands
                if i < len(fingertips):
                    curr = fingertips[i]
                    prev = prev_positions[i]

                    # Update trail
                    trails[i].append(curr)
                    if len(trails[i]) > TRAIL_LENGTH:
                        trails[i] = trails[i][-TRAIL_LENGTH:]

                    # Check for fruit slicing every frame using raw fingertip position
                    if prev is not None:
                        # Slice check: line from prev to curr intersects any fruit
                        hit = game.try_slice(prev, curr, slice_sound)

                        # Play whoosh if moving fast (throttled, cosmetic only)
                        dx = curr[0] - prev[0]
                        dy = curr[1] - prev[1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if not hit and dist > SLASH_SPEED_THRESHOLD:
                            now = time.time()
                            if now - last_whoosh_time > 0.15:
                                whoosh_sound.play()
                                last_whoosh_time = now

                    prev_positions[i] = curr
                else:
                    # Hand not detected, clear trail
                    trails[i].clear()
                    prev_positions[i] = None

            # Game logic
            game.maybe_spawn(frame_w, frame_h)
            game.update(frame_h)

            # Draw entities
            for fruit in game.fruits:
                fruit.draw(frame)
            for half in game.halves:
                half.draw(frame)
            for particle in game.particles:
                particle.draw(frame)

            # Draw trails
            for trail in trails:
                draw_trail(frame, trail)

            # Draw fingertip indicators
            for tip in fingertips:
                cv2.circle(frame, tip, max(5, int(frame_w * 0.008)), (0, 255, 255), 2)

            # Draw HUD
            draw_hud(frame, game.score, game.lives, game.difficulty_level, frame_w, frame_h)

            # Game over overlay
            if game.game_over:
                draw_game_over(frame, game.score, frame_w, frame_h)

            cv2.imshow(WINDOW_NAME, frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r") and game.game_over:
                game.reset()
                trails = [[], []]
                prev_positions = [None, None]

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        pygame.mixer.quit()
        print(f"\nSession ended. Final score: {game.score}")


if __name__ == "__main__":
    main()
