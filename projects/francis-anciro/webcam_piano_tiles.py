"""Webcam Piano Tiles — an interactive falling-tiles game controlled by your fingers.

Uses OpenCV for video capture and display, MediaPipe Hands for fingertip tracking,
and pygame for low-latency audio playback of piano notes.

Usage (from the repo root, venv active):
    python projects/francis-anciro/webcam_piano_tiles.py

Controls:
    - Move your index finger (landmark 8) or middle finger (landmark 12) into a
      lane's hit zone to strike a falling tile.
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

# Lane setup
NUM_LANES = 4
LANE_MARGIN_RATIO = 0.0625  # dead space on each side as fraction of frame width (40/640)

# Tile dimensions and speed
TILE_WIDTH_RATIO = 0.8  # tile width as fraction of lane width
TILE_HEIGHT_RATIO = 0.083  # tile height as fraction of frame height (~60/720)
TILE_SPEED_RATIO = 0.006  # pixels-per-frame as fraction of frame height (~4/720)
SPAWN_INTERVAL = 0.8  # seconds between new tile spawns

# Difficulty progression
BASE_SPEED = 10  # starting tile fall speed in pixels per frame
SPEED_INCREMENT = 2  # extra pixels per frame gained every SCORE_PER_LEVEL points
SCORE_PER_LEVEL = 10  # how many points before the next speed bump
MAX_SPEED = 25  # absolute ceiling so the game doesn't glitch out

# Hit zone: expressed as fractions of frame height
HIT_ZONE_TOP_RATIO = 0.86  # top of the hit zone (e.g. 0.86 * 720 ≈ 619)
HIT_ZONE_HEIGHT_RATIO = 0.083  # thickness as fraction of height (~60/720)

# Audio: piano notes (frequencies in Hz for lanes 0-3)
# C4, E4, G4, B4 — a Cmaj7 chord spread across the lanes
NOTE_FREQUENCIES = [262, 330, 392, 494]
NOTE_DURATION_MS = 200  # how long each note plays
SAMPLE_RATE = 44100

# Colours (BGR for OpenCV)
COLOUR_TILE = (50, 50, 50)  # dark tile
COLOUR_HIT_ZONE = (0, 200, 255)  # yellow-ish hit zone line
COLOUR_LANE_LINE = (100, 100, 100)  # subtle lane dividers
COLOUR_SCORE = (0, 255, 200)  # score text
COLOUR_FLASH = [
    (0, 255, 0),    # lane 0: green
    (255, 100, 0),  # lane 1: blue-ish
    (0, 100, 255),  # lane 2: orange
    (255, 0, 255),  # lane 3: magenta
]


# ==========================================================================
# LAYOUT — dynamically computed from the actual frame dimensions
# ==========================================================================


class Layout:
    """All positional values derived from actual frame width and height."""

    def __init__(self, frame_w: int, frame_h: int):
        self.w = frame_w
        self.h = frame_h

        # Lane boundaries
        margin = int(frame_w * LANE_MARGIN_RATIO)
        usable = frame_w - 2 * margin
        lane_w = usable // NUM_LANES
        self.lane_bounds: list[tuple[int, int]] = []
        for i in range(NUM_LANES):
            left = margin + i * lane_w
            right = left + lane_w
            self.lane_bounds.append((left, right))

        # Tile dimensions
        self.tile_height = int(frame_h * TILE_HEIGHT_RATIO)
        self.tile_speed = max(1, int(frame_h * TILE_SPEED_RATIO))

        # Hit zone
        self.hit_zone_y = int(frame_h * HIT_ZONE_TOP_RATIO)
        self.hit_zone_h = int(frame_h * HIT_ZONE_HEIGHT_RATIO)
        self.hit_zone_bottom = self.hit_zone_y + self.hit_zone_h

        # Font scale relative to frame height (base: 1.2 at 480px)
        self.font_scale = frame_h / 400.0
        self.font_thickness = max(1, int(self.font_scale * 2.5))


# ==========================================================================
# AUDIO — generate piano tones with pygame
# ==========================================================================


def init_audio() -> None:
    """Initialize the pygame mixer for low-latency playback."""
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()


def generate_tone(frequency: float, duration_ms: int = NOTE_DURATION_MS) -> pygame.mixer.Sound:
    """Synthesize a sine-wave tone at the given frequency."""
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    # Sine wave with an amplitude envelope (fade out) so it sounds cleaner
    envelope = np.linspace(1.0, 0.0, num_samples)
    wave = (np.sin(2 * math.pi * frequency * t) * envelope * 32767 * 0.6).astype(np.int16)
    # Convert mono to 2D stereo array (required when mixer is initialized in stereo mode)
    stereo_wave = np.column_stack((wave, wave))
    sound = pygame.sndarray.make_sound(stereo_wave)
    return sound


def build_sounds() -> list[pygame.mixer.Sound]:
    """Pre-generate a Sound object for each lane's note."""
    return [generate_tone(freq) for freq in NOTE_FREQUENCIES]


# ==========================================================================
# GAME STATE
# ==========================================================================


class Tile:
    """A single falling tile in a lane."""

    def __init__(self, lane: int, layout: Layout, speed: int):
        self.lane = lane
        left, right = layout.lane_bounds[lane]
        lane_width = right - left
        tile_w = int(lane_width * TILE_WIDTH_RATIO)
        self.x = left + (lane_width - tile_w) // 2
        self.w = tile_w
        self.y = -layout.tile_height  # start above the visible frame
        self.h = layout.tile_height
        self.speed = speed
        self.alive = True

    def update(self) -> None:
        """Move the tile downward at its assigned speed."""
        self.y += self.speed

    def is_offscreen(self, frame_h: int) -> bool:
        return self.y > frame_h

    def overlaps_hit_zone(self, layout: Layout) -> bool:
        """True if any part of the tile is inside the hit zone band."""
        return self.y + self.h >= layout.hit_zone_y and self.y <= layout.hit_zone_bottom

    def contains_point(self, px: int, py: int) -> bool:
        """True if (px, py) is inside this tile's rectangle."""
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)


class Game:
    """Manages tiles, scoring, spawn timing, and difficulty progression."""

    def __init__(self):
        self.tiles: list[Tile] = []
        self.score: int = 0
        self.last_spawn_time: float = time.time()
        # Per-lane flash timers for visual feedback
        self.flash_until: list[float] = [0.0] * NUM_LANES

    @property
    def current_speed(self) -> int:
        """Dynamic tile speed: increases every SCORE_PER_LEVEL points, capped at MAX_SPEED."""
        speed = BASE_SPEED + (self.score // SCORE_PER_LEVEL) * SPEED_INCREMENT
        return min(speed, MAX_SPEED)

    @property
    def speed_level(self) -> int:
        """Current difficulty level (1-based) for display purposes."""
        return 1 + (self.score // SCORE_PER_LEVEL)

    def maybe_spawn(self, layout: Layout) -> None:
        """Spawn a tile in a random lane at intervals, using the current difficulty speed."""
        now = time.time()
        if now - self.last_spawn_time >= SPAWN_INTERVAL:
            lane = random.randint(0, NUM_LANES - 1)
            self.tiles.append(Tile(lane, layout, self.current_speed))
            self.last_spawn_time = now

    def update(self, layout: Layout) -> None:
        """Advance all tiles and remove dead/offscreen ones."""
        for tile in self.tiles:
            tile.update()
        self.tiles = [t for t in self.tiles if t.alive and not t.is_offscreen(layout.h)]

    def check_hit(
        self, finger_x: int, finger_y: int, sounds: list[pygame.mixer.Sound], layout: Layout
    ) -> bool:
        """Check if a fingertip hits any tile in the hit zone. Returns True on hit."""
        # Finger must be inside the hit zone vertically
        if not (layout.hit_zone_y <= finger_y <= layout.hit_zone_bottom):
            return False

        for tile in self.tiles:
            if not tile.alive:
                continue
            if tile.overlaps_hit_zone(layout) and tile.contains_point(finger_x, finger_y):
                tile.alive = False
                self.score += 1
                self.flash_until[tile.lane] = time.time() + 0.15
                sounds[tile.lane].play()
                return True
        return False


# ==========================================================================
# DRAWING — all positions derived from the Layout object
# ==========================================================================


def draw_lanes(frame: np.ndarray, layout: Layout) -> None:
    """Draw subtle vertical lane dividers."""
    for left, right in layout.lane_bounds:
        cv2.line(frame, (left, 0), (left, layout.h), COLOUR_LANE_LINE, 1)
        cv2.line(frame, (right, 0), (right, layout.h), COLOUR_LANE_LINE, 1)


def draw_hit_zone(frame: np.ndarray, layout: Layout) -> None:
    """Draw the horizontal hit zone band with a transparent overlay."""
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (0, layout.hit_zone_y), (layout.w, layout.hit_zone_bottom), COLOUR_HIT_ZONE, -1
    )
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    # Solid border lines
    cv2.line(frame, (0, layout.hit_zone_y), (layout.w, layout.hit_zone_y), COLOUR_HIT_ZONE, 2)
    cv2.line(
        frame, (0, layout.hit_zone_bottom), (layout.w, layout.hit_zone_bottom), COLOUR_HIT_ZONE, 2
    )


def draw_tiles(frame: np.ndarray, tiles: list[Tile]) -> None:
    """Draw all active tiles."""
    for tile in tiles:
        if not tile.alive:
            continue
        cv2.rectangle(
            frame,
            (tile.x, max(0, tile.y)),
            (tile.x + tile.w, tile.y + tile.h),
            COLOUR_TILE,
            -1,
        )
        # Subtle border
        cv2.rectangle(
            frame,
            (tile.x, max(0, tile.y)),
            (tile.x + tile.w, tile.y + tile.h),
            (200, 200, 200),
            1,
        )


def draw_flashes(frame: np.ndarray, flash_until: list[float], layout: Layout) -> None:
    """Flash a lane colour briefly on a successful hit."""
    now = time.time()
    for i, deadline in enumerate(flash_until):
        if now < deadline:
            left, right = layout.lane_bounds[i]
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (left, layout.hit_zone_y),
                (right, layout.hit_zone_bottom),
                COLOUR_FLASH[i],
                -1,
            )
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)


def draw_score(frame: np.ndarray, score: int, layout: Layout, speed_level: int) -> None:
    """Display the score and current speed level in the top-left corner."""
    cv2.putText(
        frame, f"Score: {score}", (15, int(layout.h * 0.06)),
        cv2.FONT_HERSHEY_SIMPLEX, layout.font_scale, COLOUR_SCORE,
        layout.font_thickness, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Speed: Level {speed_level}", (15, int(layout.h * 0.12)),
        cv2.FONT_HERSHEY_SIMPLEX, layout.font_scale * 0.6, (180, 180, 255),
        max(1, layout.font_thickness - 1), cv2.LINE_AA,
    )


def draw_fingertips(frame: np.ndarray, points: list[tuple[int, int]], layout: Layout) -> None:
    """Draw small circles at detected fingertip positions."""
    radius = max(5, int(layout.w * 0.012))
    for px, py in points:
        cv2.circle(frame, (px, py), radius, (255, 255, 0), 2)


# ==========================================================================
# HAND TRACKING
# ==========================================================================


def get_fingertip_positions(
    results: Any, frame_w: int, frame_h: int
) -> list[tuple[int, int]]:
    """Extract index (8) and middle (12) fingertip pixel coords from MediaPipe results."""
    points: list[tuple[int, int]] = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark_id in (8, 12):  # INDEX_FINGER_TIP, MIDDLE_FINGER_TIP
                lm = hand_landmarks.landmark[landmark_id]
                px = int(lm.x * frame_w)
                py = int(lm.y * frame_h)
                points.append((px, py))
    return points


# ==========================================================================
# MAIN LOOP
# ==========================================================================

WINDOW_NAME = "Webcam Piano Tiles"


def main() -> None:
    # Initialise audio
    init_audio()
    sounds = build_sounds()

    # Initialise MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # Open webcam and request HD resolution from the camera hardware
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera index or permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create a resizable window so it scales properly when maximized
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    game = Game()
    layout: Layout | None = None

    print("Webcam Piano Tiles started! Press 'q' to quit.")
    print(f"Lanes: {NUM_LANES} | Notes (Hz): {NOTE_FREQUENCIES}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Mirror so movement feels natural
            frame = cv2.flip(frame, 1)

            # Compute layout from the actual frame dimensions
            frame_h, frame_w = frame.shape[:2]
            if layout is None or layout.w != frame_w or layout.h != frame_h:
                layout = Layout(frame_w, frame_h)
                print(f"Frame: {frame_w}x{frame_h} | Hit zone: y={layout.hit_zone_y}-{layout.hit_zone_bottom}")

            # Hand tracking (MediaPipe expects RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Get fingertip positions
            fingertips = get_fingertip_positions(results, frame_w, frame_h)

            # Game logic
            game.maybe_spawn(layout)
            game.update(layout)

            # Check collisions for each fingertip
            for px, py in fingertips:
                game.check_hit(px, py, sounds, layout)

            # Draw everything
            draw_lanes(frame, layout)
            draw_hit_zone(frame, layout)
            draw_tiles(frame, game.tiles)
            draw_flashes(frame, game.flash_until, layout)
            draw_fingertips(frame, fingertips, layout)
            draw_score(frame, game.score, layout, game.speed_level)

            cv2.imshow(WINDOW_NAME, frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        pygame.mixer.quit()
        print(f"\nGame over! Final score: {game.score}")


if __name__ == "__main__":
    main()
