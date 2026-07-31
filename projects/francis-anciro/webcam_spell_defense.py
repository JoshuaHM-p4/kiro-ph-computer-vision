"""Webcam Spell Defense — dual-hand AR tower defense with spellcasting.

Defend your Mana Crystal from waves of enemies using both hands:
- Left hand: Shield that destroys enemy projectiles on contact
- Right hand: Charge and cast spell missiles toward enemies

Usage (from the repo root, venv active):
    python projects/francis-anciro/webcam_spell_defense.py

Controls:
    - Left hand index finger: Shield (destroys enemies on contact)
    - Right hand: Hold still to charge, flick to cast spell missile
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
# CONFIGURATION — game balance variables
# ==========================================================================

# Base / Mana Crystal
BASE_HEALTH = 100
BASE_RADIUS_RATIO = 0.04  # radius as fraction of frame width

# Enemy settings
ENEMY_BASE_SPEED = 1.5  # pixels per frame at wave 1
ENEMY_SPEED_PER_WAVE = 0.4  # extra speed per wave (+0.4 px/frame each wave)
ENEMY_MAX_SPEED = 8.0  # absolute speed cap
ENEMY_BASE_HP = 2  # hit points at wave 1
ENEMY_HP_SCALE_INTERVAL = 3  # every N waves, +1 HP
ENEMY_RADIUS_RATIO = 0.015  # radius as fraction of frame width
ENEMY_DAMAGE = 10  # damage to base when enemy reaches it

# Wave system
WAVE_BASE_COUNT = 4  # enemies in wave 1
WAVE_COUNT_INCREMENT = 2  # extra enemies per wave
WAVE_SPAWN_INTERVAL_BASE = 0.8  # base seconds between spawns within a wave
WAVE_SPAWN_INTERVAL_REDUCTION = 0.03  # seconds shaved off per wave
WAVE_SPAWN_INTERVAL_MIN = 0.25  # fastest allowed spawn interval
WAVE_COOLDOWN = 3.0  # seconds between waves
MAX_ENEMIES_BASE = 10  # base max enemies on screen
MAX_ENEMIES_PER_WAVE = 2  # extra max enemies per wave
MAX_ENEMIES_CAP = 30  # absolute cap

# Shield (left hand)
SHIELD_RADIUS_RATIO = 0.045  # shield circle radius as fraction of frame width

# Spell system (right hand)
CHARGE_TIME = 1.5  # seconds to fully charge (auto-fires at 100%)
SPELL_SPEED = 14.0  # missile speed in pixels per frame
SPELL_RADIUS_RATIO = 0.01  # missile radius as fraction of frame width
SPELL_DAMAGE = 1  # damage per spell hit

# Particles
PARTICLE_COUNT = 8  # particles per explosion
PARTICLE_SPEED = 5.0
PARTICLE_LIFETIME = 0.35  # seconds

# Audio
SAMPLE_RATE = 44100

# Colours (BGR)
COLOUR_BASE = (255, 200, 100)  # mana crystal
COLOUR_BASE_RING = (255, 150, 50)
COLOUR_SHIELD = (255, 255, 0)  # cyan/neon shield
COLOUR_SPELL = (0, 180, 255)  # orange spell missile
COLOUR_ENEMY = (0, 0, 200)  # red enemies
COLOUR_ENEMY_STRONG = (0, 0, 120)  # darker strong enemies
COLOUR_CHARGE = (200, 100, 255)  # purple charge aura
COLOUR_HUD = (255, 255, 255)
COLOUR_HEALTH_BAR = (0, 255, 0)
COLOUR_HEALTH_BG = (60, 60, 60)
COLOUR_GAME_OVER = (0, 0, 255)


# ==========================================================================
# AUDIO
# ==========================================================================


def init_audio() -> None:
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()


def _make_tone(freq: float, duration_ms: int, volume: float = 0.4) -> pygame.mixer.Sound:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    env = np.linspace(1.0, 0.0, n)
    wave = (np.sin(2 * math.pi * freq * t) * env * 32767 * volume).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((wave, wave)))


def _make_noise(duration_ms: int, volume: float = 0.3) -> pygame.mixer.Sound:
    n = int(SAMPLE_RATE * duration_ms / 1000)
    env = np.exp(-np.linspace(0, 8, n))
    wave = (np.random.uniform(-1, 1, n) * env * 32767 * volume).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((wave, wave)))


class Sounds:
    """Pre-built sound effects."""

    def __init__(self):
        self.charge = _make_tone(220, 200, 0.2)
        self.cast = _make_tone(660, 100, 0.5)
        self.shield_hit = _make_noise(80, 0.4)
        self.enemy_die = _make_tone(440, 120, 0.35)
        self.base_hit = _make_noise(150, 0.5)


# ==========================================================================
# GAME ENTITIES
# ==========================================================================


class Particle:
    """Short-lived explosion particle."""

    def __init__(self, x: float, y: float, colour: tuple):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(PARTICLE_SPEED * 0.5, PARTICLE_SPEED)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.colour = colour
        self.born = time.time()
        self.alive = True

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy
        if time.time() - self.born > PARTICLE_LIFETIME:
            self.alive = False

    def draw(self, frame: np.ndarray) -> None:
        if not self.alive:
            return
        alpha = 1.0 - (time.time() - self.born) / PARTICLE_LIFETIME
        r = max(1, int(4 * alpha))
        cv2.circle(frame, (int(self.x), int(self.y)), r, self.colour, -1)


class Enemy:
    """An enemy moving toward the center base."""

    def __init__(self, x: float, y: float, target_x: float, target_y: float,
                 speed: float, hp: int, radius: int):
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.speed = speed
        self.hp = hp
        self.max_hp = hp
        self.radius = radius
        self.alive = True

        # Calculate direction vector
        dx = target_x - x
        dy = target_y - y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0:
            self.vx = (dx / dist) * speed
            self.vy = (dy / dist) * speed
        else:
            self.vx = 0.0
            self.vy = 0.0

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy

    def distance_to_base(self) -> float:
        dx = self.x - self.target_x
        dy = self.y - self.target_y
        return math.sqrt(dx * dx + dy * dy)

    def take_damage(self, dmg: int) -> None:
        self.hp -= dmg
        if self.hp <= 0:
            self.alive = False

    def draw(self, frame: np.ndarray) -> None:
        if not self.alive:
            return
        # Colour based on HP ratio
        ratio = self.hp / self.max_hp
        colour = COLOUR_ENEMY if ratio > 0.5 else COLOUR_ENEMY_STRONG
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, colour, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 1)
        # HP bar above enemy
        if self.max_hp > 1:
            bar_w = self.radius * 2
            bar_h = 4
            bx = int(self.x) - self.radius
            by = int(self.y) - self.radius - 8
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (60, 60, 60), -1)
            fill = int(bar_w * (self.hp / self.max_hp))
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), (0, 255, 0), -1)


class Projectile:
    """A spell missile fired toward an enemy."""

    def __init__(self, x: float, y: float, target_x: float, target_y: float, radius: int):
        self.x = x
        self.y = y
        self.radius = radius
        self.alive = True
        # Direction toward target
        dx = target_x - x
        dy = target_y - y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 0:
            self.vx = (dx / dist) * SPELL_SPEED
            self.vy = (dy / dist) * SPELL_SPEED
        else:
            self.vx = 0.0
            self.vy = -SPELL_SPEED

    def update(self, frame_w: int, frame_h: int) -> None:
        self.x += self.vx
        self.y += self.vy
        # Die if offscreen
        if self.x < -50 or self.x > frame_w + 50 or self.y < -50 or self.y > frame_h + 50:
            self.alive = False

    def draw(self, frame: np.ndarray) -> None:
        if not self.alive:
            return
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, COLOUR_SPELL, -1)
        # Glow effect
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius + 3, COLOUR_SPELL, 1)


# ==========================================================================
# GAME MANAGER
# ==========================================================================


class GameManager:
    """Orchestrates enemies, projectiles, particles, waves, and scoring."""

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.base_x = frame_w // 2
        self.base_y = frame_h // 2
        self.base_radius = max(15, int(frame_w * BASE_RADIUS_RATIO))
        self.base_health = BASE_HEALTH

        self.score = 0
        self.wave = 1
        self.game_over = False

        self.enemies: list[Enemy] = []
        self.projectiles: list[Projectile] = []
        self.particles: list[Particle] = []

        # Wave spawning state
        self.enemies_to_spawn = 0
        self.last_spawn_time = 0.0
        self.wave_active = False
        self.wave_cooldown_start = time.time()

        # Spell charging state (right hand)
        self.charge_level = 0.0  # 0..1
        self.right_prev_pos: tuple[int, int] | None = None

        # Shield state (left hand)
        self.shield_pos: tuple[int, int] | None = None
        self.shield_radius = max(20, int(frame_w * SHIELD_RADIUS_RATIO))

        # Spell missile radius
        self.spell_radius = max(5, int(frame_w * SPELL_RADIUS_RATIO))
        self.enemy_radius = max(10, int(frame_w * ENEMY_RADIUS_RATIO))

        self._start_wave()

    def reset(self) -> None:
        """Full restart."""
        self.base_health = BASE_HEALTH
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.charge_level = 0.0
        self.right_prev_pos = None
        self.shield_pos = None
        self._start_wave()

    def _start_wave(self) -> None:
        """Begin spawning a new wave of enemies."""
        self.enemies_to_spawn = WAVE_BASE_COUNT + (self.wave - 1) * WAVE_COUNT_INCREMENT
        self.wave_active = True
        self.last_spawn_time = time.time()

    def _enemy_speed(self) -> float:
        """Enemy speed scales with wave number, capped at ENEMY_MAX_SPEED."""
        return min(ENEMY_BASE_SPEED + (self.wave - 1) * ENEMY_SPEED_PER_WAVE, ENEMY_MAX_SPEED)

    def _enemy_hp(self) -> int:
        return ENEMY_BASE_HP + (self.wave - 1) // ENEMY_HP_SCALE_INTERVAL

    def _spawn_enemy(self) -> None:
        """Spawn one enemy from a random screen edge."""
        side = random.randint(0, 3)
        if side == 0:  # top
            x = random.randint(0, self.frame_w)
            y = -self.enemy_radius
        elif side == 1:  # bottom
            x = random.randint(0, self.frame_w)
            y = self.frame_h + self.enemy_radius
        elif side == 2:  # left
            x = -self.enemy_radius
            y = random.randint(0, self.frame_h)
        else:  # right
            x = self.frame_w + self.enemy_radius
            y = random.randint(0, self.frame_h)

        self.enemies.append(Enemy(
            x, y, self.base_x, self.base_y,
            self._enemy_speed(), self._enemy_hp(), self.enemy_radius,
        ))

    def update_right_hand(self, pos: tuple[int, int] | None, sounds: Sounds) -> None:
        """Handle spell charging: continuously charges while enemies exist, auto-fires at 100%."""
        if pos is None:
            self.charge_level = 0.0
            self.right_prev_pos = None
            return

        # Continuously charge as long as there are enemies on screen
        if self.enemies:
            self.charge_level += (1.0 / (CHARGE_TIME * 30))  # ~30fps assumption
            self.charge_level = min(1.0, self.charge_level)

            # Auto-cast at 100% charge
            if self.charge_level >= 1.0:
                self._cast_spell(pos, sounds)
                self.charge_level = 0.0
        else:
            # No enemies — charge decays
            self.charge_level = max(0.0, self.charge_level - 0.02)

        self.right_prev_pos = pos

    def _cast_spell(self, origin: tuple[int, int], sounds: Sounds) -> None:
        """Fire a spell missile toward the nearest enemy."""
        if not self.enemies:
            return

        # Find nearest alive enemy
        nearest: Enemy | None = None
        nearest_dist = float("inf")
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            dx = enemy.x - origin[0]
            dy = enemy.y - origin[1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < nearest_dist:
                nearest_dist = d
                nearest = enemy

        if nearest is not None:
            self.projectiles.append(Projectile(
                origin[0], origin[1], nearest.x, nearest.y, self.spell_radius
            ))
            sounds.cast.play()

    def update_shield(self, pos: tuple[int, int] | None, sounds: Sounds) -> None:
        """Update shield position and check for enemy collisions."""
        self.shield_pos = pos
        if pos is None:
            return

        # Check if shield touches any enemy
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            dx = enemy.x - pos[0]
            dy = enemy.y - pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.shield_radius + enemy.radius:
                enemy.take_damage(enemy.max_hp)  # shield one-shots
                self.score += 10
                sounds.shield_hit.play()
                # Explosion particles
                for _ in range(PARTICLE_COUNT):
                    self.particles.append(Particle(enemy.x, enemy.y, COLOUR_SHIELD))

    def update(self, sounds: Sounds) -> None:
        """Main game tick."""
        if self.game_over:
            return

        now = time.time()

        # Dynamic spawn interval and max enemies based on wave
        spawn_interval = max(
            WAVE_SPAWN_INTERVAL_MIN,
            WAVE_SPAWN_INTERVAL_BASE - (self.wave - 1) * WAVE_SPAWN_INTERVAL_REDUCTION,
        )
        max_enemies = min(
            MAX_ENEMIES_CAP,
            MAX_ENEMIES_BASE + (self.wave - 1) * MAX_ENEMIES_PER_WAVE,
        )

        # Spawn enemies in the current wave
        if self.wave_active and self.enemies_to_spawn > 0:
            if now - self.last_spawn_time >= spawn_interval and len(self.enemies) < max_enemies:
                self._spawn_enemy()
                self.enemies_to_spawn -= 1
                self.last_spawn_time = now

        # Check if wave is complete (all spawned and all dead)
        if self.wave_active and self.enemies_to_spawn <= 0 and not any(e.alive for e in self.enemies):
            self.wave_active = False
            self.wave_cooldown_start = now

        # Start next wave after cooldown
        if not self.wave_active and not self.game_over:
            if now - self.wave_cooldown_start >= WAVE_COOLDOWN:
                self.wave += 1
                self._start_wave()

        # Update enemies
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            enemy.update()
            # Check if enemy reached base
            if enemy.distance_to_base() < self.base_radius + enemy.radius:
                enemy.alive = False
                self.base_health -= ENEMY_DAMAGE
                sounds.base_hit.play()
                for _ in range(PARTICLE_COUNT):
                    self.particles.append(Particle(self.base_x, self.base_y, COLOUR_GAME_OVER))
                if self.base_health <= 0:
                    self.base_health = 0
                    self.game_over = True

        self.enemies = [e for e in self.enemies if e.alive]

        # Update projectiles
        for proj in self.projectiles:
            proj.update(self.frame_w, self.frame_h)
            if not proj.alive:
                continue
            # Check collision with enemies
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                dx = proj.x - enemy.x
                dy = proj.y - enemy.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < proj.radius + enemy.radius:
                    enemy.take_damage(SPELL_DAMAGE)
                    proj.alive = False
                    if not enemy.alive:
                        self.score += 10
                        sounds.enemy_die.play()
                        for _ in range(PARTICLE_COUNT):
                            self.particles.append(Particle(enemy.x, enemy.y, COLOUR_SPELL))
                    break

        self.projectiles = [p for p in self.projectiles if p.alive]

        # Update particles
        for particle in self.particles:
            particle.update()
        self.particles = [p for p in self.particles if p.alive]


# ==========================================================================
# DRAWING
# ==========================================================================


def draw_base(frame: np.ndarray, gm: GameManager) -> None:
    """Draw the Mana Crystal at the center."""
    # Outer glow ring
    cv2.circle(frame, (gm.base_x, gm.base_y), gm.base_radius + 8, COLOUR_BASE_RING, 2)
    # Crystal body
    cv2.circle(frame, (gm.base_x, gm.base_y), gm.base_radius, COLOUR_BASE, -1)
    # Inner highlight
    cv2.circle(frame, (gm.base_x - 5, gm.base_y - 5), gm.base_radius // 3, (255, 255, 255), -1)


def draw_shield(frame: np.ndarray, gm: GameManager) -> None:
    """Draw the left hand shield as a glowing neon circle."""
    if gm.shield_pos is None:
        return
    x, y = gm.shield_pos
    # Outer glow
    cv2.circle(frame, (x, y), gm.shield_radius + 4, COLOUR_SHIELD, 2)
    # Semi-transparent fill via overlay
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), gm.shield_radius, COLOUR_SHIELD, -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    # Crisp border
    cv2.circle(frame, (x, y), gm.shield_radius, COLOUR_SHIELD, 2)


def draw_charge_aura(frame: np.ndarray, pos: tuple[int, int] | None, charge: float) -> None:
    """Draw a glowing charge bar below the right index finger tip."""
    if pos is None or charge < 0.01:
        return
    x, y = pos

    # Charge bar dimensions
    bar_w = 60
    bar_h = 10
    bar_x = x - bar_w // 2
    bar_y = y + 20  # below the fingertip

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)

    # Fill (colour shifts from purple to bright orange as charge increases)
    fill_w = int(bar_w * charge)
    fill_colour = (
        int(200 * (1 - charge) + 0 * charge),
        int(100 * (1 - charge) + 180 * charge),
        int(255 * (1 - charge) + 255 * charge),
    )
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_colour, -1)

    # Border glow
    border_colour = COLOUR_CHARGE if charge < 1.0 else (0, 255, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), border_colour, 1)

    # Percentage text
    cv2.putText(
        frame, f"{int(charge * 100)}%", (bar_x + bar_w + 5, bar_y + bar_h - 1),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOUR_CHARGE, 1, cv2.LINE_AA,
    )

    # Pulsing dots around finger when nearly full
    if charge > 0.7:
        num_dots = int((charge - 0.7) / 0.3 * 8)
        radius = 25
        for i in range(num_dots):
            angle = (time.time() * 4 + i * (2 * math.pi / max(1, num_dots))) % (2 * math.pi)
            px = int(x + math.cos(angle) * radius)
            py = int(y + math.sin(angle) * radius)
            cv2.circle(frame, (px, py), 3, COLOUR_CHARGE, -1)


def draw_hud(frame: np.ndarray, gm: GameManager, frame_w: int, frame_h: int) -> None:
    """Render the full HUD overlay."""
    font_scale = frame_h / 500.0
    thickness = max(1, int(font_scale * 2))

    # Score — top left
    cv2.putText(
        frame, f"Score: {gm.score}", (15, int(frame_h * 0.05)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, COLOUR_HUD, thickness, cv2.LINE_AA,
    )

    # Wave — below score
    cv2.putText(
        frame, f"Wave: {gm.wave}", (15, int(frame_h * 0.10)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (180, 220, 255), max(1, thickness - 1), cv2.LINE_AA,
    )

    # Active spells count
    spell_count = len(gm.projectiles)
    cv2.putText(
        frame, f"Spells: {spell_count}", (15, int(frame_h * 0.15)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, COLOUR_SPELL, max(1, thickness - 1), cv2.LINE_AA,
    )

    # Base health bar — top center
    bar_w = int(frame_w * 0.3)
    bar_h = int(frame_h * 0.025)
    bar_x = (frame_w - bar_w) // 2
    bar_y = 15
    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOUR_HEALTH_BG, -1)
    # Fill
    fill_w = int(bar_w * (gm.base_health / BASE_HEALTH))
    health_colour = COLOUR_HEALTH_BAR if gm.base_health > 30 else (0, 100, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), health_colour, -1)
    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    # Label
    cv2.putText(
        frame, f"Base: {gm.base_health}/{BASE_HEALTH}",
        (bar_x + bar_w + 10, bar_y + bar_h - 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.45, COLOUR_HUD, 1, cv2.LINE_AA,
    )


def draw_game_over(frame: np.ndarray, gm: GameManager, frame_w: int, frame_h: int) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font_scale = frame_h / 300.0
    thickness = max(2, int(font_scale * 3))

    text = "MANA CRYSTAL DESTROYED"
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, thickness)[0]
    tx = (frame_w - ts[0]) // 2
    ty = frame_h // 2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, COLOUR_GAME_OVER, thickness, cv2.LINE_AA)

    info = f"Score: {gm.score}  |  Waves survived: {gm.wave - 1}"
    is_ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.4, thickness - 1)[0]
    cv2.putText(frame, info, ((frame_w - is_[0]) // 2, ty + int(frame_h * 0.07)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.4, COLOUR_HUD, max(1, thickness - 1), cv2.LINE_AA)

    prompt = "Press 'R' to restart  |  'Q' to quit"
    ps = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.35, 1)[0]
    cv2.putText(frame, prompt, ((frame_w - ps[0]) // 2, ty + int(frame_h * 0.13)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.35, (180, 180, 180), 1, cv2.LINE_AA)


# ==========================================================================
# HAND TRACKING
# ==========================================================================


def get_hand_positions(
    results: Any, frame_w: int, frame_h: int
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Get index finger tip positions for left and right hands.

    Returns (left_pos, right_pos) where each is (x, y) or None.
    MediaPipe labels hands from the camera's perspective, so we flip labels
    since the frame is mirrored.
    """
    left_pos: tuple[int, int] | None = None
    right_pos: tuple[int, int] | None = None

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return left_pos, right_pos

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        # MediaPipe label (from camera view) — mirrored means we swap
        label = handedness.classification[0].label  # "Left" or "Right"
        lm = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
        px = int(lm.x * frame_w)
        py = int(lm.y * frame_h)

        # Since frame is mirrored: camera's "Left" = user's left hand
        if label == "Left":
            right_pos = (px, py)  # camera left = user right in mirrored
        else:
            left_pos = (px, py)  # camera right = user left in mirrored

    return left_pos, right_pos


# ==========================================================================
# MAIN LOOP
# ==========================================================================

WINDOW_NAME = "Webcam Spell Defense"


def main() -> None:
    # Initialise audio
    init_audio()
    sounds = Sounds()

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

    gm: GameManager | None = None

    print("Webcam Spell Defense started!")
    print("Left hand = Shield | Right hand = Charge & Cast")
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # Mirror for intuitive interaction
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # Initialize game on first frame
            if gm is None:
                gm = GameManager(frame_w, frame_h)

            # Hand tracking
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            left_pos, right_pos = get_hand_positions(results, frame_w, frame_h)

            # Update hands
            gm.update_shield(left_pos, sounds)
            gm.update_right_hand(right_pos, sounds)

            # Update game state
            gm.update(sounds)

            # Draw everything
            draw_base(frame, gm)

            for enemy in gm.enemies:
                enemy.draw(frame)
            for proj in gm.projectiles:
                proj.draw(frame)
            for particle in gm.particles:
                particle.draw(frame)

            draw_shield(frame, gm)
            draw_charge_aura(frame, right_pos, gm.charge_level)

            # Fingertip indicators
            if left_pos:
                cv2.circle(frame, left_pos, 6, COLOUR_SHIELD, 2)
            if right_pos:
                cv2.circle(frame, right_pos, 6, COLOUR_SPELL, 2)

            draw_hud(frame, gm, frame_w, frame_h)

            if gm.game_over:
                draw_game_over(frame, gm, frame_w, frame_h)

            cv2.imshow(WINDOW_NAME, frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r") and gm.game_over:
                gm.reset()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        pygame.mixer.quit()
        print(f"\nSession ended. Score: {gm.score if gm else 0}")


if __name__ == "__main__":
    main()
