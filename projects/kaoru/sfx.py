"""Sound effects — procedurally generated, no audio files needed.

Uses sounddevice + numpy to generate tones on the fly.
All playback is non-blocking so it never stalls the game loop.
"""

from __future__ import annotations

import threading

import numpy as np

try:
    import sounddevice as sd
    _HAS_AUDIO = True
except (ImportError, OSError):
    _HAS_AUDIO = False


SAMPLE_RATE = 22050


def _play_async(samples: np.ndarray) -> None:
    """Play audio samples in a background thread (non-blocking)."""
    if not _HAS_AUDIO:
        return

    def _play():
        try:
            sd.play(samples, samplerate=SAMPLE_RATE, blocking=True)
        except Exception:
            pass

    t = threading.Thread(target=_play, daemon=True)
    t.start()


def countdown_beep(number: int) -> None:
    """Short beep for countdown numbers. Higher pitch for 'GO!'."""
    freq = 440 if number > 0 else 880
    duration = 0.1 if number > 0 else 0.2
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    # Quick fade out
    fade = np.linspace(1, 0, len(tone))
    _play_async(tone * fade)


def distraction_hit() -> None:
    """Harsh buzz/zap when a distraction fires."""
    duration = 0.15
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # Mix of harsh frequencies
    tone = (
        0.3 * np.sin(2 * np.pi * 150 * t)
        + 0.2 * np.sin(2 * np.pi * 300 * t)
        + 0.2 * np.random.uniform(-1, 1, len(t))  # noise
    ).astype(np.float32)
    _play_async(tone * 0.6)


def blink_death() -> None:
    """Descending death sound when you blink."""
    duration = 0.8
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # Descending frequency from 600 to 80 Hz
    freq = np.linspace(600, 80, len(t))
    tone = (0.6 * np.sin(2 * np.pi * freq * t / SAMPLE_RATE * np.cumsum(np.ones(len(t))))).astype(np.float32)
    # Actually do it properly
    phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
    tone = (0.6 * np.sin(phase)).astype(np.float32)
    # Add some grit
    noise = np.random.uniform(-0.2, 0.2, len(t)).astype(np.float32)
    tone = np.clip(tone + noise * np.linspace(0, 1, len(t)), -1, 1).astype(np.float32)
    _play_async(tone)


def game_start() -> None:
    """Rising tone — game is starting."""
    duration = 0.3
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    freq = np.linspace(300, 800, len(t))
    phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
    tone = (0.4 * np.sin(phase)).astype(np.float32)
    _play_async(tone)


def tension_drone(intensity: float) -> None:
    """Low rumbling drone that gets louder/higher with intensity (0..1).

    Call this periodically during gameplay for atmosphere.
    """
    duration = 0.3
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    base_freq = 50 + intensity * 100
    tone = (
        intensity * 0.15 * np.sin(2 * np.pi * base_freq * t)
        + intensity * 0.1 * np.sin(2 * np.pi * (base_freq * 1.5) * t)
    ).astype(np.float32)
    _play_async(tone)
