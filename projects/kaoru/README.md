# 👁️ Don't Blink

> ⚠️ **EPILEPSY WARNING:** This game contains rapid flashing lights, strobes,
> screen color inversions, and sudden visual effects. **Not suitable for people
> with photosensitive epilepsy.**

A staring contest against your computer. MediaPipe Face Mesh watches your eyes —
the moment you blink, it's over. Meanwhile, the screen throws increasingly chaotic
distractions at you trying to make you flinch.

## How it works

1. Press SPACE — 3-2-1 countdown begins.
2. **Keep your eyes open.** Your score is how many seconds you survive.
3. Distractions start after 3 seconds and escalate:
   - White/red flashes
   - Screen shake
   - Scary text popping up
   - Color inversion
   - TV static
   - Zoom in
   - Strobe lights
4. When you blink → **fake BSOD** with your survival time.
5. High score persists across attempts in the same session.

## Detection

Uses Eye Aspect Ratio (EAR) from MediaPipe's 468-point Face Mesh:
- EAR > 0.25 = eyes open
- EAR < 0.21 for 2+ frames = blink detected
- Hysteresis between thresholds prevents flicker

No model download needed — MediaPipe ships with the library.

## Run it

From the **repo root**:

```bash
# Desktop (fullscreen OpenCV window)
python -m projects.kaoru.main

# Web (Flask dashboard at http://127.0.0.1:5001)
python -m projects.kaoru.web
```

## Controls

| Key     | Action |
|---------|--------|
| `Space` | Start / Retry after blink |
| `R`     | Restart |
| `Q`     | Quit (desktop only) |

## Run tests

```bash
python -m pytest projects/kaoru/test_core.py -v
```

20 tests, no camera needed.

## Tuning

All knobs in `config.py`:

| Setting | Default | What it does |
|---------|---------|--------------|
| `ear_blink_threshold` | 0.21 | EAR below this = blink |
| `ear_open_threshold` | 0.25 | EAR above this = open (hysteresis) |
| `blink_frames` | 2 | Consecutive frames needed to confirm blink |
| `distraction_start_delay` | 3.0s | Grace period before chaos begins |
| `distraction_interval_initial` | 4.0s | Time between distractions (start) |
| `distraction_interval_min` | 0.5s | Fastest distraction rate |
| `distraction_interval_decay` | 0.9 | Speed multiplier each time |
| `penalty_style` | "bsod" | "bsod" or "jumpscare" |

## Architecture

```
config.py     — all tunables in a dataclass
core.py       — EAR computation, blink detection, game engine (no I/O)
main.py       — desktop: MediaPipe + fullscreen cv2 + distraction rendering + audio
sfx.py        — procedurally generated sound effects (no audio files)
web.py        — Flask + WebSocket streaming
templates/    — browser dashboard with CSS distractions
test_core.py  — 20 tests with synthetic EAR data
```

## Sound effects

All audio is procedurally generated via `sounddevice` + numpy — no audio files:
- **Countdown beeps** (3-2-1-GO, ascending pitch)
- **Distraction zap** (harsh buzz on each distraction)
- **Death sound** (descending frequency + noise)
- **Tension drone** (low rumble that intensifies over time)

## Distractions (16 types)

| Effect | What it does |
|--------|-------------|
| flash_white | Blinding white flash |
| flash_red | Blood red flash |
| flash_green | Green flash |
| shake | Screen jolts randomly |
| jumpscare_text | "BOO!", "BEHIND YOU", etc. at random sizes |
| static | TV static overlay |
| invert | Inverts all colors |
| zoom | Sudden zoom into your face |
| strobe | Rapid color cycling |
| mirror | Flips your image (disorienting) |
| pixelate | Pixelates the frame |
| blackout | Brief total darkness |
| face_warp | Barrel distortion on your face |
| split | Horizontal glitch shift |
| red_eye | Everything goes blood red |
| tilt | Rotates the frame |

Plus taunting text that appears with increasing frequency.

## What surprised me

MediaPipe Face Mesh runs at 10-20ms/frame on CPU — basically instant compared to
YOLO. The EAR approach (ratio of eye vertical to horizontal distance) is dead simple
but works remarkably well. Hysteresis between open/close thresholds is essential or
it triggers on half-blinks.

## Known issues

- **The game segfaults at ~50 seconds.** That's right — if you survive 50 seconds,
  the game literally crashes. Not because of the chaos effects. Not because of bad
  code. Because Google's TensorFlow Lite C++ engine corrupts its own memory after
  running for too long. We tried fixing it. We recreate the Face Mesh every 500
  frames. We wrap everything in try/except. We made the arrays contiguous. TFLite
  doesn't care. It dies anyway. In C++. Where Python can't catch it.

  **So if you survive 50 seconds, you win by default. The game fears you.**

  Consider the segfault the ultimate jumpscare. You were staring, focused, in the
  zone — and then your terminal says `SEGV`. Your heart rate spikes harder than any
  flash or fake BSOD ever could. You're welcome.

  (This is a MediaPipe/TFLite bug. Blame Google. We just ship it as a feature.)
