# Posture Checker 🧍

Real-time posture analysis using MediaPipe Pose. Detects bad posture from any camera angle (front or side view), scores your posture continuously, tracks streaks, and saves video replays when you slouch.

## Features

- **Multi-angle detection** — automatically detects whether the camera sees your front or side, and applies the right postural checks
- **Posture scoring** — continuous 0–100 score based on head position, shoulder alignment, and torso lean
- **Streak timers** — see how long you've maintained good (or bad) posture
- **Slouch replays** — automatically saves short video clips when bad posture persists (5s before + 3s after the trigger)
- **Two interfaces** — desktop OpenCV window and Flask web dashboard

## What it checks

| Metric | Front View | Side View |
|--------|-----------|-----------|
| Head position | Nose above midpoint of shoulders | Ear-shoulder-hip angle |
| Shoulder alignment | Tilt from horizontal | — |
| Torso lean | Mid-shoulder to mid-hip angle | Shoulder-to-hip angle from vertical |

All thresholds use **hysteresis** (separate enter/release thresholds) to prevent flickering.

## Quick Start

From the repository root (with the virtual environment activated):

```bash
cd projects/PreciousCTRL

# Desktop version (OpenCV window)
python main.py

# Web dashboard (Flask)
python web.py
# Open http://127.0.0.1:5000/ in your browser
```

### Command-line options

**Desktop (`main.py`)**
```
--camera 0        Camera index (try 1 if default doesn't work)
--no-replay       Disable slouch replay recording
--width 1280      Capture width
--height 720      Capture height
```

**Web (`web.py`)**
```
--camera 0        Camera index
--port 5000       Web server port
--host 127.0.0.1  Host to bind (use 0.0.0.0 with caution)
```

## How it works

```
config.py          All tunables in a dataclass
core.py            Pure logic: landmarks in → posture state out (no I/O)
main.py            Desktop: camera → MediaPipe → core → draw → display
web.py             Flask: background camera thread → MJPEG stream + WebSocket
templates/         Dashboard HTML with live chart and replay gallery
replays/           Auto-saved slouch clips (git-ignored)
```

The **core logic** (`PostureChecker` class) takes normalized landmarks and a timestamp, returns a `PostureState` with:
- Score (0–100)
- Detected view type (front/left_side/right_side)
- Per-metric flags (head_forward, shoulders_uneven, torso_leaning)
- Good/bad streak durations
- Slouch event trigger signal

No camera code, no display code — just measurements and state.

## Testing

```bash
# From the project directory
pytest test_core.py -v

# Or from the repo root
pytest projects/PreciousCTRL -v
```

Tests use synthetic landmark arrays (no webcam needed) to verify:
- View detection (front vs side)
- Good/bad posture scoring
- Streak accumulation and reset
- Slouch event triggering and cooldown
- Hysteresis behaviour

## Tuning

All thresholds live in `config.py`. Key ones to adjust:

| Setting | Default | What it does |
|---------|---------|--------------|
| `head_forward_angle_bad` | 155° | Below this = head too forward (side view) |
| `shoulder_tilt_bad` | 8° | Above this = uneven shoulders (front view) |
| `torso_lean_bad` | 12° | Above this = leaning forward |
| `slouch_trigger_seconds` | 3.0 | How long before bad posture triggers a replay save |
| `slouch_cooldown_seconds` | 10.0 | Minimum time between saved clips |

## Dependencies

Uses the project's shared `requirements.txt`. Key packages:
- `mediapipe` — Pose landmark detection
- `opencv-python` — Camera capture and display
- `flask` + `flask-sock` — Web dashboard and WebSocket
- `numpy` — Array operations

## What surprised me

The biggest challenge was making posture detection work from any angle. The solution: use ear visibility to detect the camera view, then apply different angle measurements for front vs. side. Scale-relative thresholds (fractions of body measurements instead of pixels) make it work at any distance.
