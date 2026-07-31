# AI Posture & Slouch Coach 🧘

Real-time posture monitoring using MediaPipe Pose. Works from **both frontal and side/profile** camera angles — sit beside your webcam or face it, either way it catches slouching.

## What it does

- Tracks upper-body landmarks (nose, ears, shoulders, hips) via MediaPipe Pose
- **Automatically detects** whether you're in front or side view and adapts metrics
- Draws a full anatomical skeleton with joint nodes, dashed spine line, and neck arrow
- Green skeleton = good posture; red glowing skeleton = slouching
- Fires a pulsing **"SLOUCH ALERT! SIT UP STRAIGHT!"** banner after 3s of sustained bad posture
- Displays posture score (0–100%), metrics, and a slouch timer in the HUD

## How detection works

| Metric | Frontal view | Side view |
|--------|-------------|-----------|
| Neck angle | Nose dropping toward ear level | Nose/ears ahead of shoulders |
| Forward head | Vertical proxy (nose drop) | Horizontal offset of ears from shoulder line |
| Spine angle | Shoulder-to-hip angle from vertical | Trunk lean (shoulders ahead of hips) |
| Shoulder tilt | Left/right shoulder height difference | (not weighted in side view) |

**View detection:** When shoulder width is small relative to torso length (< 45%), the tracker switches to side-view mode and reweights the metrics.

All thresholds are normalized by **torso length** (shoulder-midpoint to hip-midpoint) so they work at any distance.

## The skeleton

The improved skeleton includes:
- Full upper-body connections: head, neck, shoulders, arms, torso, hips, upper legs
- **Dashed spine line** (shoulder-mid → hip-mid) showing trunk alignment
- **Neck arrow** (ear-mid → nose) showing head position
- Sized joint nodes (larger for key landmarks like shoulders/hips)
- Multi-layer glow effect when posture is bad
- Region-tinted coloring for visual clarity

## Running

```bash
python projects/posture_checker/app.py
python projects/posture_checker/app.py --camera 1
```

**Side-view tip:** Position the camera at desk level, 90° to your side. The tracker auto-detects the view angle — you'll see "SIDE VIEW" or "FRONT VIEW" in the HUD.

## Controls

| Key | Action |
|-----|--------|
| C | Calibrate baseline (sit up straight first!) |
| S | Save screenshot to `screenshots/` |
| Q / ESC | Quit |

## Calibration

Press **C** while sitting with good posture. Works in both views — the tracker saves your natural angles as reference and adjusts all thresholds relative to your baseline.

## Testing

```bash
pytest projects/posture_checker/test_posture.py -v
```

32 tests covering frontal classification, side-view classification, view detection, scoring, slouch timer, calibration, and edge cases. All use synthetic landmarks — no webcam needed.

## File layout

```
projects/posture_checker/
├── app.py              Camera loop, skeleton drawing, HUD, alert banner
├── tracker.py          Pure logic: view-adaptive posture classification
├── test_posture.py     32 unit tests (frontal + side view)
├── screenshots/        Saved screenshots (git-ignored)
└── README.md           This file
```
