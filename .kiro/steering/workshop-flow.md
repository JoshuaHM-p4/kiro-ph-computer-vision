# Workshop flow

How to guide a participant through their Build Night project.

## Step 1: Choose an idea

Point them at `docs/prompts/README.md`. The prompt closest to their idea is the
starting point. If nothing fits, the "how to prompt" guide (00) gives the shape.

## Step 2: Get a camera working

The starter app in `projects/your_example_app/` already opens a webcam and draws
on the frame. Copy it as a baseline rather than starting from zero.

```bash
cp -r projects/your_example_app projects/<username>
cd projects/<username>
python main.py
```

## Step 3: Replace the logic

The starter tracks brightness. Replace `logic.py` with whatever their idea needs
(hand tracking, face mesh, object detection, etc.), keeping the same shape:
a class with an `update(measurements, timestamp) -> state` method.

## Step 4: Test it

Write tests that feed synthetic data. `pytest projects/<username>` should pass.

## Step 5: Optional web layer

Only if they want it. Point them at `docs/prompts/07-flask-web-layer.md` and the
three architecture choices there. For most Build Night projects the desktop window
is enough.

## Common problems

| Symptom | Fix |
|---------|-----|
| "No matching distribution for mediapipe" | venv is Python 3.13+; rebuild with 3.12 |
| Gesture flickers | Add hysteresis (two thresholds) |
| Works close but not far | Divide by a body measurement, not pixels |
| Counted twice | Edge-trigger with a cooldown |
| "Could not open camera" | Try `--camera 1`; check `ls /dev/video*` |

## What NOT to do

- Do not modify `demos/` to suit one participant's project
- Do not install heavy new dependencies without checking compatibility
- Do not commit model weights or API keys
- Do not assume a GPU is available; prefer CPU inference
