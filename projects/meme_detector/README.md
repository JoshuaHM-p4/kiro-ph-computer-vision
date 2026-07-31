# Emotion Analyzer & Meme Popup

Real-time facial expression detection using MediaPipe Face Mesh with matching meme overlays.

## What it does

- Captures webcam video and analyzes facial expressions in real-time
- Detects four emotions: **Happy**, **Surprised**, **Sad**, **Neutral**
- Displays the detected emotion as text overlay on the video feed
- Shows a matching meme image in a side panel that updates when your expression changes

## How it works

The app uses MediaPipe Face Mesh (468 landmarks) and classifies emotions based on:
- **Smile curvature** — mouth corner height relative to lip center (normalized by interocular distance)
- **Mouth aspect ratio** — vertical mouth opening vs width
- **Eye aspect ratio** — eyelid gap vs eye width

All thresholds are scale-relative (divided by interocular distance), so detection works at any distance from the camera. Hysteresis and a settle filter prevent flickering.

## Running

From the repository root with the venv active:

```bash
# Generate placeholder memes (optional, app creates them on-the-fly if missing)
python projects/MacXenix/make_memes.py

# Run the app
python projects/MacXenix/app.py
python projects/MacXenix/app.py --camera 1    # if you have multiple cameras
python projects/MacXenix/app.py --no-mirror   # disable mirror mode
```

Press **q** or **ESC** to quit.

## Testing

```bash
pytest projects/MacXenix/test_emotion.py -v
```

All tests use synthetic landmark data — no webcam needed.

## Custom memes

Replace the files in `memes/` with your own images:
- `memes/happy.jpg`
- `memes/surprised.jpg`
- `memes/sad.jpg`
- `memes/neutral.jpg`

PNG and JPEG are both supported.

## File layout

```
projects/MacXenix/
├── app.py            Camera loop, MediaPipe, display
├── emotion.py        Pure logic: emotion classification (no I/O)
├── test_emotion.py   Unit tests with synthetic landmarks
├── make_memes.py     Generates placeholder meme images
├── memes/            Meme images (one per emotion)
└── README.md         This file
```
