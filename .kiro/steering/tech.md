# Technology stack

## Language

Python 3.10–3.12 (hard ceiling: mediapipe 0.10.x has no wheels for 3.13+)

## Core libraries

| Library | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.10.0.84 | Image processing, display, drawing |
| opencv-contrib-python | 4.10.0.84 | Extended modules |
| mediapipe | 0.10.14 | Hand, face and pose landmark tracking |
| numpy | 1.26.4 | Array operations (must be < 2 for mediapipe) |
| ultralytics | 8.4.111 | YOLO object detection (scavenger hunt) |
| torch | 2.13.0+cpu | Tensor backend for ultralytics |
| flask | 3.1.3 | Web layer for demos |
| flask-sock | 0.7.0 | WSGI-native WebSocket |
| pytest | 9.1.1 | Testing |

## Install order matters

torch must come from the CPU index FIRST, then the rest from PyPI:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

Without that, pip pulls the CUDA build (3-5 GB of nvidia wheels).

## Model weights

- YOLO weights: downloaded at runtime by ultralytics into `models/` (git-ignored)
- SAM 3.1 weights: downloaded by transformers into `~/.cache/huggingface/hub/`
- Never committed, never pip-installed, always git-ignored

## MediaPipe API

The demos use the **legacy `mp.solutions` namespace** (Hands, FaceMesh, Pose). It
works across the whole 0.10.x line. Do not migrate to the Tasks API unless there is
a reason: the legacy one is what participants will find in tutorials.

## Browser-side vision

The web demos run `@mediapipe/tasks-vision` 0.10.21 in the browser via a CDN (or
vendored locally). Landmarks stream to Flask over a WebSocket as JSON.
