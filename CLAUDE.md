# CLAUDE.md

> **This file mirrors [`AGENTS.md`](AGENTS.md), which is the canonical source of
> project conventions.** If the two ever disagree, **AGENTS.md wins**. Update
> `AGENTS.md` first, then propagate changes here.

## Quick Orientation

`kiro-computer-vision` is a **workshop repository**. Participants fork it and build
their own computer vision app in `projects/<their-username>/` with Kiro's help.
`demos/` holds six worked OpenCV + MediaPipe + YOLO examples, and
[`docs/prompts/`](docs/prompts/) holds the prompts that produced them, for
participants to reuse. See [`docs/PRD.md`](docs/PRD.md).

Before making changes, read `AGENTS.md` and `docs/PRD.md`.

## Tech Stack

- **Python 3.10-3.12** — mediapipe 0.10.x has no wheels for 3.13+.
- **OpenCV (`opencv-python` + `opencv-contrib-python`), MediaPipe, NumPy < 2** —
  the core CV stack.
- **Flask + `flask-sock`** — web layer for the `demos/` suite only; WSGI-native, so
  no eventlet/gevent.
- **Ultralytics YOLO** — object detection for the scavenger hunt; downloads
  COCO-pretrained weights on demand. Install the **CPU** torch build first, or
  plain PyPI pulls multi-gigabyte CUDA wheels.
- **User-supplied pretrained model** (`*.pt`, `*.onnx`, `*.tflite`, …) — never
  committed, never pip-installed.
- **Testing:** `pytest`.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Build / Test / Run

```bash
# Tests
pytest                              # application suite (tests/)
pytest -c demos/pytest.ini          # demo suite, no camera needed
pytest -c demos/pytest.ini -k pinch # focused run

# Demo suite (runnable today, from the repo root)
python -m demos.home.web            # Flask hub, all four demos on port 5000
python -m demos.home.opencv_menu    # OpenCV launcher for the desktop demos
python -m demos.air_canvas.desktop  # a single desktop demo

# Run the main app (entry point is TBD — see PRD)
# python -m app --source 0 --model path/to/model
```

## Conventions (summary — see AGENTS.md for the full text)

- **Style:** PEP 8; format with `black`, lint with `ruff`; type hints on public
  signatures; docstrings on modules/public functions/classes.
- **Structure:** separate I/O (camera, file, display) from pure processing logic
  so the latter is unit-testable; no hidden global state.
- **Dependencies:** pin exact versions in `requirements.txt`; the
  OpenCV/MediaPipe/NumPy combination is fragile, so verify compatibility and note
  why any new dependency was added.
- **Testing:** application tests in `tests/`, demo tests in `demos/tests/`, all
  named `test_*.py`; test pure functions with small fixture arrays; mock the
  capture layer instead of using a live camera.
- **Demos:** each demo's `core.py` stays pure — normalized landmarks and a clock
  in, state dict out — so the desktop window and the Flask/WebSocket path are
  interchangeable adapters over one implementation. Tunables belong in the demo's
  `config.py`. See `demos/README.md`.
- **Models & large binaries:** user-provided model weights are git-ignored and
  referenced by a configurable path (CLI arg or env var); generated media goes in
  the git-ignored `outputs/`.
- **Never commit** secrets (`.env`) or model weights.

## Agent Working Agreements

- Do not invent product scope — record uncertainty under **Open Questions** in
  `docs/PRD.md`.
- When helping a participant, work inside `projects/<username>/`. Leave `demos/` and
  the shared docs alone unless asked.
- Check `docs/prompts/` before answering "how do I build X" — it is likely covered.
- Make minimal, focused changes; run lint + tests before finishing.
- Keep `README.md`, `AGENTS.md`, and this file consistent when conventions change.
