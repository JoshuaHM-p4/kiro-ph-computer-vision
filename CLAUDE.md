# CLAUDE.md

> **This file mirrors [`AGENTS.md`](AGENTS.md), which is the canonical source of
> project conventions.** If the two ever disagree, **AGENTS.md wins**. Update
> `AGENTS.md` first, then propagate changes here.

## Quick Orientation

`kiro-computer-vision` is a Python computer vision app for **image and video
stream processing** — an image preprocessing pipeline plus real-time inference
over a live video stream, using a **user-provided pretrained model**. Exact
behavior is still being defined; see [`docs/PRD.md`](docs/PRD.md).

Before making changes, read `AGENTS.md` and `docs/PRD.md`.

## Tech Stack

- **Python 3.10+**
- **OpenCV (`opencv-python`), MediaPipe, NumPy** — the core CV stack.
- **User-supplied pretrained model** (`*.pt`, `*.onnx`, `*.tflite`, …) — never
  committed, never pip-installed.
- **Testing:** `pytest`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Build / Test / Run

```bash
# Tests
pytest                              # full suite
pytest tests/test_x.py -k name      # focused run

# Run the app (entry point is TBD — see PRD)
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
- **Testing:** tests in `tests/` named `test_*.py`; test pure functions with small
  fixture arrays; mock the capture layer instead of using a live camera.
- **Models & large binaries:** user-provided model weights are git-ignored and
  referenced by a configurable path (CLI arg or env var); generated media goes in
  the git-ignored `outputs/`.
- **Never commit** secrets (`.env`) or model weights.

## Agent Working Agreements

- Do not invent product scope — record uncertainty under **Open Questions** in
  `docs/PRD.md`.
- Make minimal, focused changes; run lint + tests before finishing.
- Keep `README.md`, `AGENTS.md`, and this file consistent when conventions change.
