# AGENTS.md

> **Canonical source of project conventions.** This file is the single source of
> truth for how humans and AI agents should work in this repository. Other agent
> guidance files (e.g. `CLAUDE.md`) mirror this document — if they disagree,
> **AGENTS.md wins**. Update this file first, then propagate.

## Project Purpose

`kiro-computer-vision` is a Python computer vision application for **image and
video stream processing**. It combines an image preprocessing pipeline with
real-time inference over a live video stream, using a **user-provided pretrained
model**. The precise application behavior is still being defined — see
[`docs/PRD.md`](docs/PRD.md) for scope and open questions.

## Tech Stack

- **Language:** Python 3.10+
- **Core libraries:** OpenCV (`opencv-python`), MediaPipe, NumPy
- **Model:** a user-supplied pretrained model (e.g. `*.pt`, `*.onnx`, `*.tflite`).
  Not committed to the repo and not installed via pip.
- **Testing:** `pytest`

## Directory Layout

```
kiro-computer-vision/
├── .gitignore
├── README.md            # Overview + setup (see also this file for conventions)
├── AGENTS.md            # Canonical conventions (this file)
├── CLAUDE.md            # Mirror of AGENTS.md
├── requirements.txt     # Pinned Python dependencies
├── docs/
│   └── PRD.md           # Product requirements (draft)
├── src/                 # Application code (to be created)
├── tests/               # pytest test suite (to be created)
└── models/              # User-provided model weights (git-ignored)
```

Keep this layout in sync with the actual tree as the project grows, and mirror
any structural changes into `README.md`'s Project Structure section.

## Coding Style

- Follow **PEP 8**. Prefer an autoformatter (`black`) and a linter (`ruff`).
- Use **type hints** on all public function/method signatures.
- Write **docstrings** for modules, public functions, and classes (short summary +
  args/returns where non-obvious).
- Keep functions small and single-purpose; separate I/O (camera, file, display)
  from pure processing logic so the latter is testable.
- Name things descriptively; avoid abbreviations except well-known CV terms
  (e.g. `bgr`, `rgb`, `roi`, `fps`).
- Prefer explicit over implicit; avoid hidden global state.

## Dependency Management

- All runtime dependencies live in `requirements.txt` with **pinned versions**.
- Develop inside a virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate        # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- When adding a dependency: pin the exact version, verify it coexists with the
  existing CV stack (OpenCV/MediaPipe/NumPy compatibility is fragile), and note
  why it was added.
- Do **not** add heavy ML frameworks casually — prefer the existing stack.

## Testing

- Tests live in `tests/`, named `test_*.py`, run with `pytest`.
  ```bash
  pytest                # run the full suite
  pytest tests/test_x.py -k name   # focused run
  ```
- Write tests for pure processing functions (preprocessing, transforms, geometry).
  Use small fixture images/arrays; avoid depending on a live camera in unit tests.
- For camera/stream code, isolate the capture layer behind an interface so it can
  be mocked.
- Add or update tests alongside any behavior change; ensure the suite passes
  before committing.

## Running the App

> Entry point is **TBD** until the application scope is finalized (see PRD).
> Expected shape once it exists:
> ```bash
> python -m app --source 0 --model path/to/model
> ```
> Update this section (and `README.md` Usage) when the real command lands.

## Handling Models & Large Binaries

- Model weights are **user-provided** and must **never be committed**. They are
  ignored via patterns in `.gitignore` (`*.pt`, `*.pth`, `*.onnx`, `*.tflite`,
  `*.pb`, `*.h5`, `models/`, etc.).
- Reference models by a configurable path (CLI arg or env var), not a hardcoded
  location.
- Generated media (recorded video, exported frames) belongs in `outputs/`, which
  is git-ignored.
- If a large asset genuinely must be tracked, raise it explicitly rather than
  force-adding an ignored file.

## Working Agreements for Agents

- Read this file and `docs/PRD.md` before making changes.
- Do not invent product scope — capture uncertainty under **Open Questions** in
  the PRD instead.
- Make minimal, focused changes; verify (lint + tests) before finishing.
- Keep `README.md`, `AGENTS.md`, and `CLAUDE.md` consistent when conventions change.
- Never commit secrets (`.env`) or model weights.
