# AGENTS.md

> **Canonical source of project conventions.** This file is the single source of
> truth for how humans and AI agents should work in this repository. Other agent
> guidance files (e.g. `CLAUDE.md`) mirror this document — if they disagree,
> **AGENTS.md wins**. Update this file first, then propagate.

## Project Purpose

`kiro-computer-vision` is a **workshop repository**: participants fork it and build
their own computer vision app in `projects/<their-username>/`, using Kiro. The
`demos/` suite and `docs/prompts/` are reference material for them to borrow from,
not the product.

The demos themselves are Python computer vision applications for **image and
video stream processing**. It combines an image preprocessing pipeline with
real-time inference over a live video stream, using a **user-provided pretrained
model**. The precise application behavior is still being defined — see
[`docs/PRD.md`](docs/PRD.md) for scope and open questions.

## Tech Stack

- **Language:** Python 3.10-3.12. The upper bound is real: mediapipe 0.10.x ships
  no wheels for 3.13+, so a newer interpreter fails at install time.
- **Core libraries:** OpenCV (`opencv-python` + `opencv-contrib-python`),
  MediaPipe, NumPy (< 2, required by mediapipe 0.10.x)
- **Web layer (demo suite only):** Flask + `flask-sock`. Chosen because it is
  WSGI-native, so the plain Flask dev server serves WebSockets without eventlet
  or gevent.
- **Object detection:** Ultralytics YOLO (`ultralytics`), which ships
  COCO-pretrained weights and downloads them on demand. torch comes with it —
  install the **CPU build** from PyTorch's index first, or plain PyPI drags in
  3-5 GB of CUDA wheels for no benefit on a laptop webcam demo.
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
├── requirements.txt     # Pinned Python dependencies (all of them, demos included)
├── docs/
│   ├── PRD.md           # Product requirements
│   └── prompts/         # Reusable prompts that produced the demos
├── projects/            # Workshop participants build here, one folder each
│   └── your_example_app/  # Starter layout: main.py + logic.py + test_logic.py
├── models/              # Model weights, downloaded at runtime (git-ignored)
└── demos/               # OpenCV + MediaPipe demo suite (self-contained)
    ├── README.md        # Demo docs: controls, ports, tuning, troubleshooting
    ├── pytest.ini       # Demo test config (rootdir-scoped)
    ├── common/          # Shared core: geometry, landmarks, gestures, HUD, webapp
    ├── air_canvas/      # One package per demo:
    ├── slide_presenter/ #   config.py  tunables
    ├── six_seven_counter/ # core.py    pure logic, no I/O
    ├── pngtuber/        #   desktop.py OpenCV window
    ├── scavenger_hunt/  #   YOLO detection game (weights downloaded to models/)
    ├── home/            #   web.py     Flask blueprint + standalone runner
    ├── tools/           # Placeholder asset generators
    └── tests/           # Demo pytest suite (no camera required)
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
- Do **not** add heavy ML frameworks casually — prefer the existing stack. torch is
  the one exception, pulled in by ultralytics for the detection demo; keep it to the
  CPU build unless there is a reason not to.

## Testing

- The demo suite keeps its tests under `demos/tests/` with a dedicated config.
  Participant projects keep their tests in their own folder.

  ```bash
  pytest -c demos/pytest.ini                  # demo suite (no camera needed)
  pytest -c demos/pytest.ini -k pinch         # focused run
  pytest projects/<username>                  # a participant's own tests
  ```

- Write tests for pure processing functions (preprocessing, transforms, geometry).
  Use small fixture images/arrays; avoid depending on a live camera in unit tests.
- For camera/stream code, isolate the capture layer behind an interface so it can
  be mocked.
- Add or update tests alongside any behavior change; ensure the suite passes
  before committing.

## Running the App

> The main application entry point is **TBD** until its scope is finalized (see
> PRD). Expected shape once it exists:
>
> ```bash
> python -m app --source 0 --model path/to/model
> ```
>
> Update this section (and `README.md` Usage) when the real command lands.

The demo suite is already runnable, from the repository root:

```bash
python -m demos.home.web            # Flask hub, all four demos on port 5000
python -m demos.home.opencv_menu    # OpenCV launcher for the desktop versions
python -m demos.<demo>.desktop      # one desktop demo
python -m demos.<demo>.web          # one demo standalone, ports 5001-5004
```

Its Flask apps bind to `127.0.0.1` and have **no authentication**; `--host
0.0.0.0` would expose a webcam-driven service to the network and prints a warning.

## Demo Suite Conventions

The demos follow one rule that is worth preserving in new demos: **each demo's
logic is a pure `core.py`** that takes normalized (0..1) landmarks plus a clock
and returns a state dict. No camera, no window, no Flask inside it. The desktop
loop and the WebSocket handler are both thin adapters over that core, which is
what keeps gesture behavior identical between them and lets the whole suite be
tested with synthetic landmarks instead of a webcam.

Supporting conventions:

- Landmarks stay normalized everywhere; convert to pixels only at draw time.
- Thresholds are scale-relative (hand span, interocular distance, torso length)
  so distance from the camera does not change behavior.
- Any gesture that toggles uses hysteresis (separate enter/release thresholds);
  anything menu-like uses dwell-to-activate.
- Tunables live in the demo's `config.py` dataclass, not inline in the logic.

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
- Work inside `projects/<username>/` when helping a participant build. Do not
  restructure `demos/` or the shared docs to suit one project.
- When a participant asks "how do I...", check `docs/prompts/` first: the answer is
  often already written down there.
- Do not invent product scope — capture uncertainty under **Open Questions** in
  the PRD instead.
- Make minimal, focused changes; verify (lint + tests) before finishing.
- Keep `README.md`, `AGENTS.md`, and `CLAUDE.md` consistent when conventions change.
- Never commit secrets (`.env`) or model weights.
