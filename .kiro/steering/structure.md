# Project structure

```
kiro-computer-vision/
├── .kiro/steering/         This folder: Kiro reads these automatically
├── README.md               Workshop landing page
├── AGENTS.md               Canonical coding conventions
├── CLAUDE.md               Mirror of AGENTS.md
├── requirements.txt        ALL pinned dependencies
├── docs/
│   ├── PRD.md              Product requirements
│   └── prompts/            Reusable prompts that produced the demos
│       ├── 00-how-to-prompt.md
│       ├── 01-image-lab.md ... 09-sam-labeler.md
│       └── README.md       Index linking all prompts
├── projects/               WHERE PARTICIPANTS BUILD
│   ├── .gitkeep
│   ├── README.md           Instructions for participants
│   └── your_example_app/   Starter layout (main.py + logic.py + test_logic.py)
├── demos/                  REFERENCE IMPLEMENTATIONS (do not modify for a participant)
│   ├── README.md           Full demo docs
│   ├── pytest.ini          Demo test config
│   ├── __init__.py         Demo registry
│   ├── common/             Shared: geometry, landmarks, gestures, HUD, camera, webapp
│   ├── image_lab/          Interactive cv2 playground
│   ├── air_canvas/         Gesture painting
│   ├── slide_presenter/    Pinch to advance slides
│   ├── six_seven_counter/  Wrist-tilt rep counting
│   ├── pngtuber/           Head yaw + expression avatar
│   ├── scavenger_hunt/     COCO detection game
│   ├── sam_labeler/        SAM 3.1 text-prompted segmentation
│   ├── home/               Flask hub + OpenCV launcher menu
│   └── tools/              Asset generators
└── models/                 Runtime model weights (git-ignored)
```

## Where to work

- Helping a participant: `projects/<username>/`
- Adding to the demos: `demos/` (rare, requires updating tests)
- Updating conventions: `AGENTS.md` first, then propagate to `CLAUDE.md`

## Naming conventions

- Python: PEP 8, type hints on public signatures, docstrings on modules/classes
- Demos: each has `config.py`, `core.py`, `desktop.py`, `web.py`, `templates/`
- Tests: `test_*.py` in `demos/tests/`, driven by synthetic fixtures
