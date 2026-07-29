# your_example_app

A minimal shape for a build-night project. Copy the folder, rename it, replace the
contents.

```
your_example_app/
├── README.md      what you built, and how to run it
├── main.py        entry point: argument parsing and the camera loop
├── logic.py       the decisions, as pure functions
└── test_logic.py  tests for logic.py, no webcam required
```

## Run it

```bash
# from the repository root, with the venv active
python projects/your_example_app/main.py
python -m pytest projects/your_example_app
```

## Why the split

`main.py` owns the messy parts — opening the camera, showing a window, reading keys.
`logic.py` owns the decisions and touches none of that, which is what makes
`test_logic.py` possible at all. Every demo in this repository is built the same way,
and it is the one structural habit worth copying.

Press `q` to quit. Add your own keys in `main.py`.
