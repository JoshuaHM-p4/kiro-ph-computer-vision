# Scavenger Hunt 🔍

A webcam game that asks you to find real objects on your desk using
[YOLOv8](https://github.com/ultralytics/ultralytics) object detection.

**"BRING ME A CELL PHONE"** — you hold the item up; the model confirms it.
Five rounds, speed scoring, and a streak bonus.

---

## ▶ How to start the program (step by step)

**Step 1 — Open a terminal in this folder**
Right-click the `scavenger_hunt` folder → "Open in Terminal" (or open PowerShell and `cd` to it).

**Step 2 — Install the dependencies (first time only)**
```
py -m pip install -r requirements.txt
```
This downloads ~200 MB of packages (PyTorch, OpenCV, Ultralytics). Only needed once.

**Step 3 — Run the game**
```
py main.py
```
A window will open showing your webcam feed. The first run also downloads the YOLOv8 model (~6 MB) automatically.

**Step 4 — Start playing**
Press **SPACE** to begin. The game will tell you to find an object — hold it up to the camera!

**Step 5 — Quit**
Press **q** or **ESC** to close the window.

> 💡 If `py` doesn't work, try `python main.py` instead.

---

## Quick start

```bash
# 1 — create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2 — install dependencies
#   ⚠ ~200 MB first install (torch CPU build + ultralytics)
pip install -r requirements.txt

# 3 — run
python main.py
```

On first run the YOLOv8n weights (~6 MB) are downloaded automatically and
saved to `weights/yolov8n.pt` (git-ignored).

---

## Controls

| Key | Action |
|-----|--------|
| SPACE | Start the game |
| n | Skip the current item |
| q / ESC | Quit |

---

## Command-line options

```
python main.py --camera 1          # use a different webcam
python main.py --detect-every 5    # run YOLO every 5 frames (default 3)
python main.py --no-mirror         # don't horizontally flip the feed
python main.py --weights-dir PATH  # custom directory for model weights
```

---

## Running the tests

No webcam or model required:

```bash
python -m pytest test_logic.py -v
```

---

## How it works

```
main.py  ──► reads camera frames, throttles YOLO inference,
             calls game.update(labels, now), draws HUD
logic.py ──► pure state machine: no camera, no model, no cv2
             receives a list of labels + timestamp, returns state dict
weights/ ──► git-ignored; yolov8n.pt downloaded here on first run
```

### Scoring

- Up to **100 points** for finding an item instantly, down to **10 points**
  if you find it with only 1 second left on the clock.
- **+20 streak bonus** for each consecutive round you find the item
  (×1 for 1 in a row, ×2 for 2, …).
- Skipped or timed-out rounds score 0 and reset the streak.

### Hold timer

The target must stay visible for **1 second** before it counts.
One lucky frame cannot win a round.

### Detection throttle

YOLO runs every **3 frames** by default. The last result is reused on the
other frames. Adjust with `--detect-every` if your CPU struggles.

---

## Project structure

```
scavenger_hunt/
├── main.py          # camera loop, drawing, key handling
├── logic.py         # pure game state — no I/O
├── test_logic.py    # pytest suite for logic.py
├── requirements.txt
├── README.md
├── .gitignore       # excludes weights/, *.pt, runs/
└── weights/         # created at runtime, git-ignored
    └── yolov8n.pt   # downloaded on first run (~6 MB)
```

---

## What I learned

- Separating game logic (pure class, testable) from I/O (camera + model)
  makes the rules trivially testable without any hardware.
- A hold timer prevents one flukey detection frame from triggering a score.
- Throttling inference to every Nth frame keeps a CPU laptop at 30 fps while
  still detecting objects in near-real-time.
- `ultralytics` makes YOLOv8 inference a single function call; the hard part
  is the game state machine, not the model.
