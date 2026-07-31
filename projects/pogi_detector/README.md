# Pogi Detector 😎

A SAM-powered webcam app that segments people and objects using **Tagalog slang** as prompts.

Press **P** and the app translates "pogi" → "handsome person / face", runs segmentation, and highlights the detected person with a glowing outline and a big **"😎 POGI DETECTED! 😎"** banner.

## What it does

- Translates Tagalog slang to English object descriptions
- Runs SAM (Segment Anything Model) text-prompted segmentation — or a deterministic stub when offline
- Draws a stylish multi-layer glowing outline around segmented regions
- Shows an animated HUD banner when a detection triggers
- Works fully offline with the stub backend (no model download needed)

## Slang Dictionary

| Tagalog | English (SAM prompt) | Key |
|---------|---------------------|-----|
| pogi | handsome person / face | P |
| ganda | beautiful person | G |
| chibog | food / snack | C |
| tsismis | cell phone | T |

## Running

From the repository root with the venv active:

```bash
# Default (stub backend, works offline)
python projects/pogi_detector/app.py

# Specify camera
python projects/pogi_detector/app.py --camera 1

# With real SAM model (requires HF token + model download)
export HF_TOKEN=hf_your_token_here
python projects/pogi_detector/app.py --token $HF_TOKEN

# Force stub mode even if SAM is available
python projects/pogi_detector/app.py --stub
```

## Controls

| Key | Action |
|-----|--------|
| P | Toggle real-time Pogi scan (press again to stop) |
| G | Toggle real-time Ganda scan |
| C | Toggle real-time Chibog scan |
| T | Toggle real-time Tsismis scan |
| X | Stop scanning (clear all highlights) |
| S | Save screenshot to `screenshots/` |
| Q / ESC | Quit |

Once a scan is active, segmentation runs continuously on every frame (throttled by `--scan-interval`, default 0.3s) so the highlight tracks the subject in real time. Press the same key again or **X** to stop.

## Testing

```bash
pytest projects/pogi_detector/test_pogi.py -v
```

All tests use synthetic data — no webcam or SAM model needed.

## Architecture

```
projects/pogi_detector/
├── app.py            OpenCV camera loop, HUD drawing, key controls
├── segmentor.py      SAM backend (real + stub fallback), Instance dataclass
├── translator.py     Tagalog slang → English dictionary mapper
├── test_pogi.py      Unit tests for translator and segmentor
├── screenshots/      Saved screenshots (git-ignored)
└── README.md         This file
```

### Design principles

- **I/O separated from logic**: `translator.py` and the segmentor's classification logic are pure — no camera, no display. The camera loop in `app.py` is a thin adapter.
- **Never crashes on missing model**: `load_backend()` always returns a working backend (stub if SAM is unavailable).
- **Scale-relative rendering**: The glow outline adapts to mask size, not pixel counts.
- **Hysteresis-free for scan mode**: Since detection is user-triggered (not continuous), there's no flicker problem.

## Using real SAM

1. Get a [Hugging Face token](https://huggingface.co/settings/tokens) with read access
2. Accept the model license at the model page
3. Run with `--token` or set `HF_TOKEN` environment variable
4. First run downloads weights (~2 GB) to `~/.cache/huggingface/hub/`

The stub backend produces the same UI experience with synthetic ellipse masks, so you can explore the full app without waiting for the download.
