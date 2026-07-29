# Demo suite

Reference implementations for the [kiro-computer-vision](../README.md) workshop. These
are here to read and borrow from — **you build your own thing in
[`projects/`](../projects/)**, and the prompts that produced these demos are in
[`docs/prompts/`](../docs/prompts/).

Six interactive OpenCV + MediaPipe demos. Each one runs two ways:

- **desktop** — an OpenCV window, vision by MediaPipe Python
- **web** — a Flask page where MediaPipe runs *in the browser* and streams
  landmarks to Flask over a WebSocket

| Demo | What it does | Desktop | Web |
|---|---|---|---|
| Image Lab | Upload an image, chain OpenCV operations, reveal the equivalent code | `python -m demos.image_lab.desktop` | `/image-lab/` |
| Air Canvas | Paint with your fingertip; dwell on a palette to pick color, size, opacity; pinch to erase | `python -m demos.air_canvas.desktop` | `/air-canvas/` |
| Slide Presenter | Index finger is a laser; pinch right for next, left for previous | `python -m demos.slide_presenter.desktop` | `/slides/` |
| 6-7 Counter | Counts alternating-hand reps from pose | `python -m demos.six_seven_counter.desktop` | `/six-seven/` |
| PNGTuber | Head yaw + expression pick a sprite | `python -m demos.pngtuber.desktop` | `/pngtuber/` |
| Scavenger Hunt | "Bring me a cup" - find real objects on camera before the timer ends | `python -m demos.scavenger_hunt.desktop` | `/scavenger-hunt/` |

## Setup

The demos run on the repository's root virtualenv and use the pins in the root
`requirements.txt` — no separate dependency file. Build it with **Python 3.12**:
mediapipe 0.10.x publishes no wheels for 3.13+, so a newer interpreter fails with
*"No matching distribution found for mediapipe"*.

```bash
/usr/bin/python3.12 -m venv .venv          # python3.12 -m venv .venv on Windows
.venv/bin/pip install -r requirements.txt
```

Use `.venv/bin/python` explicitly, or `source .venv/bin/activate` first — a bare
`pip` often resolves to `~/.local/bin/pip` and installs to the wrong place.

Run every command below from the repository root, since the demos import as
`demos.*`.

Generate the assets. They are **not** committed — each has a generator, and every
demo detects them missing and prints the command — so run these once after cloning:

```bash
.venv/bin/python -m demos.tools.make_sample_slides         # the 6-slide project deck
.venv/bin/python -m demos.tools.make_sample_images         # image lab samples
.venv/bin/python -m demos.tools.make_placeholder_sprites   # 12 sprites
```

## Run

```bash
# Web hub with all four demos on one port
.venv/bin/python -m demos.home.web                # http://127.0.0.1:5000/

# Futuristic OpenCV menu that launches the desktop demos
.venv/bin/python -m demos.home.opencv_menu
.venv/bin/python -m demos.home.opencv_menu --no-camera   # keyboard only

# A single demo, standalone
.venv/bin/python -m demos.air_canvas.web          # http://127.0.0.1:5001/air-canvas/
.venv/bin/python -m demos.air_canvas.desktop
```

Port map: hub `5000`, air canvas `5001`, slides `5002`, 6-7 counter `5003`,
PNGTuber `5004`, image lab `5005`, scavenger hunt `5006`, echo diagnostic `5009`. Change any of them with `--port`.

### Image Lab

The odd one out: it is not landmark-driven, and it is the place to learn the OpenCV
API itself. Load a sample, upload your own image, or grab a single webcam frame,
then stack up to eight operations and watch the result update.

27 operations across eight categories:

| Category | Operations |
|---|---|
| Color | grayscale, brightness/contrast, hue shift, equalise / CLAHE |
| Blur | Gaussian, median, bilateral, sharpen (`filter2D`) |
| Threshold | fixed & Otsu, adaptive, HSV `inRange` mask |
| Edges | Canny, Sobel, Laplacian |
| Morphology | erode, dilate, open, close, gradient, tophat, blackhat |
| Geometry | resize, rotate (`warpAffine`), flip, crop |
| Features | contours + boxes + centroids, Hough lines, corner points |
| Drawing | rectangle, circle, line/arrow, polygon + vertex markers, text |

**Reveal code** prints a runnable script for the pipeline you built, with your
parameter values baked in:

```python
import cv2
import numpy as np

img = cv2.imread("shapes.png")

# 1. Gaussian blur - Weighted average with a Gaussian kernel: the default smoother.
img = cv2.GaussianBlur(img, (9, 9), 0.0)

# 2. Canny edges - Gradient edges with hysteresis between two thresholds.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 180, apertureSize=3)
img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

cv2.imwrite("output.png", img)
```

The snippet and the preview come from the same table in
`demos/image_lab/operations.py` — each operation declares how to run itself *and*
how to describe itself — so the code shown cannot drift from the image displayed.
Adding an operation means adding one `Operation` entry; the UI, the desktop
trackbars, the code output and the tests all pick it up automatically.

The desktop version uses **`cv2.createTrackbar`**, the native OpenCV way to
explore parameters: one operation at a time, every parameter on a trackbar, `n`/`p`
to move through the catalog and `r` to print the code. `--list` prints the whole
catalog with the cv2 function each entry maps to.

Sample images are generated to suit particular operations — shapes for contours, a
noisy gradient for the blurs, an unevenly lit document that defeats a global
threshold but not an adaptive one, and colour blocks for HSV masking:

```bash
.venv/bin/python -m demos.tools.make_sample_images
```

### Scavenger Hunt

A game on top of object detection. "BRING ME A CELL PHONE" appears with a 30 second
timer; hold the real thing up to your webcam, or upload a photo of it. Five rounds,
points for speed, a streak bonus, and a scoreboard at the end. Targets are drawn
from the COCO classes you plausibly have at a desk — cup, cell phone, scissors,
potted plant, book, keyboard, remote, banana and so on — which is what makes it
work with zero setup data: the model already knows these 80 classes.

**No model file to supply.** Ultralytics ships COCO-pretrained weights and fetches
them on first use, so this just works:

```bash
.venv/bin/python -m demos.scavenger_hunt.web        # downloads yolo26n.pt (5 MB)
.venv/bin/python -m demos.scavenger_hunt.desktop
```

Weights land in the git-ignored `models/` folder rather than the working directory,
and are never committed (see `AGENTS.md`). Swap the model any time:

```bash
--model yolo11n.pt            # any name ultralytics knows, downloaded on demand
--model ~/mine/best.pt        # your own checkpoint
--model models/yolo26n.onnx   # an ONNX export: runs on cv2.dnn, no torch needed
export SCAVENGER_MODEL=yolo26s.pt
```

Note on install size: torch comes in as an ultralytics dependency, and installing
it plainly from PyPI pulls the CUDA build plus 3-5 GB of nvidia wheels. The CPU
build is about 250 MB and is all these demos need, which is why `requirements.txt`
asks you to install it from PyTorch's CPU index first. Skip that step if you want
GPU acceleration.

If loading ever fails — no internet on first run, a corrupt checkpoint, ultralytics
missing — `load_detector` degrades to a null backend and the page explains what
happened instead of returning a 500. The game still runs; nothing is detected.

The ONNX path exists for torch-free deployments. Its decoder sniffs the output shape
rather than assuming one layout, since that changed between YOLO generations:
`(1, 4+classes, anchors)` and its transpose are decoded and passed through NMS, while
`(1, n, 6)` rows of `[x1, y1, x2, y2, score, class]` — what end-to-end, NMS-free
exports such as YOLO26 produce — are taken as-is.

**Settings, in game.** *Reveal settings* on the page opens sliders for the round
timer (5-180 s), the **confidence needed to pass** (5-95%), rounds per game, and the
hold window. The desktop app has the same controls on keys: `+`/`-` for the timer and
`[`/`]` for confidence, echoed to the terminal and shown in the status strip.

Settings are **per player**: each browser session gets its own copy of the config, so
one person shortening their timer cannot change anyone else's game. Values are clamped
server side against the same bounds the sliders publish, so a hand-written request
cannot set a 0% threshold or a one-hour round.

Confidence applies immediately, since it only changes what counts as a detection.
The timer, round count and hold window apply **from the next round** — shortening the
clock mid-round would otherwise retroactively end a round you are still playing.
Lowering the confidence also discards any hold progress earned under the old
threshold rather than crediting it.

Two details that make the game feel fair:

* **A hold window.** The target must be visible for ~0.4 s before it counts, so one
  flukey frame cannot win a round. Uploaded photos take a separate path
  (`submit_photo`) because a still cannot be "held".
* **The server owns the clock.** Timers use `time.monotonic()` on the server, so a
  client cannot win by lying about elapsed time.

This demo also inverts the usual arrangement here: YOLO has no browser build in this
project, so the page posts **JPEG frames** at about 4 fps and the server runs
inference, instead of streaming landmarks over a WebSocket.

## Controls

Shared desktop keys: `q`/`ESC` quit, `d` debug overlay, `SPACE` mirror,
`s` screenshot (written to `demos/screenshots/`).

| Demo | Extra keys | Gestures |
|---|---|---|
| Image Lab | `n` / `p` next / previous operation, `r` reveal code, `c` compare, `s` save, SPACE re-grab webcam | none: mouse and trackbars |
| Air Canvas | `c` clear, `z` undo, `[` `]` size, `-` `=` opacity, `g` gesture legend, `f` face mesh, `h` hand skeleton | **index only** = draw; **index + middle** = hover/select on the rails; **thumb-index pinch** = erase; fist = idle |
| Slide Presenter | `n`/`→` next, `p`/`←` previous, `r` reload deck, `c` webcam inset, `w` swap hands | point = laser; pinch right = next; pinch left = previous |
| 6-7 Counter | `r` reset, `a` add one, `p` re-prepare, `g` see-saw overlay, `k` skeleton | alternate hands like a see-saw: as one rises the other drops |
| PNGTuber | `c` recalibrate neutral, `b` background (camera/solid/chroma), `v` camera inset, `1`-`4` preview an expression | turn your head; smile / raise brows + open mouth / lower brows or squint |
| Scavenger Hunt | SPACE start / restart, `n` skip the item, `m` model info, `+`/`-` timer, `[`/`]` confidence | hold the named object up to the camera |

The air canvas keeps an on-screen **gesture legend** that highlights whichever
hand shape it currently recognises, which doubles as feedback that the gesture
landed. The size and opacity bars are labelled with their name and value
underneath, so a fingertip resting on a bar never hides the reading.

The presenter ships with a **six-slide deck about this project** — a title card,
four content slides (what the camera gives us, the architecture, making gestures
reliable, what we built) and a closing card. Regenerate it with
`demos.tools.make_sample_slides`, edit the copy in `DECK` at the top of that file,
or just drop your own images into `demos/slide_presenter/assets/slides/` and press
`r` to reload. Files are read in natural filename order.

The presenter shows a **webcam picture-in-picture in the bottom-left corner** in
both the desktop window and the browser, so you can see what the tracker sees
while the slide fills the rest of the frame.

### How the counter measures a rep

It watches the **difference** between the two wrist heights, not each wrist
against a fixed line:

```
tilt = (right_wrist_y - left_wrist_y) / torso_length     # positive: left higher

tilt >= +deadband  ->  side = left
tilt <= -deadband  ->  side = right
in between         ->  hold the previous side

one rep = one change of side
```

Three things fall out of that, which is why it replaced the earlier
threshold-per-wrist approach:

* **Nothing to calibrate.** There is no "high enough" to tune, so it works for
  tall and short users, seated or standing.
* **Body movement cancels.** Raising or lowering both arms together, or the whole
  body drifting in frame, does not change the difference — only a genuine swap
  does.
* **Alternation is intrinsic.** A sign flip is impossible unless both hands take
  part, so pumping one hand twice *cannot* register. No extra rule needed.

The shaded band in the on-screen tilt meter is the deadband: with hands roughly
level the current side is held, so jitter never counts. `tilt_enter` (default 0.15
torso lengths, `--tilt` on the desktop app) sets how big a swap has to be.

The counter opens in **prepare mode**: it waits until both shoulders and both
hands are inside the frame, held steady for about a second and a half, before it
starts counting, and it tells you what is missing ("SHOW BOTH HANDS"). Hips are
*not* required — you can stand close enough to fill the frame. When they are out
of shot the scale reference falls back from shoulder-to-hip length to shoulder
width (times `shoulder_to_torso`), so the deadband keeps the same meaning. The rep count is drawn over the video, not just in the side panel. A
brief tracking dropout is tolerated; only a longer one sends it back to prepare,
and the count is never lost.

OpenCV menu: arrow keys or `h j k l` to move, `ENTER` to launch, `1`-`4` direct,
or hover a card with your hand and hold still.

## How it fits together

Every demo keeps its logic in a pure `core.py`: normalized landmarks and a clock
go in, a state dict comes out. Nothing in a core touches a camera, a window, or
Flask, so the desktop loop and the WebSocket handler are both thin adapters over
one implementation — the gestures cannot behave differently between them, and the
logic is testable without hardware.

```
desktop:  cv2.VideoCapture -> mp.solutions -> LandmarkFrame -> core -> cv2 HUD -> window
web:      getUserMedia -> tasks-vision (JS) -> WebSocket {seq, landmarks}
                                             -> FrameGuard -> core -> state JSON -> canvas
                                                                   -> /snapshot -> PNG
```

Shared building blocks live in `demos/common/`:

| Module | Contents |
|---|---|
| `geometry.py` | distances, EMA smoothing, `HysteresisLatch`, `EdgeTrigger`, `DwellTimer`, FPS |
| `landmarks.py` | landmark index tables, `Hand`/`Face`/`Pose`/`LandmarkFrame`, JSON codec |
| `gestures.py` | pinch, finger extension, EAR/MAR, head pose, torso-relative wrist height |
| `detectors.py` | `VisionPipeline` wrapping `mp.solutions` Hands/FaceMesh/Pose |
| `camera.py` | `CameraLoop`: capture, mirroring, keys, teardown |
| `hud.py` | the neon theme: panels, glow, gauges, landmark overlays, alpha compositing |
| `webapp.py` | blueprint factory, WebSocket channel, `FrameGuard`, `/snapshot` |

### Web endpoints per demo

`GET /` page · `GET /ws` landmark stream · `POST /landmarks` HTTP fallback ·
`POST /command` · `POST /reset` · `GET /state` · `GET /snapshot` PNG ·
`GET /health`. Sessions are keyed by a `?sid=` the browser generates, so several
tabs stay independent and `/snapshot` returns *your* canvas.

### Frame guard

Each message carries a monotonically increasing `seq`. The server drops anything
that is not newer, and answers with the previous state marked
`_meta.skipped = true`, so a congested socket degrades into a lower frame rate
instead of corrupted gesture state. The browser also skips a frame whenever the
socket still has buffered bytes, rather than queueing stale landmarks.

### Robustness patterns used throughout

- **Hysteresis everywhere.** Pinch, wrist up/down, and yaw buckets all use
  separate enter and release thresholds, so a hand parked at a boundary cannot
  chatter.
- **Scale invariance.** Hand thresholds divide by the hand span, face ratios by
  the interocular distance, pose thresholds by torso length. Moving closer to the
  camera does not change behaviour.
- **Dwell to activate.** Palette cells and menu cards need a sustained hover, so
  a fingertip sweeping past does not trigger everything it crosses.
- **Yaw compensation.** Every PNGTuber expression ratio divides a vertical
  distance by a horizontal one, and turning your head foreshortens the horizontal
  denominator by `cos(yaw)`. Left uncorrected, a relaxed face at 40 degrees reads
  as surprised, so the ratios are multiplied back by `cos(yaw)` using the current
  frame's angle. Disable it with `yaw_compensation=False` to see the difference.

## Browser models and offline use

By default the web pages load `@mediapipe/tasks-vision@0.10.21` from jsDelivr and
the `.task` models from `storage.googleapis.com`. That is roughly **15 MB on a
cold cache**, so the first load can sit on "loading model" for a minute on a slow
link — and it needs internet at all.

Vendor the assets locally instead (strongly recommended):

```bash
.venv/bin/python -m demos.tools.vendor_web_assets          # ~36 MB, once
.venv/bin/python -m demos.tools.vendor_web_assets --check   # verify
```

`landmark-stream.js` probes `/shared/static/vendor/vision_bundle.mjs` on startup
and switches to the local copies automatically when they exist — no code change
needed. The files are git-ignored, and the status line shows *local assets* once
running so you can tell which path is in use.

While loading, the page reports each stage (`loading runtime`, `loading model
hands (~7.8 MB)`, `opening camera`, `connecting`) and, if a stage takes more than
six seconds, explains that it is a large first-time download. A stage that fails
turns the pill red and prints the error, rather than sitting on the last message.

The GPU delegate is tried first and falls back to CPU automatically, so browsers
with hardware acceleration disabled still work (slower).

## Tuning

Each demo has a `config.py` dataclass; the desktop apps expose the most useful
values as CLI flags.

| Where | Knob | Effect |
|---|---|---|
| `air_canvas/config.py` | `dwell_seconds`, `size_min/max`, `eraser_scale`, `min_move` | palette responsiveness, brush range |
| `slide_presenter/config.py` | `advance_cooldown`, `pinch_start_ratio`, `laser_alpha` | how eagerly pinches register, laser steadiness |
| `scavenger_hunt/config.py` | `round_seconds`, `confidence`, `rounds`, `hold_seconds`, `SETTING_BOUNDS` | game defaults, and the limits the in-game settings panel clamps to |
| `six_seven_counter/config.py` | `tilt_enter` (0.15 body lengths), `min_swap_seconds`, `prepare_seconds`, `lost_grace_seconds`, `shoulder_to_torso` | how big a swap must be, how fast swaps may come, how long the prepare hold lasts, how much lost tracking is tolerated |
| `pngtuber/config.py` | `yaw_enter`/`yaw_release`, `*_delta`, `calibration_seconds`, `yaw_compensation` | bucket width, expression sensitivity, yaw/expression decoupling |
| any | `--swap-handedness` | fixes inverted left/right hand labels |

## Tests

```bash
.venv/bin/python -m pytest -c demos/pytest.ini        # 674 tests, no camera or model needed
```

Everything is driven by synthetic landmarks from `demos/tests/fixtures.py`
(`make_hand`, `make_face`, `make_pose`), so gesture behaviour, HTTP routes, and
renderers are all verified headlessly. The only interactive parts are the camera
loops themselves.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No matching distribution found for mediapipe` | venv is not Python 3.10-3.12. Rebuild it with `/usr/bin/python3.12 -m venv .venv`. |
| `Could not open camera 0` | Another program holds the webcam, or it is a different index. Try `--camera 1`; check `ls /dev/video*`. |
| Next/previous slides are swapped | Desktop: press `w` or pass `--swap-handedness`. Browser: press **Swap hands** on the page. MediaPipe assigns handedness assuming a mirrored selfie image, and some cameras still report it inverted. |
| Stuck on "loading model" or "loading runtime" | The cold-cache CDN download is ~15 MB. Watch the status line for the stage and size; to make it instant and offline-capable, run `.venv/bin/python -m demos.tools.vendor_web_assets`. |
| Pill turns red with an error | The error text is printed under the video. A timeout there means a blocked or very slow CDN — vendor the assets as above. |
| Camera works in the desktop app but not the browser | Browsers only allow `getUserMedia` on `https://` or `localhost`. Use `127.0.0.1`, not a LAN IP. |
| Everything reads as one expression (PNGTuber) | The neutral baseline captured a non-neutral face. Press `c` (or *Recalibrate*) with a relaxed face. |
| Expression changes when you turn your head | Calibrate facing the camera. If it persists, raise `min_yaw_cosine` or the `*_delta` thresholds in `pngtuber/config.py`. |
| Reps do not count | Check the phase: while it says *prepare*, get your shoulders and both hands in frame and hold still (or press **Skip prepare**). Then make sure one hand is clearly higher than the other — the tilt marker has to leave the shaded deadband. |
| Reps count too easily / not easily enough | Change `tilt_enter` (`--tilt` on the desktop app). It is the wrist height difference needed to claim a side, in torso lengths: lower is more sensitive, higher demands bigger swaps. |
| `mediapipe` prints TFLite/XNNPACK warnings at startup | Normal. Harmless if the window opens. |

## Security

The Flask apps bind to `127.0.0.1` and have **no authentication**. Anything that
can reach the port can drive the demo state and read `/snapshot`. Passing
`--host 0.0.0.0` exposes an unauthenticated, webcam-driven service to your whole
network; the runner prints a warning if you do. Keep it on loopback unless you
add auth yourself.