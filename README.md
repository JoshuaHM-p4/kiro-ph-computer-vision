# kiro-computer-vision

**A workshop repository for building your own computer vision app with Kiro.**

You are here to build something. Fork this repo, make a folder in
[`projects/`](projects/), and use Kiro to write the OpenCV and model code with you.
Everything else here — the prompt library, the working demos — is reference material to
borrow from.

> **New here?** Jump to [Workshop instructions](#-workshop-instructions) and start at
> Step 1. It takes about five minutes to get to a running camera.

---

## Kiroverse Week 7: Build Nights — Computer Vision Applications with Kiro

Building on what we've covered, it's time to build a computer vision app powered by
Kiro! In this intermediate module, we are diving into image and video stream processing
using Python and OpenCV.

From setting up image preprocessing pipelines to integrating real-time object detection
models, you will learn how to guide Kiro's AI agents to write clean vision code and
handle complex dependencies.

### 🚀 The Agenda

* 🧠 **Foundations:** Understand key computer vision concepts, model architectures, and image processing basics.
* 🤖 **Model Integration:** Build & run object detection and classification models using OpenCV.
* 📑 **Hands-On Building:** Work together with Kiro to build a vision app, write preprocessing scripts, and run live inference.
* 💡 **Project Showcase & Q&A:** Present your computer vision project, ask questions, debug your code, and get feedback from facilitators.

---

## 🛠️ Workshop instructions

For this Build Night you will fork this repository, build your computer vision app in
your own folder, and use Kiro to help write your code.

### Step 1: Fork and clone

1. Click **Fork** at the top right of this repository to create a copy on your account.
2. Clone your fork:

```bash
git clone https://github.com/<YOUR-USERNAME>/kiro-computer-vision.git
cd kiro-computer-vision
```

### Step 2: Create your workspace

Make a folder in `projects/` named after your GitHub username. All your code lives
there, so pulling updates to the shared parts never conflicts with your work.

```bash
mkdir projects/<your-github-username>
```

### Step 3: Set up your environment

From the repository root. **Python 3.10–3.12** — mediapipe publishes no wheels for
3.13+, so a newer interpreter fails at install time with
*"No matching distribution found for mediapipe"*.

```bash
python3.12 -m venv .venv

source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# torch arrives with ultralytics. Install the CPU build first unless you want
# 3-5 GB of CUDA wheels you will not use:
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
```

Check it worked:

```bash
python -c "import cv2, mediapipe, numpy; print(cv2.__version__, mediapipe.__version__)"
```

### Step 4: Start building with Kiro

Run Kiro from inside your own folder so it works in your project's context:

```bash
cd projects/<your-github-username>
kiro-cli chat
```

Now open the **[prompt library](docs/prompts/)** and pick the prompt closest to your
idea. Those are the actual prompts that produced the demos in this repo, cleaned up for
reuse — swap in your subject matter and keep the constraints.

If you want a file layout to start from, copy
[`projects/your_example_app/`](projects/your_example_app/): a runnable camera loop, its
logic split into a pure module, and tests that need no webcam.

---

## 📁 Repository layout

```
kiro-computer-vision/
├── README.md               You are here
├── requirements.txt        Pinned dependencies for everything
├── AGENTS.md               Conventions Kiro follows in this repo
├── docs/
│   ├── PRD.md              What this project is
│   └── prompts/            ⬅️ THE PROMPTS THAT BUILT THE DEMOS
├── demos/                  Reference implementations to borrow from
│   ├── README.md           Demo docs: controls, ports, tuning, troubleshooting
│   ├── image_lab/          Interactive cv2 playground
│   ├── air_canvas/         Fingertip painting
│   ├── slide_presenter/    Pinch to change slides
│   ├── six_seven_counter/  Rep counting from pose
│   ├── pngtuber/           Head yaw + expression avatar
│   ├── scavenger_hunt/     COCO object detection game
│   ├── common/             Shared building blocks
│   └── tools/              Asset generators
└── projects/               ⬅️ WHERE YOU BUILD
    ├── your_example_app/   A file layout that works
    └── <your-username>/    Your project
```

## 🧠 The prompt library

[`docs/prompts/`](docs/prompts/) is the most useful thing here if you are building
something of your own. Nine documents, one per demo plus the general patterns:

| Prompt | Teaches |
|---|---|
| [How to prompt for vision code](docs/prompts/00-how-to-prompt.md) | The shape all the others follow, and the constraints worth reusing verbatim |
| [Image lab](docs/prompts/01-image-lab.md) | The OpenCV API itself: filters, thresholds, contours, drawing |
| [Air canvas](docs/prompts/02-air-canvas.md) | Hand landmarks and turning them into intent |
| [Slide presenter](docs/prompts/03-slide-presenter.md) | Handedness, edge-triggered actions, cooldowns |
| [Rep counter](docs/prompts/04-rep-counter.md) | Choosing a signal that cannot drift |
| [PNGTuber](docs/prompts/05-pngtuber.md) | Head pose, facial ratios, per-user calibration |
| [Scavenger hunt](docs/prompts/06-scavenger-hunt.md) | Pretrained detection models, and the install trap |
| [Flask web layer](docs/prompts/07-flask-web-layer.md) | Getting any of it into a browser |
| [Testing vision code](docs/prompts/08-testing-vision-code.md) | A suite that runs with no webcam |

Each one records the follow-up questions that actually improved the demo, because the
first answer is never the finished thing.

## 🎥 The demos

Six worked examples, each runnable as a desktop OpenCV window and as a Flask web app.
Read them, run them, take what you need — they are reference material, not the point of
the repository.

```bash
# One-time asset setup
python -m demos.tools.make_sample_slides
python -m demos.tools.make_sample_images
python -m demos.tools.make_placeholder_sprites

# All six in the browser, on one port
python -m demos.home.web            # http://127.0.0.1:5000/

# Or an OpenCV launcher menu for the desktop versions
python -m demos.home.opencv_menu
```

**Full documentation is in [`demos/README.md`](demos/README.md)** — controls, gesture
maps, port numbers, tuning knobs, architecture and troubleshooting.

## ✅ Testing

```bash
python -m pytest -c demos/pytest.ini    # the demo suite: no camera or model needed
python -m pytest projects/<your-name>   # your own tests
```

## 📋 Prerequisites

* Python 3.10–3.12 and `pip` / `venv`
* A webcam, or sample images and video files
* Kiro CLI
* Internet on first run: MediaPipe browser assets and YOLO weights download on demand

## 🎤 Showcase

When you are done, make sure your `projects/<your-username>/README.md` says what you
built, how to run it, and what surprised you. That last part is usually the most
interesting thing in the room.

## Contributing

See [`AGENTS.md`](AGENTS.md) for the coding conventions Kiro follows in this repository,
and how to work effectively with it here.

## License

MIT License
