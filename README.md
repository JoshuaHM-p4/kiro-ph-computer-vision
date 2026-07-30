# kiro-computer-vision

**A workshop repository for building your own computer vision app with Kiro.**

You are here to build something. Fork this repo, make a folder in
[`projects/`](projects/), and use Kiro to write the OpenCV and model code with you.
Everything else here — the prompt library, the working demos — is reference material to
borrow from.

> **New here?** The full step-by-step walkthrough lives in the
> [**workshop guide**](https://kiroversew7.notion.site/) (mirrored locally in
> [`docs/Notion.md`](docs/Notion.md)). It takes about five minutes to get to a
> running camera.

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

The full step-by-step instructions — forking, environment setup, prerequisites
(Hugging Face token, Python 3.10–3.12), and building with Kiro — live in the
**[workshop guide](https://kiroversew7.notion.site/)**, mirrored in this repo at
[`docs/Notion.md`](docs/Notion.md).

The short version:

1. **Fork and clone** this repository, then `cd kiro-computer-vision`.
2. **Set up the environment** from the repo root (Python 3.10–3.12):

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   pip install -r requirements.txt
   python check_setup.py              # verify everything is ready
   ```

3. **Make your folder** `projects/<your-github-username>/` and build there.
4. **Start Kiro from the repo root** (not inside your project folder — the
   `.kiro/steering` and `.kiro/skills` context only loads from the current
   directory):

   ```bash
   kiro-cli chat --trust-all-tools
   ```

   Then tell Kiro to build in `projects/<your-github-username>/`.

Open the **[prompt library](docs/prompts/)** and pick the prompt closest to your
idea — those are the actual prompts that produced the demos, cleaned up for reuse.
If you want a file layout to start from, copy
[`projects/your_example_app/`](projects/your_example_app/): a runnable camera loop,
its logic split into a pure module, and tests that need no webcam.

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
│   ├── sam_labeler/        SAM 3.1 text-prompted segmentation
│   ├── common/             Shared building blocks
│   └── tools/              Asset generators
└── projects/               ⬅️ WHERE YOU BUILD
    ├── your_example_app/   A file layout that works
    └── <your-username>/    Your project
```

## 🧠 The prompt library

[`docs/prompts/`](docs/prompts/) is the most useful thing here if you are building
something of your own. Ten documents, one per demo plus the general patterns:

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
| [SAM labeler](docs/prompts/09-sam-labeler.md) | Text-prompted segmentation, gated models, and what to do with a mask |

Each one records the follow-up questions that actually improved the demo, because the
first answer is never the finished thing.

## 🎥 The demos

Seven worked examples, each runnable as a desktop OpenCV window and as a Flask web app.
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
