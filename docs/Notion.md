# Workshop Overview

## Objectives

By the end of the session you will have:

- Used Kiro (CLI or IDE) to generate, scaffold, and debug a vision project on your own machine
- Chained OpenCV operations on a live webcam feed (resize, threshold, mask, draw)
- Run YOLO26n for real-time 80-class object detection, entirely on CPU
- Seen SAM 3.1 segment objects from a text prompt (facilitator demo; optional to try yourself if bandwidth allows)
- Built and presented a working webcam application of your own design

## What we'll build

We cover the foundations together (OpenCV, real-time detection, segmentation), then you fork this repo, make a folder in `projects/`, and build whatever webcam app you want. Kiro writes most of the code; you steer it.

The seven demo apps in `demos/` are there to run, read, and borrow from. You can rebuild one, mash two together, or ignore them and do something completely different. The prompt library (`docs/prompts/`) has the exact prompts that produced each demo, cleaned up so you can paste them into Kiro and swap in your own subject.

| Technology | Role in this workshop |
| --- | --- |
| Python 3.10-3.12 and OpenCV | Webcam capture, frame manipulation, all on-screen drawing |
| Ultralytics YOLO26 (PyTorch, CPU) | Real-time COCO detection. The nano model (`yolo26n.pt`, ~5 MB) runs end-to-end without a separate NMS step and downloads itself on first use |
| Meta SAM 3.1 | Text-prompted segmentation. Type "laptop" and get the pixel mask back. The model is ~6.5 GB, so we show it on a pre-loaded machine rather than downloading in the room |
| MediaPipe | Hand, face, and pose landmark tracking (21 / 478 / 33 points). Powers the gesture demos |
| Flask + flask-sock | Lightweight web framework already included. Every demo has a browser version you can run locally as a dashboard, and you can do the same for your own app. See prompt 07 in the library. |
| Kiro (CLI or IDE) | Local coding agent. Writes boilerplate, manages dependencies, helps you iterate |
| GitHub | You fork the repo, build in your folder, push when done |

---

# Reminders

- **Webcam preferred, not required.** A webcam makes the demos more fun, but you can also work with uploaded images or video files. If you have a webcam, confirm it works before the session.
- **Local setup.** We are using Kiro locally, CLI or IDE, your choice. Install whichever you prefer ahead of time so we don't lose session time to setup.
- **Public repo required.** Your GitHub project must be set to public so it can be submitted.
- **Python 3.10-3.12 only.** mediapipe has no wheels for 3.13+. Check with `python --version` before arriving.

# Prerequisites

1. Laptop or desktop with a working webcam (preferred) or sample images/videos
2. Kiro account (Builder ID, GitHub, or social login)
3. GitHub account
4. Hugging Face account (free, at [huggingface.co](https://huggingface.co)):
    - Sign up, then go to <https://huggingface.co/facebook/sam3.1> and fill out the access form (name, date of birth, country, affiliation, job title) and accept Meta's license and privacy policy. Submit it; approval is usually instant.
    - Create a read token at <https://huggingface.co/settings/tokens> (you will paste it into the SAM Labeler demo).
5. Python 3.10-3.12 installed
6. Kiro installed and signed in (pick whichever you prefer, both give you the same agent):

    Kiro CLI (terminal):

    ```bash
    curl -fsSL https://cli.kiro.dev/install | bash
    ```

    (Windows users: see the PowerShell install steps at kiro.dev/docs/cli/installation. Native support, WSL not required.)

    Kiro IDE (desktop app): download from [kiro.dev/downloads](https://kiro.dev/downloads) and sign in.

---

# Key concepts

- **Image preprocessing pipeline** — The steps a raw webcam frame goes through (resize, colour conversion, normalization) before you do anything useful with it.

- **Detection vs. classification** — Classification says "there is a face in this image." Detection says "there is a face, and it is at these coordinates."

- **Segmentation** — Every pixel gets a label. SAM 3.1 does this from a text prompt alone, no training data needed.

- **Landmarks** — MediaPipe outputs 21 hand points, 478 face points, or 33 body points, all as 0..1 coordinates. The gesture demos turn those into intent (pinch, point, raise).

- **Steering files** — Markdown in `.kiro/steering/` that Kiro loads on every session. They carry project conventions so you don't repeat yourself. Already configured in this repo.

---

# Project setup

1. Confirm Kiro (CLI or IDE) is installed and signed in (see Prerequisites).

2. Fork the repository on GitHub:
    - Open <https://github.com/JoshuaHM-p4/kiro-ph-computer-vision> in your browser.
    - Click the **Fork** button in the top-right corner of the page.
    - On the "Create a new fork" screen, leave the defaults and click **Create fork**.
    - You now have your own copy at `github.com/<YOUR-USERNAME>/kiro-ph-computer-vision`.

3. Clone your fork to your machine:
    - On your fork's page, click the green **Code** button and copy the HTTPS URL.
    - Open a terminal and run:

    ```bash
    git clone https://github.com/<YOUR-USERNAME>/kiro-ph-computer-vision.git
    cd kiro-computer-vision
    ```

4. Set up the environment:

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate

    # Option A: CPU only (recommended for most laptops, ~250 MB):
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

    # Option B: If you have an NVIDIA GPU and want CUDA acceleration, skip the line
    # above and install torch normally (pip will pull the CUDA build, 3-5 GB):
    # pip install torch torchvision
    #
    # Check your CUDA version with: nvidia-smi
    # For a specific CUDA version (e.g. 12.4):
    # pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

    # Then install everything else:
    pip install -r requirements.txt
    ```

5. Run the setup check:

    ```bash
    python check_setup.py
    ```

    Fix whatever it flags. Once everything passes, you're ready to build.

6. Open Kiro **from the repository root** and confirm it can see the `.kiro/steering` files:

    ```bash
    # Stay in the repo root — do NOT cd into projects/<username>.
    # Kiro only loads .kiro/steering and .kiro/skills from the current
    # working directory, and they live at the repo root.
    kiro-cli chat
    ```

    Then just tell Kiro to build in your folder, e.g. "Build my project in
    `projects/<your-github-username>/`." Or open the root folder in Kiro IDE.

---

# Challenge: Build a computer vision app

## Build it with Kiro

1. **Pick your concept.** Recreate a demo, remix one, or invent something new. Ideas:
    - A gesture-controlled music player
    - A posture checker that warns when you slouch
    - A sign language letter recogniser
    - Face filters (hats, glasses, moustaches)
    - A "don't blink" challenge game
    - A colour-based object sorter
    - Anything with a webcam that makes you want to show it off

2. **Open the prompt library** (`docs/prompts/`) and grab the closest starting prompt. Swap in your subject matter, keep the constraints.

3. **Work through it with Kiro:**
    - Open Kiro **from the repository root** (if you haven't already). (Don't `cd` into your project folder)
      Then Kiro will load `.kiro/steering` and `.kiro/skills` only from the current directory, and they live at the root:

      ```bash
      kiro-cli chat --trust-all-tools
      ```

    - Then just tell Kiro what you want to build in plain language:
      - "I want to build a posture checker"
      - "Help me recreate the scavenger hunt"
      - "I want to make a gesture-controlled music player"
    - Kiro will ask you a few questions about your idea (input source, model choice, what the output looks like), then scaffold the project and build it with you step by step. The Flask web layer for your presentation is included automatically.
    - You can also paste a prompt from `docs/prompts/` directly if you prefer to skip the questions.
    - Test locally with your webcam running to confirm it actually works, not just that the code compiles.

4. **Iterate.** The first output is never the finished thing. Describe what you see:
    - "It flickers between two states when my hand is still"
    - "It only works when I sit close to the camera"
    - "It counted twice when I only did it once"

    Each of those has a known fix. Kiro will reach for the right pattern if you describe the symptom plainly.

5. **Web dashboard.** Kiro includes a Flask web layer automatically when it scaffolds your project (it's required for the showcase). If you need to customise it further, see [prompt 07 - Flask web layer](prompts/07-flask-web-layer.md).

6. **Run it.** Make sure your venv is active, then:

    ```bash
    source .venv/bin/activate          # Windows: .venv\Scripts\activate

    # Desktop version (OpenCV window):
    python projects/<your-username>/main.py

    # Web version (Flask dashboard, open in browser):
    python projects/<your-username>/web.py
    ```

7. **Test it.** `python -m pytest projects/<your-username>` should pass without a webcam plugged in.

> **Reminder:** Be ready! Participants may be randomly selected to demo their build live on their webcam during the Project Showcase.

---

# Workshop recap

What you did today:

- Explored OpenCV interactively and saw the code behind every slider
- Ran YOLO26 detection on real objects from your own desk
- Watched SAM 3.1 segment objects from typed words
- Built your own webcam app by prompting Kiro with a reusable template
- Debugged real issues (flickering, drift, false positives) by describing symptoms instead of guessing at code

What you practiced:

- Preprocessing, detection, segmentation, and landmark tracking
- Running a local AI coding agent from a terminal or IDE
- Turning a prompt template into a working app through iteration
- The patterns that make vision code reliable (hysteresis, scale invariance, separating logic from I/O)

---

## What's next?

- **Submit your project and feedback form:** <https://forms.gle/HVej5YSJ3NLtorSZA>
- **Share your experience:** Post your build on LinkedIn. Tag Kiro Community and AWS User Group PH. Use #kiroverse.
- **Stay connected:**
  - Facebook: <https://www.facebook.com/kirocommunity/>
  - LinkedIn: <https://linkedin.com/company/kirocommunity>
  - Luma (events): <https://luma.com/kirocommunity>

---

*Kiro is a trademark of Amazon Web Services, Inc. (AWS) or its affiliates. The Kiroverse name is used by AWS Community Philippines with permission from AWS.*

References:

- <https://kiro.dev/cli/>
- <https://kiro.dev/docs/cli/>
- <https://kiro.dev/downloads/>
- <https://docs.ultralytics.com/models/yolo26/>
- <https://huggingface.co/facebook/sam3>
