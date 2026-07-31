# Presentation Outline: Computer Vision & Kiroverse Sandbox

### Slide 1: Title and Welcome

* **Visual:** The Kiroverse Week 7 logo, an OpenCV-processed image, and a QR code linking to your GitHub repository.
* **Speaker Notes:** "Welcome to Build Nights. Tonight we work with computer vision in Python, using a webcam and OpenCV to process a live video stream and run models on it. By the end you will have built and run your own vision application on your machine."

### Slide 2: What is Computer Vision?

* **Visual:** A side-by-side of how a human sees an image (a dog) vs. how a computer sees it (a matrix of RGB pixel values [0-255]).
* **Speaker Notes:** "Computer vision turns pixels into meaning: an image reaches the computer as a matrix of RGB values, and the task is to recognize edges, shapes, and objects in it. The same techniques appear in self-driving cars, camera filters, and medical image analysis."

### Slide 3: The Evolution of Vision

* **Visual:** A timeline.
* *Past:* Classical CV (OpenCV, Haar Cascades, manual edge detection).
* *Transition:* Deep Learning (CNNs, early YOLO).
* *Present:* Foundation Models (zero-shot learning, promptable segmentation).

* **Speaker Notes:** "Classical computer vision required hand-written rules for each feature, such as edge and line detectors. Deep learning let models learn those features from data, and today's foundation models generalize to objects and prompts they were not explicitly trained on."

### Slide 4a: 2026 Trends: Efficient Detection and Promptable Segmentation

* **Visual:** Logos for Ultralytics YOLO26 and Meta SAM 3.1.
* **Bullets:**
* **Hyper-efficient edge AI:** YOLO26 runs real-time detection on CPU, reporting up to 43% faster CPU inference than earlier versions by removing post-processing steps such as Non-Maximum Suppression.
* **Promptable segmentation:** Meta's SAM 3.1 segments an object from a text prompt and tracks it across video frames, so you describe the object in words instead of training a detector.
* **Speaker Notes:** "Two trends define efficient vision today. YOLO26 makes real-time detection practical on a standard laptop CPU, and SAM 3.1 lets you segment an object by describing it in text rather than training a model for that class."

### Slide 4b: 2026 Trends: Visual Language Models and 3D Foundation Models

* **Visual:** Logos or example outputs for NVIDIA LocateAnything and LingBot-Map.
* **Bullets:**
* **Visual language models and visual reasoning:** models that connect language to what they see, so you locate an object by describing it. NVIDIA LocateAnything grounds and detects objects from natural-language instructions (open-sourced June 2026), and SAM 3.1 segments from text prompts.
* **3D foundation models:** models that reconstruct a scene's 3D structure from ordinary video. LingBot-Map, open-sourced by Robbyant (Ant Group) in April 2026, rebuilds a scene frame by frame from a single streaming video at around 20 FPS.
* **Speaker Notes:** "Beyond flat images, two newer directions are visual language models and 3D reconstruction. Visual language models such as NVIDIA LocateAnything and SAM 3.1 link text to pixels so you can locate or segment objects by describing them, while 3D foundation models such as LingBot-Map reconstruct a scene's geometry in real time from a single video."

### Slide 5: Real-World Use Cases (My Projects)

* **Visual:** Action shots or system diagrams of your engineering work.
* **Speaker Notes:** "These models run on modest hardware, not only large servers. I built a UAV-based river mapping system that ran YOLO on an embedded Jetson Nano for real-time edge inference, and I integrated IoT monitoring into the Luntian biofilter project to track environmental health."

### Slide 6: Objectives

* **Visual:** A checklist framing "what you will have done by the end."
* **Bullets:**
* Used Kiro (CLI or IDE) to generate, scaffold, and debug a vision project on your machine.
* Chained OpenCV operations on a live webcam feed: resize, threshold, mask, and draw.
* Run YOLO26n for real-time 80-class object detection on CPU.
* Seen SAM 3.1 segment objects from a text prompt (facilitator demo; optional to try yourself).
* Built and presented a working webcam application of your own design.
* **Speaker Notes:** "This slide sets the target for the session. By the end you will have used Kiro to build a vision project, chained OpenCV operations on a live feed, run YOLO26 detection on CPU, seen SAM 3.1 segment from text, and presented an application of your own."

### Slide 7: What We Are Building Today

* **Visual:** A grid showing fast GIFs/images of the workshop demos.
* **Bullets:**
* Cover the foundations together: OpenCV, real-time detection, and segmentation.
* Fork the repository, create a folder in `projects/`, and build your own webcam app; Kiro writes most of the code and you direct it.
* Seven demos in `demos/` are reference material to run, read, and borrow from: Image Lab, Air Canvas, Slide Presenter, 6-7 Counter, PNGTuber, Scavenger Hunt, and SAM Labeler.
* The prompt library in `docs/prompts/` holds the prompts that produced each demo.
* **Speaker Notes:** "We cover the foundations together, then you build your own application in your projects folder while Kiro handles most of the code. The seven demos in demos/ are there to run and borrow from, so you can rebuild one, combine two, or build something different."

### Slide 8: Prepare Your Prerequisites

* **Visual:** A checklist of prerequisites with checkable boxes.
* **Bullets:**
* [ ] Laptop or desktop with a working webcam, or sample images and videos.
* [ ] Kiro account (Builder ID, GitHub, or social login) and Kiro installed (CLI or IDE).
* [ ] GitHub account, with your fork set to public for submission.
* [ ] Hugging Face account, SAM 3.1 license accepted, and a read token created.
* [ ] Python 3.10 to 3.12 installed (mediapipe has no wheels for 3.13+).
* **Speaker Notes:** "Confirm each item before we start so we do not lose session time to setup. The webcam is preferred but not required, the Python version must be 3.10 to 3.12, and the Hugging Face token is only needed for the SAM 3.1 demo."

### Slide 9: Project Setup

* **Visual:** A terminal showing the setup commands.
* **Commands:**

```bash
git clone https://github.com/<YOUR-USERNAME>/kiro-computer-vision.git
cd kiro-computer-vision
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt
python check_setup.py
```

* **Speaker Notes:** "From the repository root, create a Python 3.10 to 3.12 virtual environment and install the CPU build of torch first, then the rest of the requirements. Run check_setup.py to verify Python, the libraries, and your webcam before building."

### Slide 10: Start Kiro (from the repo root) and its context

* **Visual:** A terminal at the repository root running Kiro.
* **Commands:**

```bash
# Run from the repository root, not inside projects/<username>.
kiro-cli chat
# Then tell Kiro to build in projects/<your-github-username>/
```

* **Bullets:**
* `.kiro/steering`, what it is: markdown files Kiro loads automatically at the start of every session, holding the repo's conventions (tech stack, structure, computer vision patterns, workshop flow).
* `.kiro/steering`, what it does: it makes Kiro follow those conventions without reminders, such as keeping landmarks normalized, using scale-relative thresholds, separating logic from I/O, and working inside `projects/<your-username>/`.
* `.kiro/skills`, what it is: the vision-project-builder skill, whose description loads at startup and whose full instructions load on demand.
* `.kiro/skills`, what it does: when you say you want to build something, it asks structured questions, produces a plan, and scaffolds your project with a Flask web layer.
* **Speaker Notes:** "Start Kiro from the repository root, because it loads .kiro/steering and .kiro/skills only from the current directory, then ask it to build in your projects folder. Steering carries the repo's conventions so Kiro applies them automatically, and the skill guides you from an idea to a scaffolded project."

### Slide 11: Tell Kiro What to Build

* **Visual:** A Kiro CLI prompt with a plain-language request.
* **Bullets:**
* Describe your app in plain language, for example: "I want to build a posture checker", "Help me recreate the scavenger hunt", or "I want a gesture-controlled music player".
* Or use the vision-project-builder skill (say "/build" or "I want to build..."); it asks about your input source, model choice, and output, then scaffolds the project.
* You can also paste a prompt from `docs/prompts/` if you prefer to skip the questions.
* **Speaker Notes:** "Tell Kiro what you want in plain language, or use the vision-project-builder skill to be guided through the same questions. Kiro asks about your input source, model, and output, then scaffolds the project including the Flask web layer."

### Slide 12: Iterate

* **Visual:** A before/after of a gesture demo, one flickering and one stable.
* **Bullets:**
* The first output is rarely finished; describe what you see and Kiro applies the right fix.
* "It flickers between two states when my hand is still." The fix is hysteresis.
* "It only works when I sit close to the camera." The fix is scale-relative thresholds.
* "It counted twice when I did it once." The fix is an edge trigger with a cooldown.
* **Speaker Notes:** "Improvement comes from describing the symptom rather than guessing at code. Flicker points to missing hysteresis, close-only behavior points to pixel thresholds that should be scale-relative, and double counting points to a missing cooldown."

### Slide 13: Web Dashboard (Flask)

* **Visual:** A browser showing a demo running as a Flask dashboard.
* **Bullets:**
* Kiro adds a Flask web layer automatically when it scaffolds your project, and it is required for the showcase.
* The web version runs locally in the browser as a dashboard.
* To customize it, see `docs/prompts/07-flask-web-layer.md`.
* **Speaker Notes:** "Every project gets a Flask web layer so you can present it in the browser, and Kiro includes it during scaffolding. If you need to change it, prompt 07 in the library covers the web layer."

### Slide 14: Test It

* **Visual:** A terminal running pytest with passing results.
* **Command:**

```bash
python -m pytest projects/<your-username>
```

* **Bullets:**
* Write tests that feed synthetic data so they run without a webcam.
* Ask Kiro to help write and run these tests.
* **Speaker Notes:** "Tests should drive your logic with synthetic inputs so they pass without a camera connected. Ask Kiro to write them, then run python -m pytest against your project folder."

### Slide 15: Run It

* **Visual:** Side-by-side of the desktop OpenCV window and the browser dashboard.
* **Commands:**

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python projects/<your-username>/main.py   # desktop OpenCV window
python projects/<your-username>/web.py    # Flask dashboard in the browser
```

* **Bullets:**
* Activate the virtual environment first.
* The desktop version opens an OpenCV window; the web version opens the Flask dashboard.
* Test with your webcam running to confirm it works, not only that the code compiles.
* **Speaker Notes:** "With the virtual environment active, run main.py for the desktop OpenCV window or web.py for the browser dashboard. Confirm it works with your webcam running rather than only checking that the code compiles."

### Slide 16: Workshop Recap

* **What you did:**
* Explored OpenCV interactively.
* Ran YOLO26 detection on real objects.
* Watched SAM 3.1 segment from typed words.
* Built your own webcam app by prompting Kiro.
* Debugged real issues such as flickering, drift, and false positives.
* **What you practiced:**
* Preprocessing, detection, segmentation, and landmark tracking.
* Running a local AI coding agent.
* Turning a prompt template into a working app through iteration.
* The patterns that make vision code reliable, including hysteresis, scale invariance, and separating logic from I/O.
* **Speaker Notes:** "Today you worked through OpenCV, YOLO26 detection, and SAM 3.1 segmentation, then built and debugged your own application with Kiro. Along the way you practiced the patterns that keep vision code reliable, including hysteresis, scale invariance, and separating logic from input and output."

### Slide 17: What's Next

* **Bullets:**
* Submit your project and feedback form: <https://forms.gle/HVej5YSJ3NLtorSZA>
* Share your build on LinkedIn, tag Kiro Community and AWS User Group PH, and use #kiroverse.
* Stay connected: Facebook (<https://www.facebook.com/kirocommunity/>), LinkedIn (<https://linkedin.com/company/kirocommunity>), Luma (<https://luma.com/kirocommunity>).
* **Speaker Notes:** "Submit your project and the feedback form using the link, and share what you built on LinkedIn with the community tags. The Facebook, LinkedIn, and Luma links keep you connected for future events."

---

### Presentation Pro-Tips for a 3-Hour Workshop

* **Keep it under 15 minutes:** People came to a Build Night to write code, not to sit through a lecture.
* **Show, don't tell:** When you reach the demos slide, stand in front of the projector camera and run Air Canvas or Slide Presenter live. A live demo sets the energy for the rest of the night.

To help participants understand the end-to-end model training process beyond using the pretrained weights, [Training Ultralytics YOLO26 Model on Custom Dataset](https://www.youtube.com/watch?v=7lZa3Yi2kbo) walks through the Google Colab workflow for fine-tuning the latest architecture.
