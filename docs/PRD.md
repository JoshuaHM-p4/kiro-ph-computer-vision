# Product Requirements Document (Draft)

**Project:** kiro-computer-vision
**Status:** Draft — scope not yet finalized
**Last updated:** 2026-07-29

> This is an early draft. Undecided details are captured under
> [Open Questions](#open-questions) rather than assumed. Do not invent scope;
> extend this document as decisions are made.

## 1. Summary

`kiro-computer-vision` is a Python application for **image and video stream
processing**. It pairs an **image preprocessing pipeline** with **real-time
inference over a live video stream**, powered by a **user-provided pretrained
model**. The project is the deliverable for *Kiroverse Week 7: Build Nights —
Computer Vision Applications with Kiro*, emphasizing clean vision code and
disciplined dependency management with the help of Kiro's AI agents.

## 2. Background & Context

Kiroverse Week 7 focuses on building a computer vision app with Python and
OpenCV: understanding CV foundations and model architectures, integrating object
detection/classification models, hands-on building of preprocessing scripts and
live inference, and presenting the result. This PRD frames that work as a
concrete, evolvable product.

## 3. Goals & Non-Goals

### Goals
- Provide a reusable **image preprocessing pipeline** (resize, color-space
  conversion, normalization, etc.) usable for both still images and video frames.
- Run **real-time inference on a live video stream** (e.g. webcam) using a
  user-provided pretrained model, overlaying results on the frame.
- Keep processing logic **modular and testable**, isolated from I/O.
- Manage the **OpenCV / MediaPipe / NumPy** dependency stack reliably with pinned
  versions.

### Non-Goals (for now)
- Training or fine-tuning models (the model is provided by the user).
- Building a full GUI or web frontend.
- Cloud deployment / distributed inference.
- Bundling or distributing model weights.

## 4. Target Users

- Workshop participants building a CV app during Kiroverse Week 7.
- Developers who want a clean starting point for OpenCV + MediaPipe pipelines
  with a swappable pretrained model.

## 5. Functional Requirements

### 5.1 Image Preprocessing Pipeline
- Load images from disk and accept in-memory frames from a video source.
- Provide composable preprocessing steps (resize, crop/ROI, color-space
  conversion, normalization, optional augmentation).
- Expose pure, unit-testable functions independent of camera/display I/O.

### 5.2 Real-Time Video Stream Inference
- Capture from a configurable source (webcam index or file path).
- Feed preprocessed frames to the user-provided model and obtain predictions.
- Overlay results (e.g. bounding boxes, labels, landmarks) on the live stream.
- Maintain interactive frame rates and expose basic FPS feedback.

### 5.3 Model Integration
- Accept a model by **configurable path** (CLI argument or environment variable).
- Support the model format(s) implied by the chosen inference backend (TBD).
- Never commit or pip-install model weights.

### 5.4 Configuration & CLI
- Configure source, model path, and output options via CLI/env.
- (Expected shape, entry point TBD:)
  `python -m app --source 0 --model path/to/model`

## 6. Tech Stack

- **Language:** Python 3.10+
- **Core libraries:** OpenCV (`opencv-python`), MediaPipe, NumPy
- **Model:** user-provided pretrained model (format TBD — e.g. `*.pt`, `*.onnx`,
  `*.tflite`)
- **Testing:** `pytest`

## 7. Open Questions

- **What is the concrete application?** (e.g. object detection, hand/pose/face
  tracking, classification, segmentation) — *not yet decided*.
- **Which model and format** will the user provide, and **which inference
  backend** loads it (OpenCV DNN, MediaPipe, ONNX Runtime, PyTorch, …)?
- **What classes / tasks** does the model perform?
- **Target platform & hardware** (CPU-only vs GPU; desktop vs edge)?
- **Performance targets** (minimum acceptable FPS, resolution)?
- **Output requirements** — live display only, or also recording/exporting frames
  to `outputs/`?
- **Multiple input sources** (single webcam vs files vs streams) — required scope?

## 8. Success Criteria

- A user can install dependencies from `requirements.txt` in a clean virtualenv.
- The preprocessing pipeline runs on both a still image and video frames, with
  passing unit tests.
- Live inference runs against a user-provided model and displays results in real
  time.
- No model weights or secrets are committed to the repository.

## 9. Milestones (tentative)

1. **Scaffolding** — docs, conventions, dependency pins, git init. *(current)*
2. **Preprocessing pipeline** — composable, tested transforms.
3. **Capture layer** — mockable video/image source abstraction.
4. **Model integration** — load user model, run inference on frames.
5. **Live overlay & CLI** — real-time display, configurable entry point.
6. **Showcase & polish** — demo, docs, Q&A feedback incorporated.
