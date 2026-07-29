# Product Requirements Document

**Project:** kiro-computer-vision  
**Status:** Finalized  
**Last updated:** 2026-07-29  

## 1. Summary

`kiro-computer-vision` is a Python application for **image and video stream processing**. It pairs an **image preprocessing pipeline** with **real-time inference over a live video stream**, powered by a **user-provided pretrained model**. The project is the deliverable for *Kiroverse Week 7: Build Nights — Computer Vision Applications with Kiro*, emphasizing clean vision code and disciplined dependency management with the help of Kiro's AI agents.

## 2. Background & Context

Kiroverse Week 7 focuses on building a computer vision app with Python and OpenCV: understanding CV foundations and model architectures, integrating object detection/classification models, hands-on building of preprocessing scripts and live inference, and presenting the result. This PRD frames that work as a concrete, evolvable product.

## 3. Goals & Non-Goals

### Goals

* Provide a reusable **image preprocessing pipeline** (resize, color-space conversion, normalization, etc.) usable for both still images and video frames.
* Run **real-time inference on a live video stream** (e.g., webcam or video file) using a pre-trained YOLO model and MediaPipe, overlaying results on the frame.
* Keep processing logic **modular and testable**, isolated from I/O.
* Manage the **OpenCV / MediaPipe / NumPy / Ultralytics** dependency stack reliably with pinned versions.

### Non-Goals

* Training or fine-tuning models.
* Building a full GUI or web frontend.
* Cloud deployment / distributed inference.
* Bundling or distributing model weights in the repository.

## 4. Target Users

* Workshop participants building a CV app during Kiroverse Week 7.
* Developers who want a clean starting point for OpenCV + MediaPipe pipelines with a swappable pretrained model.

## 5. Functional Requirements

### 5.1 Image Preprocessing Pipeline

* Load images from disk and accept in-memory frames from a video source.
* Provide composable preprocessing steps [resize, crop/ROI, color-space conversion, normalization, optional augmentation](cite: 1).
* Expose pure, unit-testable functions independent of camera/display I/O.

### 5.2 Real-Time Video Stream Inference

* Capture from a configurable source [webcam index or file path](cite: 1).
* Feed preprocessed frames to the pre-trained model (YOLO) and MediaPipe to obtain predictions.
* Overlay results (bounding boxes, labels, hand landmarks) on the live stream.
* Maintain interactive frame rates and expose basic FPS feedback.

### 5.3 Model Integration

* Accept a model by **configurable path** [CLI argument or environment variable](cite: 1).
* Support `.pt` or `.onnx` formats via the Ultralytics backend.
* Never commit or pip-install model weights.

### 5.4 Configuration & CLI

* Configure source, model path, and output options via CLI/env.
* Expected entry point: `python -m app --source 0 --model models/yolov8n.pt`

## 6. Tech Stack

* **Language:** Python 3.10+
* **Core libraries:** OpenCV (`opencv-python`), MediaPipe, NumPy, Ultralytics [for YOLO](cite: 1)
* **Model:** Pre-trained YOLO model (`*.pt`)
* **Testing:** `pytest`

## 7. Resolved Decisions

* **What is the concrete application?** A hybrid application featuring real-time object detection (via YOLO) and optional hand tracking (via MediaPipe).
* **Which model and format?** A pre-trained lightweight YOLO model (e.g., YOLOv8n or YOLO11n) in `.pt` format, loaded via the Ultralytics library.
* **Target platform & hardware:** CPU-friendly inference to accommodate all participant hardware, with GPU acceleration if available.
* **Performance targets:** Minimum 15-20 FPS on a standard laptop CPU.
* **Multiple input sources:** Support for both webcam (`--source 0`) and video file paths (e.g., `--source data/sample_video.mp4`).

## 8. Success Criteria

* A user can install dependencies from `requirements.txt` in a clean virtualenv.
* The preprocessing pipeline runs on both a still image and video frames, with passing unit tests.
* Live inference runs against a user-provided model and displays results in real time.
* No model weights or secrets are committed to the repository.
