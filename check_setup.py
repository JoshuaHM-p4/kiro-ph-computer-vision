#!/usr/bin/env python3
"""Run this to verify your environment is ready for the workshop.

    python check_setup.py

It checks Python version, imports, the webcam, and prints what to fix if
anything is wrong. Takes about 5 seconds.
"""

import sys

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"
errors = []


def check(label, condition, fix=""):
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}")
        if fix:
            print(f"      Fix: {fix}")
        errors.append(label)


def warn(label, message):
    print(f"  {WARN} {label}: {message}")


print("\n🔍 Checking your setup...\n")

# Python version
major, minor = sys.version_info[:2]
check(
    f"Python {major}.{minor} (need 3.10-3.12)",
    major == 3 and 10 <= minor <= 12,
    "Install Python 3.12: https://www.python.org/downloads/\n"
    "      Then: python3.12 -m venv .venv && source .venv/bin/activate",
)

# Core imports
for package, module in [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("mediapipe", "mediapipe"),
    ("flask", "flask"),
    ("ultralytics", "ultralytics"),
    ("torch", "torch"),
    ("pytest", "pytest"),
]:
    try:
        __import__(module)
        check(f"{package} installed", True)
    except ImportError:
        check(
            f"{package} installed",
            False,
            "pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision\n"
            "      pip install -r requirements.txt",
        )

# Versions
try:
    import cv2
    import mediapipe as mp
    import numpy as np
    import torch

    print(f"\n  Versions: cv2={cv2.__version__}  mediapipe={mp.__version__}  "
          f"numpy={np.__version__}  torch={torch.__version__}")
except Exception:
    pass

# Webcam
print()
try:
    import cv2

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            check(f"Webcam opens (camera 0, {frame.shape[1]}x{frame.shape[0]})", True)
        else:
            check("Webcam readable", False, "Camera opened but returned no frame")
        cap.release()
    else:
        cap.release()
        # Try camera 1
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            warn("Webcam", "camera 0 failed but camera 1 works. Use --camera 1")
            cap.release()
        else:
            cap.release()
            check("Webcam opens", False, "No camera found. You can still upload images.")
except Exception as e:
    check("Webcam", False, str(e))

# YOLO model
print()
try:
    from pathlib import Path

    models_dir = Path("models")
    if (models_dir / "yolo26n.pt").exists():
        check("YOLO weights cached (models/yolo26n.pt)", True)
    else:
        warn("YOLO weights", "Will download ~5 MB on first use. Internet needed.")
except Exception:
    pass

# Kiro CLI
import shutil

kiro = shutil.which("kiro-cli")
check("kiro-cli on PATH", kiro is not None, "Install from https://kiro.dev/downloads/")

# Summary
print()
if errors:
    print(f"  ⚠️  {len(errors)} issue(s) to fix before we start.")
    print("  Run the fix commands above, then re-run this script.\n")
else:
    print("  🎉 All good! You're ready for the workshop.\n")
    print("  Next steps:")
    print("    mkdir projects/<your-github-username>")
    print("    cd projects/<your-github-username>")
    print("    kiro-cli chat")
    print()

sys.exit(1 if errors else 0)
