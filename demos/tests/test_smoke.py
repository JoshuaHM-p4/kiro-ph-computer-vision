"""Environment smoke tests for the demo suite.

These assert on *capability* rather than exact versions: the pins live in
``requirements.txt``, and the demos work across the mediapipe 0.10.x line. The
one hard constraint is the interpreter, because mediapipe 0.10.x publishes no
wheels for Python 3.13+.
"""

from __future__ import annotations

import sys


def test_python_version_supports_mediapipe():
    major, minor = sys.version_info[:2]
    assert major == 3 and 10 <= minor <= 12, (
        "mediapipe 0.10.x has no wheels for Python 3.13+; build the venv with "
        "python3.12 (see requirements.txt)."
    )


def test_vision_stack_imports():
    import cv2
    import mediapipe as mp
    import numpy as np

    assert cv2.__version__.startswith("4.")
    assert mp.__version__.startswith("0.10.")
    assert np.__version__.startswith("1."), "mediapipe 0.10.x requires numpy < 2"


def test_legacy_mediapipe_solutions_available():
    import mediapipe as mp

    # The demos use the legacy solutions API, which every 0.10.x release ships.
    assert hasattr(mp.solutions, "hands")
    assert hasattr(mp.solutions, "face_mesh")
    assert hasattr(mp.solutions, "pose")


def test_face_mesh_supports_refined_landmarks():
    """The PNGTuber needs iris landmarks, i.e. refine_landmarks support."""
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    face_mesh.close()


def test_web_stack_imports():
    from importlib.metadata import version

    import flask_sock

    assert version("flask").startswith("3.")
    assert flask_sock is not None


def test_demo_registry_is_complete():
    from demos import DEMOS

    slugs = {demo.slug for demo in DEMOS}
    assert slugs == {
        "image-lab",
        "air-canvas",
        "slide-presenter",
        "six-seven",
        "pngtuber",
        "scavenger-hunt",
        "sam-labeler",
    }
    ports = [demo.port for demo in DEMOS]
    assert len(set(ports)) == len(DEMOS), "each demo needs its own port"
    assert DEMOS[0].slug == "image-lab", "the OpenCV primer comes first"
