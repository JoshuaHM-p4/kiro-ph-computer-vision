"""Synthetic landmark builders used by the headless tests.

These produce anatomically plausible normalized landmarks without a camera, so
gesture logic can be exercised deterministically.
"""

from __future__ import annotations

import math

from ..common import landmarks as lm


def make_hand(
    *,
    label: str = "Right",
    center: tuple[float, float] = (0.5, 0.5),
    span: float = 0.12,
    pinch: float = 1.0,
    extended: tuple[str, ...] = ("index", "middle", "ring", "pinky", "thumb"),
    rotation_deg: float = 0.0,
) -> lm.Hand:
    """Build a 21-point hand.

    ``pinch`` scales the thumb-index gap (1.0 = wide open, 0.05 = pinched).
    ``extended`` lists which fingers stick out; the rest are curled so their tip
    falls short of the PIP joint. ``rotation_deg`` rotates the whole hand about
    its wrist, which is how the tests confirm finger-extension detection does not
    depend on the hand being upright.
    """
    points: list[tuple[float, float]] = [(0.0, 0.0)] * 21

    # Local frame: wrist at origin, fingers pointing along -y (up in image space).
    def place(index: int, x: float, y: float) -> None:
        points[index] = (x, y)

    place(lm.WRIST, 0.0, 0.0)
    place(lm.MIDDLE_MCP, 0.0, -1.0)  # one span from the wrist by definition

    finger_x = {"index": -0.35, "middle": 0.0, "ring": 0.32, "pinky": 0.62}
    joints = {
        "index": (lm.INDEX_MCP, lm.INDEX_PIP, lm.INDEX_DIP, lm.INDEX_TIP),
        "middle": (lm.MIDDLE_MCP, lm.MIDDLE_PIP, lm.MIDDLE_DIP, lm.MIDDLE_TIP),
        "ring": (lm.RING_MCP, lm.RING_PIP, lm.RING_DIP, lm.RING_TIP),
        "pinky": (lm.PINKY_MCP, lm.PINKY_PIP, lm.PINKY_DIP, lm.PINKY_TIP),
    }

    for name, (mcp, pip, dip, tip) in joints.items():
        x = finger_x[name]
        place(mcp, x, -0.95)
        place(pip, x, -1.30)
        if name in extended:
            place(dip, x, -1.55)
            place(tip, x, -1.75)
        else:
            # Curled: tip tucks back toward the palm, short of the PIP joint.
            place(dip, x, -1.15)
            place(tip, x, -1.00)

    # Thumb sweeps out to the side of the index finger.
    thumb_out = 1.0 if "thumb" in extended else 0.45
    place(lm.THUMB_CMC, -0.30, -0.20)
    place(lm.THUMB_MCP, -0.55, -0.45)
    place(lm.THUMB_IP, -0.75 * thumb_out, -0.70)
    # Pinch closes the gap between the thumb tip and the index tip.
    index_tip = points[lm.INDEX_TIP]
    open_thumb = (-0.95 * thumb_out, -0.85)
    gap = max(pinch, 0.0)
    place(
        lm.THUMB_TIP,
        index_tip[0] + (open_thumb[0] - index_tip[0]) * gap,
        index_tip[1] + (open_thumb[1] - index_tip[1]) * gap,
    )

    cos_r = math.cos(math.radians(rotation_deg))
    sin_r = math.sin(math.radians(rotation_deg))
    world: list[tuple[float, float]] = []
    for x, y in points:
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r
        world.append((center[0] + rx * span, center[1] + ry * span))

    return lm.Hand(points=world, label=label, score=0.95)


def make_face(
    *,
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 0.18,
    yaw: float = 0.0,
    eye_open: float = 1.0,
    mouth_open: float = 0.0,
    corner_lift: float = 0.0,
    brow_raise: float = 0.0,
    mouth_width: float = 1.0,
) -> lm.Face:
    """Build a 478-point face with controllable expression parameters.

    The mesh is a coarse stand-in: only the landmarks the demos actually read are
    placed meaningfully, the rest fill the face oval. ``yaw`` in degrees rotates
    the x coordinates about the vertical axis so ``head_pose`` sees real
    perspective-like compression on the turning-away side.
    """
    points: list[tuple[float, float]] = [(0.0, 0.0)] * 478

    def place(index: int, x: float, y: float, z: float = 0.0) -> None:
        points[index] = (x, y, z)  # type: ignore[assignment]

    eye_half = 0.16 * eye_open
    place(lm.FACE["nose_tip"], 0.0, 0.0, 0.35)
    place(lm.FACE["chin"], 0.0, 0.62, 0.0)
    place(lm.FACE["forehead"], 0.0, -0.72, 0.0)
    place(lm.FACE["left_eye_outer"], -0.46, -0.30, -0.15)
    place(lm.FACE["left_eye_inner"], -0.16, -0.30, -0.05)
    place(lm.FACE["left_eye_top"], -0.31, -0.30 - eye_half, -0.10)
    place(lm.FACE["left_eye_bottom"], -0.31, -0.30 + eye_half, -0.10)
    place(lm.FACE["right_eye_outer"], 0.46, -0.30, -0.15)
    place(lm.FACE["right_eye_inner"], 0.16, -0.30, -0.05)
    place(lm.FACE["right_eye_top"], 0.31, -0.30 - eye_half, -0.10)
    place(lm.FACE["right_eye_bottom"], 0.31, -0.30 + eye_half, -0.10)

    brow_y = -0.30 - eye_half - 0.16 - brow_raise
    place(lm.FACE["left_brow_inner"], -0.18, brow_y, -0.05)
    place(lm.FACE["left_brow_outer"], -0.31, brow_y, -0.10)
    place(lm.FACE["right_brow_inner"], 0.18, brow_y, -0.05)
    place(lm.FACE["right_brow_outer"], 0.31, brow_y, -0.10)

    half_mouth = 0.26 * mouth_width
    place(lm.FACE["mouth_left"], -half_mouth, 0.34 - corner_lift, -0.05)
    place(lm.FACE["mouth_right"], half_mouth, 0.34 - corner_lift, -0.05)
    place(lm.FACE["upper_lip"], 0.0, 0.34 - mouth_open * 0.5, 0.0)
    place(lm.FACE["lower_lip"], 0.0, 0.34 + mouth_open * 0.5, 0.0)
    place(lm.FACE["left_cheek"], -0.62, 0.02, -0.30)
    place(lm.FACE["right_cheek"], 0.62, 0.02, -0.30)

    # Fill unplaced landmarks around the oval so index lookups never fail.
    for i in range(len(points)):
        if points[i] == (0.0, 0.0):
            angle = (i / len(points)) * math.tau
            points[i] = (0.55 * math.cos(angle), 0.65 * math.sin(angle), 0.0)  # type: ignore[assignment]

    cos_y = math.cos(math.radians(yaw))
    sin_y = math.sin(math.radians(yaw))
    world: list[tuple[float, float]] = []
    for entry in points:
        x, y = entry[0], entry[1]
        z = entry[2] if len(entry) > 2 else 0.0
        # Rotate about the vertical axis, then project orthographically.
        rx = x * cos_y + z * sin_y
        world.append((center[0] + rx * scale, center[1] + y * scale))

    return lm.Face(points=world)


def make_pose(
    *,
    left_wrist_up: float = -0.3,
    right_wrist_up: float = -0.3,
    shoulder_y: float = 0.35,
    hip_y: float = 0.85,
    center_x: float = 0.5,
) -> lm.Pose:
    """Build a 33-point pose.

    ``*_wrist_up`` is the wrist height above the shoulder line in torso lengths:
    positive is raised, negative is lowered.
    """
    points: list[tuple[float, float]] = [(center_x, 0.5)] * 33
    visibility = [1.0] * 33
    torso = hip_y - shoulder_y

    def place(name: str, x: float, y: float) -> None:
        points[lm.POSE[name]] = (x, y)

    place("nose", center_x, shoulder_y - 0.18)
    place("left_shoulder", center_x - 0.14, shoulder_y)
    place("right_shoulder", center_x + 0.14, shoulder_y)
    place("left_hip", center_x - 0.10, hip_y)
    place("right_hip", center_x + 0.10, hip_y)
    place("left_wrist", center_x - 0.28, shoulder_y - left_wrist_up * torso)
    place("right_wrist", center_x + 0.28, shoulder_y - right_wrist_up * torso)
    place("left_elbow", center_x - 0.22, shoulder_y + 0.12)
    place("right_elbow", center_x + 0.22, shoulder_y + 0.12)
    place("left_knee", center_x - 0.10, hip_y + 0.25)
    place("right_knee", center_x + 0.10, hip_y + 0.25)

    return lm.Pose(points=points, visibility=visibility)


def make_frame(
    *,
    hands: list[lm.Hand] | None = None,
    face: lm.Face | None = None,
    pose: lm.Pose | None = None,
    width: int = 640,
    height: int = 480,
    timestamp: float = 0.0,
    seq: int = 0,
) -> lm.LandmarkFrame:
    return lm.LandmarkFrame(
        hands=list(hands or []),
        face=face,
        pose=pose,
        width=width,
        height=height,
        timestamp=timestamp,
        seq=seq,
    )
