"""Gesture primitives derived from landmarks.

Every function here takes landmarks and returns numbers or small dataclasses.
The only OpenCV use is ``cv2.solvePnP`` for head pose, which needs no window and
no camera, so this module stays importable and testable headlessly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import landmarks as lm
from .geometry import (
    EdgeTrigger,
    HysteresisLatch,
    clamp,
    distance,
    midpoint,
)

# ---------------------------------------------------------------------------
# Hand gestures
# ---------------------------------------------------------------------------


def pinch_ratio(hand: lm.Hand) -> float:
    """Thumb-to-index distance divided by the hand span.

    Dividing by the span makes the value scale invariant, so one threshold works
    at any distance from the camera. Roughly 0.1 when pinched shut and 1.0+ when
    the hand is open.
    """
    return distance(hand.thumb_tip, hand.index_tip) / hand.span


def pinch_point(hand: lm.Hand) -> lm.Point:
    """Midpoint between thumb and index tips: where the pinch "grabs"."""
    return midpoint(hand.thumb_tip, hand.index_tip)


@dataclass
class PinchDetector:
    """Scale-invariant pinch with hysteresis and an edge trigger.

    ``start_ratio`` closes the latch and the larger ``release_ratio`` opens it,
    so a hand resting near the threshold cannot chatter. ``cooldown`` throttles
    ``just_pinched`` for discrete actions such as advancing a slide.
    """

    start_ratio: float = 0.38
    release_ratio: float = 0.55
    cooldown: float = 0.0
    _latch: HysteresisLatch = field(init=False)
    _trigger: EdgeTrigger = field(init=False)

    def __post_init__(self) -> None:
        self._latch = HysteresisLatch(on_below=self.start_ratio, off_above=self.release_ratio)
        self._trigger = EdgeTrigger(cooldown=self.cooldown)

    def update(self, hand: lm.Hand | None, now: float | None = None) -> "PinchState":
        if hand is None:
            # A hand that leaves the frame must not leave the pinch latched on.
            self._latch.reset()
            self._trigger.fire(False, now)
            return PinchState(active=False, just_started=False, ratio=None, point=None)

        ratio = pinch_ratio(hand)
        active = self._latch.update(ratio)
        just_started = self._trigger.fire(active, now)
        return PinchState(
            active=active,
            just_started=just_started,
            ratio=ratio,
            point=pinch_point(hand),
        )

    def reset(self) -> None:
        self._latch.reset()
        self._trigger.reset()


@dataclass(frozen=True)
class PinchState:
    active: bool
    just_started: bool
    ratio: float | None
    point: lm.Point | None


def finger_extended(hand: lm.Hand, finger: str) -> bool:
    """Whether one finger is extended.

    For the four long fingers the tip is compared against the PIP joint along
    the wrist->knuckle axis, which works even when the hand is rotated (a simple
    "tip is above PIP in y" test fails as soon as you tilt your hand). The thumb
    is measured by how far its tip sits from the index knuckle relative to the
    hand span, because the thumb folds sideways rather than curling.
    """
    if finger == "thumb":
        reach = distance(hand.point(lm.THUMB_TIP), hand.point(lm.INDEX_MCP))
        return (reach / hand.span) > 0.75

    tip_idx, pip_idx, mcp_idx = {
        "index": (lm.INDEX_TIP, lm.INDEX_PIP, lm.INDEX_MCP),
        "middle": (lm.MIDDLE_TIP, lm.MIDDLE_PIP, lm.MIDDLE_MCP),
        "ring": (lm.RING_TIP, lm.RING_PIP, lm.RING_MCP),
        "pinky": (lm.PINKY_TIP, lm.PINKY_PIP, lm.PINKY_MCP),
    }[finger]

    wrist = hand.wrist
    knuckle = hand.point(mcp_idx)
    axis = (knuckle[0] - wrist[0], knuckle[1] - wrist[1])
    axis_len = math.hypot(*axis)
    if axis_len < 1e-9:
        return False
    axis = (axis[0] / axis_len, axis[1] / axis_len)

    def projection(point: lm.Point) -> float:
        return (point[0] - wrist[0]) * axis[0] + (point[1] - wrist[1]) * axis[1]

    return projection(hand.point(tip_idx)) > projection(hand.point(pip_idx))


def extended_fingers(hand: lm.Hand) -> dict[str, bool]:
    """Extension flag for all five fingers."""
    return {name: finger_extended(hand, name) for name in lm.FINGER_NAMES}


def count_extended(hand: lm.Hand) -> int:
    return sum(extended_fingers(hand).values())


def is_pointing(hand: lm.Hand) -> bool:
    """Index extended, middle/ring/pinky curled: the drawing pose."""
    flags = extended_fingers(hand)
    return flags["index"] and not flags["middle"] and not flags["ring"] and not flags["pinky"]


def is_peace(hand: lm.Hand) -> bool:
    """Index and middle extended, ring/pinky curled: the hover/select pose."""
    flags = extended_fingers(hand)
    return flags["index"] and flags["middle"] and not flags["ring"] and not flags["pinky"]


def is_fist(hand: lm.Hand) -> bool:
    """All long fingers curled: the idle pose."""
    flags = extended_fingers(hand)
    return not any(flags[name] for name in ("index", "middle", "ring", "pinky"))


def is_open_palm(hand: lm.Hand) -> bool:
    flags = extended_fingers(hand)
    return all(flags[name] for name in ("index", "middle", "ring", "pinky"))


# ---------------------------------------------------------------------------
# Face ratios
# ---------------------------------------------------------------------------


def eye_aspect_ratio(face: lm.Face, side: str) -> float:
    """Vertical eye opening divided by eye width.

    Around 0.3 for an open eye and near 0.1 when closed. Normalizing by eye
    width keeps it independent of distance from the camera.
    """
    prefix = "left" if side == "left" else "right"
    top = face.named(f"{prefix}_eye_top")
    bottom = face.named(f"{prefix}_eye_bottom")
    outer = face.named(f"{prefix}_eye_outer")
    inner = face.named(f"{prefix}_eye_inner")
    width = distance(outer, inner)
    if width < 1e-9:
        return 0.0
    return distance(top, bottom) / width


def average_ear(face: lm.Face) -> float:
    return (eye_aspect_ratio(face, "left") + eye_aspect_ratio(face, "right")) * 0.5


def mouth_aspect_ratio(face: lm.Face) -> float:
    """Lip gap divided by mouth width: how far the mouth is open."""
    width = distance(face.named("mouth_left"), face.named("mouth_right"))
    if width < 1e-9:
        return 0.0
    return distance(face.named("upper_lip"), face.named("lower_lip")) / width


def mouth_corner_lift(face: lm.Face) -> float:
    """How far the mouth corners sit above the lip center, in eye widths.

    Positive means the corners are higher than the lip line (a smile). Image y
    grows downward, hence the inverted subtraction.
    """
    corners_y = (face.named("mouth_left")[1] + face.named("mouth_right")[1]) * 0.5
    center_y = (face.named("upper_lip")[1] + face.named("lower_lip")[1]) * 0.5
    return (center_y - corners_y) / face.interocular


def mouth_width_ratio(face: lm.Face) -> float:
    return distance(face.named("mouth_left"), face.named("mouth_right")) / face.interocular


def brow_raise(face: lm.Face) -> float:
    """Average brow-to-eye distance in eye widths.

    Rises with surprise and falls when brows are lowered in a frown.
    """
    left = distance(face.named("left_brow_outer"), face.named("left_eye_top"))
    right = distance(face.named("right_brow_outer"), face.named("right_eye_top"))
    return ((left + right) * 0.5) / face.interocular


# ---------------------------------------------------------------------------
# Head pose
# ---------------------------------------------------------------------------

# Canonical 3D face points (millimetres, nose tip at the origin) paired with the
# Face Mesh indices below. Only relative geometry matters for orientation.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),        # nose tip
        (0.0, -63.6, -12.5),    # chin
        (-43.3, 32.7, -26.0),   # left eye outer corner
        (43.3, 32.7, -26.0),    # right eye outer corner
        (-28.9, -28.9, -24.1),  # left mouth corner
        (28.9, -28.9, -24.1),   # right mouth corner
    ],
    dtype=np.float64,
)
_MODEL_INDICES = (
    lm.FACE["nose_tip"],
    lm.FACE["chin"],
    lm.FACE["left_eye_outer"],
    lm.FACE["right_eye_outer"],
    lm.FACE["mouth_left"],
    lm.FACE["mouth_right"],
)


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in degrees. ``method`` records how it was obtained."""

    yaw: float
    pitch: float
    roll: float
    method: str


def yaw_ratio(face: lm.Face) -> float:
    """Fallback yaw estimate in the range -1..1 from nose offset.

    Compares nose-to-left-cheek against nose-to-right-cheek distance. Positive
    means the nose has moved toward the right of the image, matching the sign
    convention of :func:`head_pose`.
    """
    nose = face.named("nose_tip")
    left = face.named("left_cheek")
    right = face.named("right_cheek")
    d_left = distance(nose, left)
    d_right = distance(nose, right)
    total = d_left + d_right
    if total < 1e-9:
        return 0.0
    return clamp((d_left - d_right) / total, -1.0, 1.0)


def head_pose(face: lm.Face, width: int = 640, height: int = 480) -> HeadPose:
    """Estimate yaw/pitch/roll in degrees.

    Uses ``cv2.solvePnP`` against the canonical model with an assumed pinhole
    camera; if the solve fails (degenerate or near-planar input) it falls back to
    the nose-offset ratio scaled to roughly +-60 degrees so callers always get a
    usable number.

    Sign convention: positive yaw means the head has turned toward the right of
    the image (the nose moves right on screen). Both estimators agree on this, so
    a caller can bucket on the sign without caring which path produced it.
    """
    width = max(int(width), 1)
    height = max(int(height), 1)
    image_points = np.array(
        [(face.points[i][0] * width, face.points[i][1] * height) for i in _MODEL_INDICES],
        dtype=np.float64,
    )

    focal = float(width)
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok = False
    rotation = None
    try:
        ok, rotation, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        ok = False

    if ok and rotation is not None:
        matrix, _ = cv2.Rodrigues(rotation)
        sy = math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2(matrix[2, 1], matrix[2, 2]))
            yaw = math.degrees(math.atan2(-matrix[2, 0], sy))
            roll = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        else:  # gimbal-lock branch
            pitch = math.degrees(math.atan2(-matrix[1, 2], matrix[1, 1]))
            yaw = math.degrees(math.atan2(-matrix[2, 0], sy))
            roll = 0.0
        # Pitch comes back near +-180 because the canonical model's y axis points
        # opposite to image y; fold it into a readable -90..90 range.
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        # The same y flip inverts the yaw sign relative to our convention.
        yaw = -yaw
        if abs(yaw) <= 90.0:
            return HeadPose(yaw=yaw, pitch=pitch, roll=roll, method="solvepnp")

    ratio = yaw_ratio(face)
    roll = math.degrees(
        math.atan2(
            face.named("right_eye_outer")[1] - face.named("left_eye_outer")[1],
            face.named("right_eye_outer")[0] - face.named("left_eye_outer")[0],
        )
    )
    return HeadPose(yaw=ratio * 60.0, pitch=0.0, roll=roll, method="ratio")


# ---------------------------------------------------------------------------
# Pose / rep counting
# ---------------------------------------------------------------------------


def torso_scale(pose: lm.Pose) -> float:
    """Shoulder-to-hip distance: the body's scale reference.

    Thresholds expressed as a fraction of this survive the user moving closer to
    or further from the camera.
    """
    shoulders = midpoint(pose.named("left_shoulder"), pose.named("right_shoulder"))
    hips = midpoint(pose.named("left_hip"), pose.named("right_hip"))
    scale = distance(shoulders, hips)
    if scale < 1e-6:
        # Fall back to shoulder width when hips are out of frame.
        scale = distance(pose.named("left_shoulder"), pose.named("right_shoulder"))
    return max(scale, 1e-6)


def shoulder_line_y(pose: lm.Pose) -> float:
    return (pose.named("left_shoulder")[1] + pose.named("right_shoulder")[1]) * 0.5


def wrist_height(pose: lm.Pose, side: str) -> float:
    """Wrist height above the shoulder line, in torso lengths.

    Positive means the wrist is above the shoulders. Image y grows downward, so
    the subtraction is inverted.
    """
    wrist = pose.named(f"{side}_wrist")
    return (shoulder_line_y(pose) - wrist[1]) / torso_scale(pose)
