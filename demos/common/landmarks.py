"""Landmark data model shared by the desktop and browser paths.

Both MediaPipe Python (``mp.solutions``) and MediaPipe JS (``tasks-vision``)
produce landmarks normalized to 0..1 in image space, so the demos speak that
coordinate system everywhere and only convert to pixels at draw time. This
module owns the dataclasses, the named landmark indices, and the JSON codec used
by the WebSocket channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .geometry import Point, distance, midpoint

# --- MediaPipe Hands indices ------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

FINGER_TIPS: tuple[int, ...] = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS: tuple[int, ...] = (THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")

# --- MediaPipe Face Mesh indices --------------------------------------------
FACE = {
    "nose_tip": 1,
    "chin": 152,
    "forehead": 10,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "left_brow_inner": 55,
    "left_brow_outer": 105,
    "right_brow_inner": 285,
    "right_brow_outer": 334,
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    "left_cheek": 234,
    "right_cheek": 454,
}

# Contours drawn on the desktop HUD. Kept short so the overlay stays readable.
FACE_OVAL: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)
LEFT_EYE_RING: tuple[int, ...] = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
RIGHT_EYE_RING: tuple[int, ...] = (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)
OUTER_LIPS: tuple[int, ...] = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
INNER_LIPS: tuple[int, ...] = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)
LEFT_BROW: tuple[int, ...] = (70, 63, 105, 66, 107)
RIGHT_BROW: tuple[int, ...] = (300, 293, 334, 296, 336)

# --- MediaPipe Pose indices -------------------------------------------------
POSE = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
}

POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
)


@dataclass
class Hand:
    """One detected hand in normalized image coordinates."""

    points: list[Point]
    label: str = "Unknown"  # "Left" / "Right" / "Unknown"
    score: float = 0.0

    def point(self, index: int) -> Point:
        return self.points[index]

    @property
    def wrist(self) -> Point:
        return self.points[WRIST]

    @property
    def index_tip(self) -> Point:
        return self.points[INDEX_TIP]

    @property
    def thumb_tip(self) -> Point:
        return self.points[THUMB_TIP]

    @property
    def middle_tip(self) -> Point:
        return self.points[MIDDLE_TIP]

    @property
    def palm_center(self) -> Point:
        return midpoint(self.points[WRIST], self.points[MIDDLE_MCP])

    @property
    def span(self) -> float:
        """Wrist-to-middle-knuckle distance: a scale reference for this hand.

        Thresholds expressed as a fraction of the span stay correct whether the
        hand is close to the camera or far away.
        """
        return max(distance(self.points[WRIST], self.points[MIDDLE_MCP]), 1e-6)


@dataclass
class Face:
    """One detected face; ``points`` holds all Face Mesh landmarks."""

    points: list[Point]

    def point(self, index: int) -> Point:
        return self.points[index]

    def named(self, name: str) -> Point:
        return self.points[FACE[name]]

    @property
    def interocular(self) -> float:
        """Distance between the eye centers: the face's scale reference."""
        left = midpoint(self.named("left_eye_outer"), self.named("left_eye_inner"))
        right = midpoint(self.named("right_eye_outer"), self.named("right_eye_inner"))
        return max(distance(left, right), 1e-6)


@dataclass
class Pose:
    """One detected body pose."""

    points: list[Point]
    visibility: list[float] = field(default_factory=list)

    def point(self, index: int) -> Point:
        return self.points[index]

    def named(self, name: str) -> Point:
        return self.points[POSE[name]]

    def named_visibility(self, name: str) -> float:
        index = POSE[name]
        if index < len(self.visibility):
            return self.visibility[index]
        return 1.0


@dataclass
class LandmarkFrame:
    """Everything the vision layer produced for a single frame.

    ``width``/``height`` describe the source image so consumers that need pixel
    units can convert; the landmarks themselves stay normalized.
    """

    hands: list[Hand] = field(default_factory=list)
    face: Face | None = None
    pose: Pose | None = None
    width: int = 0
    height: int = 0
    timestamp: float = 0.0
    seq: int = 0

    def hand_by_label(self, label: str) -> Hand | None:
        """First hand matching a handedness label, case insensitive."""
        wanted = label.lower()
        for hand in self.hands:
            if hand.label.lower() == wanted:
                return hand
        return None

    @property
    def primary_hand(self) -> Hand | None:
        return self.hands[0] if self.hands else None

    @property
    def aspect(self) -> float:
        if self.height <= 0:
            return 16 / 9
        return self.width / self.height


def flip_label(label: str) -> str:
    """Swap a handedness label.

    MediaPipe assigns handedness assuming a selfie-mirrored image, so whether
    "Right" means the user's right hand depends on whether the frame was
    mirrored before detection. Demos expose a config flag that routes through
    this helper instead of hardcoding an assumption.
    """
    if label.lower() == "left":
        return "Right"
    if label.lower() == "right":
        return "Left"
    return label


def _points_from_json(raw: Iterable[Any]) -> list[Point]:
    points: list[Point] = []
    for item in raw:
        if isinstance(item, dict):
            points.append((float(item.get("x", 0.0)), float(item.get("y", 0.0))))
        else:  # [x, y] pair
            points.append((float(item[0]), float(item[1])))
    return points


def _visibility_from_json(raw: Iterable[Any]) -> list[float]:
    values: list[float] = []
    for item in raw:
        if isinstance(item, dict):
            values.append(float(item.get("visibility", 1.0)))
        else:
            values.append(float(item[2]) if len(item) > 2 else 1.0)
    return values


def frame_from_json(payload: dict[str, Any]) -> LandmarkFrame:
    """Build a :class:`LandmarkFrame` from a browser WebSocket message.

    Accepts landmark points either as ``{"x":..,"y":..}`` objects (what
    tasks-vision emits) or as ``[x, y]`` pairs, and tolerates missing keys so a
    demo that only sends hands does not need to send empty face/pose fields.
    """
    frame = LandmarkFrame(
        width=int(payload.get("width", 0) or 0),
        height=int(payload.get("height", 0) or 0),
        timestamp=float(payload.get("ts", 0.0) or 0.0),
        seq=int(payload.get("seq", 0) or 0),
    )

    for raw_hand in payload.get("hands") or []:
        raw_points = raw_hand.get("points") or raw_hand.get("landmarks") or []
        points = _points_from_json(raw_points)
        if len(points) < 21:
            continue
        frame.hands.append(
            Hand(
                points=points,
                label=str(raw_hand.get("label", "Unknown")),
                score=float(raw_hand.get("score", 0.0) or 0.0),
            )
        )

    raw_face = payload.get("face")
    if raw_face:
        raw_points = raw_face.get("points") if isinstance(raw_face, dict) else raw_face
        points = _points_from_json(raw_points or [])
        if len(points) >= 468:
            frame.face = Face(points=points)

    raw_pose = payload.get("pose")
    if raw_pose:
        raw_points = raw_pose.get("points") if isinstance(raw_pose, dict) else raw_pose
        points = _points_from_json(raw_points or [])
        if len(points) >= 33:
            visibility = _visibility_from_json(raw_points or [])
            frame.pose = Pose(points=points, visibility=visibility)

    return frame


def frame_to_json(frame: LandmarkFrame) -> dict[str, Any]:
    """Serialize a frame back to the wire format (used by tests and tooling)."""
    return {
        "seq": frame.seq,
        "ts": frame.timestamp,
        "width": frame.width,
        "height": frame.height,
        "hands": [
            {
                "label": hand.label,
                "score": hand.score,
                "points": [{"x": p[0], "y": p[1]} for p in hand.points],
            }
            for hand in frame.hands
        ],
        "face": (
            {"points": [{"x": p[0], "y": p[1]} for p in frame.face.points]}
            if frame.face
            else None
        ),
        "pose": (
            {
                "points": [
                    {
                        "x": p[0],
                        "y": p[1],
                        "visibility": (
                            frame.pose.visibility[i]
                            if i < len(frame.pose.visibility)
                            else 1.0
                        ),
                    }
                    for i, p in enumerate(frame.pose.points)
                ]
            }
            if frame.pose
            else None
        ),
    }


def scale_points(points: Sequence[Point], width: int, height: int) -> list[tuple[int, int]]:
    """Convert normalized points to integer pixels."""
    return [(int(round(x * width)), int(round(y * height))) for x, y in points]
