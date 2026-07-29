"""Desktop-only MediaPipe wrappers.

Each wrapper turns MediaPipe's protobuf output into the plain
:mod:`demos.common.landmarks` dataclasses, so demo logic never sees a MediaPipe
type and behaves identically whether landmarks came from Python or the browser.

Detectors are created lazily and only for the streams a demo asks for: running
Face Mesh, Hands and Pose together triples the per-frame cost, and most demos
need one or two.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp

from . import landmarks as lm


@dataclass
class DetectorConfig:
    """Tunables for the MediaPipe graphs."""

    max_hands: int = 2
    hand_detection_confidence: float = 0.6
    hand_tracking_confidence: float = 0.5
    face_detection_confidence: float = 0.5
    face_tracking_confidence: float = 0.5
    refine_face_landmarks: bool = True
    pose_detection_confidence: float = 0.5
    pose_tracking_confidence: float = 0.5
    pose_model_complexity: int = 1
    # Handedness labels assume a selfie-mirrored image. Set this when the frame
    # handed to detect() is NOT mirrored, so labels describe the real hand.
    swap_handedness: bool = False


class VisionPipeline:
    """Runs the requested MediaPipe graphs over BGR frames.

    Usage::

        pipeline = VisionPipeline(hands=True, face=True)
        frame = pipeline.detect(bgr_image)   # -> LandmarkFrame
        pipeline.close()
    """

    def __init__(
        self,
        *,
        hands: bool = False,
        face: bool = False,
        pose: bool = False,
        config: DetectorConfig | None = None,
    ):
        self.config = config or DetectorConfig()
        self._want_hands = hands
        self._want_face = face
        self._want_pose = pose
        self._seq = 0

        self._hands = (
            mp.solutions.hands.Hands(
                max_num_hands=self.config.max_hands,
                min_detection_confidence=self.config.hand_detection_confidence,
                min_tracking_confidence=self.config.hand_tracking_confidence,
            )
            if hands
            else None
        )
        self._face = (
            mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=self.config.refine_face_landmarks,
                min_detection_confidence=self.config.face_detection_confidence,
                min_tracking_confidence=self.config.face_tracking_confidence,
            )
            if face
            else None
        )
        self._pose = (
            mp.solutions.pose.Pose(
                model_complexity=self.config.pose_model_complexity,
                min_detection_confidence=self.config.pose_detection_confidence,
                min_tracking_confidence=self.config.pose_tracking_confidence,
            )
            if pose
            else None
        )

    def detect(self, frame_bgr, timestamp: float = 0.0) -> lm.LandmarkFrame:
        """Run the enabled graphs over one BGR frame."""
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # lets MediaPipe skip a copy

        self._seq += 1
        out = lm.LandmarkFrame(width=width, height=height, timestamp=timestamp, seq=self._seq)

        if self._hands is not None:
            result = self._hands.process(rgb)
            if result.multi_hand_landmarks:
                handedness = result.multi_handedness or []
                for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    label = "Unknown"
                    score = 0.0
                    if index < len(handedness) and handedness[index].classification:
                        classification = handedness[index].classification[0]
                        label = classification.label
                        score = classification.score
                    if self.config.swap_handedness:
                        label = lm.flip_label(label)
                    out.hands.append(
                        lm.Hand(
                            points=[(p.x, p.y) for p in hand_landmarks.landmark],
                            label=label,
                            score=score,
                        )
                    )

        if self._face is not None:
            result = self._face.process(rgb)
            if result.multi_face_landmarks:
                points = [(p.x, p.y) for p in result.multi_face_landmarks[0].landmark]
                out.face = lm.Face(points=points)

        if self._pose is not None:
            result = self._pose.process(rgb)
            if result.pose_landmarks:
                out.pose = lm.Pose(
                    points=[(p.x, p.y) for p in result.pose_landmarks.landmark],
                    visibility=[p.visibility for p in result.pose_landmarks.landmark],
                )

        return out

    def close(self) -> None:
        for detector in (self._hands, self._face, self._pose):
            if detector is not None:
                detector.close()

    def __enter__(self) -> "VisionPipeline":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
