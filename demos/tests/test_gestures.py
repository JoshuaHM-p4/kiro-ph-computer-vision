"""Tests for the landmark-derived gesture primitives."""

from __future__ import annotations

import pytest

from demos.common import gestures as gs
from demos.common import landmarks as lm
from demos.tests.fixtures import make_face, make_hand, make_pose


class TestPinch:
    def test_ratio_is_small_when_closed_and_large_when_open(self):
        closed = gs.pinch_ratio(make_hand(pinch=0.05))
        open_hand = gs.pinch_ratio(make_hand(pinch=1.0))
        assert closed < 0.2
        assert open_hand > 0.5

    def test_ratio_is_scale_invariant(self):
        near = gs.pinch_ratio(make_hand(pinch=0.3, span=0.30))
        far = gs.pinch_ratio(make_hand(pinch=0.3, span=0.06))
        assert near == pytest.approx(far, rel=1e-6)

    def test_detector_latches_with_hysteresis(self):
        detector = gs.PinchDetector(start_ratio=0.3, release_ratio=0.5)
        assert detector.update(make_hand(pinch=1.0), now=0.0).active is False
        assert detector.update(make_hand(pinch=0.05), now=0.1).active is True
        # A gap between the two thresholds must not release the pinch.
        mid = make_hand(pinch=0.4)
        assert 0.3 < gs.pinch_ratio(mid) < 0.5
        assert detector.update(mid, now=0.2).active is True
        assert detector.update(make_hand(pinch=1.0), now=0.3).active is False

    def test_just_started_fires_once_per_pinch(self):
        detector = gs.PinchDetector(start_ratio=0.3, release_ratio=0.5)
        first = detector.update(make_hand(pinch=0.05), now=0.0)
        second = detector.update(make_hand(pinch=0.05), now=0.1)
        assert first.just_started is True
        assert second.just_started is False

    def test_missing_hand_releases_the_latch(self):
        detector = gs.PinchDetector(start_ratio=0.3, release_ratio=0.5)
        detector.update(make_hand(pinch=0.05), now=0.0)
        state = detector.update(None, now=0.1)
        assert state.active is False
        assert state.ratio is None
        assert state.point is None

    def test_pinch_point_sits_between_the_tips(self):
        hand = make_hand(pinch=0.2)
        point = gs.pinch_point(hand)
        assert min(hand.thumb_tip[0], hand.index_tip[0]) <= point[0] <= max(
            hand.thumb_tip[0], hand.index_tip[0]
        )


class TestFingerExtension:
    def test_detects_each_curled_finger(self):
        hand = make_hand(extended=("index",))
        flags = gs.extended_fingers(hand)
        assert flags["index"] is True
        assert flags["middle"] is False
        assert flags["ring"] is False
        assert flags["pinky"] is False

    @pytest.mark.parametrize("rotation", [0.0, 35.0, -35.0, 90.0, 180.0])
    def test_extension_survives_hand_rotation(self, rotation):
        hand = make_hand(extended=("index",), rotation_deg=rotation)
        flags = gs.extended_fingers(hand)
        assert flags["index"] is True
        assert flags["middle"] is False

    def test_pose_classifiers(self):
        assert gs.is_pointing(make_hand(extended=("index",))) is True
        assert gs.is_pointing(make_hand(extended=("index", "middle"))) is False
        assert gs.is_peace(make_hand(extended=("index", "middle"))) is True
        assert gs.is_fist(make_hand(extended=())) is True
        assert gs.is_open_palm(make_hand(extended=("index", "middle", "ring", "pinky"))) is True

    def test_count_extended(self):
        assert gs.count_extended(make_hand(extended=("index", "middle"))) == 2


class TestFaceRatios:
    def test_ear_drops_when_eyes_close(self):
        assert gs.average_ear(make_face(eye_open=1.0)) > gs.average_ear(
            make_face(eye_open=0.1)
        )

    def test_ear_is_scale_invariant(self):
        near = gs.average_ear(make_face(scale=0.4))
        far = gs.average_ear(make_face(scale=0.1))
        assert near == pytest.approx(far, rel=1e-6)

    def test_mar_rises_when_mouth_opens(self):
        assert gs.mouth_aspect_ratio(make_face(mouth_open=0.0)) < 0.1
        assert gs.mouth_aspect_ratio(make_face(mouth_open=0.4)) > 0.4

    def test_corner_lift_is_positive_for_a_smile(self):
        assert gs.mouth_corner_lift(make_face(corner_lift=0.0)) == pytest.approx(0.0, abs=1e-6)
        assert gs.mouth_corner_lift(make_face(corner_lift=0.08)) > 0.1

    def test_brow_raise_increases_with_raised_brows(self):
        assert gs.brow_raise(make_face(brow_raise=0.12)) > gs.brow_raise(
            make_face(brow_raise=0.0)
        )

    def test_mouth_width_ratio_tracks_stretch(self):
        assert gs.mouth_width_ratio(make_face(mouth_width=1.4)) > gs.mouth_width_ratio(
            make_face(mouth_width=1.0)
        )


class TestHeadPose:
    """Convention under test: positive yaw = head turned toward image-right.

    The fixture's ``yaw`` parameter rotates the mesh about the vertical axis so
    that a positive value moves the nose toward the right of the image.
    """

    def test_center_face_yaw_is_near_zero(self):
        pose = gs.head_pose(make_face(yaw=0.0), 640, 480)
        assert abs(pose.yaw) < 8.0

    def test_yaw_sign_matches_rotation_direction(self):
        left = gs.head_pose(make_face(yaw=-30.0), 640, 480)
        right = gs.head_pose(make_face(yaw=30.0), 640, 480)
        assert left.yaw < -8.0
        assert right.yaw > 8.0

    def test_yaw_magnitude_is_monotonic(self):
        small = gs.head_pose(make_face(yaw=15.0), 640, 480).yaw
        large = gs.head_pose(make_face(yaw=40.0), 640, 480).yaw
        assert large > small

    def test_ratio_fallback_agrees_with_solvepnp_sign(self):
        assert gs.yaw_ratio(make_face(yaw=-30.0)) < 0
        assert gs.yaw_ratio(make_face(yaw=30.0)) > 0
        assert gs.yaw_ratio(make_face(yaw=0.0)) == pytest.approx(0.0, abs=1e-6)
        for angle in (-40.0, -20.0, 20.0, 40.0):
            solved = gs.head_pose(make_face(yaw=angle), 640, 480).yaw
            assert (solved > 0) == (gs.yaw_ratio(make_face(yaw=angle)) > 0)

    def test_degenerate_face_falls_back_without_raising(self):
        flat = lm.Face(points=[(0.5, 0.5)] * 478)
        pose = gs.head_pose(flat, 640, 480)
        assert pose.method in {"solvepnp", "ratio"}
        assert pose.yaw == pytest.approx(0.0, abs=1e-6)


class TestPoseHelpers:
    def test_torso_scale_is_positive(self):
        assert gs.torso_scale(make_pose()) > 0

    def test_wrist_height_sign(self):
        raised = make_pose(left_wrist_up=0.6)
        assert gs.wrist_height(raised, "left") == pytest.approx(0.6, rel=1e-6)
        lowered = make_pose(left_wrist_up=-0.4)
        assert gs.wrist_height(lowered, "left") == pytest.approx(-0.4, rel=1e-6)

    def test_wrist_height_is_scale_invariant(self):
        near = make_pose(left_wrist_up=0.5, shoulder_y=0.2, hip_y=0.95)
        far = make_pose(left_wrist_up=0.5, shoulder_y=0.45, hip_y=0.60)
        assert gs.wrist_height(near, "left") == pytest.approx(
            gs.wrist_height(far, "left"), rel=1e-6
        )


class TestLandmarkModel:
    def test_hand_label_lookup_and_flip(self):
        frame_hands = [make_hand(label="Left"), make_hand(label="Right")]
        assert frame_hands[0].label == "Left"
        assert lm.flip_label("Left") == "Right"
        assert lm.flip_label("Right") == "Left"
        assert lm.flip_label("Unknown") == "Unknown"

    def test_json_round_trip_preserves_landmarks(self):
        frame = lm.LandmarkFrame(
            hands=[make_hand(label="Right")],
            face=make_face(),
            pose=make_pose(),
            width=640,
            height=480,
            seq=7,
        )
        restored = lm.frame_from_json(lm.frame_to_json(frame))
        assert restored.seq == 7
        assert restored.width == 640
        assert len(restored.hands) == 1
        assert restored.hands[0].label == "Right"
        assert restored.hands[0].index_tip == pytest.approx(frame.hands[0].index_tip)
        assert restored.face is not None
        assert restored.pose is not None
        assert restored.face.named("nose_tip") == pytest.approx(frame.face.named("nose_tip"))

    def test_json_accepts_bare_xy_pairs(self):
        hand = make_hand()
        payload = {
            "seq": 3,
            "width": 320,
            "height": 240,
            "hands": [{"label": "Left", "points": [[x, y] for x, y in hand.points]}],
        }
        restored = lm.frame_from_json(payload)
        assert len(restored.hands) == 1
        assert restored.hands[0].wrist == pytest.approx(hand.wrist)

    def test_json_drops_incomplete_landmark_sets(self):
        payload = {
            "hands": [{"label": "Left", "points": [[0.1, 0.1]] * 5}],
            "face": {"points": [[0.1, 0.1]] * 10},
            "pose": {"points": [[0.1, 0.1]] * 4},
        }
        restored = lm.frame_from_json(payload)
        assert restored.hands == []
        assert restored.face is None
        assert restored.pose is None

    def test_missing_keys_are_tolerated(self):
        restored = lm.frame_from_json({"seq": 1})
        assert restored.hands == []
        assert restored.face is None
        assert restored.pose is None
