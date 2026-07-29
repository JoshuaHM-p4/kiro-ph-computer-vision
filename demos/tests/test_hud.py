"""Tests for the HUD renderer and the desktop detector wrapper.

These run headlessly: nothing here opens a window or a camera.
"""

from __future__ import annotations

import numpy as np
import pytest

from demos.common import hud
from demos.common import landmarks as lm
from demos.tests.fixtures import make_face, make_frame, make_hand, make_pose


@pytest.fixture()
def frame() -> np.ndarray:
    return np.full((240, 320, 3), 40, dtype=np.uint8)


def _changed(before: np.ndarray, after: np.ndarray) -> bool:
    return bool(np.any(before != after))


class TestHudPrimitives:
    def test_panel_preserves_shape_and_dtype(self, frame):
        before = frame.copy()
        out = hud.panel(frame, (10, 10), (200, 80))
        assert out.shape == before.shape
        assert out.dtype == np.uint8
        assert _changed(before, out)

    def test_panel_clips_out_of_bounds_rects(self, frame):
        hud.panel(frame, (-50, -50), (1000, 1000))
        hud.panel(frame, (500, 500), (600, 600))  # entirely outside
        assert frame.shape == (240, 320, 3)

    def test_text_and_title_draw(self, frame):
        before = frame.copy()
        hud.title(frame, "DEMO", "subtitle")
        hud.text(frame, "hello", (30, 120))
        assert _changed(before, frame)

    def test_status_strip_draws_within_bounds(self, frame):
        hud.status_strip(frame, [("FPS", "30.0"), ("MODE", "DRAW")])
        assert frame.shape == (240, 320, 3)

    @pytest.mark.parametrize("value", [-1.0, 0.0, 0.5, 1.0, 2.0])
    def test_gauge_clamps_value(self, frame, value):
        hud.gauge(frame, (20, 200), (120, 10), value, label="SIZE")
        assert frame.dtype == np.uint8

    @pytest.mark.parametrize("progress", [0.0, 0.5, 1.0])
    def test_ring_draws_progress(self, frame, progress):
        hud.ring(frame, (160, 120), 20, progress)
        assert frame.shape == (240, 320, 3)

    def test_glow_brightens_the_shape(self, frame):
        def draw(layer):
            import cv2

            cv2.circle(layer, (160, 120), 20, hud.THEME.cyan, -1)

        before = frame.copy()
        hud.glow(frame, draw, blur=15)
        assert frame[120, 160].sum() > before[120, 160].sum()
        assert frame.dtype == np.uint8

    def test_scanlines_darken_alternating_rows(self, frame):
        hud.scanlines(frame, spacing=2, strength=0.5)
        assert frame[0, 0, 0] < frame[1, 0, 0]

    def test_vignette_darkens_corners_more_than_centre(self, frame):
        hud.vignette(frame, 0.5)
        assert frame[0, 0, 0] < frame[120, 160, 0]

    def test_crosshair_draws(self, frame):
        before = frame.copy()
        hud.crosshair(frame, (160, 120))
        assert _changed(before, frame)


class TestLandmarkOverlays:
    def test_draw_hand_marks_pixels(self, frame):
        before = frame.copy()
        hud.draw_hand(frame, make_hand())
        assert _changed(before, frame)

    def test_draw_face_contours_and_dense(self, frame):
        contour = frame.copy()
        hud.draw_face(contour, make_face())
        dense = frame.copy()
        hud.draw_face(dense, make_face(), dense=True)
        assert _changed(frame, contour)
        assert _changed(frame, dense)

    def test_draw_pose_marks_pixels(self, frame):
        before = frame.copy()
        hud.draw_pose(frame, make_pose())
        assert _changed(before, frame)

    def test_draw_frame_landmarks_respects_flags(self, frame):
        landmark_frame = make_frame(hands=[make_hand()], face=make_face(), pose=make_pose())
        none_drawn = frame.copy()
        hud.draw_frame_landmarks(none_drawn, landmark_frame, hands=False, face=False, pose=False)
        assert not _changed(frame, none_drawn)

        all_drawn = frame.copy()
        hud.draw_frame_landmarks(all_drawn, landmark_frame)
        assert _changed(frame, all_drawn)

    def test_landmarks_outside_the_frame_do_not_raise(self, frame):
        far = make_hand(center=(2.5, -1.5))
        hud.draw_hand(frame, far)
        assert frame.shape == (240, 320, 3)


class TestAlphaBlit:
    def _sprite(self, size=40, alpha=255):
        sprite = np.zeros((size, size, 4), dtype=np.uint8)
        sprite[:, :, :3] = (0, 0, 255)
        sprite[:, :, 3] = alpha
        return sprite

    def test_opaque_sprite_replaces_pixels(self, frame):
        hud.alpha_blit(frame, self._sprite(), (100, 100))
        assert tuple(frame[120, 120]) == (0, 0, 255)

    def test_half_alpha_blends(self, frame):
        hud.alpha_blit(frame, self._sprite(alpha=128), (100, 100))
        pixel = frame[120, 120]
        assert 0 < pixel[2] < 255
        assert pixel[0] < 40 + 1

    def test_clips_at_every_edge(self, frame):
        for position in ((-20, -20), (300, 220), (-20, 100), (300, -20)):
            hud.alpha_blit(frame, self._sprite(), position)
        assert frame.shape == (240, 320, 3)

    def test_fully_offscreen_is_a_noop(self, frame):
        before = frame.copy()
        hud.alpha_blit(frame, self._sprite(), (1000, 1000))
        assert not _changed(before, frame)

    def test_rejects_bgr_sprites(self, frame):
        with pytest.raises(ValueError):
            hud.alpha_blit(frame, np.zeros((10, 10, 3), dtype=np.uint8), (0, 0))


class TestDetectorWrapper:
    """The wrapper is exercised on a still image, so no camera is needed."""

    def test_detects_nothing_in_a_blank_image_without_raising(self):
        from demos.common.detectors import VisionPipeline

        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        with VisionPipeline(hands=True, face=True, pose=True) as pipeline:
            result = pipeline.detect(blank, timestamp=0.0)
        assert result.hands == []
        assert result.face is None
        assert result.pose is None
        assert result.width == 320
        assert result.height == 240
        assert result.seq == 1

    def test_only_requested_graphs_are_created(self):
        from demos.common.detectors import VisionPipeline

        with VisionPipeline(hands=True) as pipeline:
            assert pipeline._hands is not None
            assert pipeline._face is None
            assert pipeline._pose is None

    def test_seq_increments_per_frame(self):
        from demos.common.detectors import VisionPipeline

        blank = np.zeros((120, 160, 3), dtype=np.uint8)
        with VisionPipeline(hands=True) as pipeline:
            first = pipeline.detect(blank)
            second = pipeline.detect(blank)
        assert (first.seq, second.seq) == (1, 2)

    def test_swap_handedness_flag_flips_labels(self):
        assert lm.flip_label("Left") == "Right"
        assert lm.flip_label("Right") == "Left"
