"""Tests for CameraLoop plumbing that does not need a camera or a window."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from demos.common.camera import (
    CameraConfig,
    CameraLoop,
    add_camera_arguments,
    camera_config_from_args,
)


@pytest.fixture()
def loop() -> CameraLoop:
    config = CameraConfig(window_name="test", debug=True, mirror=True)
    instance = CameraLoop(config, hands=True)
    yield instance
    instance.pipeline.close()


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_camera_arguments(parser)
    return parser.parse_args(argv)


class TestArgumentPlumbing:
    def test_defaults(self):
        config = camera_config_from_args(_args([]), "demo")
        assert config.camera_index == 0
        assert config.mirror is True
        assert config.debug is True
        assert config.swap_handedness is False
        assert config.window_name == "demo"

    def test_flags_invert_defaults(self):
        config = camera_config_from_args(
            _args(["--camera", "1", "--no-mirror", "--no-debug", "--swap-handedness"]), "demo"
        )
        assert config.camera_index == 1
        assert config.mirror is False
        assert config.debug is False
        assert config.swap_handedness is True

    def test_resolution_flags(self):
        config = camera_config_from_args(_args(["--width", "640", "--height", "480"]), "demo")
        assert (config.width, config.height) == (640, 480)


class TestKeyHandling:
    def test_quit_keys_stop_the_loop(self, loop):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        loop.running = True
        loop._handle_key(ord("q"), frame)
        assert loop.running is False

        loop.running = True
        loop._handle_key(27, frame)
        assert loop.running is False

    def test_toggles(self, loop):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        loop._handle_key(ord("d"), frame)
        assert loop.debug is False
        loop._handle_key(ord(" "), frame)
        assert loop.mirror is False

    def test_screenshot_writes_a_file(self, loop, tmp_path: Path):
        loop.config.screenshot_dir = tmp_path
        frame = np.full((20, 20, 3), 128, dtype=np.uint8)
        path = loop.save_screenshot(frame, prefix="unit")
        assert path.exists()
        assert path.suffix == ".png"
        assert loop.last_screenshot == path

    def test_stop_sets_running_false(self, loop):
        loop.running = True
        loop.stop()
        assert loop.running is False


def test_open_reports_a_helpful_error_for_a_bad_index():
    config = CameraConfig(camera_index=99)
    loop = CameraLoop(config)
    try:
        with pytest.raises(RuntimeError, match="Could not open camera 99"):
            loop.open()
    finally:
        loop.pipeline.close()
