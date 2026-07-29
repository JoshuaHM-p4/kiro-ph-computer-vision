"""Tests for the pure geometry helpers."""

from __future__ import annotations

import pytest

from demos.common.geometry import (
    DwellTimer,
    EdgeTrigger,
    EMAPoint,
    EMAScalar,
    FPSCounter,
    HysteresisLatch,
    clamp,
    distance,
    lerp,
    lerp_point,
    natural_key,
    normalize_range,
    polyline_length,
    to_pixels,
)


def test_distance_and_clamp():
    assert distance((0, 0), (3, 4)) == pytest.approx(5.0)
    assert clamp(5, 0, 1) == 1
    assert clamp(-5, 0, 1) == 0
    assert clamp(0.5, 0, 1) == 0.5


def test_lerp_helpers():
    assert lerp(0, 10, 0.25) == pytest.approx(2.5)
    assert lerp_point((0, 0), (10, 20), 0.5) == pytest.approx((5.0, 10.0))


def test_to_pixels_rounds():
    assert to_pixels((0.5, 0.25), 640, 480) == (320, 120)
    assert to_pixels((0.999, 0.999), 100, 100) == (100, 100)


def test_normalize_range_handles_degenerate_span():
    assert normalize_range(5, 0, 10) == pytest.approx(0.5)
    assert normalize_range(-5, 0, 10) == 0.0
    assert normalize_range(50, 0, 10) == 1.0
    assert normalize_range(5, 3, 3) == 0.0


def test_natural_key_orders_numbers_numerically():
    names = ["slide10.png", "slide2.png", "slide1.png"]
    assert sorted(names, key=natural_key) == ["slide1.png", "slide2.png", "slide10.png"]


def test_polyline_length():
    assert polyline_length([]) == 0.0
    assert polyline_length([(0, 0)]) == 0.0
    assert polyline_length([(0, 0), (0, 3), (4, 3)]) == pytest.approx(7.0)


def test_ema_point_first_sample_is_exact_then_smooths():
    ema = EMAPoint(alpha=0.5)
    assert ema.update((10, 20)) == (10.0, 20.0)
    assert ema.update((20, 20)) == pytest.approx((15.0, 20.0))
    assert ema.update(None) == pytest.approx((15.0, 20.0))
    ema.reset()
    assert ema.value is None


def test_ema_scalar_converges_toward_target():
    ema = EMAScalar(alpha=0.5)
    ema.update(0.0)
    for _ in range(20):
        ema.update(1.0)
    assert ema.value == pytest.approx(1.0, abs=1e-3)


class TestHysteresisLatch:
    def test_requires_crossing_both_thresholds(self):
        latch = HysteresisLatch(on_below=0.3, off_above=0.5)
        assert latch.update(0.4) is False       # between thresholds, stays off
        assert latch.update(0.3) is True        # at the on threshold, turns on
        assert latch.update(0.4) is True        # between thresholds, stays on
        assert latch.update(0.5) is False       # at the off threshold, turns off

    def test_noise_between_thresholds_cannot_toggle(self):
        latch = HysteresisLatch(on_below=0.3, off_above=0.5)
        latch.update(0.1)
        for value in (0.35, 0.45, 0.35, 0.49, 0.31):
            assert latch.update(value) is True

    def test_inverted_latch_switches_on_when_rising(self):
        latch = HysteresisLatch(on_below=0.6, off_above=0.4, invert=True)
        assert latch.update(0.5) is False
        assert latch.update(0.6) is True
        assert latch.update(0.5) is True
        assert latch.update(0.4) is False

    def test_none_holds_state(self):
        latch = HysteresisLatch(on_below=0.3, off_above=0.5)
        latch.update(0.1)
        assert latch.update(None) is True


class TestEdgeTrigger:
    def test_fires_once_per_rising_edge(self):
        trigger = EdgeTrigger()
        assert trigger.fire(True, now=0.0) is True
        assert trigger.fire(True, now=0.1) is False
        assert trigger.fire(False, now=0.2) is False
        assert trigger.fire(True, now=0.3) is True

    def test_cooldown_suppresses_rapid_refire(self):
        trigger = EdgeTrigger(cooldown=1.0)
        assert trigger.fire(True, now=0.0) is True
        trigger.fire(False, now=0.1)
        assert trigger.fire(True, now=0.5) is False   # inside the cooldown
        trigger.fire(False, now=0.6)
        assert trigger.fire(True, now=1.5) is True    # cooldown elapsed


class TestDwellTimer:
    def test_fires_only_after_threshold_and_only_once(self):
        dwell = DwellTimer(threshold=0.5)
        assert dwell.update("red", now=0.0) is None
        assert dwell.update("red", now=0.4) is None
        assert dwell.update("red", now=0.5) == "red"
        assert dwell.update("red", now=0.9) is None   # already fired

    def test_sweeping_past_a_key_does_not_fire(self):
        dwell = DwellTimer(threshold=0.5)
        for index, key in enumerate(["red", "green", "blue", "erase"]):
            assert dwell.update(key, now=index * 0.1) is None

    def test_leaving_and_returning_refires(self):
        dwell = DwellTimer(threshold=0.2)
        dwell.update("red", now=0.0)
        assert dwell.update("red", now=0.2) == "red"
        dwell.update(None, now=0.3)
        dwell.update("red", now=0.4)
        assert dwell.update("red", now=0.7) == "red"

    def test_progress_reports_fraction(self):
        dwell = DwellTimer(threshold=1.0)
        dwell.update("red", now=0.0)
        assert dwell.progress(now=0.25) == pytest.approx(0.25)
        assert dwell.progress(now=5.0) == pytest.approx(1.0)


def test_fps_counter_reports_positive_rate():
    fps = FPSCounter()
    fps.update(now=0.0)
    for i in range(1, 40):
        value = fps.update(now=i / 30.0)
    assert value == pytest.approx(30.0, rel=0.15)
