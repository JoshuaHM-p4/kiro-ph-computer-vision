"""Tests for the Pogi Detector — no camera or SAM model needed.

Tests the translator (dictionary mapping) and segmentor (stub backend)
with synthetic data.

Run from the repository root:
    pytest projects/pogi_detector/test_pogi.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure imports work from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from translator import get_all_slang, is_known, translate, translate_or_passthrough  # noqa: E402
from segmentor import Instance, SegmentorConfig, StubBackend, load_backend  # noqa: E402


# ---------------------------------------------------------------------------
# Translator tests
# ---------------------------------------------------------------------------

class TestTranslator:
    """Tests for the Tagalog slang translator."""

    def test_translate_pogi(self):
        assert translate("pogi") == "handsome person / face"

    def test_translate_ganda(self):
        assert translate("ganda") == "beautiful person"

    def test_translate_chibog(self):
        assert translate("chibog") == "food / snack"

    def test_translate_tsismis(self):
        assert translate("tsismis") == "cell phone"

    def test_translate_case_insensitive(self):
        assert translate("POGI") == "handsome person / face"
        assert translate("Ganda") == "beautiful person"
        assert translate("CHIBOG") == "food / snack"

    def test_translate_strips_whitespace(self):
        assert translate("  pogi  ") == "handsome person / face"

    def test_translate_unknown_returns_none(self):
        assert translate("unknown_word") is None

    def test_translate_or_passthrough_known(self):
        assert translate_or_passthrough("pogi") == "handsome person / face"

    def test_translate_or_passthrough_unknown(self):
        assert translate_or_passthrough("cat") == "cat"

    def test_get_all_slang_returns_dict(self):
        slang = get_all_slang()
        assert isinstance(slang, dict)
        assert len(slang) == 4
        assert "pogi" in slang

    def test_get_all_slang_is_copy(self):
        """Modifying the returned dict should not affect the original."""
        slang = get_all_slang()
        slang["new_word"] = "test"
        assert "new_word" not in get_all_slang()

    def test_is_known(self):
        assert is_known("pogi") is True
        assert is_known("GANDA") is True
        assert is_known("unknown") is False


# ---------------------------------------------------------------------------
# Segmentor tests
# ---------------------------------------------------------------------------

class TestInstance:
    """Tests for the Instance dataclass."""

    def test_box_calculation(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:60] = True
        inst = Instance(label="pogi", english="handsome person / face", score=0.9, mask=mask)
        x, y, w, h = inst.box
        assert x == 30
        assert y == 20
        assert w == 30
        assert h == 20

    def test_box_empty_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        inst = Instance(label="pogi", english="handsome person / face", score=0.5, mask=mask)
        assert inst.box == (0, 0, 0, 0)

    def test_area(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 10:20] = True  # 10x10 = 100 pixels
        inst = Instance(label="pogi", english="handsome person / face", score=0.8, mask=mask)
        assert inst.area == 100


class TestStubBackend:
    """Tests for the stub segmentation backend."""

    def test_stub_is_ready(self):
        stub = StubBackend()
        assert stub.ready is True
        assert stub.name == "stub"

    def test_stub_returns_instances(self):
        stub = StubBackend()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = stub.segment(image, ["pogi"])
        assert len(results) == 1
        assert results[0].label == "pogi"
        assert results[0].english == "handsome person / face"

    def test_stub_mask_shape_matches_image(self):
        stub = StubBackend()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = stub.segment(image, ["ganda"])
        assert results[0].mask.shape == (480, 640)
        assert results[0].mask.dtype == bool

    def test_stub_multiple_prompts(self):
        stub = StubBackend()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = stub.segment(image, ["pogi", "ganda", "chibog"])
        assert len(results) == 3
        labels = [r.label for r in results]
        assert "pogi" in labels
        assert "ganda" in labels
        assert "chibog" in labels

    def test_stub_deterministic(self):
        """Same prompt should produce the same mask across calls."""
        stub = StubBackend()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result1 = stub.segment(image, ["pogi"])[0]
        result2 = stub.segment(image, ["pogi"])[0]
        assert np.array_equal(result1.mask, result2.mask)
        assert result1.score == result2.score

    def test_stub_different_prompts_different_masks(self):
        """Different prompts should produce different masks."""
        stub = StubBackend()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result_pogi = stub.segment(image, ["pogi"])[0]
        result_ganda = stub.segment(image, ["ganda"])[0]
        assert not np.array_equal(result_pogi.mask, result_ganda.mask)

    def test_stub_mask_has_true_pixels(self):
        """Stub masks should not be empty."""
        stub = StubBackend()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = stub.segment(image, ["pogi"])
        assert results[0].mask.any()

    def test_stub_score_in_range(self):
        stub = StubBackend()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = stub.segment(image, ["pogi"])
        assert 0.0 <= results[0].score <= 1.0

    def test_stub_empty_prompts(self):
        stub = StubBackend()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = stub.segment(image, [])
        assert results == []

    def test_stub_passthrough_unknown_term(self):
        """Unknown terms pass through untranslated."""
        stub = StubBackend()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = stub.segment(image, ["cat"])
        assert results[0].label == "cat"
        assert results[0].english == "cat"  # passthrough

    def test_stub_describe(self):
        stub = StubBackend()
        desc = stub.describe()
        assert "demo" in desc.lower() or "stub" in desc.lower()


class TestLoadBackend:
    """Tests for the load_backend factory function."""

    def test_stub_mode(self):
        backend = load_backend(stub=True)
        assert backend.name == "stub"
        assert backend.ready is True

    def test_no_token_returns_stub(self):
        """Without a token, should fall back to stub (not crash)."""
        backend = load_backend(token=None)
        assert backend.ready is True
        assert backend.name == "stub"

    def test_default_config(self):
        config = SegmentorConfig()
        assert config.confidence == 0.4
        assert config.device == "cpu"
