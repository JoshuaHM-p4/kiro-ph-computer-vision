"""Tests for the slide presenter core: deck loading, pinch navigation, laser."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from demos.common import landmarks as lm
from demos.slide_presenter.config import SlideConfig
from demos.slide_presenter.core import Deck, SlidePresenterCore, fit_into
from demos.tests.fixtures import make_frame, make_hand


@pytest.fixture()
def deck_dir(tmp_path: Path) -> Path:
    """A five-slide deck whose filenames exercise natural sorting."""
    for name in ("slide10.png", "slide2.png", "slide1.png", "slide20.png", "slide3.png"):
        image = np.full((90, 160, 3), 60, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / name), image)
    (tmp_path / "notes.txt").write_text("ignored")
    return tmp_path


@pytest.fixture()
def core(deck_dir: Path) -> SlidePresenterCore:
    config = SlideConfig(slides_dir=deck_dir, advance_cooldown=0.5)
    return SlidePresenterCore(config)


def pinch(label: str) -> lm.LandmarkFrame:
    """A pinching hand of the given handedness."""
    return make_frame(hands=[make_hand(label=label, extended=("index",), pinch=0.05)])


def open_hand(label: str = "Right") -> lm.LandmarkFrame:
    return make_frame(hands=[make_hand(label=label, extended=("index", "middle", "ring", "pinky"))])


def point_at(tip: tuple[float, float], label: str = "Right") -> lm.LandmarkFrame:
    hand = make_hand(label=label, extended=("index",), span=0.08)
    offset = (tip[0] - hand.index_tip[0], tip[1] - hand.index_tip[1])
    hand.points = [(x + offset[0], y + offset[1]) for x, y in hand.points]
    return make_frame(hands=[hand])


class TestDeck:
    def test_loads_images_in_natural_order(self, deck_dir: Path):
        deck = Deck.load(deck_dir)
        assert [p.name for p in deck.paths] == [
            "slide1.png",
            "slide2.png",
            "slide3.png",
            "slide10.png",
            "slide20.png",
        ]

    def test_ignores_non_images(self, deck_dir: Path):
        assert all(p.suffix == ".png" for p in Deck.load(deck_dir).paths)

    def test_missing_directory_is_empty_not_an_error(self, tmp_path: Path):
        deck = Deck.load(tmp_path / "nope")
        assert deck.is_empty
        assert len(deck) == 0
        assert deck.name(0) == "no slides"
        assert deck.image(0) is None

    def test_images_are_cached_after_first_read(self, deck_dir: Path):
        deck = Deck.load(deck_dir)
        first = deck.image(0)
        assert first is not None
        assert deck.image(0) is first

    def test_index_wraps_for_lookups(self, deck_dir: Path):
        deck = Deck.load(deck_dir)
        assert deck.name(len(deck)) == deck.name(0)


class TestPinchNavigation:
    def test_right_pinch_advances_exactly_once(self, core: SlidePresenterCore):
        for step in range(5):  # pinch held for five frames
            core.update(pinch("Right"), step * 0.05)
        assert core.index == 1
        assert core.last_action == "next"

    def test_left_pinch_goes_back(self, core: SlidePresenterCore):
        core.go_to(3)
        for step in range(5):
            core.update(pinch("Left"), step * 0.05)
        assert core.index == 2
        assert core.last_action == "previous"

    def test_release_and_repinch_advances_again(self, core: SlidePresenterCore):
        core.update(pinch("Right"), 0.0)
        core.update(open_hand(), 0.1)
        core.update(pinch("Right"), 1.0)  # past the cooldown
        assert core.index == 2

    def test_cooldown_blocks_a_rapid_second_pinch(self, core: SlidePresenterCore):
        core.update(pinch("Right"), 0.0)
        core.update(open_hand(), 0.05)
        core.update(pinch("Right"), 0.2)  # inside the 0.5 s cooldown
        assert core.index == 1

    def test_missing_hand_releases_the_latch(self, core: SlidePresenterCore):
        core.update(pinch("Right"), 0.0)
        core.update(make_frame(), 0.1)  # hand left the frame
        core.update(pinch("Right"), 1.0)
        assert core.index == 2

    def test_both_hands_are_tracked_independently(self, core: SlidePresenterCore):
        core.go_to(2)
        both = make_frame(
            hands=[
                make_hand(label="Right", extended=("index",), pinch=0.05),
                make_hand(label="Left", extended=("index", "middle", "ring", "pinky")),
            ]
        )
        core.update(both, 0.0)
        assert core.index == 3
        # Now the left hand pinches while the right stays closed: still one step back.
        both2 = make_frame(
            hands=[
                make_hand(label="Right", extended=("index",), pinch=0.05),
                make_hand(label="Left", extended=("index",), pinch=0.05),
            ]
        )
        core.update(both2, 1.0)
        assert core.index == 2

    def test_unknown_handedness_does_not_navigate(self, core: SlidePresenterCore):
        frame = make_frame(hands=[make_hand(label="Unknown", extended=("index",), pinch=0.05)])
        for step in range(4):
            core.update(frame, step * 0.05)
        assert core.index == 0

    def test_open_hand_never_navigates(self, core: SlidePresenterCore):
        for step in range(10):
            core.update(open_hand(), step * 0.1)
        assert core.index == 0


class TestBoundaries:
    def test_clamps_at_the_end(self, core: SlidePresenterCore):
        core.go_to(len(core.deck) - 1)
        core.next_slide()
        assert core.index == len(core.deck) - 1
        assert core.state(0.0)["atEnd"] is True

    def test_clamps_at_the_start(self, core: SlidePresenterCore):
        core.previous_slide()
        assert core.index == 0
        assert core.state(0.0)["atStart"] is True

    def test_loop_deck_wraps_both_ways(self, deck_dir: Path):
        core = SlidePresenterCore(SlideConfig(slides_dir=deck_dir, loop_deck=True))
        core.go_to(len(core.deck) - 1)
        core.next_slide()
        assert core.index == 0
        core.previous_slide()
        assert core.index == len(core.deck) - 1

    def test_empty_deck_stays_at_zero(self, tmp_path: Path):
        core = SlidePresenterCore(SlideConfig(slides_dir=tmp_path))
        core.next_slide()
        core.next_slide()
        assert core.index == 0
        assert core.state(0.0)["count"] == 0

    def test_goto_clamps_out_of_range(self, core: SlidePresenterCore):
        core.go_to(999)
        assert core.index == len(core.deck) - 1
        core.go_to(-5)
        assert core.index == 0


class TestHandednessSwap:
    def test_swap_reverses_the_mapping(self, deck_dir: Path):
        core = SlidePresenterCore(SlideConfig(slides_dir=deck_dir, swap_handedness=True))
        core.go_to(2)
        core.update(pinch("Right"), 0.0)   # labelled Right, treated as Left
        assert core.index == 1
        core.update(make_frame(), 0.5)
        core.update(pinch("Left"), 1.5)
        assert core.index == 2

    def test_state_reports_the_hand_mapping(self, core: SlidePresenterCore):
        state = core.state(0.0)
        assert state["nextHand"] == "Right"
        assert state["previousHand"] == "Left"


class TestLaser:
    def test_pointing_moves_the_laser(self, core: SlidePresenterCore):
        for step in range(8):
            state = core.update(point_at((0.7, 0.3)), step * 0.05)
        assert state["laser"] is not None
        assert state["laser"]["x"] == pytest.approx(0.7, abs=0.02)
        assert state["laser"]["y"] == pytest.approx(0.3, abs=0.02)

    def test_laser_is_smoothed_not_teleported(self, core: SlidePresenterCore):
        core.update(point_at((0.2, 0.2)), 0.0)
        state = core.update(point_at((0.9, 0.9)), 0.05)
        assert 0.2 < state["laser"]["x"] < 0.9

    def test_laser_survives_a_brief_dropout(self, core: SlidePresenterCore):
        core.update(point_at((0.5, 0.5)), 0.0)
        state = core.update(make_frame(), 0.1)  # inside laser_hold
        assert state["laser"] is not None

    def test_laser_clears_after_the_hold(self, core: SlidePresenterCore):
        core.update(point_at((0.5, 0.5)), 0.0)
        state = core.update(make_frame(), 2.0)
        assert state["laser"] is None

    def test_no_laser_without_a_pointing_gesture(self, core: SlidePresenterCore):
        state = core.update(open_hand(), 0.0)
        assert state["laser"] is None


class TestCommandsAndState:
    def test_command_map(self, core: SlidePresenterCore):
        assert core.handle_command("next", {})["index"] == 1
        assert core.handle_command("previous", {})["index"] == 0
        assert core.handle_command("goto", {"index": 3})["index"] == 3
        assert core.handle_command("reload", {})["ok"] is True
        assert core.handle_command("bogus", {})["ok"] is False

    def test_deck_command_lists_slides(self, core: SlidePresenterCore):
        body = core.handle_command("deck", {})
        assert body["count"] == 5
        assert body["names"][0] == "slide1.png"

    def test_reload_picks_up_new_files(self, core: SlidePresenterCore, deck_dir: Path):
        cv2.imwrite(str(deck_dir / "slide30.png"), np.zeros((90, 160, 3), dtype=np.uint8))
        core.reload_deck()
        assert len(core.deck) == 6

    def test_reload_clamps_the_index_when_the_deck_shrinks(self, core, deck_dir: Path):
        core.go_to(4)
        for name in ("slide10.png", "slide20.png", "slide3.png"):
            (deck_dir / name).unlink()
        core.reload_deck()
        assert core.index == len(core.deck) - 1

    def test_state_is_json_safe(self, core: SlidePresenterCore):
        import json

        core.update(point_at((0.5, 0.5)), 0.0)
        json.dumps(core.state(0.0))

    def test_reset_returns_to_the_first_slide(self, core: SlidePresenterCore):
        core.go_to(3)
        core.update(point_at((0.5, 0.5)), 0.0)
        core.reset()
        assert core.index == 0
        assert core.laser is None


class TestRendering:
    def test_canvas_matches_configured_size(self, core: SlidePresenterCore):
        canvas = core.render_canvas()
        assert canvas.shape == (core.config.canvas_height, core.config.canvas_width, 3)

    def test_empty_deck_renders_guidance(self, tmp_path: Path):
        core = SlidePresenterCore(SlideConfig(slides_dir=tmp_path))
        canvas = core.render_canvas()
        assert canvas.shape[2] == 3
        assert canvas.max() > 0, "the no-slides message should be visible"

    def test_laser_is_drawn_on_the_canvas(self, core: SlidePresenterCore):
        without = core.render_canvas().copy()
        for step in range(8):
            core.update(point_at((0.5, 0.5)), step * 0.05)
        with_laser = core.render_canvas()
        assert not np.array_equal(without, with_laser)

    def test_fit_into_letterboxes_and_preserves_aspect(self):
        canvas = np.zeros((100, 200, 3), dtype=np.uint8)
        wide = np.full((50, 200, 3), 255, dtype=np.uint8)
        fit_into(canvas, wide)
        assert tuple(canvas[50, 100]) == (255, 255, 255)   # centre filled
        assert tuple(canvas[2, 100]) == (0, 0, 0)          # letterbox bar on top

    def test_fit_into_tolerates_a_degenerate_image(self):
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        fit_into(canvas, np.zeros((0, 0, 3), dtype=np.uint8))
        assert canvas.shape == (10, 10, 3)


class TestDeckGenerator:
    """The bundled six-slide project deck."""

    def test_deck_shape(self):
        from demos.tools.make_sample_slides import CLOSING, CONTENT, DECK, TITLE

        assert len(DECK) == 6
        assert DECK[0].kind == TITLE
        assert DECK[-1].kind == CLOSING
        content = [slide for slide in DECK if slide.kind == CONTENT]
        assert len(content) == 4
        for slide in content:
            assert len(slide.bullets) == 2, f"{slide.title} should have two points"
            assert slide.title and slide.kicker

    def test_copy_is_ascii_only(self):
        """Hershey fonts draw "?" for anything else, so copy must stay ASCII."""
        from demos.tools.make_sample_slides import DECK, ascii_only

        for slide in DECK:
            for text in (slide.kicker, slide.title, slide.subtitle, slide.footer, *slide.bullets):
                assert text == ascii_only(text), f"non-ASCII text: {text!r}"

    def test_ascii_only_maps_common_typography(self):
        from demos.tools.make_sample_slides import ascii_only

        assert ascii_only("a\u00b7b") == "a-b"
        assert ascii_only("dash\u2014dash") == "dash--dash"
        assert ascii_only("quote\u2019s") == "quote's"
        assert "?" not in ascii_only("plain text")

    def test_generate_writes_six_natural_sorted_slides(self, tmp_path: Path):
        from demos.tools.make_sample_slides import generate

        written = generate(tmp_path, width=640, height=360)
        assert [p.name for p in written] == [f"slide{i:02d}.png" for i in range(1, 7)]
        for path in written:
            image = cv2.imread(str(path))
            assert image is not None
            assert image.shape == (360, 640, 3)

    def test_generated_deck_loads_in_order(self, tmp_path: Path):
        from demos.tools.make_sample_slides import generate

        generate(tmp_path, width=320, height=180)
        deck = Deck.load(tmp_path)
        assert len(deck) == 6
        assert deck.name(0) == "slide01.png"
        assert deck.name(5) == "slide06.png"

    def test_slides_are_visually_distinct(self, tmp_path: Path):
        from demos.tools.make_sample_slides import generate

        images = [cv2.imread(str(p)) for p in generate(tmp_path, width=480, height=270)]
        for index, first in enumerate(images):
            for second in images[index + 1 :]:
                assert not np.array_equal(first, second)

    def test_wrap_respects_the_width(self):
        from demos.tools.make_sample_slides import wrap

        long_text = "word " * 40
        lines = wrap(long_text, 0.8, 2, 400)
        assert len(lines) > 1
        for line in lines:
            (width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            assert width <= 400 or " " not in line
