"""Tests for the home hub and the OpenCV launcher menu."""

from __future__ import annotations

import numpy as np
import pytest

from demos import DEMOS
from demos.home import opencv_menu
from demos.home.web import BUILDERS, asset_status, create_home_app
from demos.home.opencv_menu import (
    KEY_DOWN,
    KEY_ENTER,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
    MenuModel,
    build_cards,
    draw_menu,
)


@pytest.fixture()
def client():
    # The scavenger hunt page loads its YOLO model on first request. Swap in a
    # scripted detector so the hub tests stay hermetic: no torch import, no weight
    # download, and they pass with no internet.
    from demos.scavenger_hunt import web as scavenger_web
    from demos.scavenger_hunt.detector import ScriptedDetector

    scavenger_web.set_detector(ScriptedDetector())
    app = create_home_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


class TestHomeHub:
    def test_index_lists_every_demo(self, client):
        body = client.get("/").get_data(as_text=True)
        for demo in DEMOS:
            assert demo.title in body
            assert demo.tagline in body
            assert BUILDERS[demo.slug].URL_PREFIX in body

    def test_index_shows_the_desktop_commands(self, client):
        body = client.get("/").get_data(as_text=True)
        for demo in DEMOS:
            assert demo.desktop_module in body

    def test_index_links_the_diagnostics(self, client):
        body = client.get("/").get_data(as_text=True)
        assert "/echo/" in body
        assert "demos.home.opencv_menu" in body

    def test_index_warns_about_localhost_only(self, client):
        body = client.get("/").get_data(as_text=True)
        assert "no authentication" in body.lower()

    @pytest.mark.parametrize("demo", list(DEMOS), ids=lambda d: d.slug)
    def test_every_demo_page_is_reachable(self, client, demo):
        response = client.get(f"{BUILDERS[demo.slug].URL_PREFIX}/")
        assert response.status_code == 200

    @pytest.mark.parametrize("demo", list(DEMOS), ids=lambda d: d.slug)
    def test_every_demo_health_endpoint_responds(self, client, demo):
        body = client.get(f"{BUILDERS[demo.slug].URL_PREFIX}/health").get_json()
        assert body["ok"] is True

    def test_echo_diagnostic_is_mounted(self, client):
        assert client.get("/echo/").status_code == 200

    def test_shared_static_is_served_once(self, client):
        response = client.get("/shared/static/js/landmark-stream.js")
        assert response.status_code == 200

    def test_demos_json_matches_the_registry(self, client):
        payload = client.get("/demos.json").get_json()
        assert len(payload) == len(DEMOS)
        assert {entry["slug"] for entry in payload} == {demo.slug for demo in DEMOS}
        for entry in payload:
            assert entry["path"].startswith("/")
            assert entry["desktop"].startswith("demos.")
            assert 5000 < entry["standalonePort"] < 5100

    def test_app_health(self, client):
        assert client.get("/health").get_json()["ok"] is True

    def test_websocket_routes_are_registered(self, client):
        """Every landmark-driven demo has a stream; the image lab deliberately does not."""
        rules = {rule.rule for rule in client.application.url_map.iter_rules()}
        # The image lab works on stills and the scavenger hunt runs YOLO server
        # side, so neither streams landmarks; both post images instead.
        posts_images = {
            "image-lab": "/pipeline",
            "scavenger-hunt": "/frame",
            "sam-labeler": "/run",
        }
        for demo in DEMOS:
            prefix = BUILDERS[demo.slug].URL_PREFIX
            if demo.slug in posts_images:
                assert f"{prefix}/ws" not in rules
                assert f"{prefix}{posts_images[demo.slug]}" in rules
            else:
                assert f"{prefix}/ws" in rules

    def test_asset_status_reports_generated_assets(self):
        status = asset_status()
        assert set(status) == {"slide-presenter", "pngtuber"}
        for entry in status.values():
            assert "fix" in entry
            assert isinstance(entry["ok"], bool)

    def test_missing_assets_degrade_gracefully(self, tmp_path, client):
        """A demo with no assets still serves its page, with a hint."""
        from demos.slide_presenter import web as slides_web

        original = slides_web._CONFIG.slides_dir
        slides_web._CONFIG.slides_dir = tmp_path / "empty"
        try:
            body = client.get("/").get_data(as_text=True)
            assert "make_sample_slides" in body
            assert client.get("/slides/").status_code == 200
        finally:
            slides_web._CONFIG.slides_dir = original


class TestCardLayout:
    def test_one_card_per_demo(self):
        cards = build_cards()
        assert len(cards) == len(DEMOS)

    def test_cards_stay_inside_the_frame(self):
        for card in build_cards():
            x1, y1, x2, y2 = card.rect
            assert 0.0 <= x1 < x2 <= 1.0
            assert 0.0 <= y1 < y2 <= 1.0

    def test_cards_do_not_overlap(self):
        cards = build_cards()
        for i, first in enumerate(cards):
            for second in cards[i + 1 :]:
                ax1, ay1, ax2, ay2 = first.rect
                bx1, by1, bx2, by2 = second.rect
                separated = ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1
                assert separated, f"{first.demo.slug} overlaps {second.demo.slug}"

    def test_contains_and_center(self):
        card = build_cards()[0]
        assert card.contains(card.center)
        assert not card.contains((0.99, 0.99))

    def test_single_column_layout(self):
        cards = build_cards(columns=1)
        assert len(cards) == len(DEMOS)
        xs = {card.rect[0] for card in cards}
        assert len(xs) == 1


class TestMenuKeyboard:
    def test_starts_on_the_first_card(self):
        assert MenuModel().selected == 0

    def test_arrow_keys_walk_the_grid(self):
        model = MenuModel()
        model.handle_key(KEY_RIGHT)
        assert model.selected == 1
        model.handle_key(KEY_DOWN)
        assert model.selected == 3
        model.handle_key(KEY_LEFT)
        assert model.selected == 2
        model.handle_key(KEY_UP)
        assert model.selected == 0

    def test_selection_clamps_at_the_edges(self):
        model = MenuModel()
        for _ in range(6):
            model.handle_key(KEY_LEFT)
            model.handle_key(KEY_UP)
        assert model.selected == 0
        for _ in range(6):
            model.handle_key(KEY_RIGHT)
            model.handle_key(KEY_DOWN)
        assert model.selected == len(model.cards) - 1

    def test_every_card_is_reachable_with_the_arrow_keys(self):
        """A ragged last row must not strand a card, whatever the demo count."""
        template = MenuModel()
        rows = (len(template.cards) + template.columns - 1) // template.columns
        reached = set()
        for row in range(rows):
            for column in range(template.columns):
                model = MenuModel()
                for _ in range(row):
                    model.handle_key(KEY_DOWN)
                for _ in range(column):
                    model.handle_key(KEY_RIGHT)
                reached.add(model.selected)
        assert reached == set(range(len(template.cards)))

    def test_vim_keys_work_too(self):
        model = MenuModel()
        model.handle_key(ord("l"))
        assert model.selected == 1
        model.handle_key(ord("j"))
        model.handle_key(ord("h"))
        model.handle_key(ord("k"))
        assert model.selected == 0

    def test_enter_launches_the_selection(self):
        model = MenuModel()
        model.handle_key(KEY_RIGHT)
        demo = model.handle_key(KEY_ENTER)
        assert demo is not None
        assert demo.slug == model.cards[1].demo.slug

    def test_number_keys_launch_directly(self):
        model = MenuModel()
        demo = model.handle_key(ord("3"))
        assert demo is model.cards[2].demo
        assert model.selected == 2

    def test_out_of_range_number_is_ignored(self):
        model = MenuModel()
        assert model.handle_key(ord("9")) is None

    def test_navigation_keys_do_not_launch(self):
        model = MenuModel()
        for key in (KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN):
            assert model.handle_key(key) is None


class TestMenuHandHover:
    def test_dwelling_over_a_card_launches_it(self):
        model = MenuModel(dwell_seconds=0.5)
        target = model.cards[2]
        launched = None
        now = 0.0
        while now <= 2.0 and launched is None:
            launched = model.update_pointer(target.center, now)
            now += 0.1
        assert launched is model.cards[2].demo

    def test_sweeping_across_cards_launches_nothing(self):
        model = MenuModel(dwell_seconds=0.9)
        for index, card in enumerate(model.cards):
            assert model.update_pointer(card.center, index * 0.1) is None

    def test_hover_moves_the_selection_highlight(self):
        model = MenuModel(dwell_seconds=5.0)
        for step in range(8):
            model.update_pointer(model.cards[1].center, step * 0.05)
        assert model.hovered == 1
        assert model.selected == 1

    def test_pointer_between_cards_hovers_nothing(self):
        model = MenuModel()
        for step in range(6):
            model.update_pointer((0.5, 0.02), step * 0.1)
        assert model.hovered is None

    def test_losing_the_hand_clears_the_dwell(self):
        model = MenuModel(dwell_seconds=0.5)
        card = model.cards[0]
        model.update_pointer(card.center, 0.0)
        model.update_pointer(card.center, 0.2)
        assert model.update_pointer(None, 0.3) is None
        assert model.hovered is None
        # The dwell restarts rather than completing from the earlier hover.
        assert model.update_pointer(card.center, 0.4) is None

    def test_dwell_progress_grows(self):
        model = MenuModel(dwell_seconds=1.0)
        card = model.cards[0]
        model.update_pointer(card.center, 0.0)
        model.update_pointer(card.center, 0.5)
        assert 0.0 < model.dwell_progress(0.5) <= 1.0

    def test_reset_selection_clears_state(self):
        model = MenuModel(dwell_seconds=0.2)
        card = model.cards[0]
        for step in range(6):
            model.update_pointer(card.center, step * 0.1)
        model.reset_selection()
        assert model.launched is None
        assert model.hovered is None
        assert model.cursor is None

    def test_choose_ignores_a_bad_index(self):
        model = MenuModel()
        assert model.choose(None) is None
        assert model.choose(99) is None


class TestMenuRendering:
    @pytest.mark.parametrize("camera", [True, False])
    def test_draws_without_error(self, camera):
        frame = np.full((480, 854, 3), 60, dtype=np.uint8)
        out = draw_menu(frame.copy(), MenuModel(), now=1.0, camera=camera)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_selection_changes_the_render(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        first = draw_menu(frame.copy(), MenuModel(), now=0.0, camera=False)
        model = MenuModel()
        model.handle_key(KEY_RIGHT)
        second = draw_menu(frame.copy(), model, now=0.0, camera=False)
        assert not np.array_equal(first, second)

    def test_animation_advances_over_time(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        model = MenuModel()
        early = draw_menu(frame.copy(), model, now=0.0, camera=False)
        later = draw_menu(frame.copy(), model, now=0.7, camera=False)
        assert not np.array_equal(early, later)

    def test_hover_ring_is_drawn(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        plain = draw_menu(frame.copy(), MenuModel(), now=0.0, camera=False)
        model = MenuModel(dwell_seconds=5.0)
        for step in range(6):
            model.update_pointer(model.cards[0].center, step * 0.1)
        hovered = draw_menu(frame.copy(), model, now=0.6, camera=False)
        assert not np.array_equal(plain, hovered)


class TestLaunching:
    def test_launch_runs_the_desktop_module_with_this_interpreter(self, monkeypatch):
        captured = {}

        def fake_run(command, cwd, check):
            captured["command"] = command
            captured["cwd"] = cwd

            class Result:
                returncode = 0

            return Result()

        monkeypatch.setattr(opencv_menu.subprocess, "run", fake_run)
        code = opencv_menu.launch(DEMOS[0])
        assert code == 0
        assert captured["command"][0].endswith("python") or "python" in captured["command"][0]
        assert captured["command"][1:] == ["-m", DEMOS[0].desktop_module]
        assert (captured["cwd"] / "demos").is_dir(), "must run from the repo root"

    def test_launch_propagates_the_exit_code(self, monkeypatch):
        class Result:
            returncode = 3

        monkeypatch.setattr(opencv_menu.subprocess, "run", lambda *a, **k: Result())
        assert opencv_menu.launch(DEMOS[1]) == 3
