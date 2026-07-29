"""Tests for the scavenger hunt: game rules, detector backends, web routes.

No model weights are involved: detections are scripted, which is the point of
keeping the game logic pure.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from demos.common.webapp import create_app
from demos.scavenger_hunt import web
from demos.scavenger_hunt.config import COCO_CLASSES, DESK_ITEMS, HuntConfig
from demos.scavenger_hunt.core import (
    FINISHED,
    FOUND,
    IDLE,
    MISSED,
    PLAYING,
    ScavengerHuntCore,
    draw_overlay,
)
from demos.scavenger_hunt.detector import (
    Detection,
    NullDetector,
    OnnxDetector,
    ScriptedDetector,
    load_detector,
)


def seen(label: str, confidence: float = 0.9) -> Detection:
    return Detection(label=label, confidence=confidence, box=(0.3, 0.3, 0.7, 0.8))


@pytest.fixture()
def config() -> HuntConfig:
    return HuntConfig(seed=42, rounds=3, round_seconds=10.0, hold_seconds=0.4, celebrate_seconds=1.0)


@pytest.fixture()
def core(config: HuntConfig) -> ScavengerHuntCore:
    return ScavengerHuntCore(config)


class Clock:
    """Drives the core with an advancing timestamp."""

    def __init__(self, core: ScavengerHuntCore, step: float = 0.2):
        self.core = core
        self.step = step
        self.now = 0.0

    def feed(self, detections, times: int = 1) -> dict:
        state: dict = {}
        for _ in range(times):
            state = self.core.update(list(detections), self.now)
            self.now += self.step
        return state

    def nothing(self, times: int = 1) -> dict:
        return self.feed([], times)

    def show_target(self, times: int = 4) -> dict:
        return self.feed([seen(self.core.target or "cup")], times)

    def wait(self, seconds: float) -> None:
        self.now += seconds


class TestConfig:
    def test_coco_has_eighty_classes(self):
        assert len(COCO_CLASSES) == 80
        assert COCO_CLASSES[0] == "person"
        assert COCO_CLASSES[67] == "cell phone"

    def test_item_pool_is_a_subset_of_coco(self):
        assert set(DESK_ITEMS) <= set(COCO_CLASSES)

    def test_pool_contains_the_promised_desk_items(self):
        for item in ("cell phone", "cup", "scissors", "potted plant"):
            assert item in DESK_ITEMS

    def test_hints_reference_real_items(self):
        config = HuntConfig()
        assert config.hint_for("remote")
        assert config.hint_for("cup") == ""


class TestGameFlow:
    def test_starts_idle(self, core):
        assert core.phase == IDLE
        assert core.state(0.0)["prompt"] == "PRESS START"

    def test_start_picks_a_target_from_the_pool(self, core):
        core.start(0.0)
        assert core.phase == PLAYING
        assert core.target in DESK_ITEMS
        assert core.state(0.0)["prompt"].startswith("BRING ME ")

    def test_showing_the_target_long_enough_scores(self, core):
        core.start(0.0)
        clock = Clock(core)
        clock.show_target(times=4)
        assert core.phase == FOUND
        assert core.score > 0
        assert core.results[-1].found is True

    def test_a_single_frame_is_not_enough(self, core):
        core.start(0.0)
        clock = Clock(core)
        clock.feed([seen(core.target)], times=1)
        assert core.phase == PLAYING, "one lucky frame must not win the round"

    def test_losing_sight_restarts_the_hold(self, core):
        core.start(0.0)
        clock = Clock(core)
        clock.feed([seen(core.target)], times=1)
        clock.nothing(1)
        clock.feed([seen(core.target)], times=1)
        assert core.phase == PLAYING
        assert core.state(clock.now)["holdProgress"] < 1.0

    def test_wrong_object_does_not_score(self, core):
        core.start(0.0)
        target = core.target
        other = next(item for item in DESK_ITEMS if item != target)
        clock = Clock(core)
        clock.feed([seen(other)], times=10)
        assert core.phase == PLAYING
        assert core.score == 0

    def test_low_confidence_does_not_score(self, core):
        core.start(0.0)
        clock = Clock(core)
        clock.feed([seen(core.target, confidence=0.05)], times=10)
        assert core.score == 0

    def test_timer_expiry_is_a_miss(self, core):
        core.start(0.0)
        clock = Clock(core)
        clock.wait(core.config.round_seconds + 0.1)
        clock.nothing(1)
        assert core.phase == MISSED
        assert core.results[-1].found is False
        assert core.results[-1].points == 0

    def test_rounds_advance_after_the_celebration_pause(self, core):
        core.start(0.0)
        clock = Clock(core)
        first = core.target
        clock.show_target(times=4)
        assert core.phase == FOUND
        clock.wait(core.config.celebrate_seconds + 0.1)
        clock.nothing(1)
        assert core.phase == PLAYING
        assert core.round_index == 1
        assert core.target is not None
        assert core.target != first or len(DESK_ITEMS) == 1

    def test_game_finishes_after_the_configured_rounds(self, core):
        core.start(0.0)
        clock = Clock(core)
        for _ in range(core.config.rounds):
            clock.show_target(times=4)
            clock.wait(core.config.celebrate_seconds + 0.1)
            clock.nothing(1)
        assert core.phase == FINISHED
        assert len(core.results) == core.config.rounds
        assert core.state(clock.now)["prompt"] == "GAME OVER"

    def test_skip_counts_as_a_miss_and_moves_on(self, core):
        core.start(0.0)
        clock = Clock(core)
        core.skip(clock.now)
        assert core.phase == MISSED
        assert core.results[-1].found is False
        assert "skipped" in core.last_event

    def test_submit_photo_scores_without_a_hold(self, core):
        core.start(0.0)
        state = core.submit_photo([seen(core.target)], 0.5)
        assert state["phase"] == FOUND
        assert "photo" in core.last_event

    def test_submit_photo_ignores_the_wrong_item(self, core):
        core.start(0.0)
        other = next(item for item in DESK_ITEMS if item != core.target)
        assert core.submit_photo([seen(other)], 0.5)["phase"] == PLAYING

    def test_submit_photo_outside_play_does_nothing(self, core):
        assert core.submit_photo([seen("cup")], 0.0)["phase"] == IDLE

    def test_skip_outside_play_does_nothing(self, core):
        core.skip(0.0)
        assert core.results == []

    def test_reset_returns_to_idle(self, core):
        core.start(0.0)
        Clock(core).show_target(times=4)
        core.reset()
        assert core.phase == IDLE
        assert core.score == 0
        assert core.results == []

    def test_targets_avoid_immediate_repeats(self, config):
        core = ScavengerHuntCore(config)
        core.config.rounds = 12
        core.start(0.0)
        clock = Clock(core)
        picked = [core.target]
        for _ in range(8):
            core.skip(clock.now)
            clock.wait(core.config.celebrate_seconds + 0.1)
            clock.nothing(1)
            picked.append(core.target)
        for index in range(1, len(picked)):
            assert picked[index] != picked[index - 1], picked


class TestScoring:
    def test_finding_fast_scores_more_than_finding_slow(self, config):
        def run(delay: float) -> int:
            core = ScavengerHuntCore(config)
            core.start(0.0)
            clock = Clock(core)
            clock.wait(delay)
            clock.show_target(times=4)
            return core.score

        assert run(0.0) > run(config.round_seconds * 0.8)

    def test_streak_adds_a_bonus(self, config):
        core = ScavengerHuntCore(config)
        core.config.rounds = 4
        core.start(0.0)
        clock = Clock(core)
        gains = []
        before = 0
        for _ in range(3):
            clock.show_target(times=4)
            gains.append(core.score - before)
            before = core.score
            clock.wait(core.config.celebrate_seconds + 0.1)
            clock.nothing(1)
        assert gains[1] > gains[0], "the second consecutive find should be worth more"

    def test_streak_resets_on_a_miss(self, config):
        core = ScavengerHuntCore(config)
        core.config.rounds = 5
        core.start(0.0)
        clock = Clock(core)
        clock.show_target(times=4)
        assert core.streak == 1
        clock.wait(core.config.celebrate_seconds + 0.1)
        clock.nothing(1)
        core.skip(clock.now)
        assert core.streak == 0
        assert core.best_streak == 1

    def test_missed_rounds_score_nothing(self, core):
        core.start(0.0)
        clock = Clock(core)
        clock.wait(core.config.round_seconds + 0.1)
        clock.nothing(1)
        assert core.score == 0

    def test_found_count_tracks_successes(self, config):
        core = ScavengerHuntCore(config)
        core.start(0.0)
        clock = Clock(core)
        clock.show_target(times=4)
        clock.wait(core.config.celebrate_seconds + 0.1)
        clock.nothing(1)
        core.skip(clock.now)
        assert core.found_count == 1
        assert len(core.results) == 2


class TestState:
    def test_state_is_json_safe(self, core):
        import json

        core.start(0.0)
        json.dumps(Clock(core).show_target(times=2))

    def test_time_left_counts_down(self, core):
        core.start(0.0)
        assert core.time_left(0.0) == pytest.approx(core.config.round_seconds)
        assert core.time_left(4.0) == pytest.approx(core.config.round_seconds - 4.0)
        assert core.time_left(999.0) == 0.0

    def test_time_left_is_zero_outside_play(self, core):
        assert core.time_left(0.0) == 0.0

    def test_state_reports_the_matching_detection(self, core):
        core.start(0.0)
        state = core.update([seen(core.target)], 0.0)
        assert state["match"]["label"] == core.target
        assert state["match"]["box"] == [0.3, 0.3, 0.7, 0.8]

    def test_state_lists_every_detection(self, core):
        core.start(0.0)
        state = core.update([seen("person"), seen(core.target)], 0.0)
        assert len(state["detections"]) == 2

    def test_command_map(self, core):
        assert core.handle_command("start", {"now": 0.0})["phase"] == PLAYING
        assert core.handle_command("skip", {"now": 1.0})["phase"] == MISSED
        assert core.handle_command("reset", {})["phase"] == IDLE
        assert core.handle_command("bogus", {})["ok"] is False


class TestDetectorBackends:
    def test_no_model_degrades_to_null(self):
        detector = load_detector(HuntConfig(model=""))
        assert isinstance(detector, NullDetector)
        assert detector.ready is False
        assert detector.detect(np.zeros((8, 8, 3), np.uint8)) == []
        assert "No model configured" in detector.describe()

    def test_missing_file_degrades_to_null(self, tmp_path: Path):
        detector = load_detector(HuntConfig(model=str(tmp_path / "absent.onnx")))
        assert detector.ready is False
        assert "not found" in detector.describe()

    def test_unloadable_checkpoint_degrades_to_null(self, tmp_path: Path):
        path = tmp_path / "model.pt"
        path.write_bytes(b"not a checkpoint")
        detector = load_detector(HuntConfig(model=str(path)))
        assert detector.ready is False
        assert "Could not load" in detector.describe()

    def test_corrupt_onnx_degrades_to_null_instead_of_raising(self, tmp_path: Path):
        path = tmp_path / "broken.onnx"
        path.write_bytes(b"not really an onnx file")
        detector = load_detector(HuntConfig(model=str(path)))
        assert detector.ready is False
        assert "Could not load" in detector.describe()

    def test_scripted_detector_returns_queued_then_fixed(self):
        detector = ScriptedDetector([seen("cup")])
        detector.push([seen("book")])
        frame = np.zeros((8, 8, 3), np.uint8)
        assert detector.detect(frame)[0].label == "book"
        assert detector.detect(frame)[0].label == "cup"

    def test_detection_serialises(self):
        payload = seen("cup", 0.81234).to_json()
        assert payload == {"label": "cup", "confidence": 0.812, "box": [0.3, 0.3, 0.7, 0.8]}


class TestOnnxDecoding:
    """The decoder sniffs output shape, so both YOLO layouts are covered."""

    def _detector(self) -> OnnxDetector:
        detector = OnnxDetector.__new__(OnnxDetector)  # skip loading a real model
        detector.config = HuntConfig(confidence=0.3, input_size=640)
        detector.ready = True
        return detector

    def test_end_to_end_rows_are_decoded(self):
        detector = self._detector()
        # [x1, y1, x2, y2, score, class]; 67 is "cell phone".
        rows = np.array([[64, 128, 320, 448, 0.9, 67], [0, 0, 10, 10, 0.05, 41]], dtype=np.float32)
        detections = detector._decode_end_to_end(rows)
        assert len(detections) == 1
        assert detections[0].label == "cell phone"
        assert detections[0].box == pytest.approx((0.1, 0.2, 0.5, 0.7))

    def test_end_to_end_rejects_unknown_class_ids(self):
        detector = self._detector()
        rows = np.array([[0, 0, 100, 100, 0.9, 999]], dtype=np.float32)
        assert detector._decode_end_to_end(rows) == []

    def test_yolo_rows_are_decoded_with_nms(self):
        detector = self._detector()
        rows = np.zeros((3, 84), dtype=np.float32)
        # Two heavily overlapping "cup" boxes plus one distinct "book".
        rows[0, :4] = (320, 320, 200, 200)
        rows[0, 4 + 41] = 0.9
        rows[1, :4] = (322, 318, 204, 196)
        rows[1, 4 + 41] = 0.8
        rows[2, :4] = (100, 100, 60, 60)
        rows[2, 4 + 73] = 0.7
        labels = sorted(d.label for d in detector._decode_yolo(rows))
        assert labels == ["book", "cup"], "NMS should collapse the duplicate cup"

    def test_yolo_rows_below_threshold_are_dropped(self):
        detector = self._detector()
        rows = np.zeros((2, 84), dtype=np.float32)
        rows[:, :4] = (320, 320, 100, 100)
        rows[:, 4 + 41] = 0.1
        assert detector._decode_yolo(rows) == []

    def test_degenerate_output_is_ignored(self):
        detector = self._detector()
        assert detector._decode_yolo(np.zeros((2, 4), dtype=np.float32)) == []


class TestRendering:
    def test_canvas_shows_the_score(self, core):
        canvas = core.render_canvas()
        assert canvas.shape == (core.config.canvas_height, core.config.canvas_width, 3)
        blank = int((canvas > 100).sum())
        core.start(0.0)
        Clock(core).show_target(times=4)
        assert int((core.render_canvas() > 100).sum()) != blank

    def test_overlay_draws_boxes_and_prompt(self, core):
        frame = np.full((360, 640, 3), 50, dtype=np.uint8)
        core.start(0.0)
        state = core.update([seen(core.target), seen("person")], 0.1)
        out = draw_overlay(frame.copy(), core, state)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_overlay_works_in_every_phase(self, core):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        for state in (
            core.state(0.0),
            (core.start(0.0), core.state(0.0))[1],
        ):
            assert draw_overlay(frame.copy(), core, state) is not None
        core.skip(1.0)
        assert draw_overlay(frame.copy(), core, core.state(1.0)) is not None


@pytest.fixture()
def client():
    web.set_detector(ScriptedDetector())
    web._CONFIG.rounds = 3
    web._CONFIG.round_seconds = 30.0
    # The server clock is real time, and the test client posts frames with no delay
    # between them, so a hold window would never elapse. The hold itself is covered
    # against a synthetic clock in TestGameFlow.
    web._CONFIG.hold_seconds = 0.0
    blueprint, sock, _ = web.build()
    assert sock is None, "YOLO runs server side, so there is no landmark stream"
    app = create_app([], name="hunt_test")
    app.register_blueprint(blueprint)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _jpeg(colour=(60, 90, 120)) -> bytes:
    image = np.full((120, 160, 3), colour, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


class TestWebApp:
    def test_page_renders_the_prompt_and_item_list(self, client):
        body = client.get("/scavenger-hunt/").get_data(as_text=True)
        assert "Bring me" in body
        assert "cell phone" in body
        assert "Start hunt" in body

    def test_page_explains_a_failed_model_load(self, client):
        web.set_detector(NullDetector("could not reach the internet"))
        body = client.get("/scavenger-hunt/").get_data(as_text=True)
        assert "Detection model not loaded" in body
        assert "could not reach the internet" in body
        assert "requirements.txt" in body
        web.set_detector(ScriptedDetector())

    def test_health_reports_the_backend(self, client):
        body = client.get("/scavenger-hunt/health").get_json()
        assert body["ok"] is True
        assert body["model"]["backend"] == "scripted"

    def test_start_then_frames_win_a_round(self, client):
        detector = ScriptedDetector()
        web.set_detector(detector)
        started = client.post("/scavenger-hunt/command?sid=g1", json={"command": "start"}).get_json()
        target = started["target"]
        assert started["phase"] == PLAYING

        body = {}
        for _ in range(8):
            detector.push([seen(target)])
            body = client.post(
                "/scavenger-hunt/frame?sid=g1", data={"frame": (io.BytesIO(_jpeg()), "f.jpg")},
                content_type="multipart/form-data",
            ).get_json()
            if body["phase"] == FOUND:
                break
        assert body["phase"] == FOUND
        assert body["score"] > 0

    def test_frames_with_nothing_keep_playing(self, client):
        client.post("/scavenger-hunt/command?sid=g2", json={"command": "start"})
        body = client.post(
            "/scavenger-hunt/frame?sid=g2", data={"frame": (io.BytesIO(_jpeg()), "f.jpg")},
            content_type="multipart/form-data",
        ).get_json()
        assert body["phase"] == PLAYING
        assert body["score"] == 0

    def test_undecodable_frame_is_reported_without_a_500(self, client):
        client.post("/scavenger-hunt/command?sid=g3", json={"command": "start"})
        response = client.post(
            "/scavenger-hunt/frame?sid=g3", data={"frame": (io.BytesIO(b"junk"), "f.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        assert response.get_json()["ok"] is False

    def test_upload_of_a_photo_can_win_a_round(self, client):
        detector = ScriptedDetector()
        web.set_detector(detector)
        started = client.post("/scavenger-hunt/command?sid=g4", json={"command": "start"}).get_json()
        detector.fixed = [seen(started["target"])]
        body = client.post(
            "/scavenger-hunt/upload?sid=g4", data={"image": (io.BytesIO(_jpeg()), "photo.png")},
            content_type="multipart/form-data",
        ).get_json()
        assert body["ok"] is True
        assert body["phase"] == FOUND, "a still photo cannot be 'held', so one hit is enough"

    def test_upload_ignores_a_photo_without_the_item(self, client):
        detector = ScriptedDetector([seen("person")])
        web.set_detector(detector)
        client.post("/scavenger-hunt/command?sid=g9", json={"command": "start"})
        body = client.post(
            "/scavenger-hunt/upload?sid=g9", data={"image": (io.BytesIO(_jpeg()), "photo.png")},
            content_type="multipart/form-data",
        ).get_json()
        assert body["phase"] == PLAYING
        assert body["match"] is None

    def test_upload_without_a_file_is_a_400(self, client):
        assert client.post("/scavenger-hunt/upload?sid=g5").status_code == 400

    def test_upload_of_a_non_image_is_a_400(self, client):
        response = client.post(
            "/scavenger-hunt/upload?sid=g6", data={"image": (io.BytesIO(b"junk"), "x.png")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_skip_and_reset_commands(self, client):
        client.post("/scavenger-hunt/command?sid=g7", json={"command": "start"})
        assert client.post("/scavenger-hunt/command?sid=g7", json={"command": "skip"}).get_json()["phase"] == MISSED
        assert client.post("/scavenger-hunt/reset?sid=g7").get_json()["phase"] == IDLE

    def test_snapshot_is_a_scoreboard_png(self, client):
        client.post("/scavenger-hunt/command?sid=g8", json={"command": "start"})
        response = client.get("/scavenger-hunt/snapshot?sid=g8")
        assert response.data[:8] == b"\x89PNG\r\n\x1a\n"
        assert "attachment" in response.headers["Content-Disposition"]

    def test_players_have_independent_games(self, client):
        client.post("/scavenger-hunt/command?sid=p1", json={"command": "start"})
        assert client.get("/scavenger-hunt/state?sid=p1").get_json()["phase"] == PLAYING
        assert client.get("/scavenger-hunt/state?sid=p2").get_json()["phase"] == IDLE


class TestModelDefaults:
    """The demo needs no supplied weights: ultralytics fetches them on demand.

    These stay offline — nothing here loads or downloads a real model.
    """

    def test_default_model_is_the_coco_nano_checkpoint(self):
        from demos.scavenger_hunt.config import DEFAULT_MODEL

        assert DEFAULT_MODEL == "yolo26n.pt"
        assert HuntConfig().model == DEFAULT_MODEL

    def test_environment_variable_overrides_the_default(self, monkeypatch):
        monkeypatch.setenv("SCAVENGER_MODEL", "yolo11n.pt")
        assert HuntConfig().model == "yolo11n.pt"

    def test_weights_are_kept_out_of_the_repo_root(self):
        """Ultralytics downloads bare names into the cwd; we redirect to models/."""
        from demos.scavenger_hunt.config import MODEL_DIR
        from demos.scavenger_hunt.detector import UltralyticsDetector

        resolved = Path(UltralyticsDetector._resolve("yolo26n.pt"))
        assert resolved.parent == MODEL_DIR
        assert MODEL_DIR.name == "models"

    def test_an_existing_path_is_used_as_given(self, tmp_path: Path):
        from demos.scavenger_hunt.detector import UltralyticsDetector

        weights = tmp_path / "mine.pt"
        weights.write_bytes(b"x")
        assert UltralyticsDetector._resolve(str(weights)) == str(weights)

    def test_onnx_is_looked_up_in_models_too(self, tmp_path: Path):
        detector = load_detector(HuntConfig(model="definitely-absent.onnx"))
        assert detector.ready is False
        assert "not found" in detector.describe()


class TestSettings:
    """The in-game settings panel: timer, confidence, rounds, hold window."""

    def test_defaults_are_reported(self, core):
        settings = core.settings()
        assert settings["roundSeconds"] == core.config.round_seconds
        assert settings["confidence"] == pytest.approx(core.config.confidence)
        assert settings["rounds"] == core.config.rounds

    def test_values_are_clamped_to_the_published_bounds(self, core):
        from demos.scavenger_hunt.config import SETTING_BOUNDS

        applied = core.update_settings(
            {"roundSeconds": 9999, "confidence": -3, "rounds": 999, "holdSeconds": 99}
        )
        assert applied["roundSeconds"] == SETTING_BOUNDS["roundSeconds"][1]
        assert applied["confidence"] == SETTING_BOUNDS["confidence"][0]
        assert applied["rounds"] == int(SETTING_BOUNDS["rounds"][1])
        assert applied["holdSeconds"] == SETTING_BOUNDS["holdSeconds"][1]

    def test_unknown_and_unparseable_keys_are_ignored(self, core):
        applied = core.update_settings({"bogus": 5, "confidence": "not a number"})
        assert applied == {}

    def test_state_publishes_bounds_for_the_ui(self, core):
        bounds = core.state(0.0)["bounds"]
        assert set(bounds) == {"roundSeconds", "confidence", "rounds", "holdSeconds"}
        for low, high, step in bounds.values():
            assert low < high and step > 0

    def test_timer_applies_from_the_next_round(self, config):
        core = ScavengerHuntCore(config)  # 10 s rounds
        core.start(0.0)
        core.update_settings({"roundSeconds": 60})
        assert core.state(0.0)["roundSeconds"] == 10.0, "current round keeps its clock"
        assert core.time_left(5.0) == pytest.approx(5.0)

        clock = Clock(core)
        clock.show_target(times=4)
        clock.wait(config.celebrate_seconds + 0.1)
        clock.nothing(1)
        assert core.state(clock.now)["roundSeconds"] == 60.0, "next round uses the new clock"

    def test_shortening_the_timer_cannot_retroactively_lose_a_round(self, config):
        core = ScavengerHuntCore(config)
        core.start(0.0)
        clock = Clock(core)
        clock.wait(8.0)                      # 8 s into a 10 s round
        core.update_settings({"roundSeconds": 5})
        clock.nothing(1)
        assert core.phase == PLAYING, "the round in progress keeps its original clock"

    def test_confidence_applies_immediately(self, config):
        core = ScavengerHuntCore(config)
        core.start(0.0)
        clock = Clock(core)
        weak = [seen(core.target, confidence=0.2)]
        clock.feed(weak, times=6)
        assert core.phase == PLAYING, "0.2 is below the default threshold"

        core.update_settings({"confidence": 0.1})
        clock.feed(weak, times=6)
        assert core.phase == FOUND, "lowering the threshold should let it through"

    def test_raising_confidence_stops_a_marginal_detection(self, config):
        core = ScavengerHuntCore(config)
        core.start(0.0)
        core.update_settings({"confidence": 0.9})
        clock = Clock(core)
        clock.feed([seen(core.target, confidence=0.5)], times=8)
        assert core.phase == PLAYING

    def test_a_confidence_change_discards_stale_hold_progress(self, config):
        core = ScavengerHuntCore(config)
        core.start(0.0)
        clock = Clock(core)
        clock.feed([seen(core.target)], times=1)
        assert core.state(clock.now)["holdProgress"] >= 0.0
        core.update_settings({"confidence": 0.5})
        assert core._holding_since is None

    def test_hold_of_zero_scores_on_the_first_sighting(self, config):
        core = ScavengerHuntCore(config)
        core.start(0.0)
        core.update_settings({"holdSeconds": 0.0})
        core.update([seen(core.target)], 0.1)
        assert core.phase == FOUND

    def test_rounds_setting_changes_game_length(self, config):
        core = ScavengerHuntCore(config)
        core.update_settings({"rounds": 1})
        core.start(0.0)
        clock = Clock(core)
        clock.show_target(times=4)
        clock.wait(config.celebrate_seconds + 0.1)
        clock.nothing(1)
        assert core.phase == FINISHED

    def test_settings_survive_a_reset(self, core):
        core.update_settings({"roundSeconds": 45, "confidence": 0.75})
        core.reset()
        assert core.settings()["roundSeconds"] == 45.0
        assert core.settings()["confidence"] == 0.75

    def test_settings_command_reports_what_it_applied(self, core):
        result = core.handle_command("settings", {"confidence": 0.5, "now": 0.0})
        assert result["ok"] is True
        assert result["applied"]["confidence"] == 0.5
        assert result["settings"]["confidence"] == 0.5


class TestSettingsRoutes:
    def test_settings_endpoint_applies_and_clamps(self, client):
        body = client.post(
            "/scavenger-hunt/settings?sid=s1", json={"roundSeconds": 15, "confidence": 5}
        ).get_json()
        assert body["ok"] is True
        assert body["applied"]["roundSeconds"] == 15.0
        assert body["settings"]["confidence"] == 0.95, "clamped to the maximum"

    def test_settings_are_per_player(self, client):
        client.post("/scavenger-hunt/settings?sid=p1", json={"roundSeconds": 15})
        assert client.get("/scavenger-hunt/state?sid=p1").get_json()["settings"]["roundSeconds"] == 15.0
        other = client.get("/scavenger-hunt/state?sid=p2").get_json()["settings"]["roundSeconds"]
        assert other == web._CONFIG.round_seconds, "one player must not change another's game"

    def test_page_exposes_the_bounds_to_the_sliders(self, client):
        body = client.get("/scavenger-hunt/").get_data(as_text=True)
        assert "Reveal settings" in body
        assert 'id="set-conf"' in body
        assert 'id="set-time"' in body

    def test_empty_payload_is_harmless(self, client):
        before = client.get("/scavenger-hunt/state?sid=s2").get_json()["settings"]
        after = client.post("/scavenger-hunt/settings?sid=s2", json={}).get_json()["settings"]
        assert before == after

    def test_a_new_round_picks_up_the_new_timer(self, client):
        client.post("/scavenger-hunt/command?sid=s3", json={"command": "start"})
        client.post("/scavenger-hunt/settings?sid=s3", json={"roundSeconds": 15})
        client.post("/scavenger-hunt/command?sid=s3", json={"command": "skip"})
        # After the celebrate pause the next round starts; force it by restarting.
        body = client.post("/scavenger-hunt/command?sid=s3", json={"command": "start"}).get_json()
        assert body["roundSeconds"] == 15.0
