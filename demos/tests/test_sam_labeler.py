"""Tests for the SAM labeler.

No model, no Hugging Face token, no network: the stub backend produces synthetic
masks, which is enough to exercise every effect, the label styling and all routes.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from demos.common.webapp import create_app
from demos.sam_labeler import web
from demos.sam_labeler.backend import (
    Instance,
    StubBackend,
    UnavailableBackend,
    load_backend,
)
from demos.sam_labeler.config import DETECTION, EFFECTS, PALETTE, SEGMENTATION, SamLabelerConfig
from demos.sam_labeler.core import (
    LabelStyle,
    SamLabelerCore,
    apply_blur,
    apply_cutout,
    apply_fill,
    apply_outline,
    apply_pixelate,
    apply_spotlight,
)


@pytest.fixture()
def scene() -> np.ndarray:
    """A textured image, so blur and pixelate have something to destroy."""
    rng = np.random.default_rng(11)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    for y in range(200):
        image[y, :] = (40 + y // 4, 70 + y // 5, 110 + y // 6)
    image += (rng.random(image.shape) * 40).astype(np.uint8)
    for x in range(0, 300, 20):
        cv2.line(image, (x, 0), (x - 40, 200), (230, 230, 230), 1)
    return image


@pytest.fixture()
def core(scene: np.ndarray) -> SamLabelerCore:
    instance = SamLabelerCore(SamLabelerConfig(max_dimension=2000))
    instance.set_source(scene, "scene.png")
    return instance


def blob(mask_slice=(slice(40, 120), slice(60, 160)), shape=(200, 300)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[mask_slice] = True
    return mask


class TestEffectPrimitives:
    def test_fill_blends_only_inside_the_mask(self, scene):
        frame = scene.copy()
        mask = blob()
        apply_fill(frame, mask, (0, 0, 255), 1.0)
        assert tuple(frame[80, 100]) == (0, 0, 255)
        assert np.array_equal(frame[10, 10], scene[10, 10])

    def test_fill_respects_alpha(self, scene):
        frame = scene.copy()
        apply_fill(frame, blob(), (0, 0, 255), 0.5)
        pixel = frame[80, 100]
        assert 0 < pixel[2] < 256
        assert not np.array_equal(pixel, (0, 0, 255))

    def test_fill_on_an_empty_mask_is_a_noop(self, scene):
        frame = scene.copy()
        apply_fill(frame, np.zeros((200, 300), bool), (0, 0, 255), 1.0)
        assert np.array_equal(frame, scene)

    def test_outline_draws_on_the_boundary_not_the_middle(self, scene):
        frame = scene.copy()
        apply_outline(frame, blob(), (0, 255, 0), 2)
        # The interior is untouched; the border is not.
        assert np.array_equal(frame[80, 100], scene[80, 100])
        border = frame[40:121, 60:161]
        assert not np.array_equal(border, scene[40:121, 60:161])

    def test_blur_only_touches_the_mask(self, scene):
        frame = scene.copy()
        apply_blur(frame, blob(), 31)
        assert not np.array_equal(frame[80, 100], scene[80, 100])
        assert np.array_equal(frame[10, 10], scene[10, 10])

    def test_blur_reduces_local_variance(self, scene):
        frame = scene.copy()
        apply_blur(frame, blob(), 31)
        before = scene[50:110, 70:150].std()
        after = frame[50:110, 70:150].std()
        assert after < before

    def test_blur_forces_an_odd_kernel(self, scene):
        """GaussianBlur rejects even kernels, so an even request must be corrected."""
        frame = scene.copy()
        apply_blur(frame, blob(), 30)
        assert not np.array_equal(frame[80, 100], scene[80, 100])

    def test_pixelate_creates_flat_blocks(self, scene):
        frame = scene.copy()
        apply_pixelate(frame, blob(), 20)
        patch = frame[60:80, 80:100]
        assert patch.std() < scene[60:80, 80:100].std()

    def test_spotlight_darkens_outside_only(self, scene):
        frame = scene.copy()
        apply_spotlight(frame, blob())
        assert np.array_equal(frame[80, 100], scene[80, 100])
        assert frame[10, 10].sum() < scene[10, 10].sum()

    def test_cutout_replaces_the_background(self, scene):
        frame = scene.copy()
        apply_cutout(frame, blob(), (0, 0, 0))
        assert np.array_equal(frame[80, 100], scene[80, 100])
        assert tuple(frame[10, 10]) == (0, 0, 0)

    def test_effects_never_change_shape_or_dtype(self, scene):
        for effect in (
            lambda f, m: apply_fill(f, m, (0, 255, 0), 0.5),
            lambda f, m: apply_outline(f, m, (0, 255, 0), 2),
            lambda f, m: apply_blur(f, m, 15),
            lambda f, m: apply_pixelate(f, m, 8),
            lambda f, m: apply_spotlight(f, m),
            lambda f, m: apply_cutout(f, m),
        ):
            frame = scene.copy()
            effect(frame, blob())
            assert frame.shape == scene.shape
            assert frame.dtype == np.uint8


class TestPrompts:
    def test_splits_on_commas_and_newlines(self, core):
        assert core.parse_prompts("cat, dog\nmug") == ["cat", "dog", "mug"]

    def test_normalises_case_and_whitespace(self, core):
        assert core.parse_prompts("  Coffee   MUG ") == ["coffee mug"]

    def test_drops_duplicates_and_blanks(self, core):
        assert core.parse_prompts("cat,,cat , CAT") == ["cat"]

    def test_caps_the_number_of_prompts(self):
        core = SamLabelerCore(SamLabelerConfig(max_prompts=3))
        assert len(core.parse_prompts("a,b,c,d,e")) == 3

    def test_truncates_a_very_long_prompt(self):
        core = SamLabelerCore(SamLabelerConfig(max_prompt_length=5))
        assert core.parse_prompts("abcdefghij") == ["abcde"]

    def test_setting_prompts_assigns_palette_colours(self, core):
        core.set_prompts("cat, dog")
        assert core.style_for("cat").colour == PALETTE[0]
        assert core.style_for("dog").colour == PALETTE[1]

    def test_existing_colour_choices_survive_new_prompts(self, core):
        core.set_prompts("cat")
        core.set_style("cat", colour=(1, 2, 3))
        core.set_prompts("cat, dog")
        assert core.style_for("cat").colour == (1, 2, 3)


class TestStylesAndSettings:
    def test_effect_must_be_known(self, core):
        core.set_prompts("cat")
        assert core.set_style("cat", effect="fill") is True
        assert core.set_style("cat", effect="disco") is False

    def test_colour_is_clamped(self, core):
        core.set_prompts("cat")
        core.set_style("cat", colour=(999, -5, 20))
        assert core.style_for("cat").colour == (255, 0, 20)

    def test_bad_colour_is_rejected(self, core):
        core.set_prompts("cat")
        assert core.set_style("cat", colour="green") is False

    def test_style_json_exposes_css_rgb(self, core):
        style = LabelStyle("cat", (255, 0, 0))  # BGR blue
        assert style.to_json()["css"] == "rgb(0,0,255)"

    def test_settings_are_clamped(self, core):
        applied = core.update_settings({"confidence": 5, "alpha": -1, "blurStrength": 1000})
        assert applied["confidence"] == 0.95
        assert applied["alpha"] == 0.1
        assert core.config.blur_strength % 2 == 1

    def test_unknown_settings_ignored(self, core):
        assert core.update_settings({"nope": 1}) == {}

    def test_mode_switches(self, core):
        core.update_settings({"mode": DETECTION})
        assert core.config.mode == DETECTION
        core.update_settings({"mode": "nonsense"})
        assert core.config.mode == DETECTION


class TestRunAndRender:
    def test_stub_backend_finds_one_instance_per_prompt(self, core):
        core.run(StubBackend(), "cat, laptop, mug")
        assert {instance.label for instance in core.instances} <= {"cat", "laptop", "mug"}
        assert core.instances

    def test_render_changes_the_image(self, core):
        core.run(StubBackend(), "cat")
        assert not np.array_equal(core.render(), core.source)

    @pytest.mark.parametrize("effect", list(EFFECTS))
    def test_every_effect_renders(self, core, effect):
        core.run(StubBackend(), "cat, laptop")
        for label in core.prompts:
            core.set_style(label, effect=effect)
        out = core.render()
        assert out.shape == core.source.shape
        assert out.dtype == np.uint8

    def test_hide_effect_draws_nothing(self, core):
        core.run(StubBackend(), "cat")
        for label in core.prompts:
            core.set_style(label, effect="hide")
        assert np.array_equal(core.render(labels=False), core.source)

    def test_invisible_label_draws_nothing(self, core):
        core.run(StubBackend(), "cat")
        core.set_style("cat", visible=False)
        assert np.array_equal(core.render(labels=False), core.source)

    def test_detection_mode_differs_from_segmentation(self, core):
        core.run(StubBackend(), "cat")
        core.update_settings({"mode": SEGMENTATION})
        segmented = core.render()
        core.update_settings({"mode": DETECTION})
        assert not np.array_equal(segmented, core.render())

    def test_no_prompts_is_an_error_not_a_crash(self, core):
        state = core.run(StubBackend(), "")
        assert state["instances"] == []
        assert "at least one" in core.error

    def test_no_image_is_an_error(self):
        core = SamLabelerCore()
        core.run(StubBackend(), "cat")
        assert "Load an image first" in core.error

    def test_unavailable_backend_reports_its_reason(self, core):
        core.run(UnavailableBackend("no token"), "cat")
        assert core.error == "no token"
        assert core.instances == []

    def test_a_failing_backend_is_caught(self, core):
        class Exploding:
            name = "boom"
            ready = True

            def describe(self):
                return "explodes"

            def segment(self, *args):
                raise RuntimeError("CUDA out of memory")

        core.run(Exploding(), "cat")
        assert "Segmentation failed" in core.error
        assert core.instances == []

    def test_instances_are_sorted_largest_first_in_state(self, core):
        big = Instance("big", 0.9, blob((slice(0, 150), slice(0, 250))))
        small = Instance("small", 0.9, blob((slice(0, 20), slice(0, 20))))
        core.instances = [small, big]
        labels = [entry["label"] for entry in core.state()["instances"]]
        assert labels == ["big", "small"]

    def test_counts_report_per_label_totals(self, core):
        core.set_prompts("cat, dog")
        core.instances = [
            Instance("cat", 0.9, blob()),
            Instance("cat", 0.8, blob((slice(0, 30), slice(0, 30)))),
        ]
        counts = core.state()["counts"]
        assert counts["cat"] == 2
        assert counts["dog"] == 0

    def test_placeholder_when_no_image(self):
        canvas = SamLabelerCore().render()
        assert canvas.ndim == 3
        assert canvas.max() > 0

    def test_large_images_are_downscaled(self, scene):
        core = SamLabelerCore(SamLabelerConfig(max_dimension=100))
        core.set_source(np.zeros((900, 1600, 3), np.uint8))
        assert max(core.source.shape[:2]) == 100

    def test_state_is_json_safe(self, core):
        import json

        core.run(StubBackend(), "cat")
        json.dumps(core.state())


class TestInstanceGeometry:
    def test_box_is_tight_around_the_mask(self):
        instance = Instance("x", 0.5, blob((slice(10, 30), slice(40, 90))))
        assert instance.box == (40, 10, 50, 20)

    def test_empty_mask_has_a_zero_box(self):
        assert Instance("x", 0.5, np.zeros((10, 10), bool)).box == (0, 0, 0, 0)

    def test_json_boxes_are_normalized(self):
        instance = Instance("x", 0.5, blob((slice(0, 100), slice(0, 150))))
        payload = instance.to_json(300, 200)
        assert payload["box"] == [0.0, 0.0, 0.5, 0.5]


class TestBackendSelection:
    def test_stub_requested_explicitly(self):
        backend = load_backend(SamLabelerConfig(), stub=True)
        assert backend.ready is True
        assert backend.name == "stub"

    def test_missing_transformers_explains_itself(self):
        backend = load_backend(SamLabelerConfig(), token="hf_x")
        # transformers is not a dependency of this repo, so this is the normal path.
        assert backend.ready is False
        assert "transformers" in backend.describe() or "gated" in backend.describe()

    def test_no_token_is_reported_before_any_download(self, monkeypatch):
        import demos.sam_labeler.backend as backend_module

        monkeypatch.setattr(backend_module, "Sam3Backend", object)
        backend = load_backend(SamLabelerConfig(), token=None)
        assert backend.ready is False

    def test_stub_is_deterministic(self):
        image = np.zeros((120, 160, 3), np.uint8)
        config = SamLabelerConfig()
        first = StubBackend().segment(image, ["cat"], config)
        second = StubBackend().segment(image, ["cat"], config)
        assert first[0].box == second[0].box
        assert first[0].score == second[0].score

    def test_stub_respects_the_confidence_setting(self):
        image = np.zeros((120, 160, 3), np.uint8)
        generous = StubBackend().segment(image, ["cat", "dog", "mug"], SamLabelerConfig(confidence=0.05))
        strict = StubBackend().segment(image, ["cat", "dog", "mug"], SamLabelerConfig(confidence=0.95))
        assert len(strict) < len(generous)


@pytest.fixture()
def client():
    web.set_backend(StubBackend())
    blueprint, sock, _ = web.build()
    assert sock is None, "the labeler runs the model server side, so there is no stream"
    app = create_app([], name="sam_test")
    app.register_blueprint(blueprint)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client
    web.set_backend(None)


def _png(colour=(90, 110, 130)) -> bytes:
    ok, buffer = cv2.imencode(".png", np.full((120, 160, 3), colour, np.uint8))
    assert ok
    return buffer.tobytes()


def _await_job(client, sid: str, timeout: float = 5.0) -> dict:
    """Poll /status until the worker finishes. Returns the final payload."""
    import time

    deadline = time.monotonic() + timeout
    body = client.get(f"/sam-labeler/status?sid={sid}").get_json()
    while body["job"]["running"] and time.monotonic() < deadline:
        time.sleep(0.02)
        body = client.get(f"/sam-labeler/status?sid={sid}").get_json()
    assert not body["job"]["running"], "worker did not finish in time"
    return body


class TestAsyncJobs:
    """Long operations must not block the request, or the page has no feedback."""

    def _with_image(self, client, sid: str = "j") -> None:
        client.post(
            f"/sam-labeler/upload?sid={sid}",
            data={"image": (io.BytesIO(_png()), "p.png")},
            content_type="multipart/form-data",
        )

    def test_run_returns_immediately_with_a_job(self, client):
        self._with_image(client)
        body = client.post("/sam-labeler/run?sid=j", json={"prompts": "cat"}).get_json()
        assert body["job"]["kind"] == "segment"
        assert body["job"]["state"] in ("running", "done")

    def test_status_reports_progress_fields(self, client):
        self._with_image(client)
        client.post("/sam-labeler/run?sid=j", json={"prompts": "cat"})
        body = _await_job(client, "j")
        job = body["job"]
        assert set(job) >= {"state", "stage", "elapsed", "megabytes", "rate", "running"}
        assert job["elapsed"] >= 0
        assert "object(s) found" in job["detail"]

    def test_run_without_an_image_is_refused(self, client):
        response = client.post("/sam-labeler/run?sid=empty", json={"prompts": "cat"})
        assert response.status_code == 400
        assert "upload a picture first" in response.get_json()["error"]

    def test_run_without_prompts_is_refused(self, client):
        self._with_image(client, "noprompt")
        response = client.post("/sam-labeler/run?sid=noprompt", json={"prompts": "  "})
        assert response.status_code == 400
        assert "at least one" in response.get_json()["error"]

    def test_a_second_run_while_busy_is_rejected(self, client):
        """One job at a time: a double click must not start two inferences."""

        class Slow:
            name = "slow"
            ready = True
            calls = 0

            def describe(self):
                return "slow backend"

            def segment(self, image, prompts, config):
                Slow.calls += 1
                __import__("time").sleep(0.4)
                return []

        web.set_backend(Slow())
        self._with_image(client, "busy")
        first = client.post("/sam-labeler/run?sid=busy", json={"prompts": "cat"})
        second = client.post("/sam-labeler/run?sid=busy", json={"prompts": "cat"})
        assert first.status_code == 200
        assert second.status_code == 409
        assert "Already working" in second.get_json()["error"]
        _await_job(client, "busy")
        assert Slow.calls == 1
        web.set_backend(StubBackend())

    def test_failed_segmentation_surfaces_in_the_job(self, client):
        class Exploding:
            name = "boom"
            ready = True

            def describe(self):
                return "explodes"

            def segment(self, *args):
                raise RuntimeError("no memory")

        web.set_backend(Exploding())
        self._with_image(client, "fail")
        client.post("/sam-labeler/run?sid=fail", json={"prompts": "cat"})
        body = _await_job(client, "fail")
        assert body["job"]["state"] == "error"
        assert body["job"]["error"]
        web.set_backend(StubBackend())

    def test_token_post_does_not_block_on_loading(self, client):
        body = client.post("/sam-labeler/token?sid=t", json={"token": "hf_x"}).get_json()
        assert body["ok"] is True
        assert "job" in body


class TestWebApp:
    def test_page_renders_with_the_token_field_masked(self, client):
        body = client.get("/sam-labeler/").get_data(as_text=True)
        assert 'type="password"' in body, "a credential field must not be plain text"
        assert "Demo mode" in body
        assert "Find them" in body

    def test_state_never_contains_the_token(self, client):
        client.post("/sam-labeler/token?sid=a", json={"token": "hf_secret_value"})
        state = client.get("/sam-labeler/state?sid=a").get_json()
        assert state["hasToken"] is True
        assert "hf_secret_value" not in client.get("/sam-labeler/state?sid=a").get_data(as_text=True)
        assert "token" not in {key for key in state if key != "hasToken"}

    def test_token_can_be_cleared(self, client):
        client.post("/sam-labeler/token?sid=a", json={"token": "hf_x"})
        body = client.post("/sam-labeler/token?sid=a", json={"token": ""}).get_json()
        assert body["hasToken"] is False

    def test_demo_mode_toggle(self, client):
        body = client.post("/sam-labeler/token?sid=a", json={"token": "", "demoMode": True}).get_json()
        assert body["demoMode"] is True

    def test_upload_then_run(self, client):
        """/run starts a worker, so the result arrives via /status."""
        client.post(
            "/sam-labeler/upload?sid=a",
            data={"image": (io.BytesIO(_png()), "photo.png")},
            content_type="multipart/form-data",
        )
        accepted = client.post("/sam-labeler/run?sid=a", json={"prompts": "cat, mug"}).get_json()
        assert accepted["ok"] is True
        assert accepted["hasSource"] is True

        body = _await_job(client, "a")
        assert body["instances"]
        assert {entry["label"] for entry in body["labels"]} == {"cat", "mug"}
        assert body["job"]["state"] == "done"

    def test_capture_accepts_a_frame(self, client):
        body = client.post(
            "/sam-labeler/capture?sid=a",
            data={"frame": (io.BytesIO(_png()), "frame.jpg")},
            content_type="multipart/form-data",
        ).get_json()
        assert body["ok"] is True
        assert body["sourceName"] == "webcam.png"

    def test_undecodable_capture_is_a_400(self, client):
        response = client.post(
            "/sam-labeler/capture?sid=a",
            data={"frame": (io.BytesIO(b"junk"), "frame.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_upload_without_a_file_is_a_400(self, client):
        assert client.post("/sam-labeler/upload?sid=a").status_code == 400

    def test_style_command_changes_the_render(self, client):
        client.post(
            "/sam-labeler/upload?sid=a",
            data={"image": (io.BytesIO(_png()), "p.png")},
            content_type="multipart/form-data",
        )
        client.post("/sam-labeler/run?sid=a", json={"prompts": "cat"})
        before = client.get("/sam-labeler/result.png?sid=a").data
        client.post(
            "/sam-labeler/command?sid=a",
            json={"command": "style", "payload": {"label": "cat", "effect": "cutout"}},
        )
        assert client.get("/sam-labeler/result.png?sid=a").data != before

    def test_bad_effect_is_reported(self, client):
        body = client.post(
            "/sam-labeler/command?sid=a",
            json={"command": "style", "payload": {"label": "cat", "effect": "disco"}},
        ).get_json()
        assert body["ok"] is False

    def test_unknown_command_is_a_400(self, client):
        assert client.post("/sam-labeler/command?sid=a", json={"command": "nope"}).status_code == 400

    def test_settings_command_clamps(self, client):
        body = client.post(
            "/sam-labeler/command?sid=a",
            json={"command": "settings", "payload": {"alpha": 99}},
        ).get_json()
        assert body["settings"]["alpha"] == 1.0

    def test_result_and_source_pngs(self, client):
        assert client.get("/sam-labeler/source.png?sid=a").status_code == 404
        client.post(
            "/sam-labeler/upload?sid=a",
            data={"image": (io.BytesIO(_png()), "p.png")},
            content_type="multipart/form-data",
        )
        for path in ("/sam-labeler/result.png?sid=a", "/sam-labeler/source.png?sid=a"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_snapshot_downloads(self, client):
        response = client.get("/sam-labeler/snapshot?sid=a")
        assert response.data[:4] == b"\x89PNG"
        assert "attachment" in response.headers["Content-Disposition"]

    def test_sessions_are_isolated(self, client):
        client.post(
            "/sam-labeler/upload?sid=one",
            data={"image": (io.BytesIO(_png()), "p.png")},
            content_type="multipart/form-data",
        )
        assert client.get("/sam-labeler/state?sid=one").get_json()["hasSource"] is True
        assert client.get("/sam-labeler/state?sid=two").get_json()["hasSource"] is False

    def test_health(self, client):
        assert client.get("/sam-labeler/health").get_json()["ok"] is True


class TestPageWiring:
    """Guards against the JS crash class: markup and state must line up.

    A missing element used to take down the whole ``apply()`` render, which hid the
    very error message the user was trying to read.
    """

    def _page(self, client) -> str:
        return client.get("/sam-labeler/").get_data(as_text=True)

    def test_every_slider_has_a_paired_output(self, client):
        import re

        page = self._page(client)
        sliders = set(re.findall(r'id="set-(\w+)"[^>]*type="range"', page))
        sliders |= set(re.findall(r'type="range" id="set-(\w+)"', page))
        for key in sliders:
            assert f'id="out-{key}"' in page, f"slider set-{key} has no out-{key}"

    def test_mode_control_is_a_select_without_an_output(self, client):
        """The case that crashed: a control named like a slider, but not one."""
        page = self._page(client)
        assert 'id="set-mode"' in page
        assert 'id="out-mode"' not in page
        # The script must therefore never assume out-<key> exists.
        assert "if (output)" in page

    def test_every_element_the_script_touches_exists(self, client):
        import re

        page = self._page(client)
        script = page.split("{% endblock %}")[0]
        referenced = set(re.findall(r"el\('([\w-]+)'\)", script))
        for element_id in referenced:
            assert f'id="{element_id}"' in page, f"script references missing #{element_id}"

    def test_settings_keys_are_all_renderable(self, client):
        """Every key the server sends is either a slider, the mode select, or ignored."""
        state = client.get("/sam-labeler/state?sid=w").get_json()
        page = self._page(client)
        for key in state["settings"]:
            has_input = f'id="set-{key}"' in page
            has_output = f'id="out-{key}"' in page
            assert has_input or not has_output, f"{key} has an output but no input"

    def test_error_detail_area_wraps(self, client):
        page = self._page(client)
        assert "backend-detail" in page
        assert "pre-wrap" in page
