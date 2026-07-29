"""Tests for the image lab: operations, pipeline, code generation, web routes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from demos.common.webapp import create_app
from demos.image_lab import operations as ops
from demos.image_lab import web
from demos.image_lab.config import ImageLabConfig
from demos.image_lab.core import ImageLabCore, Step
from demos.tools.make_sample_images import generate as generate_samples


@pytest.fixture()
def photo() -> np.ndarray:
    """A small image with edges, colour and shapes, so every operation does something."""
    image = np.full((180, 240, 3), (40, 30, 60), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (110, 120), (120, 255, 140), -1)
    cv2.circle(image, (170, 90), 40, (255, 200, 60), -1)
    cv2.line(image, (0, 170), (240, 150), (255, 255, 255), 2)
    return image


@pytest.fixture()
def core(photo: np.ndarray) -> ImageLabCore:
    instance = ImageLabCore(ImageLabConfig())
    instance.set_source(photo, "test.png")
    return instance


class TestCatalog:
    def test_every_operation_is_registered_once(self):
        keys = [operation.key for operation in ops.OPERATIONS]
        assert len(keys) == len(set(keys))
        assert len(keys) >= 20

    def test_every_category_has_operations(self):
        for group in ops.catalog_json():
            assert group["operations"], f"{group['category']} is empty"

    def test_every_operation_declares_a_cv2_reference(self):
        for operation in ops.OPERATIONS:
            assert operation.docs, f"{operation.key} has no docs reference"
            assert operation.summary.endswith("."), f"{operation.key} summary should be a sentence"

    def test_drawing_category_covers_shapes_points_and_text(self):
        drawing = {op.key for op in ops.OPERATIONS if op.category == "Drawing"}
        assert {"draw_rectangle", "draw_circle", "draw_line", "draw_polygon", "draw_text"} <= drawing

    @pytest.mark.parametrize("operation", list(ops.OPERATIONS), ids=lambda o: o.key)
    def test_operation_runs_with_defaults(self, operation, photo):
        result = operation.run(photo.copy(), operation.defaults())
        assert result is not None
        assert result.size > 0
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("operation", list(ops.OPERATIONS), ids=lambda o: o.key)
    def test_operation_emits_code(self, operation):
        lines = operation.code(operation.defaults())
        assert lines
        assert any("cv2." in line or "np." in line or "img" in line for line in lines)

    @pytest.mark.parametrize("operation", list(ops.OPERATIONS), ids=lambda o: o.key)
    def test_operation_accepts_a_grayscale_input(self, operation, photo):
        """Steps are chainable in any order, so each must survive a 1-channel input."""
        gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        result = operation.run(gray, operation.defaults())
        assert result is not None and result.size > 0


class TestParamCoercion:
    def test_odd_only_rounds_up(self):
        param = ops.Param("k", "K", "int", 5, 1, 31, 2, odd_only=True)
        assert param.coerce(4) == 5
        assert param.coerce(7) == 7

    def test_values_are_clamped(self):
        param = ops.Param("v", "V", "int", 5, 0, 10)
        assert param.coerce(-99) == 0
        assert param.coerce(99) == 10

    def test_garbage_falls_back_to_the_default(self):
        param = ops.Param("v", "V", "int", 7, 0, 10)
        assert param.coerce("not a number") == 7
        assert param.coerce(None) == 7

    def test_choice_rejects_unknown_values(self):
        param = ops.Param("m", "M", "choice", "a", choices=("a", "b"))
        assert param.coerce("b") == "b"
        assert param.coerce("z") == "a"

    def test_colour_is_clamped_to_bgr_bytes(self):
        param = ops.Param("c", "C", "color", (1, 2, 3))
        assert param.coerce([300, -5, 20]) == (255, 0, 20)
        assert param.coerce("nope") == (1, 2, 3)

    def test_float_rounds(self):
        param = ops.Param("f", "F", "float", 1.0, 0.0, 3.0, 0.1)
        assert param.coerce(2.123456) == 2.1235


class TestPipeline:
    def test_defaults_to_a_working_pipeline(self, core):
        assert [step.operation.key for step in core.steps] == ["gaussian_blur", "canny"]
        assert core.apply().shape == core.source.shape

    def test_add_remove_and_move(self, core):
        core.set_pipeline([])
        core.add_step("grayscale")
        core.add_step("canny")
        assert [s.operation.key for s in core.steps] == ["grayscale", "canny"]
        core.move_step(0, 1)
        assert [s.operation.key for s in core.steps] == ["canny", "grayscale"]
        core.remove_step(0)
        assert [s.operation.key for s in core.steps] == ["grayscale"]

    def test_unknown_operation_is_rejected(self, core):
        assert core.add_step("does_not_exist") is None
        assert "Unknown operation" in (core.error or "")

    def test_step_cap_is_enforced(self, core):
        core.set_pipeline([])
        for _ in range(core.config.max_steps + 4):
            core.add_step("grayscale")
        assert len(core.steps) == core.config.max_steps

    def test_disabled_steps_are_skipped(self, core):
        core.set_pipeline([{"key": "grayscale", "enabled": False}])
        assert np.array_equal(core.apply(), core.source)

    def test_order_changes_the_result(self, core):
        core.set_pipeline([{"key": "canny"}, {"key": "gaussian_blur"}])
        edges_then_blur = core.apply()
        core.set_pipeline([{"key": "gaussian_blur"}, {"key": "canny"}])
        blur_then_edges = core.apply()
        assert not np.array_equal(edges_then_blur, blur_then_edges)

    def test_invalid_parameters_skip_the_step_instead_of_crashing(self, core):
        core.set_pipeline([{"key": "median_blur"}])
        # medianBlur rejects large kernels on 3-channel images; force one through.
        core.steps[0].params["ksize"] = 99
        result = core.apply()
        assert result is not None and result.size > 0

    def test_pipeline_survives_a_geometry_change_mid_chain(self, core):
        core.set_pipeline([{"key": "resize", "params": {"scale": 0.5}}, {"key": "draw_circle"}])
        result = core.apply()
        assert result.shape[0] == pytest.approx(core.source.shape[0] // 2, abs=2)

    def test_set_pipeline_ignores_unknown_keys(self, core):
        core.set_pipeline([{"key": "nope"}, {"key": "grayscale"}])
        assert [s.operation.key for s in core.steps] == ["grayscale"]

    def test_step_create_returns_none_for_unknown(self):
        assert Step.create("nope") is None


class TestCodeGeneration:
    def test_script_is_runnable_python(self, core):
        code = core.code()
        compile(code, "pipeline.py", "exec")
        assert code.startswith("import cv2")
        assert 'cv2.imread("test.png")' in code
        assert code.rstrip().endswith("cv2.waitKey(0)")

    def test_code_reflects_the_parameters(self, core):
        core.set_pipeline([{"key": "gaussian_blur", "params": {"ksize": 21, "sigma": 3.0}}])
        assert "cv2.GaussianBlur(img, (21, 21), 3.0)" in core.code()

    def test_code_updates_when_a_parameter_changes(self, core):
        core.set_pipeline([{"key": "canny", "params": {"low": 10, "high": 20}}])
        first = core.code()
        core.update_step(0, {"low": 90})
        assert first != core.code()
        assert "cv2.Canny(gray, 90, 20" in core.code()

    def test_disabled_steps_are_left_out(self, core):
        core.set_pipeline([{"key": "canny", "enabled": False}, {"key": "grayscale"}])
        code = core.code()
        assert "Canny" not in code
        assert "COLOR_BGR2GRAY" in code

    def test_empty_pipeline_is_still_valid_python(self, core):
        core.set_pipeline([])
        compile(core.code(), "pipeline.py", "exec")
        assert "No operations enabled" in core.code()

    def test_every_operation_generates_compilable_code(self, core):
        for operation in ops.OPERATIONS:
            core.set_pipeline([{"key": operation.key}])
            compile(core.code(), "pipeline.py", "exec")

    def test_drawing_code_includes_the_colour_tuple(self, core):
        core.set_pipeline([{"key": "draw_rectangle", "params": {"color": [10, 20, 30]}}])
        assert "(10, 20, 30)" in core.code()


class TestSourceLoading:
    def test_load_bytes_decodes_a_png(self, core, photo):
        ok, buffer = cv2.imencode(".png", photo)
        assert ok
        assert core.load_bytes(buffer.tobytes(), "x.png") is True
        assert core.source_name == "x.png"

    def test_load_bytes_rejects_garbage(self, core):
        assert core.load_bytes(b"not an image", "x.png") is False
        assert "decode" in (core.error or "")

    def test_load_bytes_rejects_empty(self, core):
        assert core.load_bytes(b"", "x.png") is False

    def test_oversized_uploads_are_refused(self, photo):
        core = ImageLabCore(ImageLabConfig(max_upload_bytes=10))
        ok, buffer = cv2.imencode(".png", photo)
        assert core.load_bytes(buffer.tobytes(), "big.png") is False

    def test_large_images_are_downscaled(self):
        core = ImageLabCore(ImageLabConfig(max_dimension=200))
        core.set_source(np.zeros((900, 1600, 3), dtype=np.uint8))
        assert max(core.source.shape[:2]) == 200

    def test_small_images_are_untouched(self, core, photo):
        assert core.source.shape == photo.shape

    def test_placeholder_when_there_is_no_image(self):
        core = ImageLabCore(ImageLabConfig())
        assert core.has_source is False
        canvas = core.apply()
        assert canvas.shape[2] == 3
        assert canvas.max() > 0

    def test_samples_load_by_name(self, tmp_path: Path):
        generate_samples(tmp_path)
        core = ImageLabCore(ImageLabConfig(samples_dir=tmp_path))
        names = [path.name for path in core.samples()]
        assert "shapes.png" in names
        assert core.load_sample("shapes.png") is True
        assert core.load_sample("nope.png") is False


class TestSampleImages:
    def test_generates_four_distinct_samples(self, tmp_path: Path):
        written = generate_samples(tmp_path)
        assert len(written) == 4
        images = [cv2.imread(str(path)) for path in written]
        for image in images:
            assert image is not None
            assert image.shape[2] == 3
        for index, first in enumerate(images):
            for second in images[index + 1 :]:
                assert first.shape != second.shape or not np.array_equal(first, second)

    def test_shapes_sample_yields_contours(self, tmp_path: Path):
        generate_samples(tmp_path)
        core = ImageLabCore(ImageLabConfig(samples_dir=tmp_path))
        core.load_sample("shapes.png")
        core.set_pipeline([{"key": "contours", "params": {"min_area": 500}}])
        assert not np.array_equal(core.apply(), core.source)


@pytest.fixture()
def client(tmp_path: Path):
    generate_samples(tmp_path)
    original = web._CONFIG.samples_dir
    web._CONFIG.samples_dir = tmp_path
    blueprint, sock, _ = web.build()
    assert sock is None, "the image lab has no landmark stream"
    app = create_app([], name="image_lab_test")
    app.register_blueprint(blueprint)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client
    web._CONFIG.samples_dir = original


class TestWebApp:
    def test_page_lists_the_catalog(self, client):
        body = client.get("/image-lab/").get_data(as_text=True)
        assert "Reveal code" in body
        assert "Gaussian blur" in body
        assert "Find + draw contours" in body

    def test_state_starts_on_a_sample(self, client):
        state = client.get("/image-lab/state?sid=a").get_json()
        assert state["hasSource"] is True
        assert state["samples"]
        assert state["code"].startswith("import cv2")

    def test_result_and_source_are_pngs(self, client):
        client.get("/image-lab/state?sid=a")
        for path in ("/image-lab/result.png?sid=a", "/image-lab/source.png?sid=a"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_pipeline_post_returns_code_and_steps(self, client):
        body = client.post(
            "/image-lab/pipeline?sid=a",
            json={"steps": [{"key": "threshold", "params": {"thresh": 90}}]},
        ).get_json()
        assert [step["key"] for step in body["steps"]] == ["threshold"]
        assert "cv2.threshold(gray, 90" in body["code"]

    def test_upload_accepts_an_image(self, client, photo):
        import io

        ok, buffer = cv2.imencode(".png", photo)
        data = {"image": (io.BytesIO(buffer.tobytes()), "mine.png")}
        body = client.post(
            "/image-lab/upload?sid=a", data=data, content_type="multipart/form-data"
        ).get_json()
        assert body["ok"] is True
        assert body["sourceName"] == "mine.png"

    def test_upload_rejects_a_non_image(self, client):
        import io

        data = {"image": (io.BytesIO(b"nope"), "x.png")}
        response = client.post(
            "/image-lab/upload?sid=a", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 400
        assert response.get_json()["ok"] is False

    def test_upload_without_a_file_is_a_400(self, client):
        assert client.post("/image-lab/upload?sid=a").status_code == 400

    def test_commands(self, client):
        client.get("/image-lab/state?sid=a")
        assert client.post("/image-lab/command?sid=a", json={"command": "clear"}).get_json()["steps"] == []
        assert client.post("/image-lab/command?sid=a", json={"command": "reset"}).get_json()["steps"]
        added = client.post(
            "/image-lab/command?sid=a", json={"command": "add", "payload": {"key": "sobel"}}
        ).get_json()
        assert added["steps"][-1]["key"] == "sobel"
        assert client.post("/image-lab/command?sid=a", json={"command": "bogus"}).get_json()["ok"] is False

    def test_sample_command_switches_image(self, client):
        body = client.post(
            "/image-lab/command?sid=a", json={"command": "sample", "payload": {"name": "document.png"}}
        ).get_json()
        assert body["sourceName"] == "document.png"

    def test_code_downloads_as_python(self, client):
        client.get("/image-lab/state?sid=a")
        response = client.get("/image-lab/code?sid=a")
        assert response.status_code == 200
        assert response.mimetype == "text/x-python"
        compile(response.get_data(as_text=True), "pipeline.py", "exec")

    def test_snapshot_downloads_a_png(self, client):
        client.get("/image-lab/state?sid=a")
        response = client.get("/image-lab/snapshot?sid=a")
        assert response.data[:4] == b"\x89PNG"
        assert "attachment" in response.headers["Content-Disposition"]

    def test_samples_are_served(self, client):
        assert client.get("/image-lab/sample/shapes.png").status_code == 200
        assert client.get("/image-lab/sample/nope.png").status_code == 404

    def test_sessions_are_independent(self, client):
        client.post("/image-lab/pipeline?sid=a", json={"steps": [{"key": "canny"}]})
        client.post("/image-lab/pipeline?sid=b", json={"steps": [{"key": "grayscale"}]})
        assert client.get("/image-lab/state?sid=a").get_json()["steps"][0]["key"] == "canny"
        assert client.get("/image-lab/state?sid=b").get_json()["steps"][0]["key"] == "grayscale"

    def test_health(self, client):
        assert client.get("/image-lab/health").get_json()["ok"] is True


class TestDesktopTrackbarMapping:
    """Trackbars are integer-only and start at zero; the mapping must round-trip."""

    def _round_trip(self, param: ops.Param):
        from demos.image_lab.desktop import _from_trackbar, _trackbar_range

        low, high, initial = _trackbar_range(param)
        assert low <= initial <= max(high, low)
        return _from_trackbar(param, initial)

    def test_int_param_round_trips(self):
        assert self._round_trip(ops.Param("v", "V", "int", 7, 0, 20)) == 7

    def test_negative_minimum_round_trips(self):
        assert self._round_trip(ops.Param("v", "V", "int", -5, -20, 20)) == -5

    def test_float_param_round_trips(self):
        assert self._round_trip(ops.Param("v", "V", "float", 1.5, 0.1, 3.0, 0.1)) == 1.5

    def test_bool_param_round_trips(self):
        assert self._round_trip(ops.Param("v", "V", "bool", True)) is True

    def test_choice_param_round_trips(self):
        param = ops.Param("m", "M", "choice", "b", choices=("a", "b", "c"))
        assert self._round_trip(param) == "b"

    def test_every_catalog_param_round_trips(self):
        from demos.image_lab.desktop import _from_trackbar, _trackbar_range

        for operation in ops.OPERATIONS:
            for param in operation.params:
                if param.kind == "color":
                    continue  # colours are not exposed on trackbars
                _, _, initial = _trackbar_range(param)
                value = _from_trackbar(param, initial)
                assert param.coerce(value) == value, f"{operation.key}.{param.key}"
