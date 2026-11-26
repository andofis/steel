import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipe_inspection import PipeInspection


class PipeInspectionTest(unittest.TestCase):
    def test_requests_reenrollment_when_capture_phase_calibration_changes(self):
        holes = [{"center": [x, y], "contours": [], "area": 8, "score": .9}
                 for x, y in [(10, 30), (47, 85), (30, 190), (82, 417)]]
        source = {"width": 100, "height": 500, "pipe_id": "template"}
        sample = dict(source, pipe_id="new", phase_calibration_id="nominal-v2")
        detection = {"width": 100, "height": 500, "holes": holes}
        with tempfile.TemporaryDirectory() as temp, PipeInspection(Path(temp) / "index.json") as controller:
            with patch("pipe_inspection.segment_panorama", return_value=detection) as segment:
                controller.record_template("旧模板", b"png", source)
                controller.set_enabled(True)
                self.assertTrue(controller.submit(b"png", sample))
                result = controller.wait_result("new", timeout=15)
                self.assertEqual(result["status"], "review")
                self.assertIn("重新录入模板", result["reason"])
                self.assertEqual(segment.call_count, 1)
                controller.set_enabled(False)
                controller.record_template("新模板", b"png", sample)
                controller.set_enabled(True)
                self.assertTrue(controller.submit(b"png", dict(sample, pipe_id="next")))
                self.assertEqual(controller.wait_result("next", timeout=15)["status"], "pass")

    def test_enroll_enable_and_inspect_without_changing_the_capture(self):
        masks = [{"center": [x, y], "contours": [[[x - 2, y - 2], [x + 2, y - 2], [x + 2, y + 2]]],
                  "area": 8, "score": 0.9} for x, y in [(10, 30), (47, 85), (30, 190), (82, 417)]]
        source = {"width": 100, "height": 500, "pipe_id": "template-capture",
                  "panorama_seam": {"cut_x": 17}}
        sample = dict(source, pipe_id="line-capture")
        detection = {"width": 100, "height": 500, "backend": "cpu",
                     "holes": masks, "elapsed_ms": 25}
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / "templates.json"
            archive = Path(temp) / "line-capture"
            archive.mkdir()
            with PipeInspection(index) as controller:
                with self.assertRaisesRegex(RuntimeError, "模板"):
                    controller.set_enabled(True)
                self.assertFalse(controller.submit(b"png", sample, archive))
                with patch("pipe_inspection.segment_panorama", return_value=detection) as segment:
                    state = controller.record_template("合格样管", b"png", source)
                    self.assertEqual(len(state["templates"]), 1)
                    self.assertEqual(state["active_template"], state["templates"][0]["id"])
                    image_path = index.with_name("templates_images") / (state["active_template"] + ".png")
                    self.assertEqual(image_path.read_bytes(), b"png")
                    controller.set_enabled(True)
                    self.assertTrue(controller.submit(b"png", sample, archive))
                    result = controller.wait_result("line-capture", timeout=15)
                    self.assertEqual(result["pipe_id"], "line-capture")
                    self.assertEqual(result["status"], "pass")
                    self.assertEqual([item["kind"] for item in result["findings"]], ["ok"] * 4)
                    self.assertEqual(segment.call_args.args[2:4], ("cpu", 17))
                self.assertTrue((archive / "inspection.json").is_file())
            with PipeInspection(index) as restarted:
                state = restarted.state()
                self.assertFalse(state["enabled"])
                self.assertEqual(state["templates"][0]["name"], "合格样管")

    def test_inference_error_is_reported_for_the_right_pipe(self):
        with tempfile.TemporaryDirectory() as temp, PipeInspection(Path(temp) / "index.json") as controller:
            source = {"width": 100, "height": 500, "pipe_id": "old"}
            hole = {"center": [15, 25], "contours": [[[14, 24], [16, 24], [16, 26]]],
                    "area": 2, "score": 0.9}
            response = {"width": 100, "height": 500, "backend": "cpu", "holes": [hole], "elapsed_ms": 2}
            with patch("pipe_inspection.segment_panorama", side_effect=[response, RuntimeError("推理离线")]):
                controller.record_template("样管", b"png", source)
                controller.set_enabled(True)
                controller.submit(b"png", dict(source, pipe_id="new"))
                result = controller.wait_result("new", timeout=15)
                controller.submit(None, {"pipe_id": "incomplete", "failure_reason": "原图不完整"})
                failed_capture = controller.wait_result("incomplete")
            self.assertEqual(result["status"], "error")
            self.assertIn("推理离线", result["reason"])
            self.assertEqual(result["findings"], [])
            self.assertEqual(controller.wait_result("old")["status"], "template")
            self.assertEqual(failed_capture["status"], "error")
            self.assertEqual(failed_capture["reason"], "原图不完整")
