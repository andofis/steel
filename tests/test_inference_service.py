import struct
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from inference_service.pipeline import PackedMask, display_hole, segment
from tube_unwrap.tile_detection import TILE, WORKING_SIZE, project_tile_box

try:
    from starlette.testclient import TestClient
except ImportError:
    TestClient = None


class FakeRunner:
    batch_size = 2
    model_id = "sha256:test-model"

    def __call__(self, inputs):
        assert inputs.dtype == np.float32
        assert inputs.shape[1:] == (3, TILE, TILE)
        batch = len(inputs)
        boxes = np.tile([[108, 20, 126, 32], [108, 20, 126, 32]], (batch, 1, 1))
        marked = inputs[:, 0, 20, 108] > 0.5
        scores = np.column_stack((marked * 0.9, marked * 0.8))
        coefficients = np.ones((batch, 2, 1), np.float32)
        prototypes = np.ones((batch, 1, 2, 2), np.float32)
        return boxes, scores, coefficients, prototypes


class PipelineTests(unittest.TestCase):
    def test_edge_fragment_covered_by_a_complete_mask_is_counted_once_across_zero(self):
        width, height = WORKING_SIZE
        complete_tile, edge_tile = (864, 0), (0, 0)
        complete = project_tile_box(complete_tile, (270, 90, 306, 130, .91), width, height, TILE)
        edge = project_tile_box(edge_tile, (0, 85, 40, 145, .95), width, height, TILE)
        full_pixels = np.ones((40, 36), dtype=np.uint8)
        fragment_pixels = np.zeros((60, 40), dtype=np.uint8)
        fragment_pixels[15:35, :12] = 1
        candidates = {
            complete_tile: [(complete, PackedMask((270, 90), full_pixels.shape, np.packbits(full_pixels)))],
            edge_tile: [(edge, PackedMask((0, 85), fragment_pixels.shape, np.packbits(fragment_pixels)))],
        }
        with patch("inference_service.pipeline.WINDOWS", tuple(candidates)), patch(
            "inference_service.pipeline.tile_masks", side_effect=lambda tile, *_: iter(candidates[tile]),
        ):
            holes = segment(np.zeros((height, width), dtype=np.uint8), 0, FakeRunner())
        self.assertEqual([hole["area"] for hole in holes], [1440.0])
        self.assertEqual(holes[0]["center"], [0.0, 110.0])

    def test_discards_only_weak_thin_strips_with_an_alternate_valid_view(self):
        width, height = WORKING_SIZE
        inside_tile, edge_tile = (0, 0), (0, 384)
        candidates = []
        for x, w, h, score in [(100, 27, 10, .44), (200, 10, 10, .49), (300, 27, 10, .85)]:
            key = project_tile_box(edge_tile, (x, 0, x + w, h, score), width, height, TILE)
            pixels = np.ones((h, w), dtype=np.uint8)
            candidates.append((key, PackedMask((x, 0), pixels.shape, np.packbits(pixels))))
        image = np.zeros((height, width), dtype=np.uint8)
        with patch("inference_service.pipeline.tile_masks", side_effect=lambda tile, *_: iter(
            candidates if tile == edge_tile else ()
        )):
            with patch("inference_service.pipeline.WINDOWS", (inside_tile, edge_tile)):
                holes = segment(image, 0, FakeRunner())
            with patch("inference_service.pipeline.WINDOWS", (edge_tile,)):
                uncorroborated = segment(image, 0, FakeRunner())
        self.assertEqual(sorted(hole["area"] for hole in holes), [100.0, 270.0])
        self.assertEqual(len(uncorroborated), 3)

    def test_merges_complementary_edge_pixels_into_one_mask_across_zero(self):
        width, height = WORKING_SIZE
        views = [(0, 0), (576, 0)]
        raw = [(35, 90, 120, 130, .92), (590, 90, 640, 130, .76)]
        candidates = {}
        for tile, (left, top, right, bottom, score) in zip(views, raw, strict=True):
            box = project_tile_box(tile, (left, top, right, bottom, score), width, height, TILE)
            pixels = np.ones((bottom - top, right - left), dtype=np.uint8)
            candidates[tile] = [(box, PackedMask((left, top), pixels.shape, np.packbits(pixels)))]
        with patch("inference_service.pipeline.WINDOWS", views), patch(
            "inference_service.pipeline.tile_masks", side_effect=lambda tile, *_: iter(candidates[tile]),
        ):
            holes = segment(np.zeros((height, width), dtype=np.uint8), 0, FakeRunner())
        self.assertEqual([hole["area"] for hole in holes], [4240.0])
        self.assertEqual(holes[0]["center"], [67.0, 110.0])

    def test_restores_display_cut_before_inference_and_returns_one_wrapped_hole(self):
        width, height = WORKING_SIZE
        canonical = np.zeros((height, width), dtype=np.uint8)
        canonical[20:32, 108:126] = 255
        with patch("inference_service.pipeline.WINDOWS", ((0, 0), (288, 0), (864, 0))):
            original, = segment(canonical, 0, FakeRunner())
            wrapped, = segment(np.roll(canonical, -117, axis=1), 117, FakeRunner())

        self.assertAlmostEqual(wrapped["score"], 0.9)
        self.assertEqual(original["center"], [117.0, 26.0])
        self.assertEqual(wrapped["center"], [0.0, 26.0])
        self.assertEqual(wrapped["area"], 216.0)
        self.assertEqual(len(original["contours"]), 1)
        self.assertEqual(len(wrapped["contours"]), 2)
        for contour in wrapped["contours"]:
            self.assertGreaterEqual(len(contour), 3)
            self.assertTrue(all(0 <= x < width and 0 <= y < height for x, y in contour))
            self.assertLessEqual(max(x for x, _ in contour) - min(x for x, _ in contour), 9)
        self.assertEqual(min(x for contour in wrapped["contours"] for x, _ in contour), 0.0)

    def test_fractional_scale_splits_even_one_mask_pixel_at_display_cut(self):
        box = project_tile_box((864, 6860), (407, 639, 408, 640, 0.9), *WORKING_SIZE, TILE)
        hole = display_hole(box, np.ones((1, 1), dtype=np.uint8), (407, 639), 1536, 10000, 159)

        self.assertAlmostEqual(hole["center"][0], 1 / 3)
        self.assertAlmostEqual(hole["center"][1], 9999 + 1 / 3)
        self.assertAlmostEqual(hole["area"], 16 / 9)
        self.assertEqual(len(hole["contours"]), 2)
        for contour in hole["contours"]:
            self.assertEqual(len(contour), 4)
            self.assertTrue(all(0 <= x < 1536 and 0 <= y < 10000 for x, y in contour))


@unittest.skipIf(TestClient is None, "Install inference service and HTTP test dependencies")
class HTTPTests(unittest.TestCase):
    def setUp(self):
        from inference_service.app import create_app

        self.loaded = []

        def load(backend, _model_dir):
            self.loaded.append(backend)
            if backend == "gpu":
                raise RuntimeError("CUDA test failure")
            return FakeRunner()

        self.app = create_app(("cpu", "gpu"), Path("unused"), load_runner=load)
        ok, encoded = cv2.imencode(".png", np.zeros((320, 128), dtype=np.uint8))
        self.assertTrue(ok)
        self.png = encoded.tobytes()
        self.header = {"content-type": "image/png"}

    def test_health_reports_initialized_failed_and_disabled_backends(self):
        with TestClient(self.app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok")
            self.assertTrue(data["backends"]["cpu"]["available"])
            self.assertEqual(data["backends"]["cpu"]["model_id"], FakeRunner.model_id)
            self.assertFalse(data["backends"]["gpu"]["available"])
            self.assertIn("CUDA test failure", data["backends"]["gpu"]["detail"])
            self.assertFalse(data["backends"]["tensorrt"]["available"])
            error = client.post("/v1/segment?backend=gpu&cut_x=0", content=self.png, headers=self.header)
            self.assertEqual(error.status_code, 503)
            self.assertIn("CUDA test failure", error.json()["detail"])
        self.assertEqual(self.loaded, ["cpu", "gpu"])

    def test_health_is_unavailable_if_no_model_passed_startup(self):
        from inference_service.app import create_app

        with TestClient(create_app(("cpu",), Path("missing-assets"))) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["status"], "unavailable")
            self.assertIn("missing", response.json()["backends"]["cpu"]["detail"])

    def test_segment_schema_and_validation(self):
        path = "/v1/segment?backend=cpu&cut_x=0"
        with TestClient(self.app) as client, patch("inference_service.pipeline.WINDOWS", ((0, 0),)):
            response = client.post(path, content=self.png, headers=self.header)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(set(data), {"width", "height", "backend", "elapsed_ms", "holes", "model_id"})
            self.assertEqual((data["width"], data["height"], data["backend"]), (128, 320, "cpu"))
            self.assertEqual(data["model_id"], FakeRunner.model_id)
            self.assertGreaterEqual(data["elapsed_ms"], 0)
            self.assertEqual(data["holes"], [])
            self.assertEqual(client.post(path, content=self.png).status_code, 415)
            self.assertEqual(client.post(path, content=b"not png", headers=self.header).status_code, 400)
            self.assertEqual(client.post(path, content=self.png[:33], headers=self.header).status_code, 400)
            huge = self.png[:16] + struct.pack(">II", 100000, 100000) + self.png[24:]
            self.assertEqual(client.post(path, content=huge, headers=self.header).status_code, 413)
            for query in ("backend=cpu", "backend=cpu&cut_x=128", "backend=cpu&cut_x=1.5",
                          "backend=cpu&cut_x=0&cut_x=1", "backend=unknown&cut_x=0"):
                bad = client.post("/v1/segment?" + query, content=self.png, headers=self.header)
                self.assertEqual(bad.status_code, 422, query)
            with patch("inference_service.app.MAX_BODY_BYTES", len(self.png) - 1):
                self.assertEqual(client.post(path, content=self.png, headers=self.header).status_code, 413)
                chunks = iter((self.png[:20], self.png[20:]))
                self.assertEqual(client.post(path, content=chunks, headers=self.header).status_code, 413)


if __name__ == "__main__":
    unittest.main()
