import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from tube_unwrap.geometry import CameraCalibration, locate_body, project_contour, source_coordinates
from tube_unwrap_bridge import TubeUnwrapBridge


class TubeLocalizationTest(unittest.TestCase):
    def setUp(self):
        self.bridge = TubeUnwrapBridge()
        self.bridge.config["output_height"] = 240
        self.edges = ((613, 1010), (760, 1174), (743, 1130), (555, 953))
        self.frames = [np.full((2000, 1792), 40, dtype=np.uint8) for _ in self.edges]
        for frame, (left, right) in zip(self.frames, self.edges):
            frame[210:1700, left:right] = 130

    def unwrap(self):
        return self.bridge.process_packets({
            slot: SimpleNamespace(width=frame.shape[1], height=frame.shape[0], data=frame.tobytes())
            for slot, frame in enumerate(self.frames)
        })

    def test_unwraps_tube_when_edges_move_outside_calibrated_windows(self):
        result = self.unwrap()

        self.assertTrue(result.ok, result.message)
        self.assertEqual(result.image.shape, (240, 1536))
        self.assertAlmostEqual(result.bounds[0][0], self.edges[0][0], delta=2)
        self.assertAlmostEqual(result.bounds[2][1], self.edges[2][1], delta=2)

    def test_parallel_localization_preserves_pixels_bounds_and_camera_order(self):
        # Put distinct markings in the views so swapping healthy camera pairs cannot pass.
        for i, (frame, (left, right)) in enumerate(zip(self.frames, self.edges)):
            x, y = (left + right) // 2, 400 + i * 140
            frame[y:y + 22, x - 12:x + 12] = 5
        serial = self.unwrap()
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.bridge.localization_executor = pool
            parallel = self.unwrap()
        closed_pool = self.unwrap()  # A save already in flight during shutdown still completes.
        self.assertTrue(serial.ok, serial.message)
        self.assertGreater(serial.hole_count, 0)
        for result in (parallel, closed_pool):
            self.assertTrue(result.ok, result.message)
            self.assertEqual((result.bounds, result.hole_count), (serial.bounds, serial.hole_count))
            np.testing.assert_array_equal(result.image, serial.image)

    def test_parallel_failure_identifies_camera_in_second_pair(self):
        self.frames[2].fill(40)
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.bridge.localization_executor = pool
            result = self.unwrap()
        self.assertFalse(result.ok)
        self.assertIn("Camera 3", result.message)

    def test_unwraps_a_tube_that_drifts_sideways_along_its_length(self):
        frame = self.frames[0]
        frame.fill(40)
        for row in range(210, 1700):
            shift = (row - 210) // 25
            frame[row, 613 + shift:1010 + shift] = 130

        result = self.unwrap()

        self.assertTrue(result.ok, result.message)

    def test_unwraps_a_low_contrast_tube_within_or_outside_calibrated_windows(self):
        frame = self.frames[0]
        for left, right in ((507, 997), (613, 1010)):
            with self.subTest(left=left, right=right):
                frame.fill(40)
                frame[210:1700, left:right] = 54

                result = self.unwrap()

                self.assertTrue(result.ok, result.message)
                self.assertAlmostEqual(result.bounds[0][0], left, delta=2)
                self.assertAlmostEqual(result.bounds[0][1], right, delta=2)
                self.assertEqual(int(np.median(result.image[:, :100])), 54)

    def test_side_edge_follows_the_tube_past_a_strong_internal_stripe(self):
        frame = np.full((2000, 820), 40, dtype=np.uint8)
        frame[210:1700, 230:650] = 60
        frame[720:1050, 580:620] = 5

        right = locate_body(frame, CameraCalibration(230, 650, 210, 1700, 0, search_x=100))
        left = locate_body(frame[:, ::-1], CameraCalibration(170, 590, 210, 1700, 0, search_x=100))

        np.testing.assert_allclose(right.right[820:950], 649, atol=2)
        np.testing.assert_allclose(left.left[820:950], 170, atol=2)

    def test_side_edge_recovers_a_long_weak_edge_and_preserves_a_real_notch(self):
        frame = np.full((5600, 820), 40, dtype=np.uint8)
        calibration = CameraCalibration(230, 650, 210, 5350, 0, search_x=100)
        for tube, outer, inner, expected_right in ((60, 50, 5, 649), (60, 40, 40, 579), (80, 50, 50, 579)):
            with self.subTest(tube=tube, outer=outer, inner=inner):
                frame[210:5350, 230:650] = tube
                frame[2200:2800, 580:650] = outer
                frame[2200:2800, 580:610] = inner

                body = locate_body(frame, calibration)

                np.testing.assert_allclose(body.right[2330:2670], expected_right, atol=2)

    def test_side_edge_recovers_a_sustained_small_inward_shift(self):
        frame = np.full((5600, 820), 40, dtype=np.uint8)
        frame[210:5350, 230:650] = 60
        frame[2200:2800, 632:642] = 5

        body = locate_body(frame, CameraCalibration(230, 650, 210, 5350, 0, search_x=100))

        np.testing.assert_allclose(body.right[2330:2670], 649, atol=2)

    def test_side_edge_bridges_a_short_gap_between_verified_samples(self):
        frame = np.full((2000, 820), 40, dtype=np.uint8)
        frame[210:1700, 230:650] = 60
        frame[720:1050, 580:620] = 5
        frame[875:888, 635:650] = 40

        body = locate_body(frame, CameraCalibration(230, 650, 210, 1700, 0, search_x=100))

        np.testing.assert_allclose(body.right[865:898], 649, atol=2)

    def test_projection_keeps_scale_and_pipe_motion_through_a_one_sided_width_drop(self):
        frame = np.full((2600, 820), 40, dtype=np.uint8)
        for row in range(210, 2350):
            shift = (row - 210) // 75
            right = 610 if 820 <= row < 960 else 650
            frame[row, 230 + shift:right + shift] = 100
        calibration = CameraCalibration(230, 650, 210, 2350, 500, search_x=100)
        body = locate_body(frame, calibration)
        trim = 5.0
        rows = np.array([700, 890, 1150])
        fractions = (rows - body.top - trim) / (body.bottom - body.top - 2 * trim)

        map_x, map_y = source_coordinates(body, calibration, np.array([0.0, 0.6]), fractions, trim)
        contour, _ = project_contour(np.column_stack((map_x[:, 0], map_y[:, 0])),
                                     body, calibration, 1536, 10000, trim)

        self.assertLess(body.right[890] - body.left[890], 390)
        np.testing.assert_allclose(map_x[:, 0], 439.5 + (rows - 210) // 75, atol=2)
        np.testing.assert_allclose(map_x[:, 1] - map_x[:, 0], 126, atol=2)
        np.testing.assert_allclose(contour[:, 0], 500, atol=0.1)

    def test_refuses_to_unwrap_when_one_camera_lacks_a_complete_tube(self):
        cases = (
            (),
            ((slice(210, 1700), slice(None)),),
            ((slice(210, 1700), slice(620, None)),),
            ((slice(210, 600), slice(620, 1020)), (slice(1200, 1700), slice(620, 1020))),
        )
        for regions in cases:
            with self.subTest(regions=regions):
                self.frames[0].fill(40)
                for region in regions:
                    self.frames[0][region] = 130

                result = self.unwrap()

                self.assertEqual(result.status, "review")
                self.assertIsNone(result.image)
