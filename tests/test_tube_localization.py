import unittest
from types import SimpleNamespace

import numpy as np

from tube_unwrap.geometry import CameraCalibration, locate_body
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
