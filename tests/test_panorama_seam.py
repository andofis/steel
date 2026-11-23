import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from tube_unwrap_bridge import TubeUnwrapBridge


class PanoramaSeamTest(unittest.TestCase):
    def test_acquisition_output_joins_both_halves_without_changing_pixels_or_hole_count(self):
        bridge = TubeUnwrapBridge()
        bridge.config['output_height'] = 1025
        frames = {}
        for i, camera in enumerate(bridge.config['cameras']):
            frame = np.full((1500, 1792), 40, np.uint8)
            l, r = camera['left'], camera['right']
            frame[200:1251, l:r+1] = 155
            cv2.circle(frame, ((l+r)//2, 350+i*185), 22, 7, -1)
            frames[i] = SimpleNamespace(width=1792, height=1500, data=frame.tobytes())
        for size in [(1536, 1025), (901, 517)]:
            with self.subTest(size=size):
                bridge.config['panorama_seam'] = 'fixed'
                fixed = bridge.process_packets(frames, output_size=size)
                self.assertTrue(fixed.ok, fixed.message)
                rows = np.flatnonzero((fixed.image[:,0] < 30) & (fixed.image[:,-1] < 30))
                self.assertGreater(len(rows), 6)
                y = rows[len(rows)//2]
                bridge.config['panorama_seam'] = 'auto'
                new = bridge.process_packets(frames, output_size=size)
                self.assertTrue(new.ok, new.message)
                self.assertTrue(np.all(new.image[y,[0,-1]] > 65), 'same hole still split across both sides')
                cut = new.panorama_seam.cut_x
                self.assertEqual(new.panorama_seam.status, 'shifted')
                self.assertEqual(new.hole_count, fixed.hole_count)
                np.testing.assert_array_equal(np.roll(new.image, cut, axis=1), fixed.image)
                self.assertEqual(len(np.flatnonzero(np.diff((new.image[y] < 30).astype(int)))), 2)

    def test_gap_across_zero_safe_origin_and_no_possible_global_cut(self):
        from tube_unwrap.panorama_seam import choose_seam
        self.assertEqual(choose_seam(100, [], 8).status, 'no_candidates')
        self.assertEqual(choose_seam(100, [(20,32), (60,70)], 8).cut_x, 0)
        self.assertEqual(choose_seam(100, [(-10,10)], 8).cut_x, 50)
        narrow = choose_seam(100, [(3,98)], 8)
        self.assertEqual(narrow.status, 'limited_clearance')
        self.assertEqual(narrow.crossing_candidates, 0)
        blocked = choose_seam(100, [(0,65), (45,100), (-3,3)], 8)
        self.assertEqual(blocked.status, 'no_clear_column')
        self.assertEqual(blocked.crossing_candidates, 1)
