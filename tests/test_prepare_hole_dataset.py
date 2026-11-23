import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare_hole_dataset import Box, background_subset, build, crop_ring, grid, read_labels, seam_mask, tile_labels


class HoleDataPreparationTest(unittest.TestCase):
    config = dict(size=40, step_x=30, step_y=25, min_fraction=.5, min_visible_side_px=4)

    def test_rounding_is_tolerated_but_real_out_of_bounds_and_wrong_classes_are_not(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'sample.txt'
            f.write_text('0 0.971680 0.747500 0.056641 0.010400\n', encoding='utf-8')
            line, b, rounding = read_labels(f, 1536, 10000)[0]
            self.assertEqual((line, b.r), (1, 1536))
            self.assertAlmostEqual(rounding, 0.000768, places=7)
            for bad in ('0 0.99 0.1 0.5 0.1', '1 0.5 0.5 0.1 0.1', '0 nan 0.5 0.1 0.1',
                        '0 0.5 0.5 0 0.2', '0 0.5 0.5 0.1 0.1 0.8'):
                f.write_text(bad, encoding='utf-8')
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    read_labels(f, 1536, 10000)

    def test_ring_pixels_and_corresponding_label_wrap_horizontally_only(self):
        source = np.tile(np.arange(100, dtype=np.uint8), (70, 1))
        crop = crop_ring(source, 80, 10, 40, 40)
        np.testing.assert_array_equal(crop[0], np.r_[80:100, 0:20])
        role, rows = tile_labels(80, 10, 100, self.config, [(17, Box(5, 20, 17, 37))], [])
        self.assertEqual(role, 'positive')
        self.assertEqual(rows, [{'annotation_id': 17, 'bbox': [25, 10, 37, 27], 'visible_fraction': 1}])
        with self.assertRaises(ValueError):
            crop_ring(source, 80, 50, 40, 40)

    def test_partial_objects_either_have_a_box_or_reject_the_whole_tile(self):
        anns = [(4, Box(10, 10, 30, 30))]
        role, rows = tile_labels(20, 0, 100, self.config, anns, [])
        self.assertEqual((role, rows[0]['visible_fraction']), ('positive', .5))
        self.assertEqual(rows[0]['bbox'], [0, 10, 10, 30])
        role, rows = tile_labels(22, 0, 100, self.config, anns + [(5, Box(35, 11, 47, 20))], [])
        self.assertEqual((role, rows), ('sliver', []))

    def test_review_excludes_both_sides_of_seam_but_not_other_heights(self):
        mask = seam_mask(Box(1, 300, 75, 420), 1536, 10000)
        for x in (0, 1500):
            self.assertEqual(tile_labels(x, 300, 1536, self.config, [], [mask])[0], 'review')
            self.assertEqual(tile_labels(x, 1000, 1536, self.config, [], [mask])[0], 'negative')

    def test_group_grid_has_no_axial_gap_and_background_selection_is_reproducible(self):
        pos = [(1, Box(9, 12, 23, 32))]
        vals, _ = grid('sample', 'val', 100, 85, self.config, pos, [])
        self.assertIn(85-self.config['size'], {t['y'] for t in vals})
        neg = [t for t in vals if not t['labels']]
        self.assertEqual(background_subset(neg, 2, 'sample'), background_subset(list(reversed(neg)), 2, 'sample'))
        self.assertEqual(background_subset(neg, 0, 'sample'), [])

    def test_published_package_cannot_be_overwritten_or_written_into_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root/'input'; source.mkdir(); out = root/'ready'; out.mkdir()
            profile = root/'profile'; profile.write_text('{}')
            for target in (out, source/'nested'):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    build(source, target, profile)
            self.assertEqual(list(source.iterdir()), [])
