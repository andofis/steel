import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_hole_dataset import Box, load_json, read_labels
from tools.repair_hole_labels import PROFILE, build, check_refit_rows, choose_cut, index_patches, planar, save_labels, xml_inventory


class RepairLabelsTest(unittest.TestCase):
    def test_safe_cut_preserves_the_original_tile_grid_and_roundtrips_every_box(self):
        boxes = [Box(-48, 200, 40, 304), Box(1060, 280, 1160, 400)]
        cut = choose_cut(boxes, 1536)
        self.assertIn(cut, (384, 768, 1152))
        for box in boxes:
            out = planar(box, cut, 1536)
            self.assertTrue(0 <= out.l < out.r <= 1536)
            self.assertEqual(out.r - out.l, box.r - box.l)
            self.assertEqual((out.l + cut) % 1536, box.l % 1536)
            self.assertEqual((out.t, out.b), (box.t, box.b))

    def test_no_roll_or_global_translation_when_there_is_no_broken_hole(self):
        boxes = [Box(0, 890, 96, 990), Box(1420, 110, 1536, 240)]
        self.assertEqual(choose_cut(boxes, 1536), 0)
        self.assertEqual([planar(b, 0, 1536) for b in boxes], boxes)

    def test_fail_instead_of_truncating_or_writing_a_giant_seam_box(self):
        with self.assertRaises(ValueError):
            planar(Box(-20, 200, 40, 300), 0, 1536)
        with self.assertRaises(ValueError):
            choose_cut([Box(x-10, 0, x+10, 50) for x in (0, 384, 768, 1152)], 1536)

    def test_written_yolo_box_really_covers_the_added_b24_hole(self):
        repairs = load_json(PROFILE)
        patches = index_patches(load_json(PROFILE.with_name(repairs['base_profile'])), repairs)
        box = Box(*patches['B24', None]['ring_xyxy'])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'B24.txt';cut = 1152
            save_labels(target, [planar(box, cut, 1536)], 1536, 10000)
            _, saved, _ = read_labels(target, 1536, 10000)[0]
            self.assertLess(saved.l, (465-cut)%1536)
            self.assertGreater(saved.r, (465-cut)%1536)
            self.assertTrue(saved.t < 2602 < saved.b)
        self.assertNotIn(('AOX5', None), patches)

    def test_aox1_fifth_saved_box_fits_the_measured_opening(self):
        fixes = load_json(PROFILE)
        patches = index_patches(load_json(PROFILE.with_name(fixes['base_profile'])), fixes)
        # Row 5 of the source TXT and the measured dark extent in the recut PNG (exclusive xyxy).
        with tempfile.TemporaryDirectory() as folder:
            txt = Path(folder)/'AOX1.txt'
            txt.write_text('0 0.041667 0.878050 0.059896 0.019300\n', encoding='utf-8')
            original = read_labels(txt, 1536, 10000)[0][1]
            patch = patches.get(('AOX1', 5))
            ring = Box(*patch['ring_xyxy']) if patch else original
            save_labels(txt, [planar(ring, 384, 1536)], 1536, 10000)
            saved = read_labels(txt, 1536, 10000)[0][1]
        margins = (1150-saved.l, 8700-saved.t, saved.r-1232, saved.b-8795)
        self.assertTrue(all(1 <= m <= 5 for m in margins), f'dark opening margins: {margins}')

    def test_refit_rejects_missing_or_different_original_row(self):
        repairs = load_json(PROFILE)
        patches = index_patches(load_json(PROFILE.with_name(repairs['base_profile'])), repairs)
        for old in ({}, {5: Box(19, 8684, 111, 8877)}):
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_refit_rows('AOX1', old, patches)
        check_refit_rows('AOX1', {5: Box(18.000384, 8684, 110.00064, 8877)}, patches)

    def test_empty_voc_not_mistaken_for_yolo_and_existing_output_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'raw'; source.mkdir()
            (source/'a.xml').write_text('<annotation><filename>a.png</filename></annotation>')
            self.assertEqual(xml_inventory(source)[0]['objects'], 0)
            out = root/'v2'; out.mkdir(); sentinel = out/'keep.json'; sentinel.write_text('[1]')
            with self.assertRaises(ValueError):
                build(source, out)
            self.assertEqual(json.loads(sentinel.read_text()), [1])
