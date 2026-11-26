import json
import unittest
from collections import Counter
from copy import deepcopy

import cv2
import numpy as np

from pipe_alignment import inspect_pipe, validate_tolerances


CENTERS = [(.980, .150), (.112, .218), (.265, .285), (.391, .452),
           (.527, .496), (.643, .613), (.805, .756), (.903, .892)]
OUTLINE = [(-.013, -.009), (.014, -.009), (.005, .010), (-.010, .010)]


def pipe(centers=CENTERS, *, width=1000, height=1000, cut=0,
         flipped=False, roll=0, axial=0, wrap_contours=False):
    holes = []
    sign = -1 if flipped else 1
    for index, (x, y) in enumerate(centers):
        cx = ((sign * x + roll) * width - cut) % width
        cy = ((1 - y if flipped else y) + axial) * height
        points = [[cx + sign * dx * width, cy + sign * dy * height]
                  for dx, dy in OUTLINE]
        if wrap_contours:
            for point in points:
                point[0] %= width
        holes.append(dict(center=[cx, cy], contours=[points], score=.98,
                          area=(.00042 + index * .00001) * width * height))
    return dict(width=width, height=height, cut_x=cut, holes=holes)


class InspectPipeTest(unittest.TestCase):
    def test_surface_corrects_flip_warp_without_absorbing_a_shifted_hole(self):
        width, height = 1152, 7500
        rng = np.random.default_rng(2026)
        gray = cv2.GaussianBlur(rng.integers(20, 235, (height, width), dtype=np.uint8),
                                (0, 0), 2.0)
        template = pipe(width=width, height=height)
        sample = pipe(width=width, height=height, flipped=True, roll=.125)

        def axial_displacement(y):
            return 125 * (1 - ((y - height / 2) / (height / 2)) ** 2)

        ys, xs = np.indices(gray.shape, dtype=np.float32)
        source_y = ys
        for _ in range(3):
            source_y = ys - axial_displacement(source_y)
        image = cv2.remap(gray, (width / 8 - xs) % width, height - 1 - source_y,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        for hole in sample['holes']:
            dy = axial_displacement(hole['center'][1])
            hole['center'][1] += dy
            for x_y in hole['contours'][0]:
                x_y[1] += dy
        self.assertNotEqual(inspect_pipe(template, sample, {})['status'], 'pass')
        pngs = tuple(cv2.imencode('.png', raw)[1].tobytes() for raw in (gray, image))

        registered = inspect_pipe(template, sample, {}, surface_pngs=pngs)
        self.assertEqual(registered['status'], 'pass', registered['reason'])
        self.assertTrue(registered['alignment']['flipped'])
        self.assertGreater(registered['alignment']['surface_anchors'], 30)
        limited = inspect_pipe(template, sample, {'max_rotation_deg': 25}, surface_pngs=pngs)
        self.assertReview(limited, '周向旋转')

        moved = sample['holes'][3]
        moved['center'][0] += 18
        for x_y in moved['contours'][0]:
            x_y[0] += 18
        result = inspect_pipe(template, sample, {}, surface_pngs=pngs)
        self.assertEqual(result['status'], 'fail', result['reason'])
        self.assertEqual(Counter(row['kind'] for row in result['findings']), {'ok': 7, 'shifted': 1})

    def test_extra_at_the_pipe_end_requires_the_same_foreground_on_the_reference(self):
        width, height = 1152, 7500
        rng = np.random.default_rng(37)
        wall = cv2.GaussianBlur(rng.integers(85, 235, (height, width), dtype=np.uint8),
                                (0, 0), 2.0)
        drilled = wall.copy()
        contour = [[330, 7469], [350, 7469], [356, 7499], [327, 7499]]
        cv2.fillPoly(drilled, [np.asarray(contour)], 25)
        template = pipe(width=width, height=height)
        sample = deepcopy(template)
        sample['holes'].append({'center': [341, 7485], 'contours': [contour], 'area': 665, 'score': .9})
        clean, marked = (cv2.imencode('.png', image)[1].tobytes() for image in (wall, drilled))

        repeated = inspect_pipe(template, sample, {}, surface_pngs=(marked, marked))
        self.assertEqual(repeated['status'], 'pass', repeated['reason'])
        self.assertEqual(repeated['alignment']['confirmed_end_marks'], 1)
        new_opening = inspect_pipe(template, sample, {}, surface_pngs=(clean, marked))
        self.assertEqual(new_opening['status'], 'fail', new_opening['reason'])
        self.assertEqual([f['kind'] for f in new_opening['findings'] if f['kind'] != 'ok'], ['extra'])

    def assertReview(self, result, reason):
        self.assertEqual(result['status'], 'review', result)
        self.assertIn(reason, result['reason'].lower())
        self.assertEqual(result['findings'], [])

    def test_flip_roll_axial_and_different_cuts_and_grids(self):
        template = pipe(cut=250, wrap_contours=True)
        sample = pipe(width=1200, height=1500, cut=610, flipped=True, roll=.49, axial=.03)
        sample['holes'].reverse()
        original = deepcopy(sample)

        result = inspect_pipe(template, sample, dict(max_rotation_deg=177, max_axial_pct=3,
                                                    shape_iou=.95))

        self.assertEqual(result['status'], 'pass', result)
        self.assertTrue(result['alignment']['flipped'])
        self.assertAlmostEqual(result['alignment']['rotation_deg'], 176.4)
        self.assertAlmostEqual(result['alignment']['axial_pct'], 3)
        self.assertEqual(result['alignment']['inliers'], 8)
        self.assertEqual(Counter(f['kind'] for f in result['findings']), {'ok': 8})
        seam = min(result['findings'], key=lambda finding: finding['center'][0])
        self.assertAlmostEqual(seam['center'][0], 2)
        self.assertAlmostEqual(seam['center'][1], 1320)
        self.assertTrue(any(x > 1180 for x, _ in seam['contours'][0]))
        self.assertTrue(any(x < 20 for x, _ in seam['contours'][0]))
        self.assertTrue(all(0 <= x < 1200 for x, _ in seam['contours'][0]))
        self.assertEqual(sample, original)
        json.dumps(result, allow_nan=False)

    def test_display_seam_does_not_count_as_physical_rotation(self):
        result = inspect_pipe(pipe(cut=90), pipe(cut=640), dict(max_rotation_deg=0))
        self.assertEqual(result['status'], 'pass', result)
        self.assertAlmostEqual(result['alignment']['rotation_deg'], 0)

    def test_split_contours_at_the_display_seam_compare_as_one_mask(self):
        template, sample = pipe(), pipe(cut=980)
        template['holes'][0]['contours'] = [[[970, 140], [992, 140], [992, 160], [970, 160]]]
        sample['holes'][0]['contours'] = [[[0, 140], [12, 140], [12, 160], [0, 160]],
                                         [[990, 140], [999.999, 140], [999.999, 160], [990, 160]]]
        result = inspect_pipe(template, sample, dict(shape_iou=1))
        self.assertEqual(result['status'], 'pass', result)

    def test_subpixel_center_in_the_last_row_is_valid_and_can_flip(self):
        centers = CENTERS + [(.572, .9998)]
        template = pipe(centers)
        sample = pipe(centers, height=500, flipped=True)
        for panorama in (template, sample):
            panorama['holes'][-1]['contours'] = []
        result = inspect_pipe(template, sample, {})
        self.assertEqual(result['status'], 'pass', result)
        self.assertAlmostEqual(result['findings'][-1]['center'][1], .1)

    def test_reports_actual_defect_contours_and_projects_missing_through_flip(self):
        template = pipe(cut=250)
        sample = pipe(width=1200, height=1500, cut=610, flipped=True, roll=.49, axial=.03)
        sample['holes'].pop(0)
        shifted = sample['holes'][1]
        shifted['center'][0] += 12
        shifted['center'][1] += 3
        for point in shifted['contours'][0]:
            point[0] += 12
            point[1] += 3
        extra = pipe([(.34, .45)], width=1200, height=1500)['holes'][0]
        sample['holes'].append(extra)

        result = inspect_pipe(template, sample, {})

        self.assertEqual(result['status'], 'fail', result)
        self.assertEqual(Counter(f['kind'] for f in result['findings']),
                         {'ok': 6, 'shifted': 1, 'missing': 1, 'extra': 1})
        findings = {f['kind']: f for f in result['findings']}
        self.assertEqual(findings['shifted']['center'], shifted['center'])
        self.assertEqual(findings['shifted']['contours'], shifted['contours'])
        self.assertAlmostEqual(findings['shifted']['reference_center'][0], 860)
        self.assertAlmostEqual(findings['shifted']['reference_center'][1], 1117.5)
        self.assertAlmostEqual(findings['missing']['center'][0], 2)
        self.assertAlmostEqual(findings['missing']['center'][1], 1320)
        self.assertAlmostEqual(findings['missing']['contours'][0][0][0], 17.6)
        self.assertAlmostEqual(findings['missing']['contours'][0][1][0], 1185.2)
        self.assertEqual(findings['extra']['center'], extra['center'])
        self.assertEqual(findings['extra']['contours'], extra['contours'])

    def test_counts_size_and_centered_shape_changes_after_registration(self):
        template, sample = pipe(), pipe(roll=.04, axial=-.02)
        sample['holes'][1]['area'] *= 1.6
        shape = sample['holes'][3]
        x, y = shape['center']
        shape['contours'] = [[[x - 3, y - 32], [x + 3, y - 32],
                              [x + 3, y + 32], [x - 3, y + 32]]]
        result = inspect_pipe(template, sample, {})
        self.assertEqual(result['status'], 'fail', result)
        self.assertEqual(result['alignment']['inliers'], 8)
        self.assertEqual(Counter(f['kind'] for f in result['findings']),
                         {'ok': 6, 'size': 1, 'shape': 1})
        actual = next(f for f in result['findings'] if f['kind'] == 'shape')
        self.assertEqual(actual['contours'], shape['contours'])
        self.assertNotEqual(actual['contours'], actual['reference_contours'])

    def test_pose_does_not_drift_to_hide_a_borderline_shift(self):
        template = pipe()
        centers = list(CENTERS)
        centers[2] = (.2675, .285)  # ordinary center variation within tolerance
        centers[4] = (.5339, .496)  # beyond tolerance from the consensus
        result = inspect_pipe(template, pipe(centers), {})
        self.assertEqual(result['status'], 'fail', result)
        self.assertEqual(Counter(f['kind'] for f in result['findings']), {'ok': 7, 'shifted': 1})
        self.assertAlmostEqual(result['alignment']['rotation_deg'], 0)

    def test_out_of_bounds_pose_is_review_even_if_otherwise_exact(self):
        cases = [(pipe(roll=-.22, cut=820), dict(max_rotation_deg=45), '周向旋转'),
                 (pipe(axial=.07), dict(max_axial_pct=5), '轴向偏移')]
        for sample, tolerances, reason in cases:
            with self.subTest(reason=reason):
                self.assertReview(inspect_pipe(pipe(cut=800), sample, tolerances), reason)

    def test_axial_coordinates_do_not_wrap(self):
        sample = pipe()
        end = sample['holes'][0]
        end['center'][1] += 800
        for point in end['contours'][0]:
            point[1] += 800
        result = inspect_pipe(pipe(), sample, {})
        self.assertEqual(result['status'], 'fail', result)
        self.assertEqual(Counter(f['kind'] for f in result['findings']),
                         {'ok': 7, 'missing': 1, 'extra': 1})

    def test_opening_projected_past_sample_end_requires_review(self):
        centers = CENTERS[:-1] + [(.903, .985)]
        result = inspect_pipe(pipe(centers), pipe(CENTERS[:-1], axial=.02), {})
        self.assertReview(result, '未覆盖')

    def test_orientation_and_nearly_repeated_poses_require_review(self):
        symmetric = [(0.1, 0.17), (0.27, 0.34), (.32, .45)]
        symmetric += [((-x) % 1, 1 - y) for x, y in symmetric]
        repeated = [(x, .3 + i * .0002) for i, x in enumerate([.08, .19, .30, .41, .52, .63])]
        for centers in (symmetric, repeated):
            with self.subTest(centers=centers):
                self.assertReview(inspect_pipe(pipe(centers), pipe(centers), {}), '无法确定唯一')

    def test_insufficient_consensus_never_becomes_defect_findings(self):
        cases = [pipe([]), pipe(CENTERS[:2]),
                 pipe(CENTERS[:4] + [(x, 1 - y) for x, y in CENTERS[4:]]),
                 pipe([(.05, .31)] * 8)]
        for sample in cases:
            with self.subTest(holes=len(sample['holes'])):
                result = inspect_pipe(pipe(), sample, {})
                self.assertReview(result, '对齐')
                self.assertIsNone(result['alignment'])

    def test_competing_shifted_candidates_require_review(self):
        sample = pipe(CENTERS[:-1])
        sample['holes'].extend(pipe([(.892, .892), (.914, .892)])['holes'])
        self.assertReview(inspect_pipe(pipe(), sample, {}), '对应关系')

    def test_weak_sample_matches_can_corroborate_geometry_but_do_not_anchor(self):
        template, sample = pipe(), pipe()
        for hole in template['holes']:
            hole['score'] = .1
        sample['holes'][0]['score'] = .2
        result = inspect_pipe(template, sample, {})
        self.assertEqual(result['status'], 'pass', result)
        self.assertEqual(result['alignment']['inliers'], 7)

        sample['holes'][0]['area'] *= 1.6
        self.assertReview(inspect_pipe(template, sample, {}), '置信度不足')
        for hole in sample['holes']:
            hole['score'] = .2
        result = inspect_pipe(template, sample, {})
        self.assertReview(result, '置信度')
        self.assertIsNone(result['alignment'])

    def test_a_weak_unmatched_detection_cannot_establish_an_extra_hole(self):
        sample = pipe()
        extra = pipe([(.32, .73)])['holes'][0]
        extra['score'] = .28
        sample['holes'].append(extra)
        self.assertReview(inspect_pipe(pipe(), sample, {}), '未匹配孔洞的置信度不足')

    def test_sparse_contours_report_skipped_shape(self):
        sample = pipe()
        sample['holes'][0]['contours'] = []
        result = inspect_pipe(pipe(), sample, {})
        self.assertEqual(result['status'], 'pass', result)
        self.assertIn('未比较形状', result['reason'])

    def test_tolerance_defaults_and_validation(self):
        self.assertEqual(validate_tolerances({'max_rotation_deg': 0, 'shape_iou': .7}),
                         dict(max_rotation_deg=0, max_axial_pct=5, position_pct=.5,
                              size_pct=35, shape_iou=.7))
        for endpoints in (dict(max_rotation_deg=0, max_axial_pct=0, position_pct=.01,
                               size_pct=0, shape_iou=0),
                          dict(max_rotation_deg=180, max_axial_pct=25, position_pct=5,
                               size_pct=100, shape_iou=1)):
            self.assertEqual(validate_tolerances(endpoints), endpoints)
        for bad in ([], {'unknown': .5}, {'max_rotation_deg': 181}, {'max_axial_pct': -1},
                    {'max_axial_pct': 26}, {'position_pct': '0.5'}, {'position_pct': 0},
                    {'position_pct': 5.1}, {'size_pct': True}, {'shape_iou': float('nan')},
                    {'shape_iou': 1.1}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_tolerances(bad)
                self.assertReview(inspect_pipe(pipe(), pipe(), bad), '检测阈值')

    def test_invalid_geometry_requires_review(self):
        for name, value in (('height', True), ('width', 1), ('cut_x', 1000), ('holes', None)):
            with self.subTest(name=name):
                sample = pipe()
                sample[name] = value
                self.assertReview(inspect_pipe(pipe(), sample, {}), '待检钢管')
        sample = pipe()
        sample['holes'][0]['center'][1] = float('nan')
        self.assertReview(inspect_pipe(pipe(), sample, {}), '待检钢管')


if __name__ == '__main__':
    unittest.main()
