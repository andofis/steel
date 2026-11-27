import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from tube_unwrap.holes import find_hole_groups
from tube_unwrap_bridge import TubeUnwrapBridge


class CaptureHoleGroupingTest(unittest.TestCase):
    def test_capture_recipe_merges_close_views_across_the_panorama_edge(self):
        frame = np.full((160, 1792), 150, dtype=np.uint8)
        cv2.circle(frame, (80, 80), 40, 20, -1)
        geometry = SimpleNamespace(
            left=np.zeros(160), right=np.full(160, 1791), top=0, bottom=159,
        )
        recipe = TubeUnwrapBridge()._build_recipe((frame,) * 4, ((1, 158),) * 4)

        for separation, expected_groups in ((15.33, 1), (16.1, 2)):
            with self.subTest(separation=separation):
                def project(camera, contour):
                    if camera == 0:
                        return contour + (1450, 200), 0.81
                    return contour + (1450 + separation, 203.68), 0.57

                holes, alternates = find_hole_groups(
                    (frame, frame), (geometry, geometry), project,
                    recipe.width, recipe.height, settings=recipe.hole_settings,
                )

                self.assertEqual(len(holes), expected_groups)
                self.assertEqual(holes[0].camera, 1)
                self.assertEqual(tuple(hole.camera for hole in alternates[0]),
                                 (2,) if expected_groups == 1 else ())

    def test_merges_foreshortened_grazing_view_inside_frontal_span_only(self):
        frontal = np.full((160, 1792), 150, dtype=np.uint8)
        grazing = np.full((160, 1792), 150, dtype=np.uint8)
        cv2.circle(frontal, (80, 80), 40, 20, -1)
        cv2.ellipse(grazing, (80, 80), (20, 38), 0, 0, 360, 20, -1)
        geometry = SimpleNamespace(
            left=np.zeros(160), right=np.full(160, 1791), top=0, bottom=159,
        )
        recipe = TubeUnwrapBridge()._build_recipe((frontal,) * 4, ((1, 158),) * 4)

        # Grazing view shifted +19 px (> merge_tolerance_px=16) sits completely inside frontal [40..120].
        def project_overlapping(camera, contour):
            if camera == 0:
                return contour + (500, 200), 0.89
            return contour + (519, 204), 0.50

        holes, alternates = find_hole_groups(
            (frontal, grazing), (geometry, geometry), project_overlapping,
            recipe.width, recipe.height, settings=recipe.hole_settings,
        )
        self.assertEqual(len(holes), 1)
        self.assertEqual(holes[0].camera, 1)
        self.assertEqual(tuple(hole.camera for hole in alternates[0]), (2,))

        # Distinct smaller hole nearby (center +65 px, outside frontal span) must remain separate.
        def project_distinct(camera, contour):
            if camera == 0:
                return contour + (500, 200), 0.89
            return contour + (565, 204), 0.50

        holes_sep, alternates_sep = find_hole_groups(
            (frontal, grazing), (geometry, geometry), project_distinct,
            recipe.width, recipe.height, settings=recipe.hole_settings,
        )
        self.assertEqual(len(holes_sep), 2)
        self.assertEqual(alternates_sep, ((), ()))
