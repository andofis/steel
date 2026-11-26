import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from hikrobot_camera import CAMERA_NAMES, FramePacket, HikRobotService
from tube_unwrap import TubeUnwrapper, UnwrapStatus
from tube_unwrap.holes import find_hole_groups
from tube_unwrap_bridge import TubeUnwrapBridge
from unwrap_output import OutputSizes


class SizeGeometryTest(unittest.TestCase):
    def test_uses_recalibrated_phases_only_for_nominal_tube_width(self):
        bridge = TubeUnwrapBridge()
        legacy = TubeUnwrapBridge()
        legacy.config.pop("nominal_phase_calibration", None)
        for fraction in (1, .82):
            with self.subTest(width_fraction=fraction):
                frames = []
                for i, camera in enumerate(bridge.config["cameras"]):
                    img = np.full((1000, 1792), 40, np.uint8)
                    center = (camera["left"] + camera["right"]) / 2
                    radius = (camera["right"] - camera["left"]) * fraction / 2
                    left, right = round(center - radius), round(center + radius)
                    img[160:841, left:right + 1] = 155
                    cv2.circle(img, (round(center), 275 + i * 135), 22, 7, -1)
                    frames.append(img)
                bounds = [(160, 840)] * 4
                updated = TubeUnwrapper(bridge._build_recipe(frames, bounds, (1152, 512))).process(frames)
                baseline = TubeUnwrapper(legacy._build_recipe(frames, bounds, (1152, 512))).process(frames)
                for result in (updated, baseline):
                    self.assertEqual(result.status, UnwrapStatus.OK, result.message)
                    self.assertEqual(len(result.holes), 4)
                if fraction == 1:
                    self.assertTrue(getattr(updated, "phase_calibration_id", None))
                    self.assertIsNone(baseline.phase_calibration_id)
                    config = bridge.config["nominal_phase_calibration"]
                    for old, new, phase in zip(baseline.holes, updated.holes, config["phases"], strict=True):
                        self.assertEqual(old.camera, new.camera)
                        difference = (new.x - old.x + 576) % 1152 - 576
                        old_phase = bridge.config["cameras"][old.camera - 1]["phase"]
                        self.assertAlmostEqual(difference, .75 * (phase - old_phase), places=3)
                else:
                    self.assertIsNone(updated.phase_calibration_id)
                    np.testing.assert_array_equal(updated.image, baseline.image)

    def test_direct_grid_preserves_seam_hole_and_normalized_positions(self):
        bridge = TubeUnwrapBridge()
        bridge.config["output_height"] = 1025
        frames = []
        for i, camera in enumerate(bridge.config["cameras"]):
            img = np.full((1500, 1792), 40, np.uint8)
            left, right = camera["left"], camera["right"]
            img[200:1251, left:right + 1] = 155
            cv2.circle(img, ((left + right) // 2, 350 + i * 185), 22, 7, -1)
            if i == 1:
                # Within the camera silhouette, but its outline extends past the trimmed tube end.
                # A coarse last output row must not turn it into a new candidate.
                cv2.circle(img, ((left + right) // 2, 1237), 10, 7, -1)
            frames.append(img)
        bounds = [(200, 1250)] * 4
        base = bridge._build_recipe(frames, bounds)
        original = TubeUnwrapper(base).process(frames)
        self.assertEqual(original.status, UnwrapStatus.OK, original.message)
        self.assertEqual(len(original.holes), 4)
        for w, h in ((1152, 769), (768, 513), (901, 317), (128, 128)):
            with self.subTest(size=(w, h)):
                recipe = bridge._build_recipe(frames, bounds, (w, h))
                sx, sy = w / base.width, (h - 1) / (base.height - 1)
                self.assertEqual(recipe.output_scale, (sx, sy))
                self.assertEqual(recipe.cameras[0].phase, base.cameras[0].phase)
                self.assertEqual(recipe.blend_px, round(base.blend_px * sx))
                self.assertEqual(recipe.trim_px, base.trim_px)  # Source pixels.
                self.assertEqual(recipe.hole_settings.min_diameter_px, base.hole_settings.min_diameter_px)
                result = TubeUnwrapper(recipe).process(frames)
                self.assertEqual(result.status, UnwrapStatus.OK, result.message)
                self.assertEqual(result.image.shape, (h, w))
                self.assertEqual(result.camera_bounds, original.camera_bounds)
                for old, new in zip(original.holes, result.holes, strict=True):
                    self.assertEqual(old.camera, new.camera)
                    self.assertAlmostEqual((new.x - old.x * sx + w / 2) % w - w / 2, 0, delta=.03)
                    self.assertAlmostEqual(old.y * sy, new.y, delta=.03)
                at_seam = next(hole for hole in result.holes if hole.camera == 1)
                self.assertTrue(np.all(result.image[round(at_seam.y), [0, w - 1]] < 55))
        self.assertEqual(bridge.output_sizes.native, (1536, 1025))  # No cumulative scaling.

    def test_nonuniform_grid_preserves_reference_grouping_and_wrap(self):
        frame = np.full((160, 1792), 150, np.uint8)
        cv2.circle(frame, (80, 80), 40, 20, -1)
        body = SimpleNamespace(left=np.zeros(160), right=np.full(160, 1791), top=0, bottom=159)
        recipe = TubeUnwrapBridge()._build_recipe([frame] * 4, [(1, 158)] * 4, (729, 2161))
        for dx, dy, count in ((15.5, 15.5, 1), (16.5, 0, 2), (0, 16.5, 2)):
            with self.subTest(offset=(dx, dy)):
                def project(i, contour):
                    return contour + (1450 + dx * i, 220 + dy * i), 1 - i * .1
                holes, _ = find_hole_groups([frame] * 2, [body] * 2, project, *recipe.reference_grid,
                                            settings=recipe.hole_settings)
                self.assertEqual(len(holes), count)
                hole = holes[0].scaled(*recipe.output_scale)
                self.assertLess(hole.x, recipe.width)
                xs = [x for x, _ in hole.contour]
                self.assertTrue(min(xs) < 0 or max(xs) > recipe.width)


class SizeControlTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def service(self):
        original_load = HikRobotService._load_settings

        def load(service):
            service.settings_path = self.root / "settings.json"
            service.config["output_dir"] = str(self.root / "captures")
            original_load(service)

        with (patch.object(HikRobotService, "_initialize_sdk"), patch.object(HikRobotService, "_log"),
              patch.object(HikRobotService, "_load_settings", load)):
            service = HikRobotService()
        service._log = Mock()
        self.addCleanup(service.executor.shutdown, wait=True)
        self.addCleanup(service.preview_executor.shutdown, wait=True)
        return service

    def test_controls_persist_separately_from_camera_parameters(self):
        s = self.service()
        self.assertEqual(s.unwrap_size, (1152, 7500))
        ref = OutputSizes(1000, 8000)
        self.assertEqual([(p['width'], p['height']) for p in ref.controls(ref.native)['presets']],
                         [(1000, 8000), (750, 6000), (500, 4000), (250, 2000)])
        cameras = [c.copy() for c in s.camera_configs]
        worker = s.cameras[0].worker = Mock()
        s.mode = "acquisition"
        state = s.set_unwrap_size(913, 4127)
        self.assertEqual(state['unwrap_size']['width'], 913)
        self.assertEqual(s.camera_configs, cameras)
        worker.start.assert_not_called()
        self.assertEqual(self.service().unwrap_size, (913, 4127))

    def test_legacy_settings_without_size_keep_the_reference_grid(self):
        (self.root / 'settings.json').write_text('{"target_brightness": 80}', encoding='utf-8')
        s = self.service()
        self.assertEqual(s.unwrap_size, s.output_sizes.native)

    def test_invalid_or_unwritable_setting_does_not_replace_previous_choice(self):
        s = self.service()
        s.set_unwrap_size(768, 5000)
        old = s.settings_path.read_bytes()
        for w, h in ((True, 5000), ('768', 5000), (768.5, 5000), (127, 5000), (1537, 5000), (768, 10001)):
            with self.subTest(size=(w, h)), self.assertRaises(ValueError):
                s.set_unwrap_size(w, h)
        with patch.object(Path, 'replace', side_effect=OSError('disk full')), self.assertRaises(RuntimeError):
            s.set_unwrap_size(384, 2500)
        self.assertEqual(s.unwrap_size, (768, 5000))
        self.assertEqual(s.settings_path.read_bytes(), old)
        self.assertEqual(list(self.root.glob('.settings-*.tmp')), [])
        invalid = json.loads(old)
        invalid['unwrap_size'] = {'width': -1, 'height': 900}
        s.settings_path.write_text(json.dumps(invalid), encoding='utf-8')
        self.assertEqual(self.service().unwrap_size, (1536, 10000))

    def test_capture_keeps_trigger_size_through_queue_archive_and_next_capture(self):
        s = self.service()

        def process(frames, *, output_size):
            w, h = output_size
            return SimpleNamespace(ok=True, image=np.full((h, w), 27, np.uint8), status='ok', message='',
                                   elapsed_ms=1, hole_count=0, bounds=(), input_shapes=((7, 8),) * 4)

        def read_active_line_once():
            s.monitor_stop.set()
            return False

        s.unwrap_bridge = SimpleNamespace(process_packets=process)
        s.cameras[0].worker = SimpleNamespace(read_line0_status=read_active_line_once)
        s.set_unwrap_size(401, 777)
        for i, expected in enumerate(((401, 777), (256, 511))):
            s.waiting_for_release = False
            s.monitor_stop.clear()
            s._line_monitor_loop()  # One real active-low trigger, no camera hardware.
            frames = {j: FramePacket(j, 7, 8, 0, bytes(range(56)), captured_at=1000 + i) for j in range(4)}
            s.capture_session['frames'] = {j: [f] for j, f in frames.items()}
            if i == 0:
                s.set_unwrap_size(256, 511)  # Already inside a pipe.
            with patch.object(s.executor, 'submit') as queue:
                s._finalize_capture_session_locked()
            run, *args = queue.call_args.args
            if i == 1:
                s.set_unwrap_size(128, 128)  # Already queued.
            run(*args)
            png, info = s.get_latest_unwrapped()
            path = Path(info['file'])
            self.assertEqual(path.read_bytes(), png)
            self.assertEqual(cv2.imdecode(np.frombuffer(png, np.uint8), 0).shape, expected[::-1])
            manifest = json.loads(path.with_name('capture.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['parameters']['unwrap_size'], dict(zip(('width', 'height'), expected)))
            self.assertEqual((manifest['unwrapped']['width'], manifest['unwrapped']['height']), expected)
            for camera in CAMERA_NAMES:
                im = cv2.imdecode(np.fromfile(path.with_name(camera + '.png'), np.uint8), 0)
                np.testing.assert_array_equal(im, np.arange(56, dtype=np.uint8).reshape(8, 7))
