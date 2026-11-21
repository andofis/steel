import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from hikrobot_camera import FramePacket, HikRobotService


class ResultDeliveryTest(unittest.TestCase):
    def setUp(self):
        with (patch.object(HikRobotService, "_load_settings"),
              patch.object(HikRobotService, "_initialize_sdk"),
              patch.object(HikRobotService, "_restore_latest_unwrapped"),
              patch.object(HikRobotService, "_log")):
            self.service = HikRobotService()
        self.service._log = lambda *args: None
        out = tempfile.TemporaryDirectory()
        self.addCleanup(out.cleanup)
        self.service.config["output_dir"] = out.name
        self.addCleanup(self.service.executor.shutdown, wait=True)
        self.addCleanup(self.service.preview_executor.shutdown, wait=True)
        self.image = np.arange(24, dtype=np.uint8).reshape(6, 4)
        self.service.unwrap_bridge = SimpleNamespace(process_packets=lambda _: SimpleNamespace(
            ok=True, image=self.image, elapsed_ms=1, status="ok", message="", hole_count=0,
            bounds=(), input_shapes=((4, 6),) * 4))

    def frames(self, stamp):
        return {i: FramePacket(i, 4, 6, 0, self.image.tobytes(), captured_at=stamp) for i in range(4)}

    def test_full_resolution_result_is_readable_while_atomic_archive_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        replace = os.replace

        def slow_disk(src, dst):
            if Path(dst).name == "unwrapped.png":
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release archive")
            return replace(src, dst)

        with patch("image_codec.os.replace", side_effect=slow_disk):
            worker = self.service.executor.submit(self.service._save_pipe, self.frames(1000))
            try:
                self.assertTrue(entered.wait(3))
                payload, info = self.service.get_latest_unwrapped()
                np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(payload, np.uint8), 0), self.image)
                self.assertEqual(info["archive_status"], "pending")
                self.assertFalse(Path(info["file"]).exists())
                self.assertEqual(self.service.wait_for_unwrapped("", 0)["pipe_id"], info["pipe_id"])
            finally:
                release.set()
                worker.result(timeout=5)
        self.assertEqual(Path(info["file"]).read_bytes(), payload)
        self.assertEqual(self.service.get_latest_unwrapped()[1]["archive_status"], "saved")
        self.assertIsNone(self.service.wait_for_unwrapped(info["pipe_id"], 0))

    def test_waiter_wakes_on_publish_and_disk_restore_works_without_live_cache(self):
        entered = threading.Event()

        def subscribe():
            with self.service.result_condition:
                entered.set()
                return self.service.wait_for_unwrapped("", 3)

        result = self.service.executor.submit(subscribe)
        self.assertTrue(entered.wait(1))
        self.service._save_pipe(self.frames(1000))
        info = result.result(timeout=2)
        self.assertEqual(info["pipe_id"], self.service.latest_unwrapped_info["pipe_id"])
        out = self.service.config["output_dir"]
        with (patch.object(HikRobotService, "_load_settings", lambda s: s.config.update(output_dir=out)),
              patch.object(HikRobotService, "_initialize_sdk"), patch.object(HikRobotService, "_log")):
            restored = HikRobotService()
        self.addCleanup(restored.executor.shutdown, wait=True)
        self.addCleanup(restored.preview_executor.shutdown, wait=True)
        self.assertEqual(restored.get_latest_unwrapped(info["pipe_id"])[0], self.service.get_latest_unwrapped()[0])

    def test_old_capture_cannot_replace_newer_result_or_return_wrong_pixels(self):
        self.service._save_pipe(self.frames(2000))
        newer = self.service.get_latest_unwrapped()[1]["pipe_id"]
        self.service._save_pipe(self.frames(1000))
        self.assertEqual(self.service.get_latest_unwrapped(newer)[1]["pipe_id"], newer)
        with self.assertRaises(RuntimeError):
            self.service.get_latest_unwrapped("a different pipe")

    def test_archive_failure_does_not_hide_preview_or_count_a_missing_file(self):
        replace = os.replace

        def fail_png(src, dst):
            if Path(dst).name == "unwrapped.png":
                raise OSError("disk error")
            return replace(src, dst)

        with patch("image_codec.os.replace", side_effect=fail_png):
            self.service._save_pipe(self.frames(1000))
        _, info = self.service.get_latest_unwrapped()
        self.assertEqual(info["archive_status"], "failed")
        self.assertEqual(self.service.saved_count, 4)
        manifest = json.loads(Path(info["file"]).with_name("capture.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["unwrapped"]["file"], "")
        self.assertEqual(manifest["unwrapped"]["archive_status"], "failed")
        self.assertFalse(Path(info["file"] + ".tmp").exists())

    def test_raw_archive_error_is_visible_after_panorama_is_published(self):
        with patch("hikrobot_camera.write_mono8_png", side_effect=OSError("disk full")):
            self.service._save_pipe(self.frames(1000))
        png, info = self.service.get_latest_unwrapped()
        self.assertTrue(png)
        self.assertEqual(info["archive_status"], "saved")
        self.assertIn("disk full", info["raw_archive_error"])
        self.assertEqual(self.service.get_state()["save_phase"], "failed")
