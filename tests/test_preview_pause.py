import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from hikrobot_camera import FramePacket, HikRobotService


class PreviewPauseTest(unittest.TestCase):
    def setUp(self):
        with (patch.object(HikRobotService, "_initialize_sdk"),
              patch.object(HikRobotService, "_load_settings"),
              patch.object(HikRobotService, "_restore_latest_unwrapped"),
              patch.object(HikRobotService, "_log")):
            self.service = HikRobotService()
        self.service._log = Mock()
        self.addCleanup(self.service.executor.shutdown, wait=True)
        self.addCleanup(lambda: self.service.preview_executor.shutdown(wait=True, cancel_futures=True))
        self.service.mode = "acquisition"
        self.service.capture_session = {"collect_after": 0, "frames": {i: [] for i in range(4)}}

    def frame(self, slot=0, number=1):
        return FramePacket(slot, 4, 4, 0, bytes(range(16)), frame_number=number)

    def wait_for_previews(self):
        with self.service.preview_condition:
            self.assertTrue(self.service.preview_condition.wait_for(
                lambda: not any(self.service.preview_pending), timeout=3))

    def test_pause_drains_running_work_skips_queue_and_continues_capture(self):
        service = self.service
        service.preview_executor.shutdown(wait=True)
        service.preview_executor = ThreadPoolExecutor(max_workers=1)
        preview_started = threading.Event()
        finish_preview = threading.Event()
        pause_requested = threading.Event()
        algorithm_started = threading.Event()
        finish_algorithm = threading.Event()
        self.addCleanup(finish_preview.set)
        self.addCleanup(finish_algorithm.set)

        def encode(*args, **kwargs):
            preview_started.set()
            finish_preview.wait(3)
            return b"last preview"

        def log(level, message):
            if message.startswith("开始展开"):
                pause_requested.set()

        def unwrap():
            with service._pause_preview_for_unwrap():
                algorithm_started.set()
                finish_algorithm.wait(3)

        service._log.side_effect = log
        with patch("hikrobot_camera.mono8_to_preview_bmp", side_effect=encode) as codec:
            service._on_frame(service.cameras[0], self.frame(0))
            self.assertTrue(preview_started.wait(3))
            service._on_frame(service.cameras[1], self.frame(1))
            worker = threading.Thread(target=unwrap)
            worker.start()
            self.addCleanup(worker.join, 3)
            self.assertTrue(pause_requested.wait(3))
            self.assertFalse(algorithm_started.is_set())
            finish_preview.set()
            self.assertTrue(algorithm_started.wait(3))
            self.assertEqual(codec.call_count, 1)
            self.assertEqual(service.preview_bytes[0], b"last preview")
            self.assertTrue(service.get_state()["preview_paused"])
            for slot in range(4):
                service._on_frame(service.cameras[slot], self.frame(slot, 2))
                self.assertEqual(service.capture_session["frames"][slot][-1].frame_number, 2)
            self.assertEqual(codec.call_count, 1)
            finish_algorithm.set()
            worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertFalse(service.get_state()["preview_paused"])
            service.preview_times[0] = 0
            service._on_frame(service.cameras[0], self.frame(0, 3))
            self.wait_for_previews()
            self.assertEqual(codec.call_count, 2)

    def test_overlapping_batches_keep_preview_paused_until_both_finish(self):
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]

        def unwrap(index):
            with self.service._pause_preview_for_unwrap():
                entered[index].set()
                release[index].wait(3)

        workers = [threading.Thread(target=unwrap, args=(i,)) for i in range(2)]
        for event in release:
            self.addCleanup(event.set)
        for worker in workers:
            worker.start()
        try:
            self.assertTrue(all(event.wait(3) for event in entered))
            release[0].set()
            workers[0].join(3)
            self.assertTrue(self.service.get_state()["preview_paused"])
            self.service._on_frame(self.service.cameras[0], self.frame())
            self.assertFalse(any(self.service.preview_pending))
            release[1].set()
            workers[1].join(3)
            self.assertFalse(self.service.get_state()["preview_paused"])
        finally:
            for event in release:
                event.set()
            for worker in workers:
                worker.join(3)

    def test_preview_errors_and_cancelled_queue_do_not_block_pause(self):
        service = self.service
        with patch("hikrobot_camera.mono8_to_preview_bmp", side_effect=ValueError("test error")):
            service._on_frame(service.cameras[0], self.frame())
            self.wait_for_previews()
        with service._pause_preview_for_unwrap():
            self.assertTrue(service.get_state()["preview_paused"])

        occupied = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        service.preview_executor.shutdown(wait=True)
        service.preview_executor = ThreadPoolExecutor(max_workers=1)

        def occupy_worker():
            occupied.set()
            release.wait(3)

        task = service.preview_executor.submit(occupy_worker)
        self.assertTrue(occupied.wait(3))
        service._on_frame(service.cameras[1], self.frame(1))
        service.preview_executor.shutdown(wait=False, cancel_futures=True)
        self.assertFalse(any(service.preview_pending))
        with service._pause_preview_for_unwrap():
            pass
        # A callback during shutdown must still deliver its capture frame.
        service._on_frame(service.cameras[2], self.frame(2))
        self.assertEqual(len(service.capture_session["frames"][2]), 1)
        self.assertFalse(any(service.preview_pending))
        release.set()
        task.result(3)

    def test_save_restores_preview_on_success_and_failure(self):
        service = self.service

        def process(frames, *, output_size):
            self.assertTrue(service.get_state()["preview_paused"])
            return SimpleNamespace(
                ok=True, image=np.arange(16, dtype=np.uint8).reshape(4, 4),
                status="ok", message="", elapsed_ms=1, hole_count=0,
                bounds=(), input_shapes=((4, 4),) * 4,
            )

        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                service.config["output_dir"] = directory
                service.unwrap_bridge = SimpleNamespace(process_packets=Mock(
                    side_effect=RuntimeError("forced unwrap failure") if fail else process))
                frames = {i: self.frame(i) for i in range(4)}
                service._save_pipe(frames)
                self.assertFalse(service.get_state()["preview_paused"])
                folder = next(Path(directory).iterdir())
                manifest = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
                self.assertEqual(len(list(folder.glob("*.png"))), 4 if fail else 5)
                self.assertEqual(manifest["中文摘要"]["展开状态"], "失败" if fail else "成功")
                self.assertIn("实时预览已暂停", [row["阶段"] for row in manifest["处理时间轴"]["阶段"]])
                for slot in range(4):
                    self.assertEqual(frames[slot].data, bytes(range(16)))


if __name__ == "__main__":
    unittest.main()
