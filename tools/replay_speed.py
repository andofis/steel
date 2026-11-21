"""Read-only capture replay; never load the camera SDK or write into the input tree."""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hikrobot_camera import CAMERA_NAMES, FramePacket, HikRobotService


class ReplayService(HikRobotService):
    def __init__(self, scratch: str):
        self.scratch = scratch
        super().__init__()

    def _load_settings(self):
        self.config["output_dir"] = self.scratch
        self.settings_path = Path(self.scratch) / "settings.json"

    def _initialize_sdk(self):
        pass

    def _log(self, level, message):
        if level == "ERROR":
            print(message, file=sys.stderr)


def replay(service, folder):
    frames = {}
    for slot, name in enumerate(CAMERA_NAMES):
        path = folder / (name + ".png")
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError(f"Expected Mono8: {path}")
        h, w = image.shape
        frames[slot] = FramePacket(slot, w, h, 0, image.tobytes())
    with tempfile.TemporaryDirectory(dir=service.scratch) as capture_dir:
        service.config["output_dir"] = capture_dir
        start = time.time()
        for frame in frames.values():
            frame.captured_at = start
        service._save_pipe(frames, {"acquisition_started_at": start, "last_frame_epoch": start})
        png, info = service.get_latest_unwrapped()
        path = Path(info["file"])
        if (path.parent.parent != Path(capture_dir) or info.get("archive_status") != "saved" or
                not (path.parent / "capture.json").is_file() or path.read_bytes() != png):
            raise RuntimeError(f"Incomplete replay: {folder}: {service.last_error}")
        image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        timing = info["timeline"]["durations_ms"]
        return {"capture": folder.name, "unwrap_ms": timing["unwrap_algorithm"],
                "available_ms": timing["result_ready_from_start"], "archive_ms": timing["pipeline_total"],
                "candidates": info["hole_candidate_count"], "pixel_sha256": hashlib.sha256(image).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description="离线回放四路原图：测内存输入到展开、可取图和归档，不含现场收尾、浏览器或模型。")
    parser.add_argument("captures", type=Path, help="包含采集目录的文件夹")
    parser.add_argument("--limit", type=int, help="仅测前 N 组")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    folders = [p.parent for p in sorted(args.captures.rglob("capture.json"))
               if all((p.parent / (name + ".png")).is_file() for name in CAMERA_NAMES)][:args.limit]
    if not folders:
        parser.error("没有包含四路 PNG 的完整采集目录")
    with tempfile.TemporaryDirectory(prefix="steel-replay-") as scratch:
        service = ReplayService(scratch)
        try:
            replay(service, folders[0])  # Warm-up, not included in statistics.
            rows = [replay(service, folder) for folder in folders]
        finally:
            service.executor.shutdown(wait=True)
            service.preview_executor.shutdown(wait=True)
    metrics = {k: dict(zip(("p50", "p95", "max"), np.percentile([r[k] for r in rows], [50, 95, 100]).round(2)))
               for k in ("unwrap_ms", "available_ms", "archive_ms")}
    print(json.dumps({"samples": len(rows), "milliseconds": metrics, "captures": rows}, indent=2))


if __name__ == "__main__":
    main()
