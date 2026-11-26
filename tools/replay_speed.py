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
from pipe_inspection import BACKEND_TIMEOUTS, PipeInspection


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


def replay(service, folder, inspection=None, *, record_template=False):
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
        if tuple(reversed(image.shape)) != service.unwrap_size:
            raise RuntimeError(f"Unexpected output size: {image.shape}")
        timing = info["timeline"]["durations_ms"]
        row = {"capture": folder.name, "unwrap_ms": timing["unwrap_algorithm"],
               "available_ms": timing["result_ready_from_start"], "archive_ms": timing["pipeline_total"],
               "candidates": info["hole_candidate_count"], "pixel_sha256": hashlib.sha256(image).hexdigest()}
        if inspection is not None:
            if record_template:
                inspection.record_template(folder.name, png, info)
            else:
                backend = inspection.state()["backend"]
                result = inspection.wait_result(info["pipe_id"], timeout=BACKEND_TIMEOUTS[backend] + 15)
                if not result or result["status"] in {"pending", "error"}:
                    raise RuntimeError(f"回放检测失败，{(result or {}).get('reason') or folder.name}")
                row["inspection"] = {key: result.get(key) for key in (
                    "status", "reason", "after_last_frame_ms", "inference_ms", "hole_count", "alignment",
                )}
                row["inspection"]["defects"] = [
                    {key: item[key] for key in ("kind", "center", "reference_center") if key in item}
                    for item in result["findings"] if item["kind"] != "ok"
                ]
        return row


def main():
    parser = argparse.ArgumentParser(description="离线回放四路原图：测内存输入到展开、可取图和归档，不含现场收尾、浏览器或模型。")
    parser.add_argument("captures", type=Path, help="包含采集目录的文件夹")
    parser.add_argument("--limit", type=int, help="仅测前 N 组")
    parser.add_argument("--size", nargs=2, metavar=("WIDTH", "HEIGHT"), type=int,
                        help="按指定宽高直接展开，仅影响本次临时回放；默认使用配方原尺寸")
    parser.add_argument("--inspection-backend", choices=sorted(BACKEND_TIMEOUTS),
                        help="测试指定推理后端，首个目录用作临时模板，其余目录送入流水线")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    folders = [p.parent for p in sorted(args.captures.rglob("capture.json"))
               if all((p.parent / (name + ".png")).is_file() for name in CAMERA_NAMES)][:args.limit]
    if not folders:
        parser.error("没有包含四路 PNG 的完整采集目录")
    if args.inspection_backend and len(folders) < 2:
        parser.error("检测回放至少需要一个模板和一个验证批次")
    with tempfile.TemporaryDirectory(prefix="steel-replay-") as scratch:
        service = ReplayService(scratch)
        inspection = None
        try:
            if args.size is not None:
                try:
                    service.set_unwrap_size(*args.size)
                except ValueError as exc:
                    parser.error(str(exc))
            width, height = service.unwrap_size
            if args.inspection_backend:
                inspection = PipeInspection(Path(scratch) / "inspection_settings.json")
                service.on_panorama_ready = inspection.submit
                inspection.configure(backend=args.inspection_backend)
                replay(service, folders[0], inspection, record_template=True)
                inspection.set_enabled(True)
                replay(service, folders[0], inspection)  # Warm up the full path and discard the result.
                rows = [replay(service, folder, inspection) for folder in folders[1:]]
            else:
                replay(service, folders[0])
                rows = [replay(service, folder) for folder in folders]
        finally:
            service.executor.shutdown(wait=True)
            service.preview_executor.shutdown(wait=True)
            if inspection is not None:
                inspection.close()
    metrics = {k: dict(zip(("p50", "p95", "max"), np.percentile([r[k] for r in rows], [50, 95, 100]).round(2)))
               for k in ("unwrap_ms", "available_ms", "archive_ms")}
    if args.inspection_backend:
        latencies = [row["inspection"]["after_last_frame_ms"] for row in rows]
        metrics["after_last_frame_ms"] = dict(zip(("p50", "p95", "max"), np.percentile(latencies, [50, 95, 100]).round(2)))
    print(json.dumps({"samples": len(rows), "output_size": {"width": width, "height": height},
                      "milliseconds": metrics, "captures": rows}, indent=2))


if __name__ == "__main__":
    main()
