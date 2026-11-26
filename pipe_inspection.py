"""Local controller between panorama publication and the segmentation service."""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_URL = "http://127.0.0.1:8766"
BACKEND_TIMEOUTS = {"cpu": 180, "gpu": 30, "tensorrt": 30}
BACKENDS = frozenset(BACKEND_TIMEOUTS)
TOLERANCE_LIMITS = {
    "max_rotation_deg": (180, 0, 180),
    "max_axial_pct": (5, 0, 25),
    "position_pct": (0.5, 0.01, 5),
    "size_pct": (35, 0, 100),
    "shape_iou": (0.55, 0, 1),
}


def _tolerances(values):
    if not isinstance(values, dict) or values.keys() - TOLERANCE_LIMITS.keys():
        raise ValueError("检测阈值格式错误")
    result = {}
    for key, (default, low, high) in TOLERANCE_LIMITS.items():
        value = values.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} 必须在 {low} 至 {high} 之间")
        result[key] = float(value)
    return result


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            scratch = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        scratch.replace(path)
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)


def segment_panorama(url, png, backend, cut_x, width, height):
    request = Request(
        f"{url}/v1/segment?{urlencode({'backend': backend, 'cut_x': cut_x})}",
        data=png,
        headers={"Content-Type": "image/png", "Accept": "application/json"},
    )
    timeout = BACKEND_TIMEOUTS[backend]
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
    except HTTPError as exc:
        status = exc.code
        with exc:
            raw_error = exc.read(4096)
        try:
            error_body = json.loads(raw_error)
            message = error_body.get("error") or error_body.get("detail")
        except (ValueError, AttributeError):
            message = None
        raise RuntimeError(f"推理服务 HTTP {status}，{message or '请查看推理服务日志'}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"推理服务在 {timeout} 秒内未返回结果，请检查服务日志") from exc
    except (URLError, OSError) as exc:
        raise RuntimeError(f"无法连接推理服务 {url}，{exc}") from exc
    try:
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("响应过大")
        result = json.loads(raw)
        if (result["width"], result["height"], result["backend"]) != (width, height, backend):
            raise ValueError("推理后端或图像尺寸不一致")
        holes = result["holes"]
        if not isinstance(holes, list):
            raise ValueError("掩膜列表格式错误")
        for hole in holes:
            x, y = hole["center"]
            if not (0 <= x < width and 0 <= y < height and 0 < hole["area"] <= width * height
                    and 0 <= hole["score"] <= 1 and hole["contours"]):
                raise ValueError("孔洞坐标或掩膜无效")
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        raise RuntimeError(f"推理服务返回无效结果，{exc}") from exc
    return result


class PipeInspection:
    def __init__(self, index: Path, url: str | None = None):
        self.index = index
        self.url = (url or os.environ.get("PIPE_INFERENCE_URL") or DEFAULT_URL).rstrip("/")
        self._condition = threading.Condition()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipe-inspection")
        self._running = 0
        self._closed = False
        self._recording = False
        self._results = OrderedDict()
        self._templates = {}
        self._active = ""
        self._backend = "cpu"
        self._limits = _tolerances({})
        self._enabled = False
        self._load_error = ""
        try:
            saved = json.loads(index.read_text(encoding="utf-8"))
            templates = {item["id"]: item for item in saved["templates"]}
            active = saved["active_template"]
            backend = saved["backend"]
            limits = _tolerances(saved["tolerances"])
            if backend not in BACKENDS or (active and active not in templates):
                raise ValueError("设置内容无效")
            if any(not isinstance(item["name"], str) or not isinstance(item["pipe_id"], str)
                   or not isinstance(item["holes"], list) or not item["holes"]
                   for item in templates.values()):
                raise ValueError("模板内容无效")
            self._templates, self._active, self._backend, self._limits = templates, active, backend, limits
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._load_error = f"模板文件无法读取，{exc}"

    def close(self):
        with self._condition:
            self._closed = True
            self._enabled = False
            self._condition.notify_all()
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def state(self):
        with self._condition:
            return {
                "enabled": self._enabled,
                "recording": self._recording,
                "backend": self._backend,
                "tolerances": dict(self._limits),
                "active_template": self._active,
                "templates": [
                    {**{key: item[key] for key in ("id", "name", "pipe_id")},
                     "hole_count": len(item["holes"])} for item in self._templates.values()
                ],
                "error": self._load_error,
            }

    def _save_settings(self, templates, active, backend, limits):
        if self._load_error:
            raise RuntimeError(self._load_error)
        try:
            _write_json(self.index, {"templates": list(templates.values()), "active_template": active,
                                     "backend": backend, "tolerances": limits})
        except OSError as exc:
            raise RuntimeError(f"保存模板或检测设置失败，{exc}") from exc
        self._templates, self._active, self._backend, self._limits = templates, active, backend, limits

    def configure(self, *, backend=None, tolerances=None, active_template=None):
        with self._condition:
            if self._enabled or self._recording:
                raise RuntimeError("请在流水线停止且模板录入结束后修改设置")
            backend = self._backend if backend is None else backend
            active = self._active if active_template is None else active_template
            if not isinstance(backend, str) or backend not in BACKENDS:
                raise ValueError("请选择 CPU、GPU 或 TensorRT")
            if not isinstance(active, str) or (active and active not in self._templates):
                raise ValueError("所选模板不存在")
            if tolerances is not None and not isinstance(tolerances, dict):
                raise ValueError("检测阈值格式错误")
            limits = self._limits if tolerances is None else _tolerances({**self._limits, **tolerances})
            self._save_settings(self._templates, active, backend, limits)
            return self.state()

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("流水线开关必须为布尔值")
        with self._condition:
            if enabled and (not self._active or self._load_error or self._recording or self._closed):
                raise RuntimeError(self._load_error or "请先录入并选择模板")
            self._enabled = enabled
            return self.state()

    def _segment(self, png, info, backend):
        if not info.get("width") or not info.get("height"):
            raise ValueError("当前展开图缺少尺寸，请重新采集合格样管")
        seam = info.get("panorama_seam") or {}
        cut_x = int(seam.get("cut_x") or 0) % info["width"]
        result = segment_panorama(self.url, png, backend, cut_x, info["width"], info["height"])
        return dict(result, cut_x=cut_x, phase_calibration_id=info.get("phase_calibration_id"))

    def _template_image_path(self, template_id):
        return self.index.parent / (self.index.stem + "_images") / (uuid.UUID(template_id).hex + ".png")

    def record_template(self, name, png, info):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 60:
            raise ValueError("模板名称需要 1 至 60 个字符")
        if not info.get("pipe_id") or info["pipe_id"].startswith("incomplete_"):
            raise ValueError("请先完整采集一根合格样管")
        with self._condition:
            if self._load_error:
                raise RuntimeError(self._load_error)
            if self._enabled or self._recording or self._running or self._closed:
                raise RuntimeError("请停止流水线并等待当前检测完成后录入模板")
            if any(t["name"] == name.strip() for t in self._templates.values()):
                raise ValueError("模板名称已存在")
            backend = self._backend
            self._recording = True
        try:
            detection = self._segment(png, info, backend)
            if not detection["holes"]:
                raise RuntimeError("样管未分割出孔洞，无法建立对齐模板")
            template_id = uuid.uuid4().hex
            template = dict(detection, id=template_id, name=name.strip(), pipe_id=info["pipe_id"], surface_image=True)
            image_path = self._template_image_path(template_id)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            handle = image_path.open("xb")
            try:
                with handle:
                    handle.write(png)
                with self._condition:
                    self._save_settings({**self._templates, template_id: template}, template_id, backend, self._limits)
                    self._results.pop(template["pipe_id"], None)
            except (OSError, RuntimeError, ValueError, TypeError):
                image_path.unlink(missing_ok=True)
                raise
        finally:
            with self._condition:
                self._recording = False
        return self.state()

    def submit(self, png, info, archive: Path | None = None):
        """Called by the capture worker with the same PNG served to the browser."""
        pipe_id = info["pipe_id"]
        with self._condition:
            if not self._enabled or self._closed or pipe_id in self._results:
                return False
            template = self._templates[self._active]
            result = {
                "pipe_id": pipe_id, "template_id": template["id"], "template_name": template["name"],
                "backend": self._backend, "status": "pending", "findings": [],
                "width": info.get("width"), "height": info.get("height"),
            }
            self._results[pipe_id] = result
            while len(self._results) > 32:
                self._results.popitem(last=False)
            if png is None:
                result.update(status="error", reason=info.get("failure_reason") or "未生成可检测的展开图")
            elif template.get("phase_calibration_id") != info.get("phase_calibration_id"):
                result.update(status="review", reason="模板与当前展开图的相位标定不同，请使用当前标定重新录入模板")
            elif self._running >= 2:
                result.update(status="review", reason="上一根钢管仍在检测，当前批次需要复核")
            else:
                self._running += 1
                self._executor.submit(self._inspect, png, info, archive, template, self._limits, dict(result))
                return True
            self._archive_result(archive, result)
            self._condition.notify_all()
            return True

    @staticmethod
    def _archive_result(archive, result):
        if archive is not None:
            try:
                _write_json(archive / "inspection.json", result)
            except (OSError, ValueError, TypeError) as exc:
                result["archive_error"] = str(exc)

    def _inspect(self, png, info, archive, template, limits, result):
        started = time.perf_counter()
        try:
            from pipe_alignment import inspect_pipe

            detection = self._segment(png, info, result["backend"])
            if template.get("model_id") and detection.get("model_id") != template["model_id"]:
                raise RuntimeError("推理模型已变更，请重新录入模板")
            surface_pngs = None
            if template.get("surface_image"):
                reference = self._template_image_path(template["id"]).read_bytes()
                surface_pngs = (reference, png)
            result.update(inspect_pipe(template, detection, limits, surface_pngs=surface_pngs))
            result["inference_ms"] = detection.get("elapsed_ms")
            result["hole_count"] = len(detection["holes"])
        except Exception as exc:
            result.update(status="error", reason=str(exc), findings=[])
        result["inspection_ms"] = round((time.perf_counter() - started) * 1000, 1)
        if info.get("last_frame_epoch") is not None:
            result["after_last_frame_ms"] = round((time.time() - info["last_frame_epoch"]) * 1000, 1)
        self._archive_result(archive, result)
        with self._condition:
            self._running -= 1
            if result["pipe_id"] in self._results:
                self._results[result["pipe_id"]] = result
            self._condition.notify_all()

    def wait_result(self, pipe_id, timeout=0):
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or pipe_id not in self._results or self._results[pipe_id]["status"] != "pending",
                timeout,
            )
            result = self._results.get(pipe_id)
            if result is None and self._active:
                template = self._templates[self._active]
                if template["pipe_id"] == pipe_id:
                    return {
                        "pipe_id": pipe_id, "status": "template", "template_name": template["name"],
                        "width": template["width"], "height": template["height"],
                        "findings": [dict(hole, kind="ok") for hole in template["holes"]],
                    }
            return dict(result) if result else None
