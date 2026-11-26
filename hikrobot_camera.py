# -*- coding: utf-8 -*-
"""Hikrobot MVS line-scan acquisition service.

The module uses the official Python wrapper installed by MVS and keeps all
camera operations behind a small service API so the browser UI remains simple.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterator, List, Optional, Tuple

from image_codec import create_unwrapped_image, encode_mono8_png, mono8_to_preview_bmp, write_encoded_png, write_mono8_png
from unwrap_output import OutputSizes

ROOT = Path(__file__).resolve().parent

CAMERA_SERIALS = [
    "DB0959183",
    "DB2692106",
    "DB0426965",
    "DB0426961",
]
CAMERA_NAMES = ["cam_0", "cam_90", "cam_180", "cam_270"]
CAMERA_MODEL = "MV-CL024-91GM"

DEFAULT_CONFIG: Dict[str, Any] = {
    "exposure_us": 182.0,
    "gain_db": 9.0,
    "line_rate_hz": 4100,
    "trigger_source": "Line0",
    "trigger_polarity": "LevelLow",
    "head_delay_ms": 0,
    "tail_delay_ms": 430,
    "no_signal_ms": 80,
    "stream_align_ms": 1000,
    "output_dir": "captures",
    "folder_prefix": "pipe",
    "target_brightness": 82,
}


def _candidate_mvimport_paths() -> List[Path]:
    paths: List[Path] = []
    for env_name in ("MVCAM_COMMON_RUNENV", "MVS_HOME", "MVS_SDK_PATH"):
        value = os.environ.get(env_name)
        if value:
            base = Path(value)
            paths.extend([
                base / "Samples" / "Python" / "MvImport",
                base / "Development" / "Samples" / "Python" / "MvImport",
            ])
    paths.extend([
        Path(r"E:\MVS\Development\Samples\Python\MvImport"),
        Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
        Path(r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"),
        Path(r"D:\MVS\Development\Samples\Python\MvImport"),
        ROOT / "MvImport",
    ])
    return paths


def _candidate_runtime_paths() -> List[Path]:
    paths = [
        Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"),
        Path(r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64"),
        Path(r"E:\MVS\Applications\Win64"),
        Path(r"C:\Program Files\MVS\Applications\Win64"),
        Path(r"C:\Program Files (x86)\MVS\Applications\Win64"),
        Path(r"D:\MVS\Applications\Win64"),
    ]
    mvs_home = os.environ.get("MVS_HOME")
    if mvs_home:
        paths.append(Path(mvs_home) / "Applications" / "Win64")
    return paths


def _prepare_sdk_environment() -> Tuple[Path, Path]:
    mvimport = next((path for path in _candidate_mvimport_paths() if (path / "MvCameraControl_class.py").exists()), None)
    if mvimport is None:
        raise RuntimeError("未找到海康 MVS Python 接口，请先安装完整 MVS SDK Development")

    runtime = next((path for path in _candidate_runtime_paths() if (path / "MvCameraControl.dll").exists()), None)
    if runtime is None:
        raise RuntimeError("未找到 MvCameraControl.dll，请先安装海康 MVS Runtime")

    os.environ["PATH"] = str(runtime) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(runtime))
        except (FileNotFoundError, OSError):
            pass
    sys.path.insert(0, str(mvimport))
    return mvimport, runtime


def decode_char_array(value: Any) -> str:
    try:
        data = memoryview(value).tobytes().split(b"\x00", 1)[0]
    except Exception:
        return ""
    for encoding in ("gbk", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def ipv4_to_string(value: int) -> str:
    return "%d.%d.%d.%d" % (
        (value >> 24) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 8) & 0xFF,
        value & 0xFF,
    )


def timestamp_for_path(moment: Optional[datetime] = None) -> str:
    moment = moment or datetime.now()
    return moment.strftime("%Y%m%d_%H%M%S_%f")[:-3]


def format_timeline_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value).isoformat(timespec="milliseconds")


def add_timeline_event(timeline: Dict[str, Any], stage: str, at: Optional[float] = None) -> float:
    moment = time.time() if at is None else float(at)
    timeline.setdefault("events", []).append({
        "stage": stage,
        "timestamp": format_timeline_timestamp(moment),
        "epoch": round(moment, 3),
    })
    return moment


@dataclass
class FramePacket:
    slot: int
    width: int
    height: int
    pixel_type: int
    data: bytes
    captured_at: float = field(default_factory=time.time)
    frame_number: int = 0


@dataclass
class CameraRuntime:
    slot: int
    name: str
    serial: str
    status: str = "offline"
    model: str = CAMERA_MODEL
    ip: str = ""
    width: int = 0
    height: int = 0
    exposure_us: float = DEFAULT_CONFIG["exposure_us"]
    gain_db: float = DEFAULT_CONFIG["gain_db"]
    line_rate_hz: int = DEFAULT_CONFIG["line_rate_hz"]
    frame_count: int = 0
    last_frame_at: float = 0.0
    error: str = ""
    cam: Any = None
    worker: Any = None


    @property
    def preview_ready(self) -> bool:
        return self.last_frame_at > 0


def polarization_is_level(value: str) -> bool:
    return value in ("LevelLow", "LevelHigh")

class CameraWorker:
    def __init__(self, service: "HikRobotService", runtime: CameraRuntime, device_info: Any):
        self.service = service
        self.runtime = runtime
        self.device_info = device_info
        self.sdk = service.sdk
        self.cam = self.sdk.MvCamera()
        self.grabbing = False
        self.closed = False
        self._callback_type = self.sdk.get_platform_functype()(
            None,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(self.sdk.MV_FRAME_OUT_INFO_EX),
            ctypes.c_void_p,
        )(self._on_frame)

    def _result(self, label: str, result: int, required: bool = False) -> bool:
        if result == 0:
            return True
        message = "%s %s 失败 (0x%08x)" % (self.runtime.name, label, result)
        if required:
            raise RuntimeError(message)
        self.service._log("WARN", message)
        return False

    def open(self) -> None:
        self._result("创建句柄", self.cam.MV_CC_CreateHandle(self.device_info), required=True)
        self._result("打开设备", self.cam.MV_CC_OpenDevice(self.sdk.MV_ACCESS_Exclusive, 0), required=True)
        try:
            packet_size = self.cam.MV_CC_GetOptimalPacketSize()
            if int(packet_size) > 0:
                self._result("设置最佳包长", self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(packet_size)))
        except Exception as exc:
            self.service._log("WARN", "%s 包长设置异常: %s" % (self.runtime.name, exc))
        self._result("注册图像回调", self.cam.MV_CC_RegisterImageCallBackEx(self._callback_type, None), required=True)
        self._result("设置 Mono8 输出", self.cam.MV_CC_SetEnumValueByString("PixelFormat", "Mono8"))
        self._read_current_values()
        self.runtime.status = "online"
        self.runtime.error = ""

    def _read_current_values(self) -> None:
        try:
            exposure = self.sdk.MVCC_FLOATVALUE()
            if self.cam.MV_CC_GetFloatValue("ExposureTime", exposure) == 0:
                self.runtime.exposure_us = round(float(exposure.fCurValue), 3)
        except Exception:
            pass
        try:
            # MV-CL024-91GM 的 GainRaw 上限接近 0 dB，实际亮度增益节点是
            # DigitalShift。优先读取该节点，旧型号再回退到 GainRaw/Gain。
            gain = self.sdk.MVCC_FLOATVALUE()
            if self.cam.MV_CC_GetFloatValue("DigitalShift", gain) == 0:
                self.runtime.gain_db = round(float(gain.fCurValue), 3)
            elif self.cam.MV_CC_GetFloatValue("GainRaw", gain) == 0:
                self.runtime.gain_db = round(float(gain.fCurValue), 3)
            elif self.cam.MV_CC_GetFloatValue("Gain", gain) == 0:
                self.runtime.gain_db = round(float(gain.fCurValue), 3)
        except Exception:
            pass
        try:
            rate = self.sdk.MVCC_INTVALUE_EX()
            if self.cam.MV_CC_GetIntValueEx("AcquisitionLineRate", rate) == 0:
                self.runtime.line_rate_hz = int(rate.nCurValue)
        except Exception:
            pass
        try:
            width = self.sdk.MVCC_INTVALUE_EX()
            height = self.sdk.MVCC_INTVALUE_EX()
            if self.cam.MV_CC_GetIntValueEx("Width", width) == 0:
                self.runtime.width = int(width.nCurValue)
            if self.cam.MV_CC_GetIntValueEx("Height", height) == 0:
                self.runtime.height = int(height.nCurValue)
        except Exception:
            pass

    def _set_digital_gain(self, gain_db: float) -> bool:
        """设置 MV-CL024-91GM 真正可调的数字增益节点。"""
        enable_result = self.cam.MV_CC_SetBoolValue("DigitalShiftEnable", True)
        if enable_result != 0:
            enabled = ctypes.c_bool()
            if self.cam.MV_CC_GetBoolValue("DigitalShiftEnable", enabled) != 0 or not enabled.value:
                self.service._log("WARN", "%s DigitalShiftEnable 写入失败 (0x%08x)" % (self.runtime.name, enable_result))
                return False

        result = self.cam.MV_CC_SetFloatValue("DigitalShift", float(gain_db))
        if result != 0:
            self.service._log("WARN", "%s DigitalShift 写入失败 (0x%08x)" % (self.runtime.name, result))
            return False

        actual = self.sdk.MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue("DigitalShift", actual) != 0:
            self.service._log("WARN", "%s DigitalShift 写入后回读失败" % self.runtime.name)
            return False
        self.runtime.gain_db = round(float(actual.fCurValue), 3)
        self.service._log(
            "INFO",
            "%s DigitalShift=%.2f dB（请求 %.2f dB，范围 %.1f~%.1f dB）" % (
                self.runtime.name,
                self.runtime.gain_db,
                float(gain_db),
                float(actual.fMin),
                float(actual.fMax),
            ),
        )
        return True

    def _set_legacy_gain(self, gain_db: float) -> bool:
        """兼容没有 DigitalShift 节点的其他海康相机型号。"""
        gain_value = self.sdk.MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue("GainRaw", gain_value) != 0:
            if self.cam.MV_CC_GetFloatValue("Gain", gain_value) != 0:
                return False
        if not (float(gain_value.fMin) <= float(gain_db) <= float(gain_value.fMax)):
            return False
        node = "GainRaw" if self.cam.MV_CC_GetFloatValue("GainRaw", gain_value) == 0 else "Gain"
        result = self.cam.MV_CC_SetFloatValue(node, float(gain_db))
        if result != 0:
            self.service._log("WARN", "%s %s 写入失败 (0x%08x)" % (self.runtime.name, node, result))
            return False
        actual = self.sdk.MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue(node, actual) == 0:
            self.runtime.gain_db = round(float(actual.fCurValue), 3)
        else:
            self.runtime.gain_db = float(gain_db)
        self.service._log("INFO", "%s %s=%.2f dB" % (self.runtime.name, node, self.runtime.gain_db))
        return True

    def _apply_line_settings(self) -> None:
        cfg = self.service.camera_config(self.runtime.slot)
        self._result("设置行频", self.cam.MV_CC_SetIntValue("AcquisitionLineRate", int(cfg["line_rate_hz"])))
        self._result("启用行频", self.cam.MV_CC_SetBoolValue("AcquisitionLineRateEnable", True))
        self._result("设置曝光", self.cam.MV_CC_SetFloatValue("ExposureTime", float(cfg["exposure_us"])))
        # 当前相机模式下 Sequencer 节点可能不可访问，但内部状态已确认关闭。
        # 仅做兼容性尝试，不影响后续 DigitalShift 写入。
        self.cam.MV_CC_SetEnumValue("SequencerConfigurationMode", 0)
        self.cam.MV_CC_SetEnumValue("SequencerMode", 0)
        self.cam.MV_CC_SetEnumValue("GainAuto", 0)

        if self._set_digital_gain(float(cfg["gain_db"])):
            self.runtime.error = ""
        elif self._set_legacy_gain(float(cfg["gain_db"])):
            self.runtime.error = ""
        else:
            self.runtime.error = "增益写入失败，请检查 DigitalShift 节点是否被相机锁定"
            self.service._log("WARN", self.runtime.error)
        self._result("设置 Mono8 输出", self.cam.MV_CC_SetEnumValueByString("PixelFormat", "Mono8"))
    def _set_trigger_activation(self, polarity: str) -> None:
        if self.cam.MV_CC_SetEnumValueByString("FrameTriggerActivation", polarity) != 0:
            self._result("设置帧触发极性", self.cam.MV_CC_SetEnumValueByString("TriggerActivation", polarity))

    def configure_preview(self) -> None:
        self._apply_line_settings()
        self.cam.MV_CC_SetEnumValue("TriggerMode", self.sdk.MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetIntValue("Height", 256)
        self.cam.MV_CC_SetEnumValue("LineSelector", 0)

    def configure_acquisition(self) -> None:
        # 整管模式由软件监测 Line0，并连续收集 256 行帧后纵向拼接。
        self._apply_line_settings()
        self.cam.MV_CC_SetEnumValue("TriggerMode", self.sdk.MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_SetIntValue("Height", 256)
        self.cam.MV_CC_SetEnumValue("LineSelector", 0)

    def read_line0_status(self) -> Optional[bool]:
        try:
            self.cam.MV_CC_SetEnumValue("LineSelector", 0)
            value = self.sdk.c_bool()
            if self.cam.MV_CC_GetBoolValue("LineStatus", value) == 0:
                return bool(value.value)
        except Exception:
            pass
        try:
            value = self.sdk.c_uint()
            if self.cam.MV_CC_GetIntValue("LineStatusAll", value) == 0:
                return bool(int(value.value) & 0x1)
        except Exception:
            pass
        return None
    def start(self, mode: str) -> None:
        self.stop()
        if mode == "acquisition":
            self.configure_acquisition()
        else:
            self.configure_preview()
        self._result("开始取流", self.cam.MV_CC_StartGrabbing(), required=True)
        self.grabbing = True

    def stop(self) -> None:
        if not self.closed and self.grabbing:
            try:
                self.cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            self.grabbing = False

    def close(self) -> None:
        if self.closed:
            return
        self.stop()
        try:
            self.cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            self.cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        self.closed = True

    def _on_frame(self, p_data: Any, p_frame_info: Any, _user: Any) -> None:
        try:
            if not p_data:
                return
            frame_info = ctypes.cast(p_frame_info, ctypes.POINTER(self.sdk.MV_FRAME_OUT_INFO_EX)).contents
            width = int(getattr(frame_info, "nExtendWidth", 0) or frame_info.nWidth)
            height = int(getattr(frame_info, "nExtendHeight", 0) or frame_info.nHeight)
            pixel_type = int(frame_info.enPixelType)
            expected = width * height
            if width <= 0 or height <= 0 or expected <= 0:
                return
            if pixel_type != self.sdk.PixelType_Gvsp_Mono8:
                self.service._log("WARN", "%s 收到非 Mono8 像素格式 0x%08x，已跳过" % (self.runtime.name, pixel_type))
                return
            data = ctypes.string_at(p_data, expected)
            packet = FramePacket(
                slot=self.runtime.slot,
                width=width,
                height=height,
                pixel_type=pixel_type,
                data=data,
                captured_at=time.time(),
                frame_number=int(frame_info.nFrameNum),
            )
            self.runtime.frame_count += 1
            self.runtime.last_frame_at = packet.captured_at
            self.runtime.width = width
            self.runtime.height = height
            self.service._on_frame(self.runtime, packet)
        except Exception as exc:
            self.service._log("ERROR", "%s 图像回调异常: %s" % (self.runtime.name, exc))

class HikRobotService:
    def __init__(self) -> None:
        self.on_panorama_ready = None
        self.lock = threading.RLock()
        self.preview_lock = threading.Lock()
        self.preview_condition = threading.Condition()
        self.preview_pause_count = 0
        self.group_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hik-save")
        self.preview_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hik-preview")
        self.preview_pending = [False] * 4
        self.preview_times = [0.0] * 4
        self.preview_bytes: List[Optional[bytes]] = [None] * 4
        self.preview_frame_info: List[Dict[str, Any]] = [{} for _ in range(4)]
        self.cameras = [CameraRuntime(slot=i, name=CAMERA_NAMES[i], serial=CAMERA_SERIALS[i]) for i in range(4)]
        self.devices: Dict[str, Any] = {}
        self.device_list: Any = None
        self.sdk: Any = None
        self.sdk_ready = False
        self.sdk_error = ""
        self.sdk_version = 0
        self.mvimport_path = ""
        self.runtime_path = ""
        self.mode = "idle"
        self.pipe_count = 0
        self.saved_count = 0
        self.incomplete_count = 0
        self.capture_sequence = 0
        self.capture_history: List[Dict[str, Any]] = []
        self.last_pipe_id = ""
        self.last_error = ""
        self.started_at = time.time()
        self.logs: List[Dict[str, str]] = []
        self.log_lock = threading.Lock()
        self.config = dict(DEFAULT_CONFIG)
        global_keys = {"output_dir", "stream_align_ms"}
        camera_defaults = {key: value for key, value in DEFAULT_CONFIG.items() if key not in global_keys}
        self.camera_configs = [dict(camera_defaults) for _ in range(4)]
        self.settings_path = ROOT / "settings.json"
        self.group_frames: Dict[int, FramePacket] = {}
        self.group_started_at = 0.0
        self.group_timer: Optional[threading.Timer] = None
        self.capture_session: Optional[Dict[str, Any]] = None
        self.capture_lock = threading.RLock()
        self.monitor_stop = threading.Event()
        self.monitor_thread: Optional[threading.Thread] = None
        self.waiting_for_release = True
        self.last_line_state: Optional[bool] = None
        self.unwrap_bridge: Any = None
        self.unwrap_bridge_error = ""
        self.latest_unwrapped_path: Optional[Path] = None
        self.latest_unwrapped_info: Dict[str, Any] = {}
        self.result_condition = threading.Condition(self.lock)
        self._latest_png: Optional[bytes] = None
        self._latest_capture_epoch = -math.inf
        self.save_pipe_id = ""
        self.save_phase = "idle"
        self.save_started_at = 0.0
        self.output_sizes = OutputSizes()  # Also usable in capture-only installations.
        try:
            from tube_unwrap_bridge import TubeUnwrapBridge
            self.unwrap_bridge = TubeUnwrapBridge(localization_executor=self.preview_executor)
            self.output_sizes = self.unwrap_bridge.output_sizes
            self._log("INFO", "钢管展开模块已加载: tube_unwrap 0.1.2")
        except Exception as exc:
            self.unwrap_bridge = None
            self.unwrap_bridge_error = str(exc)
            self._log("ERROR", "钢管展开模块加载失败: %s" % exc)
        self.unwrap_size = self.output_sizes.native
        self._load_settings()
        self._restore_latest_unwrapped()
        self._initialize_sdk()

    def _restore_latest_unwrapped(self) -> None:
        try:
            root = self.output_dir()
            if not root.exists():
                return
            folders = sorted(
                (item for item in root.iterdir() if item.is_dir()),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )[:25]
            for folder in folders:
                image_path = folder / "unwrapped.png"
                manifest_path = folder / "capture.json"
                if not image_path.exists() or not manifest_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                unwrapped = manifest.get("unwrapped") or {}
                self.latest_unwrapped_path = image_path
                self.latest_unwrapped_info = {
                    "pipe_id": manifest.get("pipe_id", folder.name),
                    "file": str(image_path),
                    "width": unwrapped.get("width", 0),
                    "height": unwrapped.get("height", 0),
                    "captured_at": manifest.get("captured_at", ""),
                    "algorithm": unwrapped.get("algorithm", ""),
                    "algorithm_version": unwrapped.get("algorithm_version", ""),
                    "status": unwrapped.get("status", "ok"),
                    "message": unwrapped.get("message", ""),
                    "elapsed_ms": unwrapped.get("elapsed_ms", 0),
                    "hole_candidate_count": unwrapped.get("hole_candidate_count", 0),
                    "panorama_seam": unwrapped.get("panorama_seam"),
                }
                return
        except Exception:
            pass

    def _initialize_sdk(self) -> None:
        try:
            self.mvimport_path, runtime = _prepare_sdk_environment()
            import MvCameraControl_class as sdk  # type: ignore
            self.sdk = sdk
            self.runtime_path = str(runtime)
            self.sdk.MvCamera.MV_CC_Initialize()
            self.sdk_version = int(self.sdk.MvCamera.MV_CC_GetSDKVersion())
            self.sdk_ready = True
            self._log("INFO", "海康 MVS SDK 初始化成功: 0x%x" % self.sdk_version)
            self.refresh_devices(quiet=True)
        except Exception as exc:
            self.sdk_ready = False
            self.sdk_error = str(exc)
            self._log("ERROR", "海康 MVS SDK 初始化失败: %s" % exc)

    def _load_settings(self) -> None:
        try:
            if not self.settings_path.exists():
                self.unwrap_size = self.output_sizes.preset(3)  # New installation: balanced. Legacy settings keep native.
                return
            saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                return
            if isinstance(saved.get("camera_configs"), list):
                for index, item in enumerate(saved["camera_configs"][:4]):
                    if isinstance(item, dict):
                        allowed = set(self.camera_configs[index].keys())
                        self.camera_configs[index].update({
                            key: value for key, value in item.items() if key in allowed
                        })
            else:
                camera_keys = set(self.camera_configs[0].keys())
                legacy = {key: value for key, value in saved.items() if key in camera_keys}
                for camera_config in self.camera_configs:
                    camera_config.update(legacy)
            for key in ("output_dir", "folder_prefix", "target_brightness", "stream_align_ms"):
                if key in saved:
                    self.config[key] = saved[key]
            self.config.update(self.camera_configs[0])
            if "unwrap_size" in saved:
                size = saved["unwrap_size"]
                try:
                    if not isinstance(size, dict):
                        raise ValueError("尺寸格式无效")
                    self.unwrap_size = self.output_sizes.validate(size.get("width"), size.get("height"))
                except ValueError as exc:
                    self._log("WARN", f"已忽略无效的展开尺寸，使用配方原尺寸：{exc}")
        except Exception:
            pass

    def _write_settings(self, size: tuple[int, int]) -> None:
        payload = {
            "output_dir": self.config.get("output_dir", "captures"),
            "folder_prefix": self.config.get("folder_prefix", "pipe"),
            "target_brightness": self.config.get("target_brightness", 82),
            "stream_align_ms": self.config.get("stream_align_ms", 1000),
            "unwrap_size": {"width": size[0], "height": size[1]},
            "camera_configs": self.camera_configs,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp = None
        try:
            with NamedTemporaryFile("w", dir=self.settings_path.parent, prefix=".settings-", suffix=".tmp",
                                    delete=False, encoding="utf-8") as stream:
                tmp = Path(stream.name)
                stream.write(text)
            tmp.replace(self.settings_path)
        except OSError:
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def save_settings(self) -> None:
        try:
            with self.lock:
                self._write_settings(self.unwrap_size)
        except Exception as exc:
            self._log("WARN", "保存设置失败: %s" % exc)

    def set_unwrap_size(self, width: int, height: int) -> dict:
        size = self.output_sizes.validate(width, height)
        with self.lock:
            try:
                self._write_settings(size)
            except OSError as exc:
                raise RuntimeError(f"展开尺寸未更改，无法保存设置：{exc}") from exc
            self.unwrap_size = size  # Single immutable assignment, never restart a camera.
        self._log("INFO", f"后续采集的展开尺寸：{width} × {height}；四路原图不变")
        return self.get_state()

    def camera_config(self, slot: int) -> Dict[str, Any]:
        if slot < 0 or slot >= 4:
            raise ValueError("无效相机通道")
        return self.camera_configs[slot]

    def _log(self, level: str, message: str) -> None:
        item = {"time": datetime.now().strftime("%H:%M:%S"), "level": level, "message": message}
        with self.log_lock:
            self.logs.append(item)
            if len(self.logs) > 120:
                self.logs = self.logs[-120:]
        print("[%s] %s" % (level, message))

    def _parse_gige_device(self, info: Any) -> Tuple[str, str, str]:
        gige = info.SpecialInfo.stGigEInfo
        model = decode_char_array(gige.chModelName)
        serial = decode_char_array(gige.chSerialNumber)
        ip = ipv4_to_string(int(gige.nCurrentIp))
        return model, serial, ip

    def refresh_devices(self, quiet: bool = False) -> int:
        with self.lock:
            if not self.sdk_ready:
                return 0
            try:
                device_list = self.sdk.MV_CC_DEVICE_INFO_LIST()
                layer = (self.sdk.MV_GIGE_DEVICE | self.sdk.MV_USB_DEVICE |
                         self.sdk.MV_GENTL_CAMERALINK_DEVICE | self.sdk.MV_GENTL_CXP_DEVICE |
                         self.sdk.MV_GENTL_XOF_DEVICE | self.sdk.MV_GENTL_XOC_DEVICE)
                result = self.sdk.MvCamera.MV_CC_EnumDevices(layer, device_list)
                if result != 0:
                    raise RuntimeError("枚举设备失败 (0x%08x)" % result)
                self.device_list = device_list
                self.devices = {}
                for index in range(int(device_list.nDeviceNum)):
                    info = ctypes.cast(device_list.pDeviceInfo[index], ctypes.POINTER(self.sdk.MV_CC_DEVICE_INFO)).contents
                    if info.nTLayerType in (self.sdk.MV_GIGE_DEVICE, self.sdk.MV_GENTL_GIGE_DEVICE):
                        model, serial, ip = self._parse_gige_device(info)
                    else:
                        model, serial, ip = "Unknown", "", ""
                    if serial:
                        self.devices[serial] = {"info": info, "model": model, "ip": ip, "index": index}
                if not quiet:
                    self._log("INFO", "MVS 枚举完成，发现 %d 台设备" % len(self.devices))
                return len(self.devices)
            except Exception as exc:
                self.last_error = str(exc)
                self._log("ERROR", "刷新相机列表失败: %s" % exc)
                return 0

    def connect_all(self) -> Dict[str, Any]:
        count = self.refresh_devices(quiet=True)
        for runtime in self.cameras:
            if runtime.worker is not None:
                continue
            device = self.devices.get(runtime.serial)
            if not device:
                runtime.status = "offline"
                runtime.error = "未发现序列号 %s" % runtime.serial
                continue
            try:
                runtime.model = device["model"] or CAMERA_MODEL
                runtime.ip = device["ip"]
                runtime.status = "connecting"
                worker = CameraWorker(self, runtime, device["info"])
                worker.open()
                runtime.worker = worker
                runtime.status = "online"
                runtime.error = ""
                self._log("INFO", "%s 已连接: %s / %s" % (runtime.name, runtime.serial, runtime.ip))
            except Exception as exc:
                runtime.status = "offline"
                runtime.error = str(exc)
                if runtime.worker is not None:
                    runtime.worker.close()
                    runtime.worker = None
                self._log("ERROR", "%s 连接失败: %s" % (runtime.name, exc))
        if count == 0:
            self.last_error = "MVS 未发现任何相机，请检查 PoE 供电、网线和相机 IP"
            self._log("WARN", self.last_error)
        return self.get_state()

    def disconnect_all(self) -> Dict[str, Any]:
        self.stop_all(close_camera=True)
        self.mode = "idle"
        self._log("INFO", "全部相机已断开")
        return self.get_state()

    def connect_one(self, slot: int) -> Dict[str, Any]:
        if slot < 0 or slot >= 4:
            raise ValueError("无效相机通道")
        runtime = self.cameras[slot]
        if runtime.worker is not None:
            return self.get_state()
        if not self.devices:
            self.refresh_devices(quiet=True)
        device = self.devices.get(runtime.serial)
        if not device:
            raise RuntimeError("未发现相机 %s" % runtime.serial)
        runtime.model = device["model"] or CAMERA_MODEL
        runtime.ip = device["ip"]
        worker = CameraWorker(self, runtime, device["info"])
        try:
            worker.open()
            runtime.worker = worker
            runtime.status = "online"
            runtime.error = ""
            self._log("INFO", "%s 已连接" % runtime.name)
            return self.get_state()
        except Exception:
            worker.close()
            runtime.worker = None
            runtime.status = "offline"
            raise

    def disconnect_one(self, slot: int) -> Dict[str, Any]:
        if slot < 0 or slot >= 4:
            raise ValueError("无效相机通道")
        runtime = self.cameras[slot]
        if runtime.worker:
            runtime.worker.close()
            runtime.worker = None
        runtime.status = "offline"
        if len([camera for camera in self.cameras if camera.worker is not None]) < 4:
            self.mode = "idle"
        self._log("INFO", "%s 已断开" % runtime.name)
        return self.get_state()
    def stop_all(self, close_camera: bool = False) -> None:
        for runtime in self.cameras:
            if runtime.worker:
                runtime.worker.stop()
                if close_camera:
                    runtime.worker.close()
                    runtime.worker = None
                    runtime.status = "offline"
        self.group_frames = {}
        if self.group_timer:
            self.group_timer.cancel()
            self.group_timer = None
    def start_preview(self) -> Dict[str, Any]:
        if not any(runtime.worker for runtime in self.cameras):
            self.connect_all()
        active = [runtime for runtime in self.cameras if runtime.worker is not None]
        if not active:
            raise RuntimeError("没有已连接的相机")
        self._clear_group()
        self.mode = "preview"
        for runtime in active:
            try:
                runtime.worker.start("preview")
                runtime.status = "online"
                self._log("INFO", "%s 进入实时预览" % runtime.name)
            except Exception as exc:
                runtime.status = "error"
                runtime.error = str(exc)
                self._log("ERROR", "%s 预览启动失败: %s" % (runtime.name, exc))
        return self.get_state()

    def start_acquisition(self) -> Dict[str, Any]:
        connected = [runtime for runtime in self.cameras if runtime.worker is not None]
        if len(connected) < 4:
            self.connect_all()
            connected = [runtime for runtime in self.cameras if runtime.worker is not None]
        if len(connected) < 4:
            missing = [runtime.name for runtime in self.cameras if runtime.worker is None]
            raise RuntimeError("必须四台相机全部在线，当前缺少: %s" % ", ".join(missing))

        self._stop_line_monitor()
        self._clear_group()
        with self.capture_lock:
            self.capture_session = None
            self.waiting_for_release = True
            self.last_line_state = None
        self.mode = "acquisition"
        for runtime in self.cameras:
            try:
                runtime.worker.start("acquisition")
                runtime.status = "online"
                self._log("INFO", "%s 已进入整管连续采图" % runtime.name)
            except Exception as exc:
                self.mode = "idle"
                for other in self.cameras:
                    if other.worker:
                        other.worker.stop()
                runtime.status = "online"
                runtime.error = str(exc)
                raise RuntimeError("%s 整管采集启动失败: %s" % (runtime.name, exc))
        self.monitor_stop.clear()
        self.monitor_thread = threading.Thread(target=self._line_monitor_loop, name="line0-monitor", daemon=True)
        self.monitor_thread.start()
        self._log("INFO", "四路整管采集已启动，等待 Line0 光电触发")
        return self.get_state()

    def _stop_line_monitor(self) -> None:
        self.monitor_stop.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.monitor_thread = None

    def _line_monitor_loop(self) -> None:
        baseline_logged = False
        while not self.monitor_stop.is_set():
            master = self.cameras[0].worker
            if not master:
                time.sleep(0.05)
                continue
            raw = master.read_line0_status()
            if raw is None:
                time.sleep(0.05)
                continue
            cfg = self.camera_configs[0]
            polarity = str(cfg.get("trigger_polarity", "LevelLow"))
            active = (not raw) if polarity in ("LevelLow", "FallingEdge") else raw
            now = time.time()
            if not baseline_logged:
                self._log("INFO", "Line0 初始状态=%s, 极性=%s" % (raw, polarity))
                baseline_logged = True

            with self.capture_lock:
                if self.waiting_for_release:
                    if not active:
                        self.waiting_for_release = False
                        self.last_line_state = raw
                        self._log("INFO", "Line0 已复位，等待钢管进入")
                    time.sleep(0.01)
                    continue

                session = self.capture_session
                if session is None:
                    if active:
                        head_delay = float(cfg.get("head_delay_ms", 0)) / 1000.0
                        timeline = {
                            "acquisition_started_at": now,
                            "acquisition_started_iso": format_timeline_timestamp(now),
                            "head_delay_ms": round(head_delay * 1000.0, 1),
                            "unwrap_size": self.unwrap_size,
                            "events": [],
                        }
                        add_timeline_event(timeline, "acquisition_start", now)
                        self.capture_session = {
                            "started_at": now,
                            "collect_after": now + head_delay,
                            "inactive_since": None,
                            "align_since": None,
                            "timeline": timeline,
                            "frames": {index: [] for index in range(4)},
                        }
                        self._log("INFO", "检测到钢管进入，开始整管拼接采图 @ %s" % format_timeline_timestamp(now))
                else:
                    if active:
                        session["inactive_since"] = None
                        session["align_since"] = None
                    else:
                        if session["inactive_since"] is None:
                            session["inactive_since"] = now
                            add_timeline_event(session["timeline"], "line_inactive", now)
                        no_signal = float(cfg.get("no_signal_ms", 80)) / 1000.0
                        tail = float(cfg.get("tail_delay_ms", 430)) / 1000.0
                        if now - session["inactive_since"] >= no_signal + tail:
                            row_counts = {
                                slot: sum(packet.height for packet in packets)
                                for slot, packets in session["frames"].items()
                            }
                            present_counts = [count for count in row_counts.values() if count > 0]
                            counts_aligned = len(present_counts) == 4 and len(set(present_counts)) == 1
                            if not counts_aligned and len(present_counts) < 2:
                                counts_aligned = True
                            if not counts_aligned and len(present_counts) >= 2:
                                if session["align_since"] is None:
                                    session["align_since"] = now
                                    self._log(
                                        "INFO",
                                        "等待四路最后一帧对齐，当前行数: %s" % ", ".join(
                                            "%s=%d" % (CAMERA_NAMES[slot], count)
                                            for slot, count in sorted(row_counts.items())
                                        ),
                                    )
                                align_wait = float(self.config.get("stream_align_ms", 1000)) / 1000.0
                                if now - session["align_since"] < align_wait:
                                    counts_aligned = False
                                else:
                                    self._log(
                                        "WARN",
                                        "四路对齐等待超时，按当前帧保存: %s" % ", ".join(
                                            "%s=%d" % (CAMERA_NAMES[slot], count)
                                            for slot, count in sorted(row_counts.items())
                                        ),
                                    )
                                    counts_aligned = True
                            if counts_aligned:
                                self._finalize_capture_session_locked("line_inactive")
                                self.waiting_for_release = True
                                self._log("INFO", "钢管离开并完成尾部补采，本根钢管已自动提交保存")
                                self._log("INFO", "触发采集保持运行，继续等待下一根钢管")
                    total_rows = sum(
                        sum(packet.height for packet in packets)
                        for packets in session["frames"].values()
                    )
                    if total_rows > 60000:
                        self._finalize_capture_session_locked("row_limit")
                        self.waiting_for_release = True
                        self._log("WARN", "整管图像达到 60000 行上限，已提前结束")
            time.sleep(0.01)

    def _finalize_capture_session_locked(self, reason: str = "unknown") -> None:
        session = self.capture_session
        self.capture_session = None
        if not session:
            return
        timeline = session.setdefault("timeline", {
            "acquisition_started_at": session.get("started_at", time.time()),
            "acquisition_started_iso": format_timeline_timestamp(session.get("started_at", time.time())),
            "events": [],
        })
        timeline.setdefault("unwrap_size", self.unwrap_size)
        finalize_started = add_timeline_event(timeline, "finalize_started")
        timeline["finalize_reason"] = reason
        timeline["no_signal_ms"] = float(self.camera_configs[0].get("no_signal_ms", 80))
        timeline["tail_delay_ms"] = float(self.camera_configs[0].get("tail_delay_ms", 430))
        timeline["stream_align_ms"] = float(self.config.get("stream_align_ms", 1000))

        assemble_started = time.perf_counter()
        combined: Dict[int, FramePacket] = {}
        camera_timings: Dict[str, Any] = {}
        all_packet_times: List[float] = []
        for slot, packets in session["frames"].items():
            if not packets:
                camera_timings[CAMERA_NAMES[slot]] = {"frame_count": 0, "rows": 0}
                continue
            packets.sort(key=lambda packet: packet.captured_at)
            packet_times = [packet.captured_at for packet in packets]
            all_packet_times.extend(packet_times)
            total_height = sum(packet.height for packet in packets)
            if total_height <= 0:
                continue
            combined[slot] = FramePacket(
                slot=slot,
                width=packets[0].width,
                height=total_height,
                pixel_type=packets[0].pixel_type,
                data=b"".join(packet.data for packet in packets),
                captured_at=packets[0].captured_at,
                frame_number=packets[-1].frame_number,
            )
            camera_timings[CAMERA_NAMES[slot]] = {
                "frame_count": len(packets),
                "rows": total_height,
                "first_frame_at": format_timeline_timestamp(packet_times[0]),
                "last_frame_at": format_timeline_timestamp(packet_times[-1]),
                "first_frame_epoch": round(packet_times[0], 3),
                "last_frame_epoch": round(packet_times[-1], 3),
            }
            self._log("INFO", "%s 拼接 %d 帧，整管高度 %d 行" % (CAMERA_NAMES[slot], len(packets), total_height))
        assembly_completed = add_timeline_event(timeline, "assembly_completed")
        timeline["camera_timings"] = camera_timings
        last_frame_epoch = max(all_packet_times) if all_packet_times else None
        if last_frame_epoch is not None:
            timeline["last_frame_at"] = format_timeline_timestamp(last_frame_epoch)
            timeline["last_frame_epoch"] = round(last_frame_epoch, 3)
        timeline["durations_ms"] = {
            "capture_start_to_last_frame": round((last_frame_epoch - timeline["acquisition_started_at"]) * 1000.0, 2) if last_frame_epoch else 0.0,
            "last_frame_to_finalize": round((finalize_started - last_frame_epoch) * 1000.0, 2) if last_frame_epoch else 0.0,
            "assembly": round(time.perf_counter() - assemble_started, 4) * 1000.0,
        }
        timeline["save_queued_at"] = format_timeline_timestamp(assembly_completed)
        timeline["save_queued_epoch"] = round(assembly_completed, 3)
        if combined:
            self.executor.submit(self._save_pipe, combined, timeline)
    def trigger_once(self) -> Dict[str, Any]:
        if self.mode != "acquisition":
            raise RuntimeError("请先启动触发采集")
        triggered = 0
        for runtime in self.cameras:
            if not runtime.worker:
                continue
            commands = ["FrameTriggerSoftware", "TriggerSoftware"]
            success = False
            for command in commands:
                try:
                    if runtime.worker.cam.MV_CC_SetCommandValue(command) == 0:
                        success = True
                        triggered += 1
                        break
                except Exception:
                    pass
        if triggered == 0:
            raise RuntimeError("软触发命令发送失败")
        self._log("INFO", "已发送软触发命令: %d 路" % triggered)
        return self.get_state()
    def stop_acquisition(self) -> Dict[str, Any]:
        self._stop_line_monitor()
        with self.capture_lock:
            if self.capture_session is not None:
                self._finalize_capture_session_locked("manual_stop")
        self.mode = "idle"
        self._clear_group()
        for runtime in self.cameras:
            if runtime.worker:
                runtime.worker.stop()
        self._log("INFO", "整管光电触发采集已停止")
        return self.get_state()
    def stop_preview(self) -> Dict[str, Any]:
        self.mode = "idle"
        for runtime in self.cameras:
            if runtime.worker:
                runtime.worker.stop()
        self._log("INFO", "实时预览已停止")
        return self.get_state()
    def start_all_preview(self) -> Dict[str, Any]:
        return self.start_preview()

    def apply_config(self, values: Dict[str, Any]) -> Dict[str, Any]:
        numeric_fields = {
            "exposure_us": (float, 1.0, 1000000.0),
            "gain_db": (float, -24.0, 24.0),
            "line_rate_hz": (int, 1, 100000),
            "head_delay_ms": (int, 0, 10000),
            "tail_delay_ms": (int, 0, 10000),
            "no_signal_ms": (int, 0, 5000),
            "stream_align_ms": (int, 0, 5000),
        }
        slot_value = values.get("slot")
        if slot_value is None:
            target_indices = list(range(4))
        else:
            try:
                slot = int(slot_value)
            except (TypeError, ValueError):
                raise ValueError("无效相机通道")
            if slot < 0 or slot >= 4:
                raise ValueError("无效相机通道")
            target_indices = [slot]

        for index in target_indices:
            cfg = self.camera_configs[index]
            for key, (kind, minimum, maximum) in numeric_fields.items():
                if key == "stream_align_ms":
                    continue
                if key in values:
                    try:
                        value = kind(values[key])
                    except (TypeError, ValueError):
                        raise ValueError("参数 %s 格式不正确" % key)
                    cfg[key] = max(minimum, min(maximum, value))
            if "trigger_source" in values:
                cfg["trigger_source"] = "Software" if values["trigger_source"] == "Software" else "Line0"
            if "trigger_polarity" in values:
                polarity = str(values["trigger_polarity"])
                if polarity not in ("LevelLow", "LevelHigh", "FallingEdge", "RisingEdge"):
                    raise ValueError("不支持的触发极性")
                cfg["trigger_polarity"] = polarity
            runtime = self.cameras[index]
            runtime.exposure_us = float(cfg["exposure_us"])
            runtime.gain_db = float(cfg["gain_db"])
            runtime.line_rate_hz = int(cfg["line_rate_hz"])

        if "folder_prefix" in values:
            prefix = str(values["folder_prefix"]).strip()
            for char in '<>:"/\\|?*':
                prefix = prefix.replace(char, "_")
            self.config["folder_prefix"] = prefix[:80] or "pipe"
        if "target_brightness" in values:
            try:
                brightness = int(values["target_brightness"])
            except (TypeError, ValueError):
                raise ValueError("目标亮度格式不正确")
            self.config["target_brightness"] = max(1, min(254, brightness))
        if "stream_align_ms" in values:
            try:
                align_ms = int(values["stream_align_ms"])
            except (TypeError, ValueError):
                raise ValueError("四路对齐等待格式不正确")
            self.config["stream_align_ms"] = max(0, min(5000, align_ms))
        if "output_dir" in values:
            self.set_output_dir(str(values["output_dir"]))
        self.config.update(self.camera_configs[0])
        self.save_settings()

        if self.mode in ("preview", "acquisition"):
            mode = self.mode
            for index in target_indices:
                runtime = self.cameras[index]
                if runtime.worker:
                    try:
                        runtime.error = ""
                        runtime.worker.start(mode)
                        runtime.status = "online"
                    except Exception as exc:
                        runtime.status = "online"
                        runtime.error = str(exc)
                        self._log("ERROR", "%s 参数应用失败: %s" % (runtime.name, exc))
        names = ", ".join(CAMERA_NAMES[index] for index in target_indices)
        self._log("INFO", "采集参数已应用到: %s" % names)
        return self.get_state()
    def set_output_dir(self, value: str) -> None:
        raw_value = (value or "captures").strip().strip('"')
        path = Path(raw_value)
        if not path.is_absolute():
            path = ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        self.config["output_dir"] = raw_value if not Path(raw_value).is_absolute() else str(path)
        self.save_settings()

    def output_dir(self, *, create: bool = True) -> Path:
        path = Path(str(self.config.get("output_dir", "captures")))
        if not path.is_absolute():
            path = ROOT / path
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def select_output_dir_dialog(self) -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception as exc:
            raise RuntimeError("当前 Python 未安装 tkinter，无法打开目录选择窗口: %s" % exc)
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askdirectory(
            parent=root,
            title="选择钢管图像保存目录",
            initialdir=str(self.output_dir()),
            mustexist=True,
        )
        root.destroy()
        if not selected:
            return ""
        self.set_output_dir(selected)
        self._log("INFO", "保存目录已更改为: %s" % selected)
        return selected
    def open_output_dir(self) -> str:
        path = self.output_dir()
        if platform.system() == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            raise RuntimeError("打开目录仅支持 Windows")
        return str(path)

    def _clear_group(self) -> None:
        with self.group_lock:
            self.group_frames = {}
            self.group_started_at = 0.0
            if self.group_timer:
                self.group_timer.cancel()
                self.group_timer = None

    def _on_frame(self, runtime: CameraRuntime, packet: FramePacket) -> None:
        now = time.monotonic()
        queue_preview = False
        with self.preview_condition:
            if (self.preview_pause_count == 0
                    and now - self.preview_times[runtime.slot] >= 0.22
                    and not self.preview_pending[runtime.slot]):
                self.preview_pending[runtime.slot] = True
                self.preview_times[runtime.slot] = now
                queue_preview = True
        if queue_preview:
            try:
                future = self.preview_executor.submit(self._build_preview, runtime.slot, packet)
                future.add_done_callback(
                    lambda _future, slot=runtime.slot: self._preview_finished(slot)
                )
            except RuntimeError:
                # Executor shutdown must not interrupt the acquisition callback.
                self._preview_finished(runtime.slot)

        if self.mode != "acquisition":
            return
        with self.capture_lock:
            session = self.capture_session
            if not session or packet.captured_at < session["collect_after"]:
                return
            packets = session["frames"][runtime.slot]
            if packets and packets[-1].frame_number == packet.frame_number:
                return
            packets.append(packet)
    def _flush_group_timeout(self) -> None:
        with self.group_lock:
            frames = dict(self.group_frames)
            self.group_frames = {}
            self.group_started_at = 0.0
            self.group_timer = None
        if frames:
            self.executor.submit(self._save_pipe, frames, {"unwrap_size": self.unwrap_size})

    def _preview_finished(self, slot: int) -> None:
        with self.preview_condition:
            self.preview_pending[slot] = False
            self.preview_condition.notify_all()

    @contextmanager
    def _pause_preview_for_unwrap(self) -> Iterator[None]:
        # Reference counting keeps previews paused while any batch is unwrapping.
        with self.preview_condition:
            self.preview_pause_count += 1
            first = self.preview_pause_count == 1
        try:
            if first:
                self._log("INFO", "开始展开：暂停四路实时预览更新，相机采集继续")
            with self.preview_condition:
                self.preview_condition.wait_for(lambda: not any(self.preview_pending))
            yield
        finally:
            with self.preview_condition:
                self.preview_pause_count -= 1
                if self.preview_pause_count == 0:
                    self._log("INFO", "展开任务已结束：恢复四路实时预览更新")

    def _build_preview(self, slot: int, packet: FramePacket) -> None:
        try:
            with self.preview_condition:
                if self.preview_pause_count:
                    return
            image = mono8_to_preview_bmp(packet.width, packet.height, packet.data, max_width=720)
            with self.preview_lock:
                self.preview_bytes[slot] = image
                self.preview_frame_info[slot] = {
                    "width": packet.width,
                    "height": packet.height,
                    "captured_at": packet.captured_at,
                    "frame_number": packet.frame_number,
                }
        except Exception as exc:
            self._log("WARN", "%s 预览生成失败: %s" % (CAMERA_NAMES[slot], exc))
    def _build_chinese_timeline(self, timeline: Dict[str, Any]) -> Dict[str, Any]:
        events = {event.get("stage"): event for event in timeline.get("events", [])}
        durations = timeline.get("durations_ms") or {}

        def duration_text(value_ms: float) -> str:
            value_ms = float(value_ms)
            if value_ms >= 1000.0:
                return "%.3f 秒" % (value_ms / 1000.0)
            return "%.1f ms" % value_ms

        def timestamp_text(stage: str) -> str:
            value = events.get(stage, {}).get("timestamp", "")
            return value[11:23] if len(value) >= 23 else value

        def epoch(stage: str) -> Optional[float]:
            value = events.get(stage, {}).get("epoch")
            return float(value) if value is not None else None

        rows: List[Dict[str, str]] = []

        def add_stage(label: str, stage: str, previous: Optional[str] = None, prefix: str = "", note: str = "") -> None:
            if stage not in events:
                return
            current_epoch = epoch(stage)
            previous_epoch = epoch(previous) if previous else None
            if previous is None:
                took = "—"
            elif current_epoch is not None and previous_epoch is not None:
                took = prefix + duration_text((current_epoch - previous_epoch) * 1000.0)
            else:
                took = "—"
            row = {"阶段": label, "时间": timestamp_text(stage), "耗时": took}
            if note:
                row["说明"] = note
            rows.append(row)

        add_stage("光电触发，采集开始", "acquisition_start")
        add_stage("钢管离开", "line_inactive", "acquisition_start", "采集阶段约 ")
        add_stage("收尾开始", "finalize_started", "line_inactive", "离开后等待约 ")
        add_stage("四路帧拼接完成", "assembly_completed", "finalize_started", "拼接 ")
        add_stage("开始保存", "save_started", "assembly_completed", "排队约 ")
        add_stage("实时预览已暂停", "preview_paused", "preview_pause_requested", "等待预览结束 ", "仅暂停画面更新，相机采集继续")
        add_stage("开始生成展开图", "unwrap_started", "save_started", "保存准备 ")
        add_stage("展开算法完成", "unwrap_completed", "unwrap_started", "展开算法 ")
        add_stage("展开图已按所选尺寸发布", "result_published", "unwrap_completed", "无损编码与发布 ", "浏览器此时可取图，尚未计入浏览器解码和显示")
        add_stage("unwrapped.png 保存完成", "unwrapped_image_saved", "result_published", "展开图写盘 ")
        add_stage("展开图归档失败", "unwrapped_archive_failed", "result_published")
        add_stage("四张原图开始并行保存", "raw_save_started", "unwrapped_image_saved", "紧接着 ")
        add_stage("四张原图保存完成", "raw_images_saved", "raw_save_started", "四图并行写盘 ")
        add_stage("capture.json 完成", "manifest_saved", "raw_images_saved", "清单写入 ")
        add_stage("全部处理完成", "pipeline_completed", "manifest_saved", "最终收尾 ")
        if rows and "pipeline_completed" in events:
            rows[-1]["耗时"] = duration_text(durations.get("pipeline_total", 0.0))

        camera_rows = []
        for name, timing in (timeline.get("camera_timings") or {}).items():
            camera_rows.append({
                "相机": name,
                "首帧时间": str(timing.get("first_frame_at", ""))[11:23],
                "末帧时间": str(timing.get("last_frame_at", ""))[11:23],
                "帧数": timing.get("frame_count", 0),
                "拼接行数": timing.get("rows", 0),
            })

        started = str(timeline.get("acquisition_started_iso", ""))
        return {
            "日期": started[:10],
            "触发到展开图可见": duration_text(durations.get("result_ready_from_start", 0.0)),  # 兼容旧清单，指服务端可取图
            "四路记录末帧到可取图": duration_text(durations["result_ready_after_last_frame"]) if "result_ready_after_last_frame" in durations else "—",
            "全部处理总耗时": duration_text(durations.get("pipeline_total", 0.0)),
            "阶段": rows,
            "四路相机采集明细": camera_rows,
        }

    def _notify_inspection(self, png, info: Dict[str, Any], folder: Path) -> None:
        callback = self.on_panorama_ready
        if callback is not None:
            try:
                callback(png, info, folder)
            except Exception as exc:
                self._log("ERROR", "无法安排孔洞检测 %s: %s" % (info["pipe_id"], exc))

    def _save_pipe(self, frames: Dict[int, FramePacket], timeline: Optional[Dict[str, Any]] = None) -> None:
        complete = len(frames) == 4
        first_frame_epoch = min(packet.captured_at for packet in frames.values())
        moment = datetime.fromtimestamp(first_frame_epoch)
        prefix = str(self.config.get("folder_prefix", "pipe")).strip() or "pipe"
        for char in '<>:"/\\|?*':
            prefix = prefix.replace(char, "_")
        state_prefix = "" if complete else "incomplete_"
        pipe_id = state_prefix + prefix[:80] + "_" + timestamp_for_path(moment)
        folder = self.output_dir(create=False) / pipe_id
        timeline = timeline or {}
        output_size = timeline.get("unwrap_size", self.unwrap_size)
        timeline.setdefault("acquisition_started_at", first_frame_epoch)
        timeline.setdefault("acquisition_started_iso", format_timeline_timestamp(first_frame_epoch))
        timeline["durations_ms"] = dict(timeline.get("durations_ms") or {})
        save_started = add_timeline_event(timeline, "save_started")
        with self.lock:
            self.save_pipe_id = pipe_id
            self.save_phase = "preparing"
            self.save_started_at = save_started
        records: List[Dict[str, Any]] = []
        composite_record = {}
        try:
            try:
                add_timeline_event(timeline, "preview_pause_requested")
                with self._pause_preview_for_unwrap():
                    add_timeline_event(timeline, "preview_paused")
                    unwrap_started = add_timeline_event(timeline, "unwrap_started")
                    unwrap_started_perf = time.perf_counter()
                    if self.unwrap_bridge is None:
                        raise RuntimeError(self.unwrap_bridge_error or "钢管展开模块不可用")
                    bridge_result = self.unwrap_bridge.process_packets(frames, output_size=output_size)
                    if not bridge_result.ok or bridge_result.image is None:
                        raise RuntimeError("%s: %s" % (bridge_result.status, bridge_result.message))
                    unwrap_completed = add_timeline_event(timeline, "unwrap_completed")
                    timeline["durations_ms"]["unwrap_algorithm"] = round((time.perf_counter() - unwrap_started_perf) * 1000.0, 2)
                unwrapped_height, unwrapped_width = bridge_result.image.shape
                unwrapped_path = folder / "unwrapped.png"
                view = memoryview(bridge_result.image)
                png = encode_mono8_png(unwrapped_width, unwrapped_height,
                                       view.cast("B") if view.c_contiguous else view.tobytes())
                unwrapped_ready = add_timeline_event(timeline, "result_published")
                timeline["durations_ms"]["result_ready_from_start"] = round(
                    (unwrapped_ready - float(timeline["acquisition_started_at"])) * 1000.0,
                    2,
                )
                if timeline.get("last_frame_epoch") is not None:
                    timeline["durations_ms"]["result_ready_after_last_frame"] = round(
                        (unwrapped_ready - timeline["last_frame_epoch"]) * 1000.0, 2)
                seam = getattr(bridge_result, "panorama_seam", None)
                composite_record = {
                    "file": "",
                    "panorama_seam": asdict(seam) if seam is not None else None,
                    "phase_calibration_id": getattr(bridge_result, "phase_calibration_id", None),
                    "width": unwrapped_width,
                    "height": unwrapped_height,
                    "angle_order": [0, 90, 180, 270],
                    "algorithm": "tube_unwrap",
                    "algorithm_version": "0.1.2",
                    "status": bridge_result.status,
                    "message": bridge_result.message,
                    "elapsed_ms": round(bridge_result.elapsed_ms, 2),
                    "hole_candidate_count": bridge_result.hole_count,
                    "camera_bounds": bridge_result.bounds,
                    "input_shapes": bridge_result.input_shapes,
                }
                self._notify_inspection(png if complete else None, dict(
                    composite_record, pipe_id=pipe_id, last_frame_epoch=timeline.get("last_frame_epoch"),
                    failure_reason=None if complete else "四路原图不完整，请复核采集"
                ), folder)
                with self.result_condition:
                    # Work may overlap: a slow older capture must never replace a newer image.
                    if first_frame_epoch >= self._latest_capture_epoch:
                        self._latest_capture_epoch = first_frame_epoch
                        self._latest_png = png
                        self.latest_unwrapped_path = unwrapped_path
                        self.latest_unwrapped_info = dict(composite_record, pipe_id=pipe_id, file=str(unwrapped_path),
                            captured_at=moment.isoformat(timespec="milliseconds"), raw_images_ready=False,
                            archive_status="pending", result_ready_ms=timeline["durations_ms"]["result_ready_from_start"],
                            timeline=timeline)
                        self.result_condition.notify_all()
                    if self.save_pipe_id == pipe_id:
                        self.save_phase = "unwrapped_ready"

                # This is already a background save worker. No second compression, new queue or disk read for the viewer.
                write_started = time.perf_counter()
                try:
                    write_encoded_png(unwrapped_path, png)
                    composite_record.update(file="unwrapped.png", archive_status="saved")
                    add_timeline_event(timeline, "unwrapped_image_saved")
                except OSError as exc:
                    composite_record.update(archive_status="failed", archive_error=str(exc))
                    self.last_error = str(exc)
                    add_timeline_event(timeline, "unwrapped_archive_failed")
                    self._log("ERROR", "展开图已发布但归档失败 %s: %s" % (pipe_id, exc))
                timeline["durations_ms"]["unwrapped_image_write"] = round((time.perf_counter() - write_started) * 1000, 2)
                with self.lock:
                    if self.latest_unwrapped_path == unwrapped_path:
                        self.latest_unwrapped_info["archive_status"] = composite_record["archive_status"]
                self._log(
                    "INFO",
                    "四视角展开完成: %s，%d × %d，算法 %.0f ms，触发至出图 %.1f ms @ %s" % (
                        pipe_id, unwrapped_width, unwrapped_height, bridge_result.elapsed_ms,
                        timeline["durations_ms"]["result_ready_from_start"],
                        format_timeline_timestamp(unwrapped_ready),
                    ),
                )
                self._log("INFO", "展开结果已可查看，四张原图转入后台并行保存")
            except Exception as composite_error:
                self.last_error = str(composite_error)
                add_timeline_event(timeline, "unwrap_failed")
                self._log("ERROR", "展开图生成失败: %s" % composite_error)
                self._notify_inspection(None, {"pipe_id": pipe_id, "failure_reason":
                                               f"展开图生成失败，{composite_error}"}, folder)

            with self.lock:
                if self.save_pipe_id == pipe_id:
                    self.save_phase = "raw_saving"
            add_timeline_event(timeline, "raw_save_started")
            raw_started_perf = time.perf_counter()
            raw_results: Dict[int, Dict[str, Any]] = {}
            raw_errors: List[str] = []

            def write_raw_image(slot: int) -> Tuple[int, Dict[str, Any]]:
                packet = frames[slot]
                file_name = "%s.png" % CAMERA_NAMES[slot]
                target_path = folder / file_name
                started_epoch = time.time()
                started_perf = time.perf_counter()
                write_mono8_png(target_path, packet.width, packet.height, packet.data)
                ended_epoch = time.time()
                return slot, {
                    "file": file_name,
                    "started_at": format_timeline_timestamp(started_epoch),
                    "completed_at": format_timeline_timestamp(ended_epoch),
                    "duration_ms": round((time.perf_counter() - started_perf) * 1000.0, 2),
                    "bytes": target_path.stat().st_size if target_path.exists() else 0,
                }

            if frames:
                with ThreadPoolExecutor(max_workers=min(4, len(frames)), thread_name_prefix="hik-raw-save") as raw_executor:
                    futures = {raw_executor.submit(write_raw_image, slot): slot for slot in sorted(frames)}
                    for future in as_completed(futures):
                        slot = futures[future]
                        try:
                            result_slot, result_info = future.result()
                            raw_results[result_slot] = result_info
                        except Exception as exc:
                            raw_errors.append("%s: %s" % (CAMERA_NAMES[slot], exc))
            if raw_errors:
                message = "原图并行保存失败: %s" % "; ".join(raw_errors)
                with self.lock:
                    if self.latest_unwrapped_info.get("pipe_id") == pipe_id:
                        self.latest_unwrapped_info["raw_archive_error"] = message
                raise RuntimeError(message)

            for slot in sorted(frames):
                packet = frames[slot]
                runtime = self.cameras[slot]
                write_info = raw_results.get(slot, {})
                records.append({
                    "channel": slot,
                    "camera": CAMERA_NAMES[slot],
                    "serial": runtime.serial,
                    "ip": runtime.ip,
                    "width": packet.width,
                    "height": packet.height,
                    "frame_number": packet.frame_number,
                    "captured_at": datetime.fromtimestamp(packet.captured_at).isoformat(timespec="milliseconds"),
                    "file": write_info.get("file", "%s.png" % CAMERA_NAMES[slot]),
                    "write_started_at": write_info.get("started_at", ""),
                    "write_completed_at": write_info.get("completed_at", ""),
                    "write_duration_ms": write_info.get("duration_ms", 0.0),
                    "file_bytes": write_info.get("bytes", 0),
                })
            add_timeline_event(timeline, "raw_images_saved")
            timeline["durations_ms"]["raw_images_write"] = round((time.perf_counter() - raw_started_perf) * 1000.0, 2)
            timeline["camera_write_timestamps"] = raw_results
            timeline["camera_write_durations_ms"] = {
                CAMERA_NAMES[slot]: raw_results.get(slot, {}).get("duration_ms", 0.0)
                for slot in sorted(frames)
            }
            timeline["raw_images_parallel"] = True
            with self.lock:
                if self.save_pipe_id == pipe_id:
                    self.save_phase = "raw_saved"
                if self.latest_unwrapped_path and self.latest_unwrapped_path.parent == folder:
                    self.latest_unwrapped_info["raw_images_ready"] = True

            manifest_started_perf = time.perf_counter()
            add_timeline_event(timeline, "manifest_write_started")
            manifest = {
                "pipe_id": pipe_id,
                "complete": complete,
                "camera_count": len(frames),
                "captured_at": moment.isoformat(timespec="milliseconds"),
                "model": CAMERA_MODEL,
                "parameters": {
                    "unwrap_size": {"width": output_size[0], "height": output_size[1]},
                    "output_dir": self.config.get("output_dir", "captures"),
                    "folder_prefix": prefix,
                    "target_brightness": self.config.get("target_brightness", 82),
                    "cameras": self.camera_configs,
                },
                "cameras": records,
                "unwrapped": composite_record,
                "中文摘要": {
                    "批次号": pipe_id,
                    "采集时间": moment.isoformat(timespec="milliseconds"),
                    "四路图片": "完整" if complete else "不完整",
                    "展开状态": "成功" if composite_record.get("status") == "ok" else "失败",
                    "展开图文件": composite_record.get("file", ""),
                    "展开图归档": composite_record.get("archive_status", "未生成"),
                    "展开图尺寸": (
                        "%d × %d" % (composite_record.get("width", 0), composite_record.get("height", 0))
                        if composite_record else ""
                    ),
                },
                "处理时间轴": self._build_chinese_timeline(timeline),
            }
            (folder / "capture.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            add_timeline_event(timeline, "manifest_saved")
            timeline["durations_ms"]["manifest_write"] = round((time.perf_counter() - manifest_started_perf) * 1000.0, 2)
            pipeline_completed = add_timeline_event(timeline, "pipeline_completed")
            timeline["completed_at"] = format_timeline_timestamp(pipeline_completed)
            timeline["completed_epoch"] = round(pipeline_completed, 3)
            timeline["durations_ms"]["save_total"] = round((pipeline_completed - save_started) * 1000.0, 2)
            timeline["durations_ms"]["pipeline_total"] = round(
                (pipeline_completed - float(timeline["acquisition_started_at"])) * 1000.0,
                2,
            )
            timeline["durations_ms"]["save_queue_wait"] = round(
                (save_started - float(timeline.get("save_queued_epoch", save_started))) * 1000.0,
                2,
            )
            with self.lock:
                if self.latest_unwrapped_path and self.latest_unwrapped_path.parent == folder:
                    self.latest_unwrapped_info["timeline"] = timeline
            # 将最终耗时和“全部处理完成”阶段写回 capture.json。
            manifest["处理时间轴"] = self._build_chinese_timeline(timeline)
            (folder / "capture.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            event_times = {event["stage"]: event["timestamp"] for event in timeline.get("events", [])}
            stage_labels = {
                "acquisition_start": "光电触发",
                "line_inactive": "钢管离开",
                "finalize_started": "收尾开始",
                "assembly_completed": "四路拼接完成",
                "save_started": "开始保存",
                "unwrap_started": "开始展开",
                "unwrap_completed": "展开算法完成",
                "result_published": "展开图可取",
                "unwrapped_image_saved": "展开图归档完成",
                "raw_save_started": "原图并行保存开始",
                "raw_images_saved": "四张原图保存完成",
                "manifest_saved": "capture.json完成",
                "pipeline_completed": "全部完成",
            }
            self._log(
                "INFO",
                "采集时间轴 %s: %s" % (
                    pipe_id,
                    " | ".join(
                        "%s=%s" % (stage_labels.get(stage, stage), event_times[stage][11:23])
                        for stage in stage_labels if stage in event_times
                    ),
                ),
            )
            durations = timeline["durations_ms"]
            self._log(
                "INFO",
                "采集耗时 %s: 触发→展开图可见 %.1f ms | 展开算法 %.1f ms | 展开图写盘 %.1f ms | 四图并行写盘 %.1f ms | 全部总耗时 %.1f ms" % (
                    pipe_id,
                    float(durations.get("result_ready_from_start", 0.0)),
                    float(durations.get("unwrap_algorithm", 0.0)),
                    float(durations.get("unwrapped_image_write", 0.0)),
                    float(durations.get("raw_images_write", 0.0)),
                    float(durations.get("pipeline_total", 0.0)),
                ),
            )
            self._log("INFO", "阶段时间戳已写入: %s" % (folder / "capture.json"))
            saved_images = len(frames) + int(bool(composite_record.get("file")))
            actual_image_count = len(list(folder.glob("*.png")))
            with self.lock:
                if self.save_pipe_id == pipe_id:
                    self.save_phase = "completed"
                self.capture_sequence += 1
                self.last_pipe_id = pipe_id
                self.saved_count += saved_images
                if complete:
                    self.pipe_count += 1
                else:
                    self.incomplete_count += 1
                    missing = [CAMERA_NAMES[i] for i in range(4) if i not in frames]
                    self._log("WARN", "不完整批次 %s，缺少 %s" % (pipe_id, ", ".join(missing)))
                history_item = {
                    "序号": self.capture_sequence,
                    "文件夹": folder.name,
                    "文件夹路径": str(folder),
                    "图像数量": actual_image_count,
                    "预期图像数量": 5,
                    "采集状态": "完整" if complete else "不完整",
                    "归档失败": composite_record.get("archive_status") == "failed",
                    "展开状态": "成功" if composite_record.get("status") == "ok" else "失败",
                    "采集时间": moment.isoformat(timespec="milliseconds"),
                    "完成时间": format_timeline_timestamp(pipeline_completed),
                    "触发到展开图可见_ms": round(float(timeline["durations_ms"].get("result_ready_from_start", 0.0)), 2),
                    "全部处理总耗时_ms": round(float(timeline["durations_ms"].get("pipeline_total", 0.0)), 2),
                }
                self.capture_history.append(history_item)
                if len(self.capture_history) > 200:
                    self.capture_history = self.capture_history[-200:]
                self._log(
                    "INFO",
                    "采集记录已登记: 序号=%d 文件夹=%s 图像=%d/%d 状态=%s" % (
                        self.capture_sequence,
                        folder.name,
                        actual_image_count,
                        5,
                        history_item["采集状态"],
                    ),
                )
        except Exception as exc:
            self.last_error = str(exc)
            with self.lock:
                if self.save_pipe_id == pipe_id:
                    self.save_phase = "failed"
            self._log("ERROR", "保存钢管图像失败: %s" % exc)
    def get_gencim_xml(self, slot: int) -> str:
        if slot < 0 or slot >= 4:
            raise ValueError("无效相机通道")
        runtime = self.cameras[slot]
        if not runtime.worker:
            raise RuntimeError("相机未连接")
        cam = runtime.worker.cam
        last_error = 0
        for size in (2 * 1024 * 1024, 8 * 1024 * 1024, 32 * 1024 * 1024):
            buffer = (self.sdk.c_ubyte * size)()
            actual_len = self.sdk.c_uint()
            ret = cam.MV_XML_GetGenICamXML(buffer, size, actual_len)
            if ret == 0 and actual_len.value > 0:
                return bytes(buffer[:actual_len.value]).decode("utf-8", errors="replace")
            last_error = ret
        raise RuntimeError("读取相机 XML 失败 (0x%08x)" % last_error)
    def wait_for_unwrapped(self, after: str, timeout: float = 15) -> Optional[Dict[str, Any]]:
        """Wait for a different pipe without polling camera state or the output directory."""
        with self.result_condition:
            if not self.result_condition.wait_for(
                    lambda: bool(self.latest_unwrapped_info) and self.latest_unwrapped_info["pipe_id"] != after,
                    timeout=timeout):
                return None
            info = {key: value for key, value in self.latest_unwrapped_info.items() if key != "timeline"}
            return dict(info, url="/api/unwrapped/latest.png")

    def get_latest_unwrapped(self, pipe_id: str = "") -> Tuple[bytes, Dict[str, Any]]:
        with self.lock:
            path = self.latest_unwrapped_path
            info = dict(self.latest_unwrapped_info)
            if pipe_id and pipe_id != info.get("pipe_id"):
                raise RuntimeError("结果已更新，请获取新批次")
            if self._latest_png is not None:
                return self._latest_png, info
        if not path or not path.exists():
            raise RuntimeError("当前没有可查看的展开图")
        return path.read_bytes(), info

    def get_preview(self, slot: int) -> Tuple[Optional[bytes], Dict[str, Any]]:
        if slot < 0 or slot >= 4:
            return None, {}
        with self.preview_lock:
            return self.preview_bytes[slot], dict(self.preview_frame_info[slot])

    def get_state(self) -> Dict[str, Any]:
        with self.preview_condition:
            preview_paused = self.preview_pause_count > 0
        with self.lock:
            cameras = []
            for runtime in self.cameras:
                cameras.append({
                    "slot": runtime.slot,
                    "name": runtime.name,
                    "serial": runtime.serial,
                    "model": runtime.model,
                    "ip": runtime.ip,
                    "status": runtime.status,
                    "width": runtime.width,
                    "height": runtime.height,
                    "frame_count": runtime.frame_count,
                    "last_frame_at": runtime.last_frame_at,
                    "error": runtime.error,
                    "preview_url": "/api/preview/%d.bmp" % runtime.slot if self.preview_bytes[runtime.slot] else "",
                    "config": dict(self.camera_configs[runtime.slot]),
                })
            with self.log_lock:
                logs = list(self.logs[-60:])
            return {
                "sdk_ready": self.sdk_ready,
                "sdk_error": self.sdk_error,
                "sdk_version": "0x%x" % self.sdk_version if self.sdk_version else "",
                "mvimport_path": str(self.mvimport_path),
                "runtime_path": self.runtime_path,
                "mode": self.mode,
                "preview_paused": preview_paused,
                "pipe_count": self.pipe_count,
                "saved_count": self.saved_count,
                "incomplete_count": self.incomplete_count,
                "last_pipe_id": self.last_pipe_id,
                "last_error": self.last_error,
                "latest_unwrapped": dict(
                    self.latest_unwrapped_info,
                    url="/api/unwrapped/latest.png" if self.latest_unwrapped_path else "",
                ) if self.latest_unwrapped_info else None,
                "save_pipe_id": self.save_pipe_id,
                "save_phase": self.save_phase,
                "capture_history": list(self.capture_history[-200:]),
                "unwrap_module_error": self.unwrap_bridge_error,
                "unwrap_size": self.output_sizes.controls(self.unwrap_size),
                "config": dict(self.camera_configs[0], output_dir=self.config.get("output_dir", "captures"), folder_prefix=self.config.get("folder_prefix", "pipe"), target_brightness=self.config.get("target_brightness", 82), stream_align_ms=self.config.get("stream_align_ms", 1000)),
                "camera_configs": [dict(cfg) for cfg in self.camera_configs],
                "output_path": str(self.output_dir(create=False)),
                "cameras": cameras,
                "logs": logs,
                "uptime_seconds": round(time.time() - self.started_at, 1),
            }

    @staticmethod
    def _shutdown_executor(executor: ThreadPoolExecutor) -> None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)

    def shutdown(self) -> None:
        try:
            self.stop_all(close_camera=True)
        finally:
            if self.sdk_ready and self.sdk is not None:
                try:
                    self.sdk.MvCamera.MV_CC_Finalize()
                except Exception:
                    pass
            self._shutdown_executor(self.executor)
            self._shutdown_executor(self.preview_executor)
