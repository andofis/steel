# -*- coding: utf-8 -*-
"""Adapter that applies the supplied tube_unwrap module to acquisition packets."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tube_unwrap import HoleSettings, Recipe, TubeUnwrapper, UnwrapStatus
from tube_unwrap.geometry import CameraCalibration

ROOT = Path(__file__).resolve().parent
DEFAULT_RECIPE = ROOT / "tube_unwrap_recipe.json"


@dataclass(frozen=True)
class BridgeResult:
    status: str
    image: Optional[np.ndarray]
    message: str
    elapsed_ms: float
    hole_count: int
    bounds: Tuple[Tuple[float, float, float, float], ...]
    input_shapes: Tuple[Tuple[int, int], ...]

    @property
    def ok(self) -> bool:
        return self.status == UnwrapStatus.OK.value and self.image is not None


def _estimate_vertical_bounds(image: np.ndarray) -> Tuple[int, int]:
    """Locate persistent tube end transitions in the line-scan image."""
    height = int(image.shape[0])
    if height < 64:
        raise ValueError("图像高度过小，无法定位钢管端部")
    profile = cv2.reduce(image, 1, cv2.REDUCE_SUM, dtype=cv2.CV_64F).ravel() / image.shape[1]
    smooth = np.convolve(profile, np.ones(21, dtype=np.float64) / 21.0, mode="same")
    gradient = np.abs(np.diff(smooth))
    if gradient.size < 32:
        raise ValueError("图像高度过小，无法定位钢管端部")

    top_start = max(16, int(height * 0.03))
    top_end = min(int(height * 0.25), gradient.size)
    bottom_start = max(256, int(height * 0.70))
    bottom_end = min(int(height * 0.985), gradient.size)
    if top_end <= top_start or bottom_end <= bottom_start:
        raise ValueError("钢管端部搜索范围无效")

    top = top_start + int(np.argmax(gradient[top_start:top_end]))
    bottom = bottom_start + int(np.argmax(gradient[bottom_start:bottom_end]))
    if bottom - top < max(128, int(height * 0.3)):
        raise ValueError("未能可靠定位钢管上下端")
    return top, bottom


class TubeUnwrapBridge:
    """Build a per-capture recipe from fixed optical calibration and dynamic tube ends."""

    def __init__(self, recipe_path: Path | str = DEFAULT_RECIPE, *, localization_executor: ThreadPoolExecutor | None = None) -> None:
        self.recipe_path = Path(recipe_path)
        self.localization_executor = localization_executor
        self.config = json.loads(self.recipe_path.read_text(encoding="utf-8"))
        if len(self.config.get("cameras", [])) != 4:
            raise ValueError("拼接配方必须包含四台相机")

    def _apply_contrast(self, slot: int, image: np.ndarray) -> np.ndarray:
        settings = self.config.get("contrast_luts", {}).get(str(slot))
        if not settings:
            return image
        offset = float(settings.get("offset", 0.0))
        gain = float(settings.get("gain", 1.0))
        lut = np.clip(np.rint((np.arange(256, dtype=np.float64) - offset) * gain + offset), 0, 255).astype(np.uint8)
        return cv2.LUT(image, lut)

    def _build_recipe(self, frames: Sequence[np.ndarray], bounds: Sequence[Tuple[int, int]]) -> Recipe:
        width = int(self.config["input_width"])
        heights = [int(frame.shape[0]) for frame in frames]
        nominal_height = max(heights)
        minimum_height = min(heights)
        perspective = float(self.config.get("perspective", 0.09))
        search_x = int(self.config.get("search_x", 100))
        search_y = int(self.config.get("search_y", 75))
        cameras: List[CameraCalibration] = []
        for camera_config, (top, bottom) in zip(self.config["cameras"], bounds):
            cameras.append(CameraCalibration(
                left=float(camera_config["left"]),
                right=float(camera_config["right"]),
                top=float(top),
                bottom=float(bottom),
                phase=float(camera_config["phase"]),
                perspective=perspective,
                search_x=search_x,
                search_y=search_y,
            ))
        return Recipe(
            cameras=tuple(cameras),
            input_shape=(nominal_height, width),
            input_height_range=(minimum_height, nominal_height),
            width=int(self.config.get("output_width", 1536)),
            height=int(self.config.get("output_height", 10000)),
            trim_px=float(self.config.get("trim_px", 5.0)),
            blend_px=int(self.config.get("blend_px", 20)),
            max_view_deg=float(self.config.get("max_view_deg", 68.0)),
            hole_settings=HoleSettings(**self.config.get("hole_settings", {})),
        )

    def process_packets(self, packets: Dict[int, Any]) -> BridgeResult:
        if sorted(packets) != [0, 1, 2, 3]:
            return BridgeResult(
                status=UnwrapStatus.INVALID_INPUT.value,
                image=None,
                message="四路钢管图片不完整，无法调用展开模块",
                elapsed_ms=0.0,
                hole_count=0,
                bounds=(),
                input_shapes=(),
            )

        arrays: List[np.ndarray] = []
        shapes: List[Tuple[int, int]] = []
        for slot in range(4):
            packet = packets[slot]
            expected = int(packet.width) * int(packet.height)
            data = packet.data
            if expected <= 0 or len(data) < expected:
                return BridgeResult(
                    status=UnwrapStatus.INVALID_INPUT.value,
                    image=None,
                    message="相机 %d 图像数据不足" % (slot + 1),
                    elapsed_ms=0.0,
                    hole_count=0,
                    bounds=(),
                    input_shapes=tuple(shapes),
                )
            image = np.frombuffer(data, dtype=np.uint8, count=expected).reshape(
                int(packet.height), int(packet.width)
            )
            if not isinstance(data, bytes):
                image = image.copy()
            image = self._apply_contrast(slot, image)
            arrays.append(image)
            shapes.append((int(packet.width), int(packet.height)))

        bounds_list: List[Tuple[int, int]] = []
        try:
            for image in arrays:
                bounds_list.append(_estimate_vertical_bounds(image))
            recipe = self._build_recipe(arrays, bounds_list)
            result = TubeUnwrapper(recipe).process(arrays, localization_executor=self.localization_executor)
        except Exception as exc:
            return BridgeResult(
                status=UnwrapStatus.REVIEW.value,
                image=None,
                message=str(exc),
                elapsed_ms=0.0,
                hole_count=0,
                bounds=tuple((0.0, 0.0, float(top), float(bottom)) for top, bottom in bounds_list),
                input_shapes=tuple(shapes),
            )

        calibration_bounds = result.camera_bounds or tuple(
            (float(camera.left), float(camera.right), float(camera.top), float(camera.bottom))
            for camera in recipe.cameras
        )
        return BridgeResult(
            status=result.status.value,
            image=result.image,
            message=result.message,
            elapsed_ms=float(result.elapsed_ms),
            hole_count=len(result.holes),
            bounds=calibration_bounds,
            input_shapes=tuple(shapes),
        )
