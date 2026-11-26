"""In-memory four-camera round tube unwrapping."""

from __future__ import annotations

import json
import math
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from importlib.resources import files
from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np

from .geometry import CameraCalibration, _row_profiles, locate_body_with_fallback, project_contour, source_coordinates
from .hole_blending import blend_holes
from .holes import Hole, HoleSettings, find_hole_groups, find_holes


class UnwrapStatus(str, Enum):
    OK = "ok"
    INVALID_INPUT = "invalid_input"
    REVIEW = "review"


@dataclass(frozen=True)
class PhaseCalibration:
    id: str
    width_fraction: tuple[float, float]
    phases: tuple[float, float, float, float]


@dataclass(frozen=True)
class Recipe:
    cameras: tuple[CameraCalibration, ...]
    input_shape: tuple[int, int]
    width: int
    height: int
    trim_px: float = 5.0
    blend_px: int = 12
    max_view_deg: float = 68.0
    hole_settings: HoleSettings = field(default_factory=HoleSettings)
    input_height_range: tuple[int, int] | None = None
    reference_size: tuple[int, int] | None = None  # Calibrated width/height, not an intermediate image.
    nominal_phase_calibration: PhaseCalibration | None = None

    @property
    def reference_grid(self) -> tuple[int, int]:
        return (self.width, self.height) if self.reference_size is None else self.reference_size

    @property
    def output_scale(self) -> tuple[float, float]:
        w, h = self.reference_grid
        # X is periodic; Y includes both end rows.
        return self.width / w, (self.height - 1) / (h - 1)

    @classmethod
    def sample(cls) -> Recipe:
        return cls._from_data(json.loads(files(__package__).joinpath("sample_recipe.json").read_text(encoding="utf-8")))

    @classmethod
    def from_json(cls, path: str | Path) -> Recipe:
        return cls._from_data(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def _from_data(cls, data: dict) -> Recipe:
        calibration = data.get("nominal_phase_calibration")
        return cls(
            cameras=tuple(CameraCalibration(**camera) for camera in data["cameras"]),
            input_shape=tuple(data["input_shape"]),
            width=data["width"],
            height=data["height"],
            reference_size=tuple(data["reference_size"]) if data.get("reference_size") is not None else None,
            trim_px=data.get("trim_px", 5.0),
            blend_px=data.get("blend_px", 12),
            max_view_deg=data.get("max_view_deg", 68.0),
            hole_settings=HoleSettings(**data.get("hole_settings", {})),
            input_height_range=(tuple(data["input_height_range"])
                                if "input_height_range" in data else None),
            nominal_phase_calibration=PhaseCalibration(**calibration) if calibration is not None else None,
        )


@dataclass(frozen=True)
class UnwrapResult:
    status: UnwrapStatus
    image: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    source_camera: np.ndarray | None = None
    holes: tuple[Hole, ...] = ()
    message: str = ""
    elapsed_ms: float = 0.0
    camera_bounds: tuple[tuple[float, float, float, float], ...] = ()
    phase_calibration_id: str | None = None


@dataclass(frozen=True)
class _View:
    columns: np.ndarray
    ratio: np.ndarray
    local_column: np.ndarray
    primary_spans: tuple[tuple[slice, slice], ...]


def _primary_spans(columns: np.ndarray, local: np.ndarray) -> tuple[tuple[slice, slice], ...]:
    runs = np.split(columns, np.flatnonzero(np.diff(columns) != 1) + 1)
    return tuple((slice(run[0], run[-1] + 1), slice(local[run[0]], local[run[-1]] + 1))
                 for run in runs if run.size)


class TubeUnwrapper:
    """Process a group in camera order. Reuse the instance for subsequent groups."""

    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        self._validate_recipe()
        axis = np.arange(recipe.width, dtype=np.float32)
        sx, _ = recipe.output_scale
        phases = np.array([camera.phase * sx for camera in recipe.cameras], dtype=np.float32)
        offset = (axis - phases[:, None] + recipe.width / 2) % recipe.width - recipe.width / 2
        angles = offset * np.float32(2 * np.pi / recipe.width)
        distances = np.abs(offset)
        rank = np.argsort(distances, axis=0, kind="stable")
        self._primary, self._secondary = rank[0], rank[1]
        best = np.take_along_axis(np.abs(angles), self._primary[None, :], axis=0)[0]
        limit = np.deg2rad(recipe.max_view_deg)
        if np.any(best > limit):
            raise ValueError("Camera recipe has gaps in circumferential coverage")
        self._alpha = np.zeros(recipe.width, dtype=np.float32)
        if recipe.blend_px:
            nearest = np.take_along_axis(distances, rank[:2], axis=0)
            weight = np.clip(1 - (nearest[1] - nearest[0]) / (2 * recipe.blend_px), 0, 1)
            alternate = np.take_along_axis(np.abs(angles), self._secondary[None, :], axis=0)[0]
            self._alpha = 0.5 * weight * (alternate <= limit)
        self._views = []
        for i, (camera, theta) in enumerate(zip(recipe.cameras, angles)):
            columns = np.flatnonzero(np.abs(theta) <= limit)
            active = theta[columns]
            k = camera.perspective
            ratio = np.sqrt(1 - k*k) * np.sin(active) / (1 - k*np.cos(active))
            local = np.full(recipe.width, -1, dtype=np.int32)
            local[columns] = np.arange(len(columns), dtype=np.int32)
            spans = _primary_spans(np.flatnonzero(self._primary == i), local)
            self._views.append(_View(columns, ratio.astype(np.float32), local, spans))
        self._rows = np.linspace(0, 1, recipe.height, dtype=np.float32)

    def _validate_recipe(self) -> None:
        r = self.recipe
        if len(r.cameras) != 4 or len(r.input_shape) != 2:
            raise ValueError("Recipe must specify four cameras and one input shape (height, width)")
        if len(r.reference_grid) != 2:
            raise ValueError("Reference grid must have two dimensions")
        dimensions = (*r.input_shape, r.width, r.height, *r.reference_grid)
        if any(not isinstance(value, int) or not 1 < value < 32767 for value in dimensions):
            raise ValueError("Input and output dimensions must be integers between 2 and 32766")
        if r.input_height_range is not None:
            bounds = r.input_height_range
            if (len(bounds) != 2 or any(not isinstance(value, int) for value in bounds)
                    or not 1 < bounds[0] <= r.input_shape[0] <= bounds[1] < 32767):
                raise ValueError("input_height_range must contain the nominal input height")
        if not isinstance(r.blend_px, int) or not 0 <= r.blend_px < r.width / 8:
            raise ValueError("blend_px must be a nonnegative integer smaller than width/8")
        if not math.isfinite(r.trim_px) or r.trim_px < 0 or not 0 < r.max_view_deg < 90:
            raise ValueError("Invalid trim or view angle")
        profile = r.nominal_phase_calibration
        if profile is not None:
            if (not isinstance(profile.id, str) or not profile.id.strip() or len(profile.width_fraction) != 2
                    or not all(map(math.isfinite, profile.width_fraction))
                    or not 0 < profile.width_fraction[0] <= 1 <= profile.width_fraction[1]
                    or len(profile.phases) != len(r.cameras)
                    or not all(0 <= phase < r.reference_grid[0] for phase in profile.phases)):
                raise ValueError("Invalid nominal camera phase calibration")
        for c in r.cameras:
            values = (c.left, c.right, c.top, c.bottom, c.phase, c.perspective)
            if not all(map(math.isfinite, values)) or not 0 <= c.phase < r.reference_grid[0]:
                raise ValueError("Invalid camera coordinates or phase")
            if not (0 <= c.left < c.right < r.input_shape[1]
                    and 0 <= c.top < c.bottom < r.input_shape[0]
                    and 0 <= c.perspective < math.cos(math.radians(r.max_view_deg))
                    and c.bottom - c.top > 2 * r.trim_px):
                raise ValueError("Invalid camera silhouette bounds or perspective")

    def process(self, frames: Sequence[np.ndarray], *, localization_executor: ThreadPoolExecutor | None = None) -> UnwrapResult:
        """Accept four uint8 Mono8 or BGR arrays. OK reports processing, not part acceptance."""
        started = perf_counter()

        def failure(status: UnwrapStatus, message: str) -> UnwrapResult:
            return UnwrapResult(status, message=message, elapsed_ms=(perf_counter()-started)*1000)

        if isinstance(frames, np.ndarray) or not hasattr(frames, "__len__") or len(frames) != 4:
            return failure(UnwrapStatus.INVALID_INPUT, "Expected four camera frames in recipe order")
        gray = []
        nominal_height, width = self.recipe.input_shape
        min_height, max_height = self.recipe.input_height_range or (nominal_height, nominal_height)
        for index, frame in enumerate(frames, 1):
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.ndim not in (2, 3) or frame.shape[1] != width
                    or not min_height <= frame.shape[0] <= max_height
                    or (frame.ndim == 3 and frame.shape[2] != 3)):
                return failure(UnwrapStatus.INVALID_INPUT, f"Camera {index}: unexpected shape or pixel format")
            gray.append(frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        def locate_pair(start: int):
            pair = []
            for i in (start, start + 1):
                try:
                    pair.append(locate_body_with_fallback(gray[i], self.recipe.cameras[i]))
                except ValueError as exc:
                    raise ValueError(f"Camera {i + 1}: {exc}") from exc
            return pair

        # Only one additional worker per capture. In the console this borrows a
        # drained preview worker; standalone callers can keep the serial path.
        later = None
        if localization_executor is not None:
            try:
                later = localization_executor.submit(locate_pair, 2)
            except RuntimeError:  # Executor already shutting down; finish this capture serially.
                pass
        try:
            first = locate_pair(0)
            try:
                second = later.result() if later is not None else locate_pair(2)
            except CancelledError:
                second = locate_pair(2)
        except ValueError as exc:
            return failure(UnwrapStatus.REVIEW, str(exc))
        finally:
            if later is not None and not later.done():
                # Drain the other pair even on an early failure before resuming live previews.
                try:
                    later.exception()
                except CancelledError:
                    pass
        geometries, cameras = zip(*(first + second), strict=True)

        r = self.recipe
        projection = self
        phase_calibration_id = None
        profile = r.nominal_phase_calibration
        if profile is not None:
            # Evaluate projected silhouette width on the same central rows in every view.
            fractions = np.linspace(.2, .8, 9, dtype=np.float32)
            widths = [np.median(_row_profiles(body, fractions, r.trim_px)[1]) / nominal.radius
                      for body, nominal in zip(geometries, r.cameras, strict=True)]
            low, high = profile.width_fraction
            if all(low <= width <= high for width in widths):
                cameras = tuple(replace(camera, phase=phase)
                                for camera, phase in zip(cameras, profile.phases, strict=True))
                projection = TubeUnwrapper(replace(r, cameras=cameras, nominal_phase_calibration=None))
                phase_calibration_id = profile.id
        ref_w, ref_h = r.reference_grid

        def project(camera_index: int, points: np.ndarray) -> tuple[np.ndarray, float]:
            return project_contour(points, geometries[camera_index], cameras[camera_index],
                                   ref_w, ref_h, r.trim_px)

        # Group/clamp the few contours at calibrated coordinates, not at integer output rows.
        # Otherwise the last output pixel can admit a contour outside the calibrated pipe end.
        holes, alternates = find_hole_groups(gray, geometries, project, ref_w, ref_h, settings=r.hole_settings)
        if (r.width, r.height) != (ref_w, ref_h):
            sx, sy = r.output_scale
            holes = tuple(h.scaled(sx, sy) for h in holes)
            alternates = tuple(tuple(h.scaled(sx, sy) for h in group) for group in alternates)
        strips = []
        for i, (frame, body, camera, view) in enumerate(zip(gray, geometries, cameras, projection._views), 1):
            strip = np.empty((r.height, view.columns.size), dtype=np.uint8)
            for start in range(0, r.height, 256):
                rows = slice(start, start + 256)
                map_x, map_y = source_coordinates(body, camera, view.ratio, projection._rows[rows], r.trim_px)
                if (not 0 <= map_x.min() <= map_x.max() <= frame.shape[1]-1
                        or not 0 <= map_y.min() <= map_y.max() <= frame.shape[0]-1):
                    return failure(UnwrapStatus.REVIEW, f"Camera {i}: estimated mapping falls outside image")
                cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, dst=strip[rows])
            strips.append(strip)
        image, source = projection._assemble(strips, holes, alternates)
        valid = np.ones_like(image, dtype=bool)
        camera_bounds = tuple((c.left, c.right, c.top, c.bottom) for c in cameras)
        return UnwrapResult(UnwrapStatus.OK, image, valid, source, holes,
                            elapsed_ms=(perf_counter()-started)*1000, camera_bounds=camera_bounds,
                            phase_calibration_id=phase_calibration_id)

    def _assemble(
        self, strips: list[np.ndarray], holes: tuple[Hole, ...],
        alternate_contours: tuple[tuple[Hole, ...], ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        r = self.recipe
        image = np.empty((r.height, r.width), dtype=np.uint8)
        source = np.broadcast_to(self._primary + 1, image.shape).astype(np.uint8)
        for i, view in enumerate(self._views):
            for columns, local in view.primary_spans:
                image[:, columns] = strips[i][:, local]
        for i, view in enumerate(self._views):
            columns = np.flatnonzero((self._secondary == i) & (self._alpha > 0))
            if columns.size:
                alpha = self._alpha[columns]
                mix = image[:, columns] * (1-alpha) + strips[i][:, view.local_column[columns]] * alpha
                image[:, columns] = np.rint(mix).astype(np.uint8)

        blend_holes(image, source, strips, tuple(view.local_column for view in self._views),
                    self._primary, self._alpha, holes, alternate_contours,
                    r.hole_settings.dark_threshold, output_scale=r.output_scale)
        return image, source
