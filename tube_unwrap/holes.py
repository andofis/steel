"""Extract dark exterior openings in the original views before panorama blending."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


class _BodyGeometry(Protocol):
    left: NDArray[np.floating]
    right: NDArray[np.floating]
    top: float
    bottom: float


_ProjectContour = Callable[[int, NDArray[np.float32]], tuple[NDArray[np.float32], float]]


@dataclass(frozen=True, slots=True)
class HoleSettings:
    """Thresholds calibrated against the acquisition lighting and minimum target hole."""

    dark_threshold: int = 55
    min_diameter_px: float = 8.0
    minimum_quality: float = 0.30
    minimum_compactness: float = 0.25
    merge_tolerance_px: float = 15.0

    def __post_init__(self) -> None:
        if not 0 < self.dark_threshold < 255:
            raise ValueError("dark_threshold must be in 1..254")
        if not np.isfinite(self.min_diameter_px) or self.min_diameter_px < 3:
            raise ValueError("min_diameter_px must be finite and at least 3")
        if not 0 <= self.minimum_quality <= 1:
            raise ValueError("minimum_quality must be in 0..1")
        if not 0 <= self.minimum_compactness <= 1:
            raise ValueError("minimum_compactness must be in 0..1")
        if not np.isfinite(self.merge_tolerance_px) or self.merge_tolerance_px < 0:
            raise ValueError("merge_tolerance_px must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class Hole:
    """Exterior opening bounds in the unfolded image, in pixels.

    camera is one-based. x is wrapped to the panorama width. The contour is
    unwrapped locally around x, so points may cross either horizontal edge.
    """

    x: float
    y: float
    width: float
    height: float
    camera: int
    contour: tuple[tuple[float, float], ...]


def _raw_contours(
    frame: NDArray[np.uint8], geometry: _BodyGeometry, settings: HoleSettings
) -> list[NDArray[np.float32]]:
    margin = 2
    first_row = max(0, int(np.ceil(geometry.top)))
    last_row = min(frame.shape[0], int(np.floor(geometry.bottom)) + 1)
    rows = np.arange(first_row, last_row, dtype=np.int32)
    if rows.size == 0:
        return []
    left = np.ceil(geometry.left[rows]).astype(np.int32) + margin
    right = np.floor(geometry.right[rows]).astype(np.int32) - margin
    x0 = max(0, int(left.min()))
    x1 = min(frame.shape[1], int(right.max()) + 1)
    if x1 <= x0:
        return []

    dark = cv2.inRange(frame[first_row:last_row, x0:x1], 0, settings.dark_threshold - 1)
    local_rows = rows - first_row
    polygon = np.concatenate(
        [np.column_stack((left - x0, local_rows)), np.column_stack((right - x0, local_rows))[::-1]]
    )
    interior = cv2.fillPoly(np.zeros_like(dark), [polygon], 255)
    cv2.bitwise_and(dark, interior, dst=dark)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 0.30 * settings.min_diameter_px**2
    accepted = []
    for contour in contours:
        if len(contour) < 3:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < settings.min_diameter_px or y == 0 or y + h >= len(rows):
            continue
        points = contour.reshape(-1, 2)
        cols = points[:, 0] + x0
        bounds_rows = points[:, 1]
        if np.any((cols <= left[bounds_rows] + 1) | (cols >= right[bounds_rows] - 1)):
            continue
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        if 4 * np.pi * area / perimeter**2 < settings.minimum_compactness:
            continue
        accepted.append((points + (x0, first_row)).astype(np.float32))
    return accepted


def _project_hole(
    points: NDArray[np.float32], camera: int, width: int, height: int
) -> Hole | None:
    if points.shape[0] < 3 or not np.isfinite(points).all():
        return None
    points = np.array(points, dtype=np.float32, order="C", copy=True)
    anchor = points[0, 0]
    points[:, 0] = (points[:, 0] - anchor + width / 2) % width - width / 2 + anchor
    moment = cv2.moments(points)
    if not moment["m00"] or points[:, 1].min() < 0 or points[:, 1].max() >= height:
        return None
    x = moment["m10"] / moment["m00"]
    y = moment["m01"] / moment["m00"]
    points[:, 0] += x % width - x
    return Hole(
        x=x % width,
        y=y,
        width=float(np.ptp(points[:, 0])),
        height=float(np.ptp(points[:, 1])),
        camera=camera + 1,
        contour=tuple((float(u), float(v)) for u, v in points),
    )


def find_holes(
    frames: Sequence[NDArray[np.uint8]],
    geometries: Sequence[_BodyGeometry],
    project_contour: _ProjectContour,
    width: int,
    height: int,
    *,
    settings: HoleSettings = HoleSettings(),
) -> tuple[Hole, ...]:
    """Return candidate openings and select the frontmost full view of duplicates.

    project_contour maps an Nx2 raw contour and returns its projected coordinates
    and cosine-based view quality (0 grazing, 1 frontal). Detection expects dark
    openings fully within the selected body and projects their exterior outline.
    """
    holes, _ = find_hole_groups(frames, geometries, project_contour, width, height, settings=settings)
    return holes


def find_hole_groups(
    frames: Sequence[NDArray[np.uint8]],
    geometries: Sequence[_BodyGeometry],
    project_contour: _ProjectContour,
    width: int,
    height: int,
    *,
    settings: HoleSettings = HoleSettings(),
) -> tuple[tuple[Hole, ...], tuple[tuple[Hole, ...], ...]]:
    """Keep other projections of the same hole for compositing only."""
    candidates = []
    for index, (frame, geometry) in enumerate(zip(frames, geometries, strict=True)):
        for contour in _raw_contours(frame, geometry, settings):
            points, quality = project_contour(index, contour)
            if not np.isfinite(quality) or quality < settings.minimum_quality:
                continue
            hole = _project_hole(points, index, width, height)
            if hole is not None:
                candidates.append((float(quality), hole))

    selected: list[tuple[Hole, list[Hole]]] = []
    for _, hole in sorted(candidates, key=lambda candidate: candidate[0], reverse=True):
        for other, duplicates in selected:
            if hole.camera == other.camera:
                continue
            dy = abs(hole.y - other.y)
            if dy > min(settings.merge_tolerance_px, 0.35 * min(hole.height, other.height)):
                continue
            dx = abs((hole.x - other.x + width / 2) % width - width / 2)
            if dx <= min(settings.merge_tolerance_px, 0.35 * min(hole.width, other.width)):
                duplicates.append(hole)
                break
        else:
            selected.append((hole, []))
    selected.sort(key=lambda group: (group[0].y, group[0].x))
    return (tuple(hole for hole, _ in selected),
            tuple(tuple(duplicates) for _, duplicates in selected))
