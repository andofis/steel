"""Locate the tube in each camera and map between image and surface pixels."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Nominal silhouette and the camera's frontal column in the panorama."""

    left: float
    right: float
    top: float
    bottom: float
    phase: float
    perspective: float = 0.09
    search_x: int = 32
    search_y: int = 54

    def __post_init__(self) -> None:
        if not np.isfinite((self.left, self.right, self.top, self.bottom,
                            self.phase, self.perspective)).all():
            raise ValueError("Camera calibration must contain finite numbers")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("Camera calibration bounds are reversed or empty")
        if not 0 <= self.perspective < 1:
            raise ValueError("Camera perspective must be in [0, 1)")
        if any(not isinstance(span, int) or span < 4 for span in (self.search_x, self.search_y)):
            raise ValueError("Camera search windows must be integers of at least 4 pixels")

    @property
    def center(self) -> float:
        return (self.left + self.right) / 2

    @property
    def radius(self) -> float:
        return (self.right - self.left) / 2


@dataclass(frozen=True)
class BodyGeometry:
    """Current side boundaries and cubic end curves in source image coordinates."""

    left: np.ndarray
    right: np.ndarray
    top: float
    bottom: float
    top_curve: np.ndarray
    bottom_curve: np.ndarray


def _gradient(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 1)
    return cv2.Sobel(blurred, cv2.CV_16S, dx, dy, ksize=3)


def _side_boundary(
    image: np.ndarray, rows: np.ndarray, x: int, span: int, sign: int,
) -> np.ndarray:
    start = x - span
    region = image[rows[0]-3:rows[-1]+4, start-3:x+span+4]
    gradient = sign * _gradient(region, 1, 0)[3:-3, 3:-3]
    # An interior mark can be stronger than the silhouette on individual rows.
    # Find the outermost persistent edge first, then refine close to that edge.
    persistent = gradient.astype(np.float32)
    for _ in range(3):
        cv2.blur(persistent, (1, 133), dst=persistent)
    threshold = np.maximum(25, persistent.max(axis=1, keepdims=True) * 0.44)
    columns = np.arange(gradient.shape[1])
    candidates = (persistent >= threshold) & (columns > 1) & (columns < 2 * span - 1)
    has_edge = candidates.any(axis=1)
    if sign > 0:
        outer = candidates.argmax(axis=1)
    else:
        outer = gradient.shape[1] - 1 - candidates[:, ::-1].argmax(axis=1)
    narrow = np.abs(columns - outer[:, None]) <= 5
    peaks = np.where(narrow, gradient, np.iinfo(gradient.dtype).min).argmax(axis=1)
    strengths = gradient[np.arange(len(rows)), peaks]
    usable = has_edge & (strengths >= 40) & (peaks > 1) & (peaks < 2 * span - 1)
    if usable.mean() < 0.70:
        raise ValueError("Tube side edge is weak or outside the calibrated search window")

    points = np.interp(rows, rows[usable], peaks[usable]).astype(np.float32)
    points = cv2.medianBlur(np.rint(points).astype(np.uint16)[:, None], 5)
    smoothed = cv2.GaussianBlur(points.astype(np.float32), (1, 111), 18.5).ravel()
    return np.interp(np.arange(image.shape[0]), rows, smoothed + start).astype(np.float32)


def _end_curve(
    image: np.ndarray,
    calibration: CameraCalibration,
    columns: np.ndarray,
    nominal_y: float,
    sign: int,
) -> np.ndarray:
    y = round(nominal_y)
    span = calibration.search_y
    region = image[y-span-3:y+span+4, columns[0]-3:columns[-1]+4]
    gradient = sign * _gradient(region, 0, 1)[3:-3, 3:-3]
    peaks = gradient.argmax(axis=0)
    strengths = gradient[peaks, np.arange(len(columns))]
    x = (columns - calibration.center) / calibration.radius
    positions = peaks + y - span
    usable = (strengths >= max(32, np.percentile(strengths, 25)))
    usable &= (peaks > 1) & (peaks < 2 * span - 1)
    for _ in range(4):
        if usable.mean() < 0.45 or np.ptp(x[usable]) < 0.95:
            raise ValueError("Tube end edge has insufficient support in the search window")
        curve = np.polynomial.polynomial.polyfit(x[usable], positions[usable], 3)
        residual = np.abs(np.polynomial.polynomial.polyval(x, curve) - positions)
        usable &= residual < max(2.0, 2 * np.percentile(residual[usable], 70))
    if usable.mean() < 0.45 or np.ptp(x[usable]) < 0.95:
        raise ValueError("Tube end profile could not be fitted reliably")
    return np.polynomial.polynomial.polyfit(x[usable], positions[usable], 3).astype(np.float32)


def locate_body(image: np.ndarray, calibration: CameraCalibration) -> BodyGeometry:
    """Fit the side and end profiles. Raise ValueError if a profile is unreliable."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Tube localization requires a 2-D uint8 image")
    h, w = image.shape
    xl, xr, yt, yb = map(round, (calibration.left, calibration.right,
                                 calibration.top, calibration.bottom))
    sx, sy = calibration.search_x, calibration.search_y
    if xl - sx < 3 or xr + sx + 3 >= w or yt - sy < 3 or yb + sy + 3 >= h:
        raise ValueError("Calibrated silhouette search windows exceed the input image")
    if xr - xl < 35 or yb - yt <= 2 * (sy + 30) + 8:
        raise ValueError("Calibrated silhouette is too small for the configured search windows")

    rows = np.arange(yt + sy + 30, yb - sy - 30)
    left = _side_boundary(image, rows, xl, sx, 1)
    right = _side_boundary(image, rows, xr, sx, -1)
    widths = (right - left)[rows]
    if np.mean((0.8 * (xr - xl) <= widths) & (widths <= 1.2 * (xr - xl))) < 0.95:
        raise ValueError("Observed tube width is inconsistent with its calibration")

    columns = np.arange(xl + 15, xr - 14)
    top_curve = _end_curve(image, calibration, columns, calibration.top, 1)
    bottom_curve = _end_curve(image, calibration, columns, calibration.bottom, -1)
    test_x = np.linspace(-1, 1, 33)
    lengths = np.polynomial.polynomial.polyval(test_x, bottom_curve - top_curve)
    if np.any(np.abs(lengths - (yb - yt)) > 2 * sy):
        raise ValueError("Observed tube length is inconsistent with its calibration")

    for array in (left, right, top_curve, bottom_curve):
        array.flags.writeable = False
    return BodyGeometry(left, right, float(top_curve[0]), float(bottom_curve[0]),
                        top_curve, bottom_curve)


def _row_profiles(
    body: BodyGeometry, row_fraction: np.ndarray, trim: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = body.top + trim + row_fraction * (body.bottom - body.top - 2 * trim)
    lower = np.clip(y.astype(np.intp), 0, len(body.left) - 2)
    fraction = y - lower.astype(np.float32)
    left = body.left[lower] + fraction * (body.left[lower + 1] - body.left[lower])
    right = body.right[lower] + fraction * (body.right[lower + 1] - body.right[lower])
    return (left + right) * 0.5, (right - left) * 0.5


def _evaluate_curve(coordinate: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return ((coefficients[3] * coordinate + coefficients[2]) * coordinate
            + coefficients[1]) * coordinate + coefficients[0]


def source_coordinates(
    body: BodyGeometry,
    calibration: CameraCalibration,
    projected_ratio: np.ndarray,
    output_rows: np.ndarray,
    trim: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV remap grids for the supplied projection ratios and row fractions."""
    row_fraction = np.asarray(output_rows, dtype=np.float32)
    ratios = np.asarray(projected_ratio, dtype=np.float32)
    center, radius = _row_profiles(body, row_fraction, trim)
    map_x = center[:, None] + radius[:, None] * ratios[None, :]
    normalized_x = (map_x - calibration.center) / calibration.radius
    start = _evaluate_curve(normalized_x, body.top_curve) + trim
    map_y = _evaluate_curve(normalized_x, body.bottom_curve)
    map_y -= trim
    map_y -= start
    map_y *= row_fraction[:, None]
    map_y += start
    return map_x, map_y


def project_contour(
    points: np.ndarray,
    body: BodyGeometry,
    calibration: CameraCalibration,
    width: int,
    height: int,
    trim: float,
) -> tuple[np.ndarray, float]:
    """Map camera contour points to wrapped panorama pixels and report frontality."""
    x, y = np.asarray(points, dtype=np.float32).T
    normalized_x = (x - calibration.center) / calibration.radius
    start = _evaluate_curve(normalized_x, body.top_curve) + trim
    end = _evaluate_curve(normalized_x, body.bottom_curve) - trim
    row_fraction = (y - start) / (end - start)
    center, radius = _row_profiles(body, np.clip(row_fraction, 0, 1), trim)
    ratio = np.clip((x - center) / radius, -1, 1)

    # Rearranging the perspective model yields a sine with a phase offset.
    k = calibration.perspective
    s = np.float32(np.sqrt(1 - k * k))
    offset = k * ratio
    angle = np.arcsin(np.clip(ratio / np.hypot(s, offset), -1, 1)) - np.arctan2(offset, s)
    panorama_x = (calibration.phase + angle * (width / (2 * np.pi))) % width
    panorama_y = row_fraction * (height - 1)
    return np.column_stack((panorama_x, panorama_y)), float(np.cos(angle).mean())
