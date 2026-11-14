"""Composite protected hole contours using a local transition on the tube wall."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .holes import Hole


_GUARD_PX = 4
_FEATHER_PX = 12
_MAX_BRIGHTNESS_OFFSET = 24


@dataclass(frozen=True)
class _Patch:
    rows: slice
    columns: np.ndarray
    camera: int
    donor: np.ndarray
    opening: np.ndarray
    protected: np.ndarray
    weight: np.ndarray
    blended: np.ndarray


def _hole_polygons(hole: Hole, alternates: tuple[Hole, ...], width: int) -> list[np.ndarray]:
    polygons = []
    for candidate in (hole, *alternates):
        delta = (candidate.x - hole.x + width / 2) % width - width / 2
        polygon = np.asarray(candidate.contour, dtype=np.float32).copy()
        polygon[:, 0] += hole.x + delta - candidate.x
        polygons.append(polygon)
    return polygons


def _make_patch(
    image: np.ndarray, strip: np.ndarray, local_column: np.ndarray, primary: np.ndarray,
    seam_alpha: np.ndarray, hole: Hole, alternates: tuple[Hole, ...], dark_threshold: int,
) -> _Patch | None:
    height, width = image.shape
    polygons = _hole_polygons(hole, alternates, width)
    coordinates = np.concatenate(polygons)
    pad = _GUARD_PX + _FEATHER_PX + 2
    x0, y0 = np.floor(coordinates.min(axis=0)).astype(int) - pad
    x1, y1 = np.ceil(coordinates.max(axis=0)).astype(int) + pad + 1
    y0, y1 = max(0, y0), min(height, y1)
    if y1 <= y0 or x1 - x0 > width:
        return None

    rows = slice(y0, y1)
    cols = np.arange(x0, x1) % width
    local = local_column[cols]
    valid = local >= 0
    already_single_view = np.all((primary[cols] == hole.camera - 1) & (seam_alpha[cols] == 0))

    union = np.zeros((y1 - y0, len(cols)), dtype=np.uint8)
    for index, polygon in enumerate(polygons):
        points = np.rint((polygon - (x0, y0)) * 16).astype(np.int32)
        cv2.fillPoly(union, [points], 1, shift=4)
        if index == 0:
            opening = union != 0

    if np.any(opening[:, ~valid]):
        return None
    donor = strip[rows, np.maximum(local, 0)]
    if already_single_view:
        return _Patch(rows, cols, hole.camera, donor, opening, opening,
                      np.zeros_like(union), donor)

    distance = cv2.distanceTransform(1 - union, cv2.DIST_L2, cv2.DIST_MASK_5)
    protected = (distance <= _GUARD_PX) & valid[None, :]

    feather = np.clip((distance - _GUARD_PX) / _FEATHER_PX, 0, 1)
    weight = 1 - feather * feather * (3 - 2 * feather)
    if not valid.all():
        # A feather can extend beyond the camera strip. Fade it before that edge.
        available = cv2.distanceTransform(valid[None, :].astype(np.uint8), cv2.DIST_L1, 3)[0]
        clearance = np.maximum(available - 1, 0)
        coverage = np.divide(clearance, clearance + feather * _FEATHER_PX,
                             out=np.ones_like(weight), where=feather > 0)
        weight *= coverage
        weight[:, ~valid] = 0

    background = image[rows, cols]
    outer_wall = ((feather >= 0.5) & (feather < 1) & valid[None, :]
                  & (donor >= dark_threshold) & (background >= dark_threshold)
                  & (donor < 250) & (background < 250))
    offset = 0.0
    if np.count_nonzero(outer_wall) >= 16:
        differences = background[outer_wall].astype(np.int16) - donor[outer_wall]
        offset = float(np.clip(np.median(differences), -_MAX_BRIGHTNESS_OFFSET, _MAX_BRIGHTNESS_OFFSET))
    adjusted = np.clip(donor + offset * feather, 0, 255)
    blended = np.rint(background * (1 - weight) + adjusted * weight).astype(np.uint8)
    return _Patch(rows, cols, hole.camera, donor, opening, protected, weight, blended)


def blend_holes(
    image: np.ndarray, source: np.ndarray, strips: list[np.ndarray],
    local_columns: tuple[np.ndarray, ...], primary: np.ndarray, seam_alpha: np.ndarray,
    holes: tuple[Hole, ...], alternate_contours: tuple[tuple[Hole, ...], ...], dark_threshold: int,
) -> None:
    patches = []
    for index, hole in enumerate(holes):
        camera = hole.camera - 1
        alternates = alternate_contours[index] if alternate_contours else ()
        patch = _make_patch(image, strips[camera], local_columns[camera], primary, seam_alpha,
                            hole, alternates, dark_threshold)
        if patch is not None:
            patches.append(patch)

    # Put all protected pixels back after all feathers. A nearby hole's
    # feather or duplicate contour must not blend away the chosen opening.
    for patch in patches:
        y, x = np.nonzero((patch.weight > 0) & ~patch.protected)
        image[patch.rows.start + y, patch.columns[x]] = patch.blended[y, x]
        source[patch.rows.start + y, patch.columns[x]] = np.where(
            patch.weight[y, x] >= 0.5, patch.camera, primary[patch.columns[x]] + 1,
        )
    for mask in ("protected", "opening"):
        for patch in patches:
            y, x = np.nonzero(getattr(patch, mask))
            image[patch.rows.start + y, patch.columns[x]] = patch.donor[y, x]
            source[patch.rows.start + y, patch.columns[x]] = patch.camera
