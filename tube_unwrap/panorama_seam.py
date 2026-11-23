"""Pick a publication origin on a circular panorama; camera calibration never changes."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Iterable, Sequence

import numpy as np

from .holes import Hole


@dataclass(frozen=True)
class PanoramaSeam:
    # Published column 0 is this column of the calibrated image, in output pixels.
    cut_x: int
    status: str
    clearance_px: int | None
    crossing_candidates: int | None


def choose_seam(width: int, spans: Iterable[tuple[float, float]], margin_px: int) -> PanoramaSeam:
    """Use one straight cut for every row. Spans are locally unwrapped (e.g. -30..12)."""
    if width < 2 or margin_px < 1:
        raise ValueError('Invalid panorama width or seam margin')
    # Difference array counts how many protected silhouettes each cut would touch.
    diff = np.zeros(width + 1, dtype=np.int32)
    for left, right in spans:
        if not (isfinite(left) and isfinite(right) and left <= right):
            raise ValueError('Invalid hole longitude span')
        first, last = floor(left), ceil(right)  # conservative to the nearest outside column
        length = min(width, last - first + 1)
        start = first % width
        stop = start + length
        diff[start] += 1
        diff[min(stop, width)] -= 1
        if stop > width:
            diff[0] += 1
            diff[stop - width] -= 1
    counts = np.cumsum(diff[:-1]); occupied = np.flatnonzero(counts)
    if not occupied.size:
        return PanoramaSeam(0, 'no_candidates', None, 0)
    columns = np.arange(width)
    if counts.min() > 0:
        # No straight cut can leave every candidate whole. Minimize splits; report it.
        choices = columns[counts == counts.min()]
        cut = int(choices[np.argmin(np.minimum(choices, width - choices))])
        return PanoramaSeam(cut, 'no_clear_column', 0, int(counts[cut]))
    idx = np.searchsorted(occupied, columns)
    clearance = np.minimum((columns - occupied[(idx-1) % occupied.size]) % width,
                           (occupied[idx % occupied.size] - columns) % width)
    if clearance[0] >= margin_px:
        return PanoramaSeam(0, 'kept', int(clearance[0]), 0)
    best = int(clearance.max())
    choices = columns[clearance == best]
    cut = int(choices[np.argmin(np.minimum(choices, width - choices))])
    return PanoramaSeam(cut, 'limited_clearance' if best < margin_px else 'shifted', best, 0)


def present_panorama(image: np.ndarray, holes: Sequence[Hole], mode: str) -> tuple[np.ndarray, PanoramaSeam]:
    """Presentation only: do not modify calibrated contours, source map or raw frames."""
    if mode == 'fixed':
        return image, PanoramaSeam(0, 'fixed', None, None)
    if mode != 'auto':
        raise ValueError('panorama_seam must be auto or fixed')
    width = image.shape[1]
    spans = ((min(x for x, _ in h.contour), max(x for x, _ in h.contour)) if h.contour
             else (h.x - h.width/2, h.x + h.width/2) for h in holes)
    seam = choose_seam(width, spans, max(2, round(width/64)))
    return (np.roll(image, -seam.cut_x, axis=1) if seam.cut_x else image), seam
