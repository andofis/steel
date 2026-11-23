"""Merge tiled *hole detector* boxes on a cylinder; independent of the camera candidate count."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np


WORKING_SIZE = (1152, 7500)  # Training grid (width, height) of pipe_holes_v2.1.
TILE, STEP_X, STEP_Y = 640, 288, 384


@dataclass(frozen=True)
class TileBox:
    # Continuous pixel edges in the periodic, calibrated working grid. x may cross zero.
    left: float
    top: float
    right: float
    bottom: float
    score: float
    tile: tuple[int, int]
    at_edge: bool

    @property
    def area(self) -> float:
        return (self.right-self.left) * (self.bottom-self.top)


def project_tile_box(tile: tuple[int, int], row: Sequence[float], width: int, height: int, side: int) -> TileBox:
    x, y = tile
    l, t, r, b, score = map(float, row)
    if not (0 < side <= min(width,height) and 0 <= x < width and 0 <= y <= height-side
            and 0 <= l < r <= side and 0 <= t < b <= side and 0 <= score <= 1):
        raise ValueError('Invalid tile window or detection coordinates')
    half = (r-l)/2
    center = (x + (l+r)/2) % width
    # The tube head and tail are real boundaries, not internal tile cuts.
    edge = ((side < width and (l <= 4 or r >= side-4)) or (y > 0 and t <= 4)
            or (y+side < height and b >= side-4))
    return TileBox(center-half, y+t, center+half, y+b, score, tile, edge)


def _intersection(a: TileBox, b: TileBox, period: int) -> float:
    height = min(a.bottom,b.bottom) - max(a.top,b.top)
    if height <= 0:
        return 0.
    return height * sum(max(0., min(a.right,b.right+dx) - max(a.left,b.left+dx))
                        for dx in (-period,0,period))


def merge_holes(boxes: Sequence[TileBox], period: int, *, nms_iou: float = .45,
                covered_fraction: float = .70) -> list[TileBox]:
    """Prefer complete views; never reject a lone detection just for being at a tile edge."""
    if period < 2 or not 0 < nms_iou < 1 or not 0 < covered_fraction <= 1:
        raise ValueError('Invalid ring or merge thresholds')

    def redundant(a, b, *, containment=False):
        hit = _intersection(a,b,period)
        return (hit > nms_iou*(a.area+b.area-hit)
                or (containment and hit >= covered_fraction*a.area))

    def nms(items):
        kept = []
        for box in sorted(items, key=lambda b: -b.score):
            if not any(redundant(box, k) for k in kept):
                kept.append(box)
        return kept

    full = nms(b for b in boxes if not b.at_edge)
    edges = nms(b for b in boxes if b.at_edge and not any(
        b.tile != k.tile and redundant(b,k,containment=True) for k in full))
    return full + edges


def output_hole(box: TileBox, working: tuple[int,int], output: tuple[int,int], cut_x: int) -> dict:
    """Return one logical hole even if drawing it on this image takes two rectangles."""
    w,h = output
    sx,sy = w/working[0], h/working[1]
    left = (box.left*sx - cut_x) % w
    span = (box.right-box.left)*sx
    top,bottom = box.top*sy,box.bottom*sy
    right = left+span
    segments = [[left,top,min(float(w),right),bottom]]
    if right > w+1e-7:
        segments.append([0.,top,right-w,bottom])
    return {'center_xy': [(left+span/2)%w,(top+bottom)/2], 'size_wh': [span,bottom-top],
            'confidence':box.score, 'edge_only':box.at_edge, 'segments_xyxy':segments}


def detect_on_grid(image: np.ndarray, predict_batch: Callable, *, cut_x: int = 0, batch: int = 8) -> list[TileBox]:
    """Return merged detections on the fixed, calibrated grid, regardless of the display cut.

    predict_batch takes BGR tiles and yields one iterable of local (l,t,r,b,score) rows per tile.
    Restoring the calibrated image *before* resampling avoids subpixel drift for arbitrary display cuts.
    """
    if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 2:
        raise ValueError('Expected a Mono8 panorama')
    h,w = image.shape
    if type(cut_x) is not int or not 0 <= cut_x < w or type(batch) is not int or batch < 1:
        raise ValueError('Invalid display origin or batch size')
    ww,wh = WORKING_SIZE
    canonical = np.roll(image, cut_x, axis=1) if cut_x else image
    if (w,h) != WORKING_SIZE:
        canonical = cv2.resize(canonical, WORKING_SIZE, interpolation=cv2.INTER_AREA)
    ys = sorted({*range(0,wh-TILE+1,STEP_Y),wh-TILE})
    windows = [(x,y) for y in ys for x in range(0,ww,STEP_X)]
    candidates = []
    for offset in range(0,len(windows),batch):
        group = windows[offset:offset+batch]; chips = []
        for x,y in group:
            rows = canonical[y:y+TILE]
            chip = rows[:,x:x+TILE] if x+TILE <= ww else np.concatenate((rows[:,x:],rows[:,:x+TILE-ww]),axis=1)
            chips.append(cv2.cvtColor(chip,cv2.COLOR_GRAY2BGR))
        for tile, predictions in zip(group,predict_batch(chips),strict=True):
            candidates.extend(project_tile_box(tile,row,ww,wh,TILE) for row in predictions)
    return sorted(merge_holes(candidates,ww), key=lambda b: ((b.top+b.bottom)/2, (b.left+b.right)/2))
