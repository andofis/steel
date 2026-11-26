"""YOLO-World mask decoding and cylindrical panorama projection."""

from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np

from tube_unwrap.tile_detection import (
    STEP_X,
    STEP_Y,
    TILE,
    WORKING_SIZE,
    TileBox,
    merge_holes,
    project_tile_box,
)

WORK_WIDTH, WORK_HEIGHT = WORKING_SIZE
WINDOWS = tuple(
    (x, y)
    for y in sorted({*range(0, WORK_HEIGHT - TILE + 1, STEP_Y), WORK_HEIGHT - TILE})
    for x in range(0, WORK_WIDTH, STEP_X)
)
CONFIDENCE = 0.25
MAX_MASK_BYTES = 64 * 1024 * 1024
EDGE_CONTACT = 4
STRIP_CONTEXT = 16
EDGE_FRAGMENT_SCORE = 0.60
FRAGMENT_MASK_COVERAGE = 0.70
COMPLEMENTARY_MASK_COVERAGE = 0.55


class MaskBudgetExceeded(ValueError):
    """The panorama produced more candidate mask data than the service can retain."""


@dataclass
class PackedMask:
    origin: tuple[int, int]
    shape: tuple[int, int]
    bits: np.ndarray

    def unpack(self):
        rows, cols = self.shape
        return np.unpackbits(self.bits, count=rows * cols).reshape(self.shape)


def model_inputs(canonical, windows):
    chips = []
    for x, y in windows:
        rows = canonical[y:y + TILE]
        chip = rows[:, x:x + TILE]
        if chip.shape[1] < TILE:
            chip = np.concatenate((chip, rows[:, :TILE - chip.shape[1]]), axis=1)
        chips.append(chip)
    gray = np.stack(chips)
    inputs = np.empty((len(chips), 3, TILE, TILE), dtype=np.float32)
    np.multiply(gray[:, None], np.float32(1 / 255), out=inputs)
    return inputs


def tile_masks(tile, boxes, scores, coefficients, prototypes):
    anchors = np.flatnonzero(
        (scores > CONFIDENCE) & (scores <= 1) & np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    )
    if not anchors.size:
        return

    raw_boxes = boxes[anchors].astype(np.float32)
    tile_scores = scores[anchors].astype(np.float32)
    rectangles = raw_boxes.copy()
    rectangles[:, 2:] -= rectangles[:, :2]
    indices = cv2.dnn.NMSBoxes(rectangles.tolist(), tile_scores.tolist(), CONFIDENCE, 0.70)
    indices = np.asarray(indices, dtype=np.intp).ravel()[:300]
    if not indices.size:
        return

    proto = prototypes.astype(np.float32, copy=False)
    weights = coefficients[anchors[indices]].astype(np.float32, copy=False)
    logits = cv2.gemm(weights, proto.reshape(proto.shape[0], -1), 1, None, 0)
    np.clip(logits, -80, 80, out=logits)
    probabilities = 1 / (1 + np.exp(-logits))
    for box, score, probability in zip(raw_boxes[indices], tile_scores[indices], probabilities, strict=True):
        left, top, right, bottom = np.clip(box, 0, TILE)
        x1, y1, x2, y2 = map(ceil, (left, top, right, bottom))
        if x1 == x2 or y1 == y2:
            continue
        # The benchmark interpolates probabilities before thresholding and applies the box at pixel indices.
        resized = cv2.resize(probability.reshape(proto.shape[1:]), (TILE, TILE), interpolation=cv2.INTER_LINEAR)
        mask = resized[y1:y2, x1:x2] > 0.5
        if mask.any():
            key = project_tile_box(tile, (left, top, right, bottom, score), *WORKING_SIZE, TILE)
            yield key, PackedMask((x1, y1), mask.shape, np.packbits(mask))


def _working_origin(tile, origin):
    return tile[0] + origin[0], tile[1] + origin[1]


def display_hole(box, mask, origin, width, height, cut_x):
    native_x, native_y = _working_origin(box.tile, origin)
    # Keep the seam arithmetic integral until projection, including cuts inside a working-grid pixel.
    first_edge = native_x * width - cut_x * WORK_WIDTH
    last_edge = first_edge + mask.shape[1] * width
    period = width * WORK_WIDTH
    max_x, max_y = np.nextafter([float(width), float(height)], 0)
    contours = []
    for band in range(first_edge // period, (last_edge - 1) // period + 1):
        left = band * period
        start = max(0, (left - first_edge) // width)
        stop = min(mask.shape[1], -((first_edge - left - period) // width))
        # A 2x raster gives even a one-pixel mask four vertices on the cell edges.
        doubled = np.repeat(np.repeat(mask[:, start:stop], 2, axis=0), 2, axis=1)
        outlines, _ = cv2.findContours(doubled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for outline in outlines:
            edges = (outline[:, 0] + 1) // 2
            x = ((edges[:, 0] + start) * width + first_edge - left) / WORK_WIDTH
            y = (edges[:, 1] + native_y) * (height / WORK_HEIGHT)
            points = np.column_stack((np.clip(x, 0, max_x), np.clip(y, 0, max_y)))
            distinct = (points != np.roll(points, 1, axis=0)).any(axis=1)
            contours.append(points[distinct].tolist())

    moments = cv2.moments(mask, binaryImage=True)
    pixels = moments["m00"]
    center_x = (moments["m10"] / pixels + 0.5) * width + first_edge
    center_y = (moments["m01"] / pixels + 0.5 + native_y) * (height / WORK_HEIGHT)
    return {
        "score": box.score,
        "center": [min(max_x, (center_x % period) / WORK_WIDTH), min(max_y, center_y)],
        "contours": contours,
        "area": pixels * (width / WORK_WIDTH) * (height / WORK_HEIGHT),
    }


def _overlap_with_full_mask(pixels, origin, full, full_origin):
    """Return the foreground overlap and the fragment's offset into the full mask's grid."""
    x, y = origin
    rows, cols = pixels.shape
    other_x, other_y = full_origin
    other_rows, other_cols = full.shape
    top, bottom = max(y, other_y), min(y + rows, other_y + other_rows)
    if top >= bottom:
        return 0, 0
    mask = None
    best_overlap, offset = 0, 0
    for shift in (-WORK_WIDTH, 0, WORK_WIDTH):
        start = other_x + shift
        left, right = max(x, start), min(x + cols, start + other_cols)
        if left >= right:
            continue
        if mask is None:
            mask = full.unpack()
        part = pixels[top - y:bottom - y, left - x:right - x]
        complete = mask[top - other_y:bottom - other_y, left - start:right - start]
        overlap = np.count_nonzero(part & complete)
        if overlap > best_overlap:
            best_overlap, offset = overlap, -shift
    return best_overlap, offset


def _weak_edge_strip(box, pixels, origin):
    """Reject thin, weak responses near a cut when another tile sees that region inside."""
    if box.score >= EDGE_FRAGMENT_SCORE:
        return False
    offset_x, offset_y, width, height = cv2.boundingRect(pixels)
    left, top = origin[0] + offset_x, origin[1] + offset_y
    right, bottom = left + width, top + height
    tile_x, tile_y = box.tile
    near_x = left - tile_x <= EDGE_CONTACT or tile_x + TILE - right <= EDGE_CONTACT
    near_y = ((tile_y > 0 and top - tile_y <= EDGE_CONTACT)
              or (tile_y + TILE < WORK_HEIGHT and tile_y + TILE - bottom <= EDGE_CONTACT))
    strip_x = 0 < width <= STRIP_CONTEXT and height >= 2 * width and near_x
    strip_y = 0 < height <= STRIP_CONTEXT and width >= 2 * height and near_y
    if not (strip_x or strip_y):
        return False
    for other_x, other_y in WINDOWS:
        interior_top = other_y + STRIP_CONTEXT if other_y > 0 else 0
        interior_bottom = other_y + TILE - STRIP_CONTEXT if other_y + TILE < WORK_HEIGHT else WORK_HEIGHT
        if (other_x, other_y) == box.tile or top < interior_top or bottom > interior_bottom:
            continue
        if any(other_x + STRIP_CONTEXT <= left + shift
               and right + shift <= other_x + TILE - STRIP_CONTEXT
               for shift in (-WORK_WIDTH, 0, WORK_WIDTH)):
            return True
    return False


def _combine_masks(origin, pixels, fragments):
    planes = [(origin, pixels), *fragments]
    left = min(x for (x, _), _ in planes)
    top = min(y for (_, y), _ in planes)
    right = max(x + p.shape[1] for (x, _), p in planes)
    bottom = max(y + p.shape[0] for (_, y), p in planes)
    merged = np.zeros((bottom - top, right - left), dtype=np.uint8)
    for (x, y), p in planes:
        x0, y0 = x - left, y - top
        merged[y0:y0 + p.shape[0], x0:x0 + p.shape[1]] |= p
    return (left, top), merged


def _resolve_edge_masks(kept, found):
    complete = [(key, found[key], _working_origin(key.tile, found[key].origin))
                for key in kept if not key.at_edge]
    discarded, joins = set(), {}
    for key in kept:
        if not key.at_edge:
            continue
        packed = found[key]
        pixels = packed.unpack()
        origin = _working_origin(key.tile, packed.origin)
        if _weak_edge_strip(key, pixels, origin):
            discarded.add(key)
            continue
        area = np.count_nonzero(pixels)
        matches = [(other, *_overlap_with_full_mask(pixels, origin, mask, full_origin))
                   for other, mask, full_origin in complete if other.tile != key.tile]
        if any(overlap >= FRAGMENT_MASK_COVERAGE * area for _, overlap, _ in matches):
            discarded.add(key)
            continue
        partners = [(other, offset) for other, overlap, offset in matches
                    if overlap >= COMPLEMENTARY_MASK_COVERAGE * area]
        if key.score >= EDGE_FRAGMENT_SCORE and len(partners) == 1 and partners[0][0].score >= key.score:
            other, offset = partners[0]
            joins.setdefault(other, []).append(((origin[0] + offset, origin[1]), pixels))
            discarded.add(key)
    return discarded, joins


def segment(image, cut_x, runner):
    height, width = image.shape
    canonical = np.roll(image, cut_x, axis=1) if cut_x else image
    if (width, height) != WORKING_SIZE:
        canonical = cv2.resize(canonical, WORKING_SIZE, interpolation=cv2.INTER_AREA)

    found: dict[TileBox, PackedMask] = {}
    mask_bytes = 0
    for offset in range(0, len(WINDOWS), runner.batch_size):
        group = WINDOWS[offset:offset + runner.batch_size]
        predictions = runner(model_inputs(canonical, group))
        for tile, output in zip(group, zip(*predictions, strict=True), strict=True):
            for key, mask in tile_masks(tile, *output):
                mask_bytes += mask.bits.nbytes
                if mask_bytes > MAX_MASK_BYTES:
                    raise MaskBudgetExceeded("Panorama exceeds the candidate mask budget")
                found[key] = mask

    kept = merge_holes(list(found), WORK_WIDTH)
    discarded, joins = _resolve_edge_masks(kept, found)
    holes = []
    for key in kept:
        if key in discarded:
            continue
        packed = found[key]
        origin, pixels = packed.origin, packed.unpack()
        if key in joins:
            plane, pixels = _combine_masks(_working_origin(key.tile, origin), pixels, joins[key])
            origin = plane[0] - key.tile[0], plane[1] - key.tile[1]
        holes.append(display_hole(key, pixels, origin, width, height, cut_x))
    return holes
