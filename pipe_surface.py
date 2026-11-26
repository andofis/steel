"""Register the pipe wall independently of the openings being inspected."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import cv2
import numpy as np


_GRID = np.array([1152, 7500])
_WIDTH, _HEIGHT = _GRID
_HALF_WIDTH, _HALF_HEIGHT = _GRID // 2
_PATCH_HALF_WIDTH, _PATCH_HALF_HEIGHT = 18, 27
_SEARCH_X, _SEARCH_Y = 10, 54
_TEXTURE_PAD = _PATCH_HALF_WIDTH + _SEARCH_X + 16


def _wrap(x, width):
    return (x + width / 2) % width - width / 2


def _canonical(png, cut, flipped=False):
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = np.roll(image, cut, axis=1) if cut else image
    if image.shape != (_HEIGHT, _WIDTH):
        image = cv2.resize(image, (_WIDTH, _HEIGHT), interpolation=cv2.INTER_AREA)
    if flipped:
        image = np.roll(image[::-1, ::-1], 1, axis=1)
    return image


def _texture(image):
    small = cv2.resize(image, (_HALF_WIDTH, _HALF_HEIGHT), interpolation=cv2.INTER_AREA)
    small = cv2.copyMakeBorder(small, 0, 0, _TEXTURE_PAD, _TEXTURE_PAD, cv2.BORDER_WRAP).astype(np.float32)
    band = cv2.GaussianBlur(small, (0, 0), 1.1) - cv2.GaussianBlur(small, (0, 0), 4, 9)
    return np.clip(band / np.sqrt(cv2.blur(band * band, (25, 37)) + 1.4), -3, 3)


@lru_cache(maxsize=2)
def _reference_surface(png, cut, flipped):
    image = _canonical(png, cut, flipped)
    return image, None if image is None else _texture(image)


def _wall_matches(a, b, template_holes, sample_holes, shift):
    half_width, half_height = _PATCH_HALF_WIDTH, _PATCH_HALF_HEIGHT
    search_x, search_y = _SEARCH_X, _SEARCH_Y
    dx, dy = np.rint(shift * (_GRID / 2)).astype(int)
    tm = template_holes[:, :2] * (_GRID / 2)
    sm = sample_holes[:, :2] * (_GRID / 2)
    # Keep the whole correlation search clear of opening pixels in both images.
    t_margin = np.maximum(template_holes[:, 2:] * (_GRID / 2) + 5, [31, 43])
    s_margin = np.maximum(sample_holes[:, 2:] * (_GRID / 2) + 5, [25, 39])
    xs = np.arange(7, _HALF_WIDTH, 35)
    offsets = (xs + dx) % _HALF_WIDTH
    t_clear_x = np.abs(_wrap(xs[:, None] - tm[:, 0], _HALF_WIDTH)) >= half_width + t_margin[:, 0]
    s_clear_x = (np.abs(_wrap(offsets[:, None] - sm[:, 0], _HALF_WIDTH)) >=
                 half_width + search_x + s_margin[:, 0])
    points = []
    for y in range(155, 3600, 85):
        sy = y + dy
        if not half_height + search_y <= sy <= _HALF_HEIGHT - half_height - search_y:
            continue
        t_near_y = np.abs(tm[:, 1] - y) < half_height + t_margin[:, 1]
        s_near_y = np.abs(sm[:, 1] - sy) < half_height + search_y + s_margin[:, 1]
        usable = t_clear_x[:, t_near_y].all(axis=1) & s_clear_x[:, s_near_y].all(axis=1)
        for index in np.flatnonzero(usable):
            x, sx = xs[index] + _TEXTURE_PAD, offsets[index] + _TEXTURE_PAD
            patch = a[y - half_height:y + half_height, x - half_width:x + half_width]
            if np.std(patch) < .45:
                continue
            roi = b[sy - half_height - search_y:sy + half_height + search_y,
                    sx - half_width - search_x:sx + half_width + search_x]
            scores = cv2.matchTemplate(roi, patch, cv2.TM_CCOEFF_NORMED)
            _, best, _, (px, py) = cv2.minMaxLoc(scores)
            if best < .43:
                continue
            scores[max(0, py - 5):py + 6, max(0, px - 2):px + 3] = -1
            if best - scores.max() >= .075:
                points.append((xs[index] * 2, y * 2, (dx + px - search_x) * 2,
                               (dy + py - search_y) * 2, best))
    return np.asarray(points, dtype=float).reshape(-1, 5)


@dataclass(frozen=True)
class SurfaceRegistration:
    reference: np.ndarray
    sample: np.ndarray
    shift: np.ndarray
    matches: np.ndarray

    @property
    def translation(self):
        """Global pose from wall matches, before any local adjustment."""
        return np.median(self.matches[:, 2:4], axis=0) / _GRID

    def displacement(self, point):
        """Predict the displacement from nearby wall patches, in normalized coordinates."""
        offsets = self.matches[:, :2] - point * _GRID
        offsets[:, 0] = _wrap(offsets[:, 0], _WIDTH)
        offsets /= [65, 230]
        distance = np.sum(offsets**2, axis=1)
        near = distance < 9
        if near.sum() < 6 or not np.any(distance < 4):
            return None
        rows = self.matches[near]
        if min(np.unique(rows[:, axis]).size for axis in (0, 1)) < 2:
            return None
        vectors = rows[:, 2:4].copy()
        vectors[:, 0] = self.shift[0] * _WIDTH + _wrap(vectors[:, 0] - self.shift[0] * _WIDTH, _WIDTH)
        design = np.column_stack((np.ones(len(rows)), offsets[near]))
        base = np.exp(-distance[near] / 2) * rows[:, 4] ** 4
        weight = base
        for _ in range(4):
            penalty = np.diag([1e-6, .05, .05]) * weight.sum()
            fit = np.linalg.solve((design.T * weight) @ design + penalty, (design.T * weight) @ vectors)
            residual = np.linalg.norm((design @ fit - vectors) / [3, 12], axis=1)
            weight = base / np.maximum(1, residual)
        if base[residual < 2.5].sum() < .5 * base.sum():
            return None
        displacement = fit[0] / _GRID
        if np.any(np.abs(displacement - self.shift) > np.array([22, 116]) / _GRID):
            return None
        return displacement

    def has_same_end_pattern(self, center, polygons):
        """Require the extra end candidate's foreground to be visible on the accepted tube."""
        x, y = np.rint(center * _GRID).astype(int)
        yy, xx = np.mgrid[max(0, y - 56):min(_HEIGHT, y + 57), x - 56:x + 57]
        mask = np.zeros(yy.shape, dtype=np.uint8)
        painted = []
        for contour in polygons:
            p = np.rint(contour * _GRID).astype(np.int32)
            p[:, 0] = _wrap(p[:, 0] - x, _WIDTH) + 56
            p[:, 1] = p[:, 1].clip(0, _HEIGHT - 1) - yy[0, 0]
            if np.any((p < 0) | (p >= np.array(mask.shape[::-1]))):
                return False
            painted.append(p)
        cv2.fillPoly(mask, painted, 1)
        inside = mask.astype(bool)
        if inside.sum() < 36:
            return False
        guard = ~cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8)).astype(bool)
        a = self.sample[yy, xx % _WIDTH].astype(float)
        estimate = self.displacement(center - self.shift)
        shift = self.shift if estimate is None else estimate
        xd, yd = np.rint(shift * _GRID).astype(int)
        for ox, oy in product(range(-6, 7), range(-10, 11, 2)):
            source_y = yy - yd + oy
            valid = (source_y >= 0) & (source_y < _HEIGHT)
            if (inside & valid).sum() < .9 * inside.sum() or (guard & valid).sum() < 36:
                continue
            b = self.reference[source_y.clip(0, _HEIGHT - 1), (xx - xd + ox) % _WIDTH].astype(float)
            a_background, b_background = (np.median(im[guard & valid]) for im in (a, b))
            a_dark = a < a_background - max(6, .22 * a_background)
            b_dark = b < b_background - max(6, .22 * b_background)
            if (a_dark & b_dark & inside & valid).sum() < .81 * inside.sum():
                continue
            left, right = a[valid] - a[valid].mean(), b[valid] - b[valid].mean()
            if left @ right > .86 * np.linalg.norm(left) * np.linalg.norm(right):
                return True
        return False


def register_surface(pngs, cuts, flipped, holes, shift):
    reference, features = _reference_surface(pngs[0], cuts[0], flipped)
    sample = _canonical(pngs[1], cuts[1])
    if reference is None or sample is None:
        return None
    matches = _wall_matches(features, _texture(sample), *holes, shift)
    # A few repeating scratches cannot establish a registration for a pipe.
    if (len(matches) < 30 or np.unique(matches[:, 0] // 288).size < 3
            or np.unique(matches[:, 1] // 1500).size < 3):
        return None
    return SurfaceRegistration(reference, sample, shift, matches)
