"""Automatic 4-camera circumferential ring alignment (direction, phases, and perspective)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .geometry import BodyGeometry, CameraCalibration, _row_profiles, source_coordinates


AUTO_RING_CALIBRATION_ID = "auto-ring-v1"

# Minimum overlap evidence required before trusting image-based ring alignment over a recipe.
_MIN_RING_SCORE = 1.20
_MIN_DIRECTION_MARGIN = 0.60
_MIN_PAIR_CORRELATION = 0.25

# Tolerances within which a matching-direction calibrated recipe is kept untouched.
_RECIPE_MAX_DELTA_DIFF_PX = 18.0
_RECIPE_MAX_PERSPECTIVE_DIFF = 0.026


@dataclass(frozen=True)
class RingAlignment:
    """Solved circumferential geometry on the calibrated reference grid."""

    phases: tuple[float, float, float, float]
    perspective: float
    direction: int
    ring_score: float
    pair_correlations: tuple[float, float, float, float]


def _ring_dp_max(curves: Sequence[np.ndarray], d_min: int, target_sum: int) -> tuple[float, list[int]]:
    """Maximize sum_i curves[i][d_i - d_min] subject to sum_i d_i == target_sum."""
    n = len(curves[0])
    target_idx = target_sum - 4 * d_min
    if target_idx < 0 or target_idx > 4 * (n - 1):
        return -1e9, [target_sum // 4] * 4
    dp: list[np.ndarray] = [curves[0].astype(np.float64, copy=True)]
    back: list[np.ndarray] = []
    for step in range(1, 4):
        prev = dp[-1]
        cur = curves[step]
        new_len = len(prev) + n - 1
        nxt = np.full(new_len, -1e9, dtype=np.float64)
        choice = np.zeros(new_len, dtype=np.int32)
        for j in range(n):
            cand = prev + cur[j]
            sl = slice(j, j + len(prev))
            better = cand > nxt[sl]
            nxt[sl] = np.where(better, cand, nxt[sl])
            choice[sl] = np.where(better, j, choice[sl])
        dp.append(nxt)
        back.append(choice)
    best_score = float(dp[-1][target_idx])
    idxs = [0] * 4
    s = target_idx
    for step in range(3, 0, -1):
        j = int(back[step - 1][s])
        idxs[step] = j
        s -= j
    idxs[0] = s
    return best_score, [idx + d_min for idx in idxs]


def _pair_ncc_curves(
    a: np.ndarray, b: np.ndarray, d_lo: int, d_hi: int, signs: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Compute max-over-dy(-1..1) normalized cross-correlation curves for adjacent camera strips."""
    w_st = a.shape[1]
    a_in = a[1:-1]
    na2 = np.sum(a_in * a_in, axis=0, dtype=np.float64)
    pa2 = np.concatenate(([0.0], np.cumsum(na2)))
    out = {s: np.full(d_hi - d_lo + 1, -1.0, dtype=np.float64) for s in signs}
    for dy in (-1, 0, 1):
        b_sh = b[1 + dy : b.shape[0] - 1 + dy]
        nb2 = np.sum(b_sh * b_sh, axis=0, dtype=np.float64)
        pb2 = np.concatenate(([0.0], np.cumsum(nb2)))
        cross = a_in.T @ b_sh
        for sign in signs:
            curve = out[sign]
            for idx, mag in enumerate(range(d_lo, d_hi + 1)):
                d = sign * mag
                num = float(np.trace(cross, offset=-d))
                if d >= 0:
                    ea = pa2[w_st] - pa2[d]
                    eb = pb2[w_st - d]
                else:
                    ea = pa2[w_st + d]
                    eb = pb2[w_st] - pb2[-d]
                den = math.sqrt(max(ea * eb, 0.0))
                val = num / den if den > 1e-6 else 0.0
                if val > curve[idx]:
                    curve[idx] = val
    return out


def _subbin_refine(curves: Sequence[np.ndarray], d_mags: Sequence[int], d_lo: int, target_sum: int) -> np.ndarray:
    """Refine discrete DP shifts with curvature-weighted parabolic interpolation under exact closure."""
    sub = np.asarray(d_mags, dtype=np.float64)
    curvatures = np.full(4, 1e-4, dtype=np.float64)
    for i in range(4):
        idx = d_mags[i] - d_lo
        c = curves[i]
        if 0 < idx < len(c) - 1:
            y0, y1, y2 = float(c[idx - 1]), float(c[idx]), float(c[idx + 1])
            denom = y0 - 2.0 * y1 + y2
            if denom < -1e-4:
                sub[i] += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
                curvatures[i] = -denom
    inv_w = 1.0 / np.maximum(curvatures, 1e-4)
    residual = float(target_sum) - float(sub.sum())
    sub += residual * (inv_w / float(inv_w.sum()))
    return sub


def estimate_ring_alignment(
    frames: Sequence[np.ndarray],
    geometries: Sequence[BodyGeometry],
    cameras: Sequence[CameraCalibration],
    nominal_cameras: Sequence[CameraCalibration],
    ref_width: int,
    trim_px: float,
    max_view_deg: float,
) -> RingAlignment | None:
    """Estimate circumferential camera order, relative phases, and perspective from seam overlaps.

    Returns None when wall texture in overlap regions is too weak to establish alignment
    (for example, uniform synthetic test frames) or when the current calibration already agrees.
    """
    if len(frames) != 4 or ref_width < 64:
        return None
    min_span = min(body.bottom - body.top - 2.0 * trim_px for body in geometries)
    if min_span < 32:
        return None

    w_small = max(128, min(384, ref_width // 4))
    scale_to_ref = ref_width / float(w_small)
    half_deg = min(max_view_deg - 2.0, 64.0)
    half = max(16, round(half_deg / 360.0 * w_small))
    w_st = 2 * half + 1
    d_lo = round(72.0 / 360.0 * w_small)
    d_hi = min(round(108.0 / 360.0 * w_small), w_st - 12)
    if d_hi <= d_lo or 4 * d_lo > w_small or 4 * d_hi < w_small:
        return None

    thetas = np.arange(-half, half + 1, dtype=np.float32) * np.float32(2.0 * np.pi / w_small)
    n_rows = min(256, max(48, int(min_span // 4)))
    rows = np.linspace(0.06, 0.94, n_rows, dtype=np.float32)
    blur_h = min(31, max(7, (n_rows // 8) | 1))

    # Scale nominal perspective linearly with observed tube radius (k = R / D).
    fractions = np.linspace(0.2, 0.8, 9, dtype=np.float32)
    obs_radii = [float(np.median(_row_profiles(body, fractions, trim_px)[1])) for body in geometries]
    nom_radii = [float(cam.radius) for cam in nominal_cameras]
    width_ratio = float(np.mean(obs_radii)) / max(float(np.mean(nom_radii)), 1.0)
    base_k = float(np.mean([cam.perspective for cam in cameras]))
    max_k = min(0.25, math.cos(math.radians(max_view_deg)) - 0.02)
    if max_k <= 0.04:
        return None
    k_init = float(np.clip(max(base_k, 0.09) * width_ratio, 0.04, max_k))

    def build_features(k: float) -> list[np.ndarray] | None:
        ratio = (math.sqrt(1.0 - k * k) * np.sin(thetas) / (1.0 - k * np.cos(thetas))).astype(np.float32)
        feats: list[np.ndarray] = []
        for frame, body, cam in zip(frames, geometries, cameras, strict=True):
            mx, my = source_coordinates(body, cam, ratio, rows, trim_px)
            if (mx.min() < 0 or mx.max() > frame.shape[1] - 1
                    or my.min() < 0 or my.max() > frame.shape[0] - 1):
                return None
            strip = cv2.remap(frame, mx, my, cv2.INTER_LINEAR).astype(np.float32)
            # Vertical high-pass removes lighting/specular bands that are constant along the pipe axis.
            hp = strip - cv2.GaussianBlur(strip, (1, blur_h), 0)
            feats.append(cv2.GaussianBlur(hp, (3, 3), 0))
        return feats

    feats0 = build_features(k_init)
    if feats0 is None:
        return None

    pos0: list[np.ndarray] = []
    neg0: list[np.ndarray] = []
    for i in range(4):
        pair = _pair_ncc_curves(feats0[i], feats0[(i + 1) % 4], d_lo, d_hi, (1, -1))
        pos0.append(pair[1])
        neg0.append(pair[-1])

    s_pos, d_pos = _ring_dp_max(pos0, d_lo, w_small)
    s_neg, d_neg = _ring_dp_max(neg0, d_lo, w_small)
    direction = 1 if s_pos >= s_neg else -1
    s_best = s_pos if direction == 1 else s_neg
    s_alt = s_neg if direction == 1 else s_pos
    curves = pos0 if direction == 1 else neg0
    d_mags = d_pos if direction == 1 else d_neg

    if s_best < _MIN_RING_SCORE or (s_best - s_alt) < _MIN_DIRECTION_MARGIN:
        return None
    free_peaks = [d_lo + int(np.argmax(c)) for c in curves]
    free_corrs = tuple(float(curves[i][free_peaks[i] - d_lo]) for i in range(4))
    if min(free_corrs) < _MIN_PAIR_CORRELATION:
        return None

    # Refine perspective k using the unconstrained loop-closure residual (sum |d_i| - w_small).
    # Do this before requiring every closed-loop pair to pass the correlation floor, since a
    # large perspective mismatch pushes one constrained pair off its peak until k is updated.
    pair_corrs = tuple(float(curves[i][d_mags[i] - d_lo]) for i in range(4))
    free_err = sum(free_peaks) - w_small
    k_est = k_init
    slope = 350.0 * (w_small / 384.0)
    k_candidate = float(np.clip(k_init + free_err / slope, 0.04, max_k))
    if abs(free_err) >= 2 and abs(k_candidate - k_init) >= 0.005:
        feats1 = build_features(k_candidate)
        if feats1 is not None:
            curves1 = [
                _pair_ncc_curves(feats1[i], feats1[(i + 1) % 4], d_lo, d_hi, (direction,))[direction]
                for i in range(4)
            ]
            s1, d1 = _ring_dp_max(curves1, d_lo, w_small)
            corrs1 = tuple(float(curves1[i][d1[i] - d_lo]) for i in range(4))
            if s1 >= s_best - 0.05 and min(corrs1) >= _MIN_PAIR_CORRELATION:
                curves, s_best, d_mags, pair_corrs, k_est = curves1, s1, d1, corrs1, k_candidate

    if min(pair_corrs) < _MIN_PAIR_CORRELATION:
        return None

    sub = _subbin_refine(curves, d_mags, d_lo, w_small)
    est_ds = [direction * float(m) * scale_to_ref for m in sub]

    # Compare against the candidate baseline calibration; keep the baseline if it already matches.
    base_ds = [
        ((cameras[(i + 1) % 4].phase - cameras[i].phase + ref_width / 2.0) % ref_width) - ref_width / 2.0
        for i in range(4)
    ]
    base_dir = 1 if sum(base_ds) >= 0 else -1
    max_delta_diff = max(abs(a - b) for a, b in zip(est_ds, base_ds, strict=True))
    if (direction == base_dir
            and max_delta_diff <= _RECIPE_MAX_DELTA_DIFF_PX
            and abs(k_est - base_k) <= _RECIPE_MAX_PERSPECTIVE_DIFF):
        return None

    # Anchor camera 0 to the recipe's reference phase so longitudinal orientation stays predictable.
    anchor = float(cameras[0].phase) % ref_width
    phases = (
        anchor,
        (anchor + est_ds[0]) % ref_width,
        (anchor + est_ds[0] + est_ds[1]) % ref_width,
        (anchor + est_ds[0] + est_ds[1] + est_ds[2]) % ref_width,
    )
    return RingAlignment(
        phases=phases,
        perspective=k_est,
        direction=direction,
        ring_score=s_best,
        pair_correlations=pair_corrs,
    )
