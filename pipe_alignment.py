"""Register and compare hole patterns from two full pipe panoramas.

Canonical longitude is (display x + cut_x) / width modulo one turn. Axial
coordinates are continuous image coordinates divided by height and never wrap. The
reported pose maps the template to the sample in canonical coordinates. A flip
maps (x, y) to (-x, 1 - y) before adding the signed roll and axial translation.

Findings use sample-display pixels. Every returned x is modulo sample width.
To draw a seam-crossing contour, unwrap its points around the corresponding
center (actual or reference) and draw copies shifted by one width as needed.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import cv2
import numpy as np


_TOLERANCES = {
    'max_rotation_deg': (180, 0, 180),
    'max_axial_pct': (5, 0, 25),
    'position_pct': (0.5, 0.01, 5),
    'size_pct': (35, 0, 100),
    'shape_iou': (0.55, 0, 1),
}
_EPS = 1e-10
_MIN_ANCHOR_SCORE = 0.35
_MAX_HOLES = 32


def validate_tolerances(tolerances):
    """Return a new dict with defaults applied or raise ValueError.

    Ranges are inclusive and match the inspection console. A shape_iou of zero
    disables shape checks.
    """
    if not isinstance(tolerances, dict):
        raise ValueError('应为对象')
    for key in tolerances:
        if key not in _TOLERANCES:
            raise ValueError(f'未知阈值项 {key!r}')
    validated = {}
    for key, (default, minimum, maximum) in _TOLERANCES.items():
        value = tolerances.get(key, default)
        if isinstance(value, bool) or not isinstance(value, Real) or not minimum <= value <= maximum:
            raise ValueError(f'{key} 必须是 {minimum} 至 {maximum} 之间的有限数值')
        validated[key] = float(value)
    return validated


@dataclass(frozen=True)
class _Hole:
    center: np.ndarray
    contours: tuple[np.ndarray, ...]
    area: float
    score: float


@dataclass(frozen=True)
class _Pipe:
    grid: np.ndarray  # circumference and axial length, in display pixels
    cut: int
    holes: tuple[_Hole, ...]
    centers: np.ndarray  # canonical fractions of the circumference/axial span


@dataclass(frozen=True)
class _Pose:
    flipped: bool
    shift: np.ndarray
    inliers: int
    scaled_rms: float  # RMS in units of the registration allowance
    rms: float  # RMS in fractions of the circumference and length

    @property
    def sign(self):
        return -1 if self.flipped else 1


def _integer(value, minimum, maximum):
    return (isinstance(value, Integral) and not isinstance(value, bool)
            and minimum <= value <= maximum)


def _points(value):
    try:
        points = np.asarray(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('坐标必须由数值组成') from None
    if points.dtype.kind not in 'iuf' or not np.isfinite(points).all():
        raise ValueError('坐标必须由有限数值组成')
    return points.astype(np.float64, copy=False)


def _read_pipe(data):
    if not isinstance(data, dict):
        raise ValueError('应为对象')
    width, height = data.get('width'), data.get('height')
    if not all(_integer(value, 2, 2**31 - 1) for value in (width, height)):
        raise ValueError('宽度和高度必须是至少为 2 的整数像素尺寸')
    cut = data.get('cut_x', 0)
    if not _integer(cut, 0, width - 1):
        raise ValueError('cut_x 必须是图像范围内的整数列号')
    rows = data.get('holes')
    if not isinstance(rows, (list, tuple)) or len(rows) > _MAX_HOLES:
        raise ValueError(f'应提供最多含 {_MAX_HOLES} 个孔洞的列表')

    holes = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'第 {index + 1} 个孔洞应为对象')
        center = _points(row.get('center'))
        if (center.shape != (2,) or any(isinstance(value, (bool, np.bool_)) for value in row['center'])
                or not 0 <= center[0] < width or not 0 <= center[1] < height):
            raise ValueError(f'第 {index + 1} 个孔洞的中心坐标无效或超出图像范围')
        area, score = row.get('area'), row.get('score')
        if isinstance(area, bool) or not isinstance(area, Real) or not 0 < area <= width * height:
            raise ValueError(f'第 {index + 1} 个孔洞的面积必须大于 0 且不得超过图像面积')
        if isinstance(score, bool) or not isinstance(score, Real) or not 0 <= score <= 1:
            raise ValueError(f'第 {index + 1} 个孔洞的置信度必须在 0 至 1 之间')
        if not isinstance(row.get('contours'), (list, tuple)):
            raise ValueError(f'第 {index + 1} 个孔洞的轮廓应为列表')
        contours = tuple(_points(contour) for contour in row['contours'])
        if any(points.ndim != 2 or points.shape[1] != 2 or not len(points)
               or np.any((points[:, 1] < 0) | (points[:, 1] >= height)) for points in contours):
            raise ValueError(f'第 {index + 1} 个孔洞的轮廓须包含有效的 [x, y] 点，且纵坐标在图像范围内')
        holes.append(_Hole(center, contours, float(area), float(score)))

    grid = np.array([width, height], dtype=float)
    centers = np.array([hole.center for hole in holes]).reshape(-1, 2) / grid
    centers[:, 0] = (centers[:, 0] + cut / width) % 1
    return _Pipe(grid, cut, tuple(holes), centers)


def _wrap_delta(x):
    return (x + 0.5) % 1 - 0.5


def _deltas(template, sample, flipped):
    oriented = [0, 1] - template.centers if flipped else template.centers
    return sample.centers[None, :, :] - oriented[:, None, :]


def _errors(deltas, shifts):
    errors = deltas - shifts[:, None, None, :]
    errors[..., 0] = _wrap_delta(errors[..., 0])
    return errors


def _mutual_pairs(errors, allowance):
    """Find reciprocal nearest neighbors for each pose, counting each endpoint once."""
    ax, ay = np.broadcast_to(allowance, (2,))
    x, y = errors[..., 0] / ax, errors[..., 1] / ay
    distance = x * x + y * y
    distance[(np.abs(x) > 1) | (np.abs(y) > 1)] = np.inf
    nearest_sample = distance.argmin(axis=2)
    nearest_template = distance.argmin(axis=1)
    match = ((nearest_sample[:, :, None] == np.arange(errors.shape[2]))
             & (nearest_template[:, None, :] == np.arange(errors.shape[1])[:, None])
             & np.isfinite(distance))
    return match, distance


def _refine(shifts, errors, match):
    """Add the component-wise median of matched residuals to each seed."""
    counts = match.sum(axis=(1, 2))
    values = np.where(match[..., None], errors, np.inf).reshape(len(shifts), -1, 2)
    ordered = np.sort(values, axis=1)
    rows = np.flatnonzero(counts)
    refined = shifts.copy()
    refined[rows] += (ordered[rows, (counts[rows] - 1) // 2]
                      + ordered[rows, counts[rows] // 2]) / 2
    refined[:, 0] = _wrap_delta(refined[:, 0])
    return refined


def _poses(deltas, flipped, allowance, minimum):
    # Enumerate center-to-center offsets, including ones outside the physical
    # limits. A strong out-of-limit pose must never turn into a weak in-limit fit.
    seeds = deltas.reshape(-1, 2)
    for start in range(0, len(seeds), 128):
        shifts = seeds[start:start + 128].copy()
        shifts[:, 0] = _wrap_delta(shifts[:, 0])
        errors = _errors(deltas, shifts)
        broad_match, _ = _mutual_pairs(errors, 2 * allowance)
        trials = np.vstack([shifts, _refine(shifts, errors, broad_match)])
        # Rank only poses centered on their own consensus. A provisional seed
        # can gain borderline matches until the new consensus recenters it.
        for step in range(4):
            trials = np.unique(np.round(trials, 12), axis=0)
            errors = _errors(deltas, trials)
            match, distance = _mutual_pairs(errors, allowance)
            refined = _refine(trials, errors, match)
            change = refined - trials
            change[:, 0] = _wrap_delta(change[:, 0])
            stable = (np.abs(change) <= _EPS).all(axis=1)
            if stable.all() or step == 3:
                break
            trials = refined
        counts = match.sum(axis=(1, 2))
        candidates = np.flatnonzero(stable & (counts >= minimum))
        residuals = np.where(match[candidates], distance[candidates], 0).sum(axis=(1, 2))
        for index, residual in zip(candidates, residuals):
            rms = np.sqrt(np.square(errors[index][match[index]]).sum(axis=1).mean())
            yield _Pose(flipped, trials[index], int(counts[index]),
                        float(np.sqrt(residual / counts[index])), float(rms))


def _fit(template, sample, position, reliable):
    # One pixel across the two grids gives a registration noise floor. Keep the
    # search tight enough to distinguish patterns even with a generous defect tolerance.
    pixel = 0.5 / template.grid + 0.5 / sample.grid
    allowance = np.clip(np.maximum(position, pixel), _EPS, .02) + _EPS
    minimum = max(3, max(len(template.holes), len(sample.holes)) // 2 + 1)
    if reliable.sum() < minimum:
        return None, '待检钢管的高置信度孔洞不足，无法可靠对齐。'
    candidates = [pose for flipped in (False, True)
                  for pose in _poses(_deltas(template, sample, flipped)[:, reliable], flipped,
                                     allowance, max(3, minimum - 1))]
    if not candidates:
        return None, '能一致支持同一对齐位置的孔洞不足。'
    candidates.sort(key=lambda pose: (-pose.inliers, pose.scaled_rms))
    best = candidates[0]
    if best.inliers < minimum:
        return None, '能一致支持同一对齐位置的孔洞不足。'
    for other in candidates[1:]:
        if other.inliers < max(3, best.inliers - 1):
            break
        difference = other.shift - best.shift
        difference[0] = _wrap_delta(difference[0])
        distinct = other.flipped != best.flipped or np.any(np.abs(difference) > allowance)
        if distinct and other.scaled_rms <= best.scaled_rms + .35:
            return None, '无法确定唯一的对齐位置，多个方向或偏移量均与孔位吻合。'
    return best, ''


def _local_contours(hole, pipe):
    contours = tuple((contour - hole.center) / pipe.grid for contour in hole.contours)
    for polygon in contours:
        polygon[:, 0] = _wrap_delta(polygon[:, 0])
    return contours


def _shape_iou(reference, actual, template_grid, sample_grid, sign):
    # Compare centered silhouettes on the template grid, preserving both axes'
    # relative scale. Require resolved polygons in each source before rasterizing.
    polygons = []
    for contours, native_grid, orientation in ((reference, template_grid, sign),
                                                (actual, sample_grid, 1)):
        valid = [polygon for polygon in contours if len(polygon) >= 3
                 and cv2.contourArea((polygon * native_grid).astype(np.float32)) > 0]
        if not valid or np.ptp(np.concatenate(valid) * native_grid, axis=0).min() < 3:
            return None
        polygons.append([polygon * template_grid * orientation for polygon in valid])

    points = np.concatenate(polygons[0] + polygons[1])
    lower = points.min(axis=0)
    span = points.max(axis=0) - lower
    scale = min(1, 256 / span.max())
    width, height = np.ceil(span * scale).astype(int) + 3
    masks = []
    for group in polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in group:
            fixed = np.rint(((polygon - lower) * scale + 1) * 16).astype(np.int32)
            cv2.fillPoly(mask, [fixed], 1, shift=4)
        if cv2.countNonZero(mask) < 12:
            return None
        masks.append(mask)
    return (cv2.countNonZero(cv2.bitwise_and(*masks))
            / cv2.countNonZero(cv2.bitwise_or(*masks)))


def _wrap_display(points, width):
    points = points.copy()
    points[..., 0] %= width
    points[..., 0] = np.where(points[..., 0] >= width, 0, points[..., 0])
    return points.tolist()


def _finding(kind, center, contours, width):
    return dict(kind=kind, center=_wrap_display(center, width),
                contours=[_wrap_display(contour, width) for contour in contours])


def _result(status, reason, alignment=None, findings=()):
    return dict(status=status, reason=reason, alignment=alignment, findings=list(findings))


def _compare(template, sample, pose, alignment, errors, limits, reliable,
             projected, surface=None):
    position = limits['position_pct'] / 100
    allowed = position + _EPS
    matches, distance = _mutual_pairs(errors[None, ...], allowed)
    matches = matches[0]
    open_reference = ~matches.any(axis=1)
    open_sample = ~matches.any(axis=0)
    if np.any(np.isfinite(distance[0]) & (open_reference[:, None] | open_sample)):
        return _result('review', '同一预期孔位附近有多个候选，无法确定孔洞对应关系。', alignment)

    # Only pair shifted holes after reserving the position-consistent matches.
    # The local window is wider than the defect limit but small on the pipe.
    shift_window = min(.05, max(.025, 3 * position))
    shifted = ((np.abs(errors) <= shift_window + _EPS).all(axis=2)
               & open_reference[:, None] & open_sample)
    if np.any(shifted.sum(axis=0) > 1) or np.any(shifted.sum(axis=1) > 1):
        return _result('review', '附近有多个错位候选，无法确定孔洞对应关系。', alignment)
    pairs = dict(zip(*np.nonzero(matches | shifted)))

    projected = projected * sample.grid - [sample.cut, 0]
    width, axial_span = sample.grid
    axial_epsilon = _EPS * axial_span
    findings = []
    skipped_shapes = 0
    area_scale = np.prod(template.grid / sample.grid)
    for index, (ref, center) in enumerate(zip(template.holes, projected)):
        ref_local = _local_contours(ref, template)
        ref_contours = [center + contour * sample.grid * pose.sign for contour in ref_local]
        if (not -axial_epsilon <= center[1] <= axial_span + axial_epsilon
                or any(np.any((c[:, 1] < -axial_epsilon) | (c[:, 1] > axial_span + axial_epsilon))
                       for c in ref_contours)):
            return _result('review', '待检图像未覆盖预期孔洞的全部轴向范围。', alignment)
        for coordinates in [center, *ref_contours]:
            coordinates[..., 1] = np.clip(coordinates[..., 1], 0, np.nextafter(axial_span, 0))
        if index not in pairs:
            findings.append(_finding('missing', center, ref_contours, width))
            continue

        actual_index = pairs[index]
        actual = sample.holes[actual_index]
        kind = 'ok'
        if shifted[index, actual_index]:
            kind = 'shifted'
        elif abs(actual.area * area_scale - ref.area) > ref.area * (limits['size_pct'] / 100 + _EPS):
            kind = 'size'
        elif limits['shape_iou'] > 0:
            iou = _shape_iou(ref_local, _local_contours(actual, sample),
                             template.grid, sample.grid, pose.sign)
            if iou is None:
                skipped_shapes += 1
            elif iou + _EPS < limits['shape_iou']:
                kind = 'shape'
        finding = _finding(kind, actual.center, actual.contours, width)
        if kind != 'ok':
            if not reliable[actual_index]:
                return _result('review', '疑似异常孔洞的置信度不足，需复核。', alignment)
            finding['reference_center'] = _wrap_display(center, width)
            finding['reference_contours'] = [_wrap_display(contour, width) for contour in ref_contours]
        findings.append(finding)

    paired = set(pairs.values())
    unmatched = [i for i in range(len(reliable)) if i not in paired]
    extras = [i for i in unmatched if surface is None or not _same_end_pattern(surface, sample, i)]
    if any(not reliable[index] for index in extras):
        return _result('review', '未匹配孔洞的置信度不足，需复核。', alignment)
    findings.extend(_finding('extra', hole.center, hole.contours, width)
                    for hole in (sample.holes[index] for index in extras))
    if len(unmatched) > len(extras):
        alignment['confirmed_end_marks'] = len(unmatched) - len(extras)
    defects = sum(finding['kind'] != 'ok' for finding in findings)
    reason = f'发现 {defects} 处孔洞异常。' if defects else '孔洞检测结果符合设定公差。'
    if skipped_shapes:
        reason += f'有 {skipped_shapes} 个孔洞的轮廓分辨率不足，未比较形状。'
    return _result('fail' if defects else 'pass', reason, alignment, findings)


def _same_end_pattern(surface, sample, index):
    hole = sample.holes[index]
    # Restrict this to censored contours at the actual axial ends.
    if not any(np.any((c[:, 1] < 1.5) | (c[:, 1] >= sample.grid[1] - 1.5))
               for c in hole.contours):
        return False
    offset = np.array([sample.cut / sample.grid[0], 0])
    contours = [c / sample.grid + offset for c in hole.contours]
    return surface.has_same_end_pattern(sample.centers[index], contours)


def _surface_holes(pipe, centers):
    margins = np.zeros_like(centers)
    for i, hole in enumerate(pipe.holes):
        contours = _local_contours(hole, pipe)
        if contours:
            margins[i] = np.max(np.abs(np.concatenate(contours)), axis=0)
    return np.column_stack((centers, margins))


def _surface_pose(template, sample, pngs, position, reliable):
    from pipe_surface import register_surface

    # The coarse search can span scan warp, while acceptance uses position_pct.
    pose, _ = _fit(template, sample, max(.02, position), reliable)
    if pose is None:
        return None
    oriented = template.centers * pose.sign + [0, int(pose.flipped)]
    holes = (_surface_holes(template, oriented), _surface_holes(sample, sample.centers))
    surface = register_surface(pngs, (template.cut, sample.cut), pose.flipped, holes, pose.shift)
    if surface is None:
        return None
    displacements = [surface.displacement(point) for point in oriented]
    if any(offset is None for offset in displacements):
        return None
    return pose, oriented + displacements, surface


def inspect_pipe(template, sample, tolerances, *, surface_pngs=None):
    """Return a JSON-ready inspection result with a pose and sample overlays.

    At least three openings and a majority of both patterns must support a unique
    pose. Enrollment establishes the reference. Sample detections with scores
    of at least 0.35 can anchor the pose. Weaker detections can corroborate an
    expected opening after alignment. A defect dependent on weak evidence,
    invalid input, an ambiguous pose or an out-of-limit pose requires review.
    Area and centered shape IoU are compared on the template grid. Shape
    comparison requires a resolved outline from each image. When both panoramas
    are supplied, corresponding wall texture can correct local projection drift.
    """
    try:
        limits = validate_tolerances(tolerances)
    except ValueError as exc:
        return _result('review', f'检测阈值无效：{exc}。')
    pipes = []
    for name, data in (('模板', template), ('待检钢管', sample)):
        try:
            pipes.append(_read_pipe(data))
        except ValueError as exc:
            return _result('review', f'{name}数据无效：{exc}。')
    template_pipe, sample_pipe = pipes

    if min(len(template_pipe.holes), len(sample_pipe.holes)) < 3:
        return _result('review', '孔洞数量不足，无法可靠对齐。')
    reliable = np.array([hole.score >= _MIN_ANCHOR_SCORE for hole in sample_pipe.holes])
    position = limits['position_pct'] / 100
    corrected = (_surface_pose(template_pipe, sample_pipe, surface_pngs, position, reliable)
                 if surface_pngs is not None else None)
    surface = None
    if corrected is None:
        pose, reason = _fit(template_pipe, sample_pipe, position, reliable)
        if pose is None:
            return _result('review', reason)
        projected = template_pipe.centers * pose.sign + pose.shift + [0, int(pose.flipped)]
    else:
        pose, projected, surface = corrected
    errors = sample_pipe.centers[None, :, :] - projected[:, None, :]
    errors[..., 0] = _wrap_delta(errors[..., 0])
    shift = pose.shift if surface is None else surface.translation
    rotation, axial = float(_wrap_delta(shift[0]) * 360), float(shift[1] * 100)
    alignment = dict(flipped=pose.flipped, rotation_deg=rotation, axial_pct=axial,
                     inliers=pose.inliers, rms_position_pct=pose.rms * 100)
    if surface is not None:
        matches, _ = _mutual_pairs(errors[None, :, reliable], position + _EPS)
        matched_errors = errors[:, reliable][matches[0]]
        alignment['inliers'] = len(matched_errors)
        alignment['surface_anchors'] = len(surface.matches)
        if len(matched_errors):
            alignment['rms_position_pct'] = float(np.sqrt(np.mean(np.sum(matched_errors**2, axis=1))) * 100)
    violations = []
    if abs(rotation) > limits['max_rotation_deg'] + 360 * _EPS:
        violations.append('周向旋转')
    if abs(axial) > limits['max_axial_pct'] + 100 * _EPS:
        violations.append('轴向偏移')
    if violations:
        return _result('review', f'对齐所需的{"和".join(violations)}超出允许范围。', alignment)

    return _compare(template_pipe, sample_pipe, pose, alignment, errors, limits, reliable, projected, surface)
