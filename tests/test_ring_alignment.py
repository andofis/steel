import math
import unittest

import cv2
import numpy as np

from tube_unwrap import TubeUnwrapper, UnwrapStatus
from tube_unwrap.ring_alignment import AUTO_RING_CALIBRATION_ID
from tube_unwrap_bridge import TubeUnwrapBridge


def _render_cylinder_views(
    surface: np.ndarray,
    centers: list[float],
    radii: list[float],
    phases_deg: list[float],
    perspective: float,
    *,
    frame_h: int = 1200,
    frame_w: int = 1792,
    top: int = 150,
    bottom: int = 1050,
) -> list[np.ndarray]:
    """Forward-project a cylindrical surface texture (H x W_pan) into four line-scan views."""
    pan_h, pan_w = surface.shape
    body_h = bottom - top + 1
    k = perspective
    s = math.sqrt(1.0 - k * k)
    frames = []
    for center, radius, phase_deg in zip(centers, radii, phases_deg, strict=True):
        frame = np.full((frame_h, frame_w), 35, dtype=np.uint8)
        x_lo = max(0, int(math.ceil(center - radius)))
        x_hi = min(frame_w - 1, int(math.floor(center + radius)))
        xs = np.arange(x_lo, x_hi + 1, dtype=np.float32)
        ratio = np.clip((xs - center) / radius, -0.999, 0.999)
        offset = k * ratio
        angle = np.arcsin(np.clip(ratio / np.hypot(s, offset), -1.0, 1.0)) - np.arctan2(offset, s)
        pan_x = ((phase_deg / 360.0) * pan_w + angle * (pan_w / (2.0 * math.pi))) % pan_w
        pan_y = np.linspace(0, pan_h - 1, body_h, dtype=np.float32)
        map_x = np.broadcast_to(pan_x[None, :], (body_h, xs.size)).astype(np.float32)
        map_y = np.broadcast_to(pan_y[:, None], (body_h, xs.size)).astype(np.float32)
        rendered = cv2.remap(surface, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        # Add realistic camera-specific vertical illumination shading to test invariance.
        shading = (1.0 - 0.18 * (ratio ** 2)).astype(np.float32)
        rendered = np.clip(rendered.astype(np.float32) * shading[None, :], 0, 255).astype(np.uint8)
        frame[top : bottom + 1, x_lo : x_hi + 1] = rendered
        frames.append(frame)
    return frames


def _make_textured_surface(pan_h: int = 900, pan_w: int = 1536) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(20261127)
    base = np.full((pan_h, pan_w), 145.0, dtype=np.float32)
    # Multi-scale surface texture + circumferential squiggle + 5 holes (including near seams).
    noise = cv2.GaussianBlur(rng.normal(0.0, 28.0, (pan_h, pan_w)).astype(np.float32), (9, 9), 0)
    surface = np.clip(base + noise, 70, 220).astype(np.uint8)
    xs = np.arange(pan_w)
    ys = (pan_h * 0.72 + 18.0 * np.sin(2 * np.pi * xs / pan_w * 5) + 9.0 * np.cos(2 * np.pi * xs / pan_w * 11)).astype(int)
    for x, y in zip(xs, ys):
        surface[y - 2 : y + 3, x] = 65
    holes = [
        (30, 200),
        (560, 220),   # Between cam_180 and cam_270 in reversed order
        (560, 420),   # Between cam_180 and cam_270 in reversed order
        (1100, 350),
        (1400, 560),
    ]
    for hx, hy in holes:
        cv2.circle(surface, (hx, hy), 26, 8, -1)
    return surface, holes


class RingAlignmentTest(unittest.TestCase):
    def test_auto_aligns_reversed_camera_order_and_thicker_tube_without_duplicate_holes(self):
        bridge = TubeUnwrapBridge()
        bridge.config["output_height"] = 1000
        surface, hole_coords = _make_textured_surface(900, 1536)

        # Reversed ring order (0 -> -90 -> -180 -> -270) with thicker diameter (+16%) and shifted centers.
        centers = [1060.0, 1030.0, 1150.0, 815.0]
        radii = [285.0, 282.0, 286.0, 289.0]
        true_phases_deg = [359.0, 270.0, 179.0, 92.0]
        frames = _render_cylinder_views(surface, centers, radii, true_phases_deg, perspective=0.125)

        recipe = bridge._build_recipe(frames, [(150, 1050)] * 4, (1152, 750))
        result = TubeUnwrapper(recipe).process(frames)

        self.assertEqual(result.status, UnwrapStatus.OK, result.message)
        self.assertEqual(result.phase_calibration_id, AUTO_RING_CALIBRATION_ID)
        big_holes = [h for h in result.holes if h.width >= 20 and h.height >= 20]
        self.assertEqual(len(big_holes), len(hole_coords))

    def test_keeps_nominal_calibration_when_recipe_already_matches(self):
        bridge = TubeUnwrapBridge()
        bridge.config["output_height"] = 1000
        surface, hole_coords = _make_textured_surface(900, 1536)

        nom = bridge.config["nominal_phase_calibration"]
        centers = [(c["left"] + c["right"]) / 2.0 for c in bridge.config["cameras"]]
        radii = [(c["right"] - c["left"]) / 2.0 for c in bridge.config["cameras"]]
        true_phases_deg = [p / 1536.0 * 360.0 for p in nom["phases"]]
        frames = _render_cylinder_views(surface, centers, radii, true_phases_deg, perspective=0.095)

        recipe = bridge._build_recipe(frames, [(150, 1050)] * 4, (1152, 750))
        result = TubeUnwrapper(recipe).process(frames)

        self.assertEqual(result.status, UnwrapStatus.OK, result.message)
        self.assertEqual(result.phase_calibration_id, nom["id"])
        big_holes = [h for h in result.holes if h.width >= 20 and h.height >= 20]
        self.assertEqual(len(big_holes), len(hole_coords))

    def test_adapts_perspective_and_phases_for_thick_pipe_in_forward_order(self):
        bridge = TubeUnwrapBridge()
        bridge.config["output_height"] = 1000
        surface, hole_coords = _make_textured_surface(900, 1536)

        centers = [820.0, 930.0, 1010.0, 810.0]
        radii = [285.0, 285.0, 285.0, 285.0]
        true_phases_deg = [359.0, 86.0, 178.0, 268.0]
        frames = _render_cylinder_views(surface, centers, radii, true_phases_deg, perspective=0.145)

        recipe = bridge._build_recipe(frames, [(150, 1050)] * 4, (1152, 750))
        result = TubeUnwrapper(recipe).process(frames)

        self.assertEqual(result.status, UnwrapStatus.OK, result.message)
        self.assertEqual(result.phase_calibration_id, AUTO_RING_CALIBRATION_ID)
        big_holes = [h for h in result.holes if h.width >= 20 and h.height >= 20]
        self.assertEqual(len(big_holes), len(hole_coords))


if __name__ == "__main__":
    unittest.main()
