import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from image_codec import encode_mono8_png, mono8_to_preview_bmp


class ImageCodecTest(unittest.TestCase):
    def test_png_native_and_stdlib_fallback_preserve_gray_values(self):
        src = np.arange(256, dtype=np.uint8).reshape(16, 16)
        pixels = src.tobytes() + b"ignored trailing packet data"
        native = encode_mono8_png(16, 16, memoryview(pixels))
        with patch.dict(sys.modules, {"cv2": None}):
            fallback = encode_mono8_png(16, 16, pixels)
        for png in (native, fallback):
            np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(png, np.uint8), -1), src)
        with self.assertRaises(ValueError):
            encode_mono8_png(16, 16, b"too short")

    def test_bmp_preserves_orientation_sampling_channels_and_row_padding(self):
        for w, h, limit in ((1, 5, 1), (11, 7, 1), (7, 9, 12), (17, 13, 6), (1792, 256, 720)):
            with self.subTest(width=w, height=h, limit=limit):
                src = np.arange(w * h, dtype=np.uint32).astype(np.uint8).reshape(h, w)
                bmp = mono8_to_preview_bmp(w, h, src.tobytes(), limit)
                image = cv2.imdecode(np.frombuffer(bmp, np.uint8), -1)
                ow, oh = min(limit, w), max(1, int(h * min(1.0, limit / w)))
                target = src[np.ix_(np.arange(oh) * h // oh, np.arange(ow) * w // ow)]
                np.testing.assert_array_equal(image, np.repeat(target[:, :, None], 3, axis=2))
