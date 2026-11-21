# -*- coding: utf-8 -*-
"""Image codecs for acquisition, with a stdlib fallback for capture-only installs."""

from __future__ import annotations

import binascii
import os
import struct
import zlib
from operator import itemgetter
from pathlib import Path
from typing import Dict


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind)
    crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def encode_mono8_png(width: int, height: int, pixels: bytes | memoryview) -> bytes:
    """Encode once, losslessly; the viewer and the archive share these bytes."""
    if width <= 0 or height <= 0:
        raise ValueError("无效图像尺寸")
    expected = width * height
    if len(pixels) < expected:
        raise ValueError(f"像素数据不足：需要 {expected} 字节，实际 {len(pixels)} 字节")
    # Capture-only installations can still archive frames without OpenCV.
    try:
        import cv2
        import numpy as np
    except ImportError:
        pass
    else:
        image = np.frombuffer(pixels, dtype=np.uint8, count=expected).reshape(height, width)
        try:
            ok, encoded = cv2.imencode(".png", image)
        except cv2.error:  # A minimal OpenCV build may lack a PNG codec.
            pass
        else:
            if ok:
                return encoded.tobytes()

    raw = bytearray((width + 1) * height)
    source_offset = 0
    target_offset = 0
    for _ in range(height):
        raw[target_offset] = 0
        target_offset += 1
        raw[target_offset:target_offset + width] = pixels[source_offset:source_offset + width]
        source_offset += width
        target_offset += width

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    compressed = zlib.compress(raw, level=4)
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")


def write_encoded_png(path: Path, payload: bytes) -> None:
    """Publish only a complete, flushed file (including on non-ASCII paths)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(temp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    except OSError:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_mono8_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    write_encoded_png(path, encode_mono8_png(width, height, pixels))


def mono8_to_preview_bmp(width: int, height: int, pixels: bytes, max_width: int = 960) -> bytes:
    """Create a compact 24-bit grayscale BMP preview for browser <img> rendering."""
    if width <= 0 or height <= 0:
        raise ValueError("无效图像尺寸")
    expected = width * height
    if len(pixels) < expected:
        raise ValueError(f"预览像素数据不足：需要 {expected} 字节，实际 {len(pixels)} 字节")

    scale = min(1.0, float(max_width) / float(width))
    out_width = max(1, int(width * scale))
    out_height = max(1, int(height * scale))
    # itemgetter gathers a whole scan line in C rather than looping through RGB pixels in Python.
    gather = itemgetter(*(x * width // out_width for x in range(out_width))) if 1 < out_width < width else None

    row_stride = ((out_width * 3 + 3) // 4) * 4
    image_size = row_stride * out_height
    file_size = 54 + image_size
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, out_width, out_height, 1, 24, 0, image_size, 2835, 2835, 0, 0)
    body = bytearray(image_size)

    for out_y in range(out_height):
        source_y = (out_height - 1 - out_y) * height // out_height
        source_base = source_y * width
        target_base = out_y * row_stride
        row = pixels[source_base:source_base + width]
        gray = bytes(gather(row)) if gather else row[:out_width]
        end = target_base + 3 * out_width
        for channel in range(3):
            body[target_base + channel:end:3] = gray
    return header + info + bytes(body)


def _sample_mean(pixels: bytes, stride: int = 97) -> float:
    if not pixels:
        return 0.0
    sample = pixels[::max(1, stride)]
    return float(sum(sample)) / float(len(sample))


def _percentile(values: list, ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * max(0.0, min(1.0, ratio)))
    return float(ordered[index])


def _median(values: list) -> float:
    return _percentile(values, 0.5)


def detect_pipe_bounds(width: int, height: int, pixels: bytes) -> tuple:
    """Estimate the non-background steel-pipe rectangle in a Mono8 frame."""
    if width <= 2 or height <= 2:
        return 0, width, 0, height

    row_step = max(1, height // 240)
    x_step = max(1, width // 500)
    col_sums = [0] * width
    col_squares = [0] * width
    col_counts = [0] * width
    for y in range(0, height, row_step):
        base = y * width
        for x in range(0, width, x_step):
            value = pixels[base + x]
            col_sums[x] += value
            col_squares[x] += value * value
            col_counts[x] += 1

    col_means = []
    col_vars = []
    for x in range(width):
        count = col_counts[x]
        mean = col_sums[x] / count if count else 0.0
        variance = (col_squares[x] / count - mean * mean) if count else 0.0
        col_means.append(mean)
        col_vars.append(max(0.0, variance))

    def smooth(values: list, radius: int = 4) -> list:
        output = []
        for index in range(width):
            left = max(0, index - radius)
            right = min(width, index + radius + 1)
            segment = values[left:right]
            output.append(sum(segment) / len(segment))
        return output

    smoothed = smooth(col_means)
    smooth_var = smooth(col_vars)
    edge_width = max(8, width // 10)
    var_background = _median(smooth_var[:edge_width] + smooth_var[-edge_width:])
    var_peak = _percentile(smooth_var, 0.94)
    var_threshold = var_background + max(1.0, (var_peak - var_background) * 0.28)

    def longest_run(mask: list, gap: int) -> tuple:
        best_start, best_end = 0, 0
        run_start = None
        last_above = -999
        for index, enabled in enumerate(mask):
            if enabled:
                if run_start is None:
                    run_start = index
                last_above = index
            elif run_start is not None and index - last_above > gap:
                if index - run_start > best_end - best_start:
                    best_start, best_end = run_start, last_above + 1
                run_start = None
        if run_start is not None and width - run_start > best_end - best_start:
            best_start, best_end = run_start, width
        return best_start, best_end

    best_start, best_end = longest_run([value >= var_threshold for value in smooth_var], 8)

    # 背景可能是缓变灰度时，用均值轮廓作为兜底。
    background = _median(smoothed[:edge_width] + smoothed[-edge_width:])
    peak = _percentile(smoothed, 0.92)
    threshold = background + max(5.0, (peak - background) * 0.2)
    mean_start, mean_end = longest_run([value >= threshold for value in smoothed], 10)
    if best_end - best_start < max(24, width // 20) or best_end - best_start > int(width * 0.72):
        best_start, best_end = mean_start, mean_end

    if best_end - best_start < max(24, width // 20):
        best_start = int(width * 0.25)
        best_end = int(width * 0.75)
    x0 = max(0, best_start - 2)
    x1 = min(width, best_end + 2)
    row_means = []
    sample_x = max(1, (x1 - x0) // 140)
    for y in range(height):
        base = y * width
        row = pixels[base + x0:base + x1:sample_x]
        row_means.append(sum(row) / len(row) if row else 0.0)

    row_background = _percentile(row_means, 0.08)
    row_peak = _percentile(row_means, 0.92)
    row_threshold = row_background + max(3.0, (row_peak - row_background) * 0.13)
    rows_above = [y for y, value in enumerate(row_means) if value >= row_threshold]
    if rows_above:
        y0 = max(0, rows_above[0] - 2)
        y1 = min(height, rows_above[-1] + 3)
    else:
        y0, y1 = 0, height
    if y1 - y0 < 20:
        y0, y1 = 0, height

    return x0, x1, y0, y1


def _crop_mono8(width: int, height: int, pixels: bytes, bounds: tuple) -> tuple:
    x0, x1, y0, y1 = bounds
    rows = []
    for y in range(y0, y1):
        base = y * width
        rows.append(pixels[base + x0:base + x1])
    return x1 - x0, y1 - y0, b"".join(rows)


def _brightness_lut(target: float, mean: float) -> bytes:
    scale = target / mean if mean > 0.5 else 1.0
    return bytes(min(255, max(0, int(index * scale))) for index in range(256))


def _resize_nearest(width: int, height: int, pixels: bytes, new_width: int, new_height: int) -> bytes:
    x_map = [min(width - 1, int(x * width / new_width)) for x in range(new_width)]
    output = bytearray(new_width * new_height)
    for out_y in range(new_height):
        source_y = min(height - 1, int(out_y * height / new_height))
        source_base = source_y * width
        target_base = out_y * new_width
        row = pixels[source_base:source_base + width]
        output[target_base:target_base + new_width] = bytes(row[x] for x in x_map)
    return bytes(output)


def create_unwrapped_image(packets: Dict[int, Any], target_brightness: float = 82.0) -> tuple:
    """Crop four angular images to the pipe surface and concatenate them horizontally."""
    segments = []
    bounds_info = []
    for slot in sorted(packets):
        packet = packets[slot]
        bounds = detect_pipe_bounds(packet.width, packet.height, packet.data)
        cropped_width, cropped_height, cropped = _crop_mono8(packet.width, packet.height, packet.data, bounds)
        mean = _sample_mean(cropped)
        segments.append({
            "slot": slot,
            "width": cropped_width,
            "height": cropped_height,
            "data": cropped,
            "mean": mean,
        })
        bounds_info.append({"slot": slot, "bounds": bounds, "mean": mean})

    if not segments:
        raise ValueError("没有可用于展开图拼接的钢管图像")

    # 不强制四段等宽：长方体可能由宽面、窄面组成，保留真实宽度比例。
    target_height = max(64, int(_median([segment["height"] for segment in segments])))
    normalized = []
    for segment in segments:
        lut = _brightness_lut(target_brightness, segment["mean"])
        pixels = segment["data"].translate(lut)
        pixels = _resize_nearest(segment["width"], segment["height"], pixels, segment["width"], target_height)
        normalized.append({"slot": segment["slot"], "width": segment["width"], "data": pixels})

    composite_width = sum(segment["width"] for segment in normalized)
    composite = bytearray(composite_width * target_height)
    for out_y in range(target_height):
        row_parts = []
        for segment in normalized:
            width = segment["width"]
            row_parts.append(segment["data"][out_y * width:(out_y + 1) * width])
        base = out_y * composite_width
        composite[base:base + composite_width] = b"".join(row_parts)
    return bytes(composite), composite_width, target_height, bounds_info
