"""
preprocessing/clip_preprocessing.py

Everything CLIP-specific: decode raw bytes -> BGR array, cap to
CLIP_MAX_DIMENSION, cut into tiles, score each tile's ink density, and
detect table/ruled-line structure (used as a hard override -- a document
with a real printed table is never classified handwritten, regardless of
what CLIP's tile vote says).

This is intentionally separate from preprocessing/printed_preprocessing.py
and preprocessing/handwritten_preprocessing.py -- each model stage resizes
and formats the image differently, and mixing the logic makes the
thresholds in config.py hard to reason about.
"""
import cv2
import numpy as np
from PIL import Image

import config


def bytes_to_bgr_array(raw_bytes):
    file_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2 could not decode image bytes")
    h, w = img.shape[:2]
    if max(h, w) > config.CLIP_MAX_DIMENSION:
        scale = config.CLIP_MAX_DIMENSION / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def make_tiles(img_bgr, grid=config.TILE_GRID):
    rows, cols = grid
    h, w = img_bgr.shape[:2]
    tiles = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(h * r / rows), int(h * (r + 1) / rows)
            x0, x1 = int(w * c / cols), int(w * (c + 1) / cols)
            tile = img_bgr[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            tiles.append(tile)
    return tiles


def bgr_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def compute_ink_density(tile_bgr):
    if tile_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except cv2.error:
        return 0.0
    ink_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.size
    return ink_pixels / total_pixels if total_pixels else 0.0


def has_table_structure(
    img_bgr,
    min_lines=config.STRUCTURAL_MIN_LINES,
    min_line_frac=config.STRUCTURAL_MIN_LINE_FRAC,
    require_both_axes=config.STRUCTURAL_REQUIRE_BOTH_AXES,
):
    """Hard override: a page with enough long horizontal + vertical ruled
    lines is treated as printed/tabular even if CLIP's tile vote leans
    handwritten (e.g. a form with lots of handwritten fill-ins)."""
    if img_bgr is None or img_bgr.size == 0:
        return False
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = int(min_line_frac * min(h, w))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return False

    horizontal_count = 0
    vertical_count = 0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length < min_len:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        is_horizontal = angle <= 5 or angle >= 175
        is_vertical = 85 <= angle <= 95
        if is_horizontal and length >= min_line_frac * w:
            horizontal_count += 1
        elif is_vertical and length >= min_line_frac * h:
            vertical_count += 1

    if require_both_axes:
        return horizontal_count >= min_lines and vertical_count >= 1
    return (horizontal_count + vertical_count) >= min_lines
