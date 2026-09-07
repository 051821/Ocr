"""
preprocessing/clip_preprocessing.py

Everything CLIP-specific: decode raw bytes -> BGR array, cap to
CLIP_MAX_DIMENSION
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
    """Hard override: a page with printed horizontal/vertical divider lines
    or table structure is treated as printed even if CLIP's tile vote leans
    handwritten (e.g. a prescription form with handwritten notes or tick marks).

    NOTE: A single axis of many, regularly-spaced horizontal lines is the
    signature of ruled/lined notebook paper, not a printed table. Real tables
    have dividers on BOTH axes. Requiring only horizontal_count >= min_lines
    (as the old `or` fallback did) misclassifies handwritten notes on ruled
    paper as printed. We now always require at least one vertical line
    before this can fire as a table override, regardless of require_both_axes.
    """
    if img_bgr is None or img_bgr.size == 0:
        return False
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Morphological line detection (robust against folds, creases, and shadows)
    thresh = cv2.adaptiveThreshold(
        ~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
    )

    min_line_w = int(min_line_frac * w)
    min_line_h = int(min_line_frac * h)

    # Horizontal lines kernel
    k_w = max(min_line_w // 3, 25)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    horiz_map = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

    # Vertical lines kernel
    k_h = max(min_line_h // 3, 25)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
    vert_map = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)

    # Count distinct horizontal lines
    contours_h, _ = cv2.findContours(horiz_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal_count = 0
    for cnt in contours_h:
        _, _, cw, _ = cv2.boundingRect(cnt)
        if cw >= min_line_w:
            horizontal_count += 1

    # Count distinct vertical lines
    contours_v, _ = cv2.findContours(vert_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_count = 0
    for cnt in contours_v:
        _, _, _, ch = cv2.boundingRect(cnt)
        if ch >= min_line_h:
            vertical_count += 1

    # 2. Complementary Hough lines detection in case lines are thin or segmented
    if horizontal_count < min_lines:
        edges = cv2.Canny(gray, 40, 120, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80,
            minLineLength=min_line_w, maxLineGap=40
        )
        if lines is not None:
            h_hough = 0
            v_hough = 0
            for line in lines[:, 0]:
                x1, y1, x2, y2 = line
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if (angle <= 8 or angle >= 172) and dx >= min_line_w:
                    h_hough += 1
                elif 82 <= angle <= 98 and dy >= min_line_h:
                    v_hough += 1
            horizontal_count = max(horizontal_count, h_hough)
            vertical_count = max(vertical_count, v_hough)

    # A real table needs structure on BOTH axes. Pages with only horizontal
    # lines (ruled/lined paper) must never qualify, regardless of how many
    # horizontal lines are present or what require_both_axes is set to.
    if horizontal_count < min_lines or vertical_count < 1:
        return False

    if require_both_axes:
        return horizontal_count >= min_lines and vertical_count >= 1

    # require_both_axes == False just relaxes how strict we are about the
    # horizontal count/vertical count relationship beyond the hard minimums
    # already enforced above -- it no longer allows a vertical-free pass.
    return True


def cv_classify_document(img_bgr):
    """Fallback zero-RAM computer vision classifier for printed vs handwritten."""
    if has_table_structure(img_bgr):
        return False, {"decision": "structural_table_override"}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    # NOTE: previously these sparse-content bailouts returned "printed"
    # (False) unconditionally on low component/label counts. A page with a
    # small amount of handwriting (e.g. a short note, a few lines of script)
    # can easily fall under these thresholds despite genuinely containing
    # handwritten content -- it just doesn't have much content at all. We now
    # check ink density first: if there IS meaningful ink on the page, we
    # don't default to "printed" just because there are few components; we
    # fall through to a lighter-weight stroke check instead of a blind
    # bailout.
    ink_density = compute_ink_density(img_bgr)
    SPARSE_INK_FLOOR = 0.01  # below this, treat the page as genuinely blank/sparse

    if num_labels <= 10:
        if ink_density < SPARSE_INK_FLOOR:
            return False, {"decision": "cv_blank_or_sparse", "ink_density": round(float(ink_density), 4)}
        # Some ink present despite few connected components (e.g. a short
        # handwritten note, large loose cursive strokes that merge into a
        # handful of blobs). Don't default to printed -- fall through to
        # stroke analysis below using whatever components exist.

    valid_stats = [s for s in stats[1:] if 15 <= s[4] <= 0.05 * h * w and 5 <= s[3] <= 150]
    if len(valid_stats) < 15:
        if ink_density < SPARSE_INK_FLOOR:
            return False, {"decision": "cv_insufficient_components", "ink_density": round(float(ink_density), 4)}
        # Same reasoning as above: real ink on the page, just not enough
        # "component-shaped" ink to hit the threshold. Rather than default
        # to printed, fall through and do our best with what's available.
        if not valid_stats:
            # Truly nothing usable to analyze despite ink_density > floor
            # (e.g. large connected wash of ink, heavy smudging). We can't
            # run stroke-shape analysis without components, so bail out
            # honestly as "undetermined" rather than silently mislabeling
            # it printed.
            return True, {
                "decision": "cv_ink_present_no_components",
                "ink_density": round(float(ink_density), 4),
            }

    heights = [int(s[3]) for s in valid_stats]
    median_h = float(np.median(heights))
    regular_h_ratio = float(sum(0.5 * median_h <= ch <= 1.8 * median_h for ch in heights)) / float(len(heights))

    proj = np.sum(thresh, axis=1) / 255.0
    whitespace_rows = float(np.sum(proj < (0.01 * w))) / float(h)

    # Third, independent signal: stroke-width irregularity via distance
    # transform. Printed fonts have a roughly constant stroke width (font
    # weight is fixed); handwriting has much more variable stroke width
    # (pen pressure, speed, pen lift/re-touch). We measure this as the
    # coefficient of variation (std / mean) of the distance-transform peak
    # values sampled along the ink -- high CV suggests handwriting.
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    stroke_samples = dist[dist > 0]
    if stroke_samples.size >= 20:
        mean_stroke = float(np.mean(stroke_samples))
        stroke_width_cv = float(np.std(stroke_samples) / mean_stroke) if mean_stroke > 0 else 0.0
    else:
        stroke_width_cv = 0.0

    # Each signal casts one vote for "handwritten". Machine-printed documents
    # tend to have high character-height uniformity, distinct/regular row
    # gaps, and uniform stroke width -- so each of these being "irregular"
    # independently suggests handwriting. Requiring ALL signals to agree
    # (the old `and` logic) meant a single noisy or ambiguous signal (e.g.
    # neat print-style handwriting with fairly uniform heights, or dense
    # notes with little blank-row whitespace) would force a false "printed"
    # result even when the other signals correctly indicated handwriting.
    # A majority vote is far more robust to any single feature being
    # inconclusive on a given page.
    votes_hw = 0
    votes_hw += int(regular_h_ratio < 0.55)
    votes_hw += int(whitespace_rows < 0.15)
    votes_hw += int(stroke_width_cv > 0.35)

    is_hw = votes_hw >= 2
    return is_hw, {
        "decision": "cv_stroke_analysis",
        "regular_h_ratio": round(float(regular_h_ratio), 3),
        "whitespace_rows": round(float(whitespace_rows), 3),
        "stroke_width_cv": round(float(stroke_width_cv), 3),
        "votes_hw": votes_hw,
    }