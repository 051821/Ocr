"""
model/paddel.py
"""
import gc
import json
import os
import re

from rapidfuzz import process, fuzz

import config
from data.data_fetcher import download_image_bytes, item_key
from preprocessing.printed_preprocessing import preprocess_for_paddle
from model.load_model import load_paddle_engines





# ---------------------------------------------------------------------------
# OCR EXECUTION + ENSEMBLE
# ---------------------------------------------------------------------------
def run_ocr_batch(model, img_arrays):
    results = model.predict(img_arrays)
    per_image = []
    for page in results:
        entries = []
        for box, text, score in zip(page["rec_boxes"], page["rec_texts"], page["rec_scores"]):
            entries.append((tuple(box.tolist()), text, float(score)))
        per_image.append(entries)
    return per_image


def ensemble_batch(en_batch_results, server_batch_results):
    """Same detected box (rounded to the nearest 5px) -> keep the higher
    confidence recognition between the two engines."""
    merged_batch = []
    for en_results, server_results in zip(en_batch_results, server_batch_results):
        merged = {}
        for box, text, score in en_results + server_results:
            key = tuple(round(v / 5) * 5 for v in box)
            if key not in merged or score > merged[key][2]:
                merged[key] = (box, text, score)
        merged_batch.append(merged)
    return merged_batch


# ---------------------------------------------------------------------------
# READING ORDER
# ---------------------------------------------------------------------------
def get_box_geometry(box):
    x_coords = box[0::2]
    y_coords = box[1::2]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    height = max(max_y - min_y, 1.0)
    return cx, cy, min_x, height


def sort_reading_order(merged):
    """
    Sorts OCR text boxes in natural reading order:
    1. Group items into horizontal rows using the running row midpoint Y.
       Using the midpoint (rather than a fixed anchor) prevents multi-word boxes
       whose center-Ys differ by a few pixels from being split across rows.
    2. Sort rows top-to-bottom by midpoint Y.
    3. Within each row, sort items left-to-right by min_x.
    """
    items = []
    for box, text, score in merged.values():
        cx, cy, min_x, height = get_box_geometry(box)
        items.append({
            "x": cx,
            "y": cy,
            "min_x": min_x,
            "height": height,
            "text": text,
            "score": score
        })

    # Sort all items top-to-bottom by Y coordinate
    items = sorted(items, key=lambda i: i["y"])

    rows = []
    row_mid_ys = []  # running midpoint Y for each row
    row_tolerance = config.ROW_TOLERANCE

    for item in items:
        matched_row = None
        for idx, row in enumerate(rows):
            # Compare against the running row midpoint Y to tolerate slight per-glyph Y variance.
            # Use 60% of the text height as tolerance (≈12px for 20px-tall text),
            # still bounded by ROW_TOLERANCE (15) to prevent snowballing across actual rows.
            mid_y = row_mid_ys[idx]
            ref_h = row[0]["height"]
            tol = max(6.0, min(row_tolerance, ref_h * 0.6))
            if abs(item["y"] - mid_y) <= tol:
                matched_row = row
                # Update the running midpoint Y incrementally
                row_mid_ys[idx] = (mid_y * len(row) + item["y"]) / (len(row) + 1)
                break
        if matched_row is not None:
            matched_row.append(item)
        else:
            rows.append([item])
            row_mid_ys.append(item["y"])

    # Sort rows top-to-bottom by their midpoint Y
    paired = sorted(zip(row_mid_ys, rows), key=lambda p: p[0])
    rows = [r for _, r in paired]

    # Sort items within each row left-to-right by min_x / x
    ordered = []
    for row in rows:
        row.sort(key=lambda i: i["min_x"])
        ordered.extend(row)

    return ordered


# ---------------------------------------------------------------------------
# VOCAB CORRECTION + VITALS TAGGING (printed-domain specific)
# ---------------------------------------------------------------------------
def correct_against_vocab(text, score, conf_gate=0.85, threshold=config.PADDLE_VOCAB_MATCH_THRESHOLD):
    if score >= conf_gate or not text.strip():
        return text
    # Only correct against vocab for words >= 4 chars to prevent short strings from distorting
    if len(text.strip()) < 4:
        return text
    match, match_score, _ = process.extractOne(text, config.LAB_VOCAB, scorer=fuzz.ratio)
    return match if match_score >= threshold else text


def tag_vitals_by_unit(text):
    lower = text.lower()
    for unit, label in config.VITALS_UNIT_HINTS.items():
        if unit in lower:
            return label
    return None


def finalize_entries(merged):
    ordered = sort_reading_order(merged)
    final = []
    for item in ordered:
        text, score = item["text"], item["score"]
        stripped = text.strip()
        if not stripped:
            continue
        # Filter phantom tokens that are purely punctuation/symbols (e.g. bare '.' from bullet chars)
        if re.fullmatch(r"[^A-Za-z0-9%]+", stripped):
            continue
        # Filter the '00' artifact: the '. %' dot printed before '%' in differential-count rows
        # is mis-recognized as '00' by one engine. Reject short all-digit tokens that cannot
        # plausibly stand alone as a clinical value (i.e. exactly "00" or "0" at low confidence).
        if stripped in ("00",) and score < 0.75:
            continue
        min_score = (
            config.PADDLE_LONG_TEXT_CONFIDENCE_THRESHOLD
            if len(stripped) >= config.PADDLE_LONG_TEXT_MIN_CHARS
            else config.PADDLE_CONFIDENCE_THRESHOLD
        )
        if score < min_score:
            continue
        corrected = correct_against_vocab(text, score)
        vitals_tag = tag_vitals_by_unit(corrected)
        entry = {"text": corrected, "score": round(score, 3)}
        if vitals_tag:
            entry["vitals_type"] = vitals_tag
        final.append(entry)
    return final


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------
def run_printed_ocr(printed_items):
    """printed_items: manifest items already decided as printed by the CLIP
    filter stage. Downloads bytes lazily (never fetched for handwritten
    images), runs the dual-engine ensemble, and writes output to database."""
    if not printed_items:
        print("[paddel] nothing new to OCR.")
        return

    print(f"[paddel] loading PaddleOCR engines on {config.PADDLE_DEVICE}...")
    ocr_en, ocr_server = load_paddle_engines()

    processed_since_save = 0
    for batch_start in range(0, len(printed_items), config.PADDLE_BATCH_SIZE):
        batch_items = printed_items[batch_start: batch_start + config.PADDLE_BATCH_SIZE]
        print(f"[paddel] batch {batch_start // config.PADDLE_BATCH_SIZE + 1} "
              f"({batch_start + 1}-{min(batch_start + config.PADDLE_BATCH_SIZE, len(printed_items))} / {len(printed_items)})")

        img_arrays, valid_items = [], []
        for item in batch_items:
            key = item_key(item)
            try:
                raw_bytes = item.get("raw_bytes") or download_image_bytes(item["file_id"])
                img_arrays.append(preprocess_for_paddle(raw_bytes))
                valid_items.append(item)
            except Exception as e:
                print(f"  SKIP (download/decode error) {key}: {e}")

        if not img_arrays:
            continue

        try:
            en_batch = run_ocr_batch(ocr_en, img_arrays)
            server_batch = run_ocr_batch(ocr_server, img_arrays)
            merged_batch = ensemble_batch(en_batch, server_batch)
        except Exception as e:
            print(f"  ERROR on batch: {e}")
            continue

        from model.db_writer import insert_extracted_document
        for item, merged in zip(valid_items, merged_batch):
            entries = finalize_entries(merged)
            text_lines = [e["text"] for e in entries]
            insert_extracted_document(
                imagename=item["file_name"],
                text_lines=text_lines,
                patient_id=item.get("patient_id"),
                visit_id=item.get("visit_id"),
            )

        processed_since_save += len(valid_items)
        if processed_since_save >= config.PADDLE_SAVE_EVERY:
            import paddle
            paddle.device.cuda.empty_cache()
            gc.collect()
            processed_since_save = 0

    print("[paddel] done.")
