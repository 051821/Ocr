"""
model/paddel.py
"""
import gc
import json
import os

import paddle
from rapidfuzz import process, fuzz

import config
from data.data_fetcher import download_image_bytes, item_key
from preprocessing.printed_preprocessing import preprocess_for_paddle
from model.load_model import load_paddle_engines


# ---------------------------------------------------------------------------
# RESUMABLE JSON I/O
# ---------------------------------------------------------------------------
def load_existing_results():
    if os.path.exists(config.OUTPUT_JSON):
        with open(config.OUTPUT_JSON, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    return {}


def save_results(all_results):
    with open(config.OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)


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
    """Same detected box (rounded to the nearest 10px) -> keep the higher
    confidence recognition between the two engines."""
    merged_batch = []
    for en_results, server_results in zip(en_batch_results, server_batch_results):
        merged = {}
        for box, text, score in en_results + server_results:
            key = tuple(round(v / 10) * 10 for v in box)
            if key not in merged or score > merged[key][2]:
                merged[key] = (box, text, score)
        merged_batch.append(merged)
    return merged_batch


# ---------------------------------------------------------------------------
# READING ORDER
# ---------------------------------------------------------------------------
def get_center(box):
    x_coords = box[0::2]
    y_coords = box[1::2]
    return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)


def cluster_into_columns(items, gap_threshold=config.COLUMN_GAP_THRESHOLD):
    items = sorted(items, key=lambda i: i["x"])
    columns, current_col, prev_x = [], [], None
    for item in items:
        if prev_x is not None and (item["x"] - prev_x) > gap_threshold:
            columns.append(current_col)
            current_col = []
        current_col.append(item)
        prev_x = item["x"]
    if current_col:
        columns.append(current_col)
    return columns


def sort_column_top_to_bottom(column, row_tolerance=config.ROW_TOLERANCE):
    column = sorted(column, key=lambda i: i["y"])
    rows, current_row, current_y = [], [], None
    for item in column:
        if current_y is None or abs(item["y"] - current_y) <= row_tolerance:
            current_row.append(item)
            current_y = item["y"] if current_y is None else current_y
        else:
            current_row.sort(key=lambda i: i["x"])
            rows.append(current_row)
            current_row = [item]
            current_y = item["y"]
    if current_row:
        current_row.sort(key=lambda i: i["x"])
        rows.append(current_row)
    ordered = []
    for row in rows:
        ordered.extend(row)
    return ordered


def sort_reading_order(merged):
    items = []
    for box, text, score in merged.values():
        x, y = get_center(box)
        items.append({"x": x, "y": y, "text": text, "score": score})
    columns = cluster_into_columns(items)
    ordered = []
    for col in columns:
        ordered.extend(sort_column_top_to_bottom(col))
    return ordered


# ---------------------------------------------------------------------------
# VOCAB CORRECTION + VITALS TAGGING (printed-domain specific)
# ---------------------------------------------------------------------------
def correct_against_vocab(text, score, conf_gate=0.85, threshold=config.PADDLE_VOCAB_MATCH_THRESHOLD):
    if score >= conf_gate or not text.strip():
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
        if not text.strip():
            continue
        min_score = (
            config.PADDLE_LONG_TEXT_CONFIDENCE_THRESHOLD
            if len(text.strip()) >= config.PADDLE_LONG_TEXT_MIN_CHARS
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
    images), runs the dual-engine ensemble, and writes/resumes output.json."""
    all_results = load_existing_results()
    done_keys = {
        f"{folder}/{fname}" for folder, files in all_results.items() for fname in files
    }
    remaining = [it for it in printed_items if item_key(it) not in done_keys]

    if not remaining:
        print("[paddel] nothing new to OCR.")
        return all_results

    print(f"[paddel] loading PaddleOCR engines on {config.PADDLE_DEVICE}...")
    ocr_en, ocr_server = load_paddle_engines()

    processed_since_save = 0
    for batch_start in range(0, len(remaining), config.PADDLE_BATCH_SIZE):
        batch_items = remaining[batch_start: batch_start + config.PADDLE_BATCH_SIZE]
        print(f"[paddel] batch {batch_start // config.PADDLE_BATCH_SIZE + 1} "
              f"({batch_start + 1}-{min(batch_start + config.PADDLE_BATCH_SIZE, len(remaining))} / {len(remaining)})")

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

        for item, merged in zip(valid_items, merged_batch):
            entries = finalize_entries(merged)
            all_results.setdefault(item["folder_name"], {})[item["file_name"]] = [e["text"] for e in entries]

        processed_since_save += len(valid_items)
        if processed_since_save >= config.PADDLE_SAVE_EVERY:
            paddle.device.cuda.empty_cache()
            gc.collect()
            save_results(all_results)
            print(f"  --- progress saved ({sum(len(v) for v in all_results.values())} images total) ---")
            processed_since_save = 0

    save_results(all_results)
    print(f"[paddel] done. {config.OUTPUT_JSON} has "
          f"{sum(len(v) for v in all_results.values())} images total.")
    return all_results
