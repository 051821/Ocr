"""
main.py
"""
import gc
import json
import os

import torch

import config
from data.data_fetcher import fetch_batch, item_key
from preprocessing.clip_preprocessing import (
    bytes_to_bgr_array,
    bgr_to_pil,
    make_tiles,
    compute_ink_density,
    has_table_structure,
)
from model.load_model import (
    load_clip_classifier,
    unload_clip_classifier,
    start_ec2_instance,
    stop_ec2_instance,
)
from model.paddel import run_printed_ocr
from model.handwritten import run_handwritten_ocr


# ---------------------------------------------------------------------------
# STAGE 1 -- CLIP CLASSIFICATION -> filter.json
# ---------------------------------------------------------------------------
def load_filter_cache():
    if os.path.exists(config.FILTER_JSON):
        with open(config.FILTER_JSON, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_filter_cache(cache):
    with open(config.FILTER_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _classify_tiles(tiles_pil, clip_bundle):
    batch = torch.stack([clip_bundle["preprocess"](t) for t in tiles_pil]).to(clip_bundle["device"])
    with torch.no_grad():
        image_features = clip_bundle["model"].encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ clip_bundle["text_features"].T
        probs = logits.softmax(dim=-1)
    return probs.cpu().numpy()


def classify_image(img_bgr, clip_bundle):
    """Returns (is_handwritten: bool, details: dict). Mirrors the tiered
    decision from the original test.py: structural override first, then a
    cheap whole-page shortcut, then a weighted tile vote as the fallback."""
    labels = clip_bundle["labels"]
    printed_idx = labels.index("printed")
    handwritten_idx = labels.index("handwritten")

    if config.STRUCTURAL_OVERRIDE_ENABLED and has_table_structure(img_bgr):
        return False, {"decision": "structural_table_override"}

    whole_probs = _classify_tiles([bgr_to_pil(img_bgr)], clip_bundle)[0]
    g_hw, g_pr = float(whole_probs[handwritten_idx]), float(whole_probs[printed_idx])

    if g_pr >= config.PRINTED_SHORTCUT_THRESHOLD and (g_pr - g_hw) >= config.PRINTED_SHORTCUT_MARGIN:
        return False, {"decision": "whole_page_printed", "global_printed_prob": round(g_pr, 3),
                        "global_handwritten_prob": round(g_hw, 3)}

    if config.HANDWRITTEN_SHORTCUT_ENABLED and g_hw >= config.HANDWRITTEN_SHORTCUT_THRESHOLD and \
            (g_hw - g_pr) >= config.HANDWRITTEN_SHORTCUT_MARGIN:
        return True, {"decision": "whole_page_handwritten", "global_printed_prob": round(g_pr, 3),
                       "global_handwritten_prob": round(g_hw, 3)}

    bgr_tiles = make_tiles(img_bgr)
    if not bgr_tiles:
        return False, {"reason": "no_tiles"}

    pil_tiles = [bgr_to_pil(t) for t in bgr_tiles]
    ink_densities = [compute_ink_density(t) for t in bgr_tiles]

    tile_probs = []
    for start in range(0, len(pil_tiles), config.CLIP_BATCH_SIZE):
        chunk = pil_tiles[start:start + config.CLIP_BATCH_SIZE]
        tile_probs.extend(_classify_tiles(chunk, clip_bundle))

    weighted_handwritten = weighted_total = 0.0
    non_blank_tiles = 0
    for row, density in zip(tile_probs, ink_densities):
        hw_p, pr_p = float(row[handwritten_idx]), float(row[printed_idx])
        if density < config.MIN_TILE_WEIGHT:
            continue
        non_blank_tiles += 1
        weight = max(density, config.MIN_TILE_WEIGHT)
        weighted_total += weight
        if hw_p >= config.HANDWRITTEN_THRESHOLD and (hw_p - pr_p) >= config.HANDWRITTEN_MARGIN:
            weighted_handwritten += weight

    if non_blank_tiles == 0 or weighted_total == 0:
        return False, {"decision": "all_tiles_blank", "global_printed_prob": round(g_pr, 3),
                        "global_handwritten_prob": round(g_hw, 3)}
    if non_blank_tiles < config.MIN_NON_BLANK_TILES_FOR_HANDWRITTEN:
        return False, {"decision": "insufficient_tile_evidence", "non_blank_tiles": non_blank_tiles,
                        "global_printed_prob": round(g_pr, 3), "global_handwritten_prob": round(g_hw, 3)}

    handwritten_ratio = weighted_handwritten / weighted_total
    is_handwritten = handwritten_ratio >= config.HANDWRITTEN_TILE_RATIO
    return is_handwritten, {
        "decision": "tile_vote", "handwritten_ratio": round(handwritten_ratio, 3),
        "non_blank_tiles": non_blank_tiles, "global_printed_prob": round(g_pr, 3),
        "global_handwritten_prob": round(g_hw, 3),
    }


def classify_batch(manifest):
    """Classifies every image in `manifest` not already cached under the
    CURRENT threshold fingerprint, writes filter.json, and splits the
    manifest into (printed_items, handwritten_items)."""
    from data.data_fetcher import download_image_bytes

    filter_cache = load_filter_cache()
    printed_items, handwritten_items = [], []
    to_classify = []

    for item in manifest:
        folder, fname = item["folder_name"], item["file_name"]
        cached = filter_cache.get(folder, {}).get(fname)
        if cached and cached.get("_cfg") == config.CLASSIFIER_CONFIG_FINGERPRINT:
            (handwritten_items if cached["is_handwritten"] else printed_items).append(item)
        else:
            to_classify.append(item)
    #edited  ...
    if to_classify:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"[classify] loading CLIP on {config.CLIP_DEVICE} for {len(to_classify)} new image(s)...")
        clip_bundle = load_clip_classifier()

        print(f"[classify] prefetching {len(to_classify)} image(s) with {config.DOWNLOAD_WORKERS} worker(s)...")
        prefetched = {}
        with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as pool:
            futures = {pool.submit(download_image_bytes, it["file_id"]): it["file_id"] for it in to_classify}
            for future in as_completed(futures):
                file_id = futures[future]
                try:
                    prefetched[file_id] = future.result()
                except Exception as e:
                    prefetched[file_id] = e

        classified_since_save = 0
        for item in to_classify:
            key = item_key(item)
            raw_bytes = prefetched.get(item["file_id"])
            if isinstance(raw_bytes, Exception):
                print(f"  SKIP (classify error) {key}: {raw_bytes}")
                is_hw, details = False, {"decision": "error", "error": str(raw_bytes)}
            else:
                try:
                    img_bgr = bytes_to_bgr_array(raw_bytes)
                    is_hw, details = classify_image(img_bgr, clip_bundle)
                    item["raw_bytes"] = raw_bytes  # carry forward -- stage 2/3 reuse this instead of re-downloading
                except Exception as e:
                    print(f"  SKIP (classify error) {key}: {e}")
                    is_hw, details = False, {"decision": "error", "error": str(e)}

            entry = {"is_handwritten": is_hw, "_cfg": config.CLASSIFIER_CONFIG_FINGERPRINT, **details}
            filter_cache.setdefault(item["folder_name"], {})[item["file_name"]] = entry
            (handwritten_items if is_hw else printed_items).append(item)

            classified_since_save += 1
            if classified_since_save >= config.CLIP_CACHE_SAVE_EVERY:
                save_filter_cache(filter_cache)
                classified_since_save = 0

        save_filter_cache(filter_cache)
        unload_clip_classifier(clip_bundle)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[classify] done, CLIP unloaded, VRAM freed.")

    print(f"[classify] {len(printed_items)} printed | {len(handwritten_items)} handwritten "
          f"({len(manifest) - len(to_classify)} from cache)")
    return printed_items, handwritten_items


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    from model.db_writer import init_pool, close_pool, retry_pending_linkage

    init_pool()
    try:
        print("=== DB RETRY: retry pending linkage ===")
        retry_pending_linkage()

        print("=== STAGE 0: fetch ===")
        manifest = fetch_batch()
        if not manifest:
            print("Nothing new to process.")
            return

        print("=== STAGE 1: CLIP classify -> filter.json ===")
        printed_items, handwritten_items = classify_batch(manifest)

        if printed_items:
            print("=== STAGE 2: PaddleOCR (printed) -> DB ===")
            run_printed_ocr(printed_items)
        else:
            print("=== STAGE 2: skipped, no printed images this run ===")

        if handwritten_items:
            print("=== STAGE 3: EC2 handwritten model -> DB ===")
            try:
                start_ec2_instance()
                run_handwritten_ocr(handwritten_items)
            finally:
                stop_ec2_instance()
        else:
            print("=== STAGE 3: skipped, no handwritten images this run ===")

        print("=== pipeline complete ===")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
