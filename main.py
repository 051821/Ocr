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
    cv_classify_document,
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
    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    with open(config.FILTER_JSON, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False, default=_json_default)


def _classify_tiles(tiles_pil, clip_bundle):
    batch = torch.stack([clip_bundle["preprocess"](t) for t in tiles_pil]).to(clip_bundle["device"])
    with torch.no_grad():
        image_features = clip_bundle["model"].encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ clip_bundle["text_features"].T
        probs = logits.softmax(dim=-1)
    return probs.cpu().numpy()


def classify_image(img_bgr, clip_bundle):
    """Returns (is_handwritten | None, details). ``None`` means non-document.
    Uses a whole-page non-document gate followed by a weighted tile vote.
    A ruled form is not proof that its contents are printed: handwritten
    forms must be allowed to reach the handwriting OCR path.

    NOTE ON BIAS FIXES (see inline comments below for detail): the tile vote
    used to have three compounding biases that all defaulted ambiguous or
    sparse pages to "printed":
      1. non_document tile weight was counted in the ratio denominator but
         could never contribute to the handwritten numerator, silently
         diluting handwritten_ratio.
      2. a tile only voted "handwritten" if it cleared BOTH an absolute
         probability threshold AND a margin over "printed" -- a tile that
         leaned handwritten but didn't clear that bar contributed to the
         denominator but nothing to either numerator, i.e. it was treated
         as a printed vote without ever actually being classified printed.
      3. insufficient/blank tile evidence (e.g. very light or sparse
         handwriting under the per-tile ink-density floor) hard-defaulted
         to printed instead of falling back to the whole-page global CLIP
         probabilities that are already computed earlier in this function.
    """
    labels = clip_bundle["labels"]
    printed_idx = labels.index("printed")
    handwritten_idx = labels.index("handwritten")
    non_document_idx = labels.index("non_document")

    whole_probs = _classify_tiles([bgr_to_pil(img_bgr)], clip_bundle)[0]
    g_hw, g_pr = float(whole_probs[handwritten_idx]), float(whole_probs[printed_idx])
    g_non_doc = float(whole_probs[non_document_idx])

    if g_non_doc >= config.NON_DOCUMENT_THRESHOLD and \
            (g_non_doc - max(g_pr, g_hw)) >= config.NON_DOCUMENT_MARGIN:
        return None, {"decision": "whole_page_non_document", "non_document_prob": round(g_non_doc, 3),
                      "global_printed_prob": round(g_pr, 3), "global_handwritten_prob": round(g_hw, 3)}

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

    # weighted_handwritten / weighted_printed / weighted_non_document are
    # mutually exclusive buckets -- every non-blank tile's weight lands in
    # exactly one of them. weighted_total tracks all non-blank tile weight
    # (used only for the non-document ratio, which legitimately needs to
    # consider the whole page). The handwritten-vs-printed ratio below uses
    # its own denominator (document_weight) that excludes non_document
    # weight, so a few ambiguous/non-document-looking tiles can no longer
    # dilute the handwritten score without ever being able to help it.
    weighted_handwritten = weighted_printed = weighted_non_document = weighted_total = 0.0
    non_blank_tiles = 0
    for row, density in zip(tile_probs, ink_densities):
        hw_p, pr_p = float(row[handwritten_idx]), float(row[printed_idx])
        non_doc_p = float(row[non_document_idx])
        if density < config.MIN_TILE_WEIGHT:
            continue
        non_blank_tiles += 1
        weight = max(density, config.MIN_TILE_WEIGHT)
        weighted_total += weight

        if non_doc_p >= config.NON_DOCUMENT_THRESHOLD and non_doc_p > max(hw_p, pr_p):
            weighted_non_document += weight
            continue

        # Every remaining (document) tile casts a full vote for whichever
        # class it leans toward, using the configured threshold/margin as a
        # measure of *how confidently* it leans handwritten rather than as
        # a hard gate that silently discards ambiguous-but-handwritten-
        # leaning tiles into a de facto "printed" bucket. A tile that
        # clears the strict handwritten bar counts fully as handwritten; a
        # tile that merely leans handwritten (hw_p > pr_p) without clearing
        # the strict bar still counts as handwritten, just via the softer
        # comparison -- it is never dropped on the floor.
        if hw_p >= config.HANDWRITTEN_THRESHOLD and (hw_p - pr_p) >= config.HANDWRITTEN_MARGIN:
            weighted_handwritten += weight
        elif hw_p > pr_p:
            weighted_handwritten += weight
        else:
            weighted_printed += weight

    if non_blank_tiles == 0 or weighted_total == 0:
        # No usable tile evidence at all (e.g. very light/faint handwriting
        # entirely under the per-tile ink-density floor). Fall back to the
        # whole-page global CLIP probabilities computed above instead of
        # silently defaulting to printed.
        fallback_hw = g_hw > g_pr
        return fallback_hw, {
            "decision": "all_tiles_blank_global_fallback",
            "global_printed_prob": round(g_pr, 3), "global_handwritten_prob": round(g_hw, 3),
        }
    if non_blank_tiles < config.MIN_NON_BLANK_TILES_FOR_HANDWRITTEN:
        # Too few non-blank tiles for a reliable tile vote (e.g. sparse
        # handwriting on an otherwise mostly-blank page). Same reasoning:
        # trust the whole-page global probabilities rather than defaulting
        # to printed.
        fallback_hw = g_hw > g_pr
        return fallback_hw, {
            "decision": "insufficient_tile_evidence_global_fallback", "non_blank_tiles": non_blank_tiles,
            "global_printed_prob": round(g_pr, 3), "global_handwritten_prob": round(g_hw, 3),
        }

    non_document_ratio = weighted_non_document / weighted_total
    if non_document_ratio >= config.HANDWRITTEN_TILE_RATIO:
        return None, {"decision": "tile_vote_non_document", "non_document_ratio": round(non_document_ratio, 3),
                      "non_blank_tiles": non_blank_tiles, "global_printed_prob": round(g_pr, 3),
                      "global_handwritten_prob": round(g_hw, 3)}

    document_weight = weighted_handwritten + weighted_printed
    if document_weight == 0:
        # Every non-blank tile landed in the non_document bucket but didn't
        # clear the non_document_ratio threshold above -- fall back to
        # global probabilities rather than dividing by zero / defaulting.
        fallback_hw = g_hw > g_pr
        return fallback_hw, {
            "decision": "no_document_tile_weight_global_fallback",
            "global_printed_prob": round(g_pr, 3), "global_handwritten_prob": round(g_hw, 3),
        }

    handwritten_ratio = weighted_handwritten / document_weight
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
    excluded_count = 0
    to_classify = []

    for item in manifest:
        folder, fname = item["folder_name"], item["file_name"]
        cached = filter_cache.get(folder, {}).get(fname)
        if cached and cached.get("_cfg") == config.CLASSIFIER_CONFIG_FINGERPRINT:
            if cached.get("is_document", True):
                (handwritten_items if cached["is_handwritten"] else printed_items).append(item)
            else:
                excluded_count += 1
        else:
            to_classify.append(item)
    #edited  ...
    if to_classify:
        from concurrent.futures import ThreadPoolExecutor, as_completed

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

        # Decode first.  Non-document gating must happen before any table/line
        # shortcut; hands and faces can contain strong edges that look like a
        # table to the CV heuristic.
        needs_clip = []
        for item in to_classify:
            raw_bytes = prefetched.get(item["file_id"])
            if isinstance(raw_bytes, Exception) or not raw_bytes:
                needs_clip.append((item, None, raw_bytes if isinstance(raw_bytes, Exception) else Exception("empty bytes")))
                continue

            try:
                img_bgr = bytes_to_bgr_array(raw_bytes)
                item["raw_bytes"] = raw_bytes  # carry forward for Stage 2/3
                needs_clip.append((item, img_bgr, None))
            except Exception as e:
                needs_clip.append((item, None, e))

        # Second pass: only load CLIP if there are non-structural images
        if needs_clip:
            clip_bundle = None
            try:
                print(f"[classify] loading CLIP on {config.CLIP_DEVICE} for {len(needs_clip)} image(s)...")
                clip_bundle = load_clip_classifier()
            except (MemoryError, OSError, Exception) as e:
                print(f"[classify] Note: CLIP model could not be loaded due to memory ({e}); using CV stroke/layout fallback.")
                clip_bundle = None

            classified_since_save = 0
            for item, img_bgr, err in needs_clip:
                key = item_key(item)
                if err is not None or img_bgr is None:
                    print(f"  SKIP (classify error) {key}: {err}")
                    is_hw, details = None, {"decision": "error", "error": str(err)}
                elif clip_bundle is not None:
                    try:
                        is_hw, details = classify_image(img_bgr, clip_bundle)
                    except Exception as e:
                        is_hw, details = None, {"decision": "classification_error", "error": str(e)}
                else:
                    # Without CLIP there is no reliable face/hand gate.  Do
                    # not send an unknown image to either OCR model.
                    is_hw, details = None, {"decision": "classifier_unavailable"}

                is_document = is_hw is not None
                entry = {"is_document": is_document, "is_handwritten": bool(is_hw),
                         "_cfg": config.CLASSIFIER_CONFIG_FINGERPRINT, **details}
                filter_cache.setdefault(item["folder_name"], {})[item["file_name"]] = entry
                if is_document:
                    (handwritten_items if is_hw else printed_items).append(item)
                else:
                    excluded_count += 1

                classified_since_save += 1
                if classified_since_save >= config.CLIP_CACHE_SAVE_EVERY:
                    save_filter_cache(filter_cache)
                    classified_since_save = 0

            if clip_bundle is not None:
                unload_clip_classifier(clip_bundle)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[classify] done, CLIP unloaded, VRAM freed.")

        save_filter_cache(filter_cache)

    print(f"[classify] {len(printed_items)} printed | {len(handwritten_items)} handwritten | {excluded_count} excluded "
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