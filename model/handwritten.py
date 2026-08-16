"""
model/handwritten.py

Runs HANDWRITTEN images (as decided by the CLIP filter in filter.json)
through the vLLM model hosted on EC2. No OCR confidence score exists here
the way it does for Paddle -- the VLM either answers or times out/retries --
so this stage has its own retry/timeout knobs (HANDWRITTEN_TIMEOUT,
HANDWRITTEN_MAX_RETRIES) instead of a score threshold.

Writes/resumes result.json in the same shape as output.json:
    { "<folder_name>": { "<file_name>": [ "line1", "line2", ... ] } }
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config
from data.data_fetcher import download_image_bytes, item_key
from preprocessing.handwritten_preprocessing import encode_image_b64


# ---------------------------------------------------------------------------
# RESUMABLE JSON I/O
# ---------------------------------------------------------------------------
def load_existing_results():
    if os.path.exists(config.RESULT_JSON):
        with open(config.RESULT_JSON, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    return {}


def save_results(all_results):
    with open(config.RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)


def text_to_lines(text):
    if text is None:
        return []
    return [line for line in (l.strip() for l in text.splitlines()) if line]


# ---------------------------------------------------------------------------
# OCR EXECUTION
# ---------------------------------------------------------------------------
def _run_single(item, endpoint):
    key = item_key(item)
    try:
        raw_bytes = item.get("raw_bytes") or download_image_bytes(item["file_id"])
        image_b64, mime = encode_image_b64(raw_bytes, item["file_name"])
    except Exception as e:
        return {"folder_name": item["folder_name"], "file_name": item["file_name"],
                "status": "failed", "text": None, "error": f"download/encode error: {e}"}

    payload = {
        "model": "ocr-engine",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config.HANDWRITTEN_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 1000,
    }

    last_error = None
    for attempt in range(1, config.HANDWRITTEN_MAX_RETRIES + 1):
        try:
            resp = requests.post(f"{endpoint}/v1/chat/completions", json=payload, timeout=config.HANDWRITTEN_TIMEOUT)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return {"folder_name": item["folder_name"], "file_name": item["file_name"],
                    "status": "ok", "text": text, "error": None}
        except Exception as e:
            last_error = str(e)
            if attempt < config.HANDWRITTEN_MAX_RETRIES:
                time.sleep(2 * attempt)
    return {"folder_name": item["folder_name"], "file_name": item["file_name"],
            "status": "failed", "text": None, "error": last_error}


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------
def run_handwritten_ocr(handwritten_items, endpoint=None):
    """handwritten_items: manifest items already decided as handwritten by
    the CLIP filter stage. Caller (main.py) is responsible for making sure
    the EC2 endpoint is up (see model/load_model.start_ec2_instance) before
    calling this."""
    endpoint = (endpoint or config.HANDWRITTEN_ENDPOINT).rstrip("/")

    all_results = load_existing_results()
    done_keys = {
        f"{folder}/{fname}" for folder, files in all_results.items() for fname in files
    }
    remaining = [it for it in handwritten_items if item_key(it) not in done_keys]

    if not remaining:
        print("[handwritten] nothing new to OCR.")
        return all_results

    print(f"[handwritten] {len(remaining)} image(s) to send to {endpoint}")
    start = time.monotonic()
    done = 0
    since_save = 0

    with ThreadPoolExecutor(max_workers=config.HANDWRITTEN_CONCURRENCY) as pool:
        futures = {pool.submit(_run_single, item, endpoint): item for item in remaining}
        for future in as_completed(futures):
            r = future.result()
            if r["status"] == "ok":
                all_results.setdefault(r["folder_name"], {})[r["file_name"]] = text_to_lines(r["text"])
            else:
                print(f"  FAILED {r['folder_name']}/{r['file_name']}: {r['error']}")
            done += 1
            since_save += 1
            if r["status"] == "failed":
                print(f"  FAILED {r['folder_name']}/{r['file_name']}: {r['error']}")
            if done % 10 == 0 or done == len(remaining):
                elapsed = time.monotonic() - start
                print(f"  {done}/{len(remaining)} done ({elapsed:.0f}s elapsed)")
            if since_save >= config.HANDWRITTEN_SAVE_EVERY:
                save_results(all_results)
                since_save = 0

    save_results(all_results)
    total_done = sum(len(v) for v in all_results.values())
    print(f"[handwritten] done. {config.RESULT_JSON} has {total_done} images total.")
    return all_results
