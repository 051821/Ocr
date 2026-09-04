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
        return {
            "folder_name": item["folder_name"],
            "file_name": item["file_name"],
            "patient_id": item.get("patient_id"),
            "visit_id": item.get("visit_id"),
            "status": "failed",
            "text": None,
            "error": f"download/encode error: {e}",
        }

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
            return {
                "folder_name": item["folder_name"],
                "file_name": item["file_name"],
                "patient_id": item.get("patient_id"),
                "visit_id": item.get("visit_id"),
                "status": "ok",
                "text": text,
                "error": None,
            }
        except Exception as e:
            last_error = str(e)
            if attempt < config.HANDWRITTEN_MAX_RETRIES:
                time.sleep(2 * attempt)
    return {
        "folder_name": item["folder_name"],
        "file_name": item["file_name"],
        "patient_id": item.get("patient_id"),
        "visit_id": item.get("visit_id"),
        "status": "failed",
        "text": None,
        "error": last_error,
    }


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------
def run_handwritten_ocr(handwritten_items, endpoint=None):
    """handwritten_items: manifest items already decided as handwritten by
    the CLIP filter stage. Caller (main.py) is responsible for making sure
    the EC2 endpoint is up (see model/load_model.start_ec2_instance) before
    calling this."""
    endpoint = (endpoint or config.HANDWRITTEN_ENDPOINT).rstrip("/")

    if not handwritten_items:
        print("[handwritten] nothing new to OCR.")
        return

    print(f"[handwritten] {len(handwritten_items)} image(s) to send to {endpoint}")
    start = time.monotonic()
    done = 0

    from model.db_writer import insert_extracted_document
    with ThreadPoolExecutor(max_workers=config.HANDWRITTEN_CONCURRENCY) as pool:
        futures = {pool.submit(_run_single, item, endpoint): item for item in handwritten_items}
        for future in as_completed(futures):
            r = future.result()
            if r["status"] == "ok":
                text_lines = text_to_lines(r["text"])
                insert_extracted_document(
                    imagename=r["file_name"],
                    text_lines=text_lines,
                    patient_id=r.get("patient_id"),
                    visit_id=r.get("visit_id"),
                )
            else:
                print(f"  FAILED {r['folder_name']}/{r['file_name']}: {r['error']}")
            done += 1
            if done % 10 == 0 or done == len(handwritten_items):
                elapsed = time.monotonic() - start
                print(f"  {done}/{len(handwritten_items)} done ({elapsed:.0f}s elapsed)")

    print("[handwritten] done.")
