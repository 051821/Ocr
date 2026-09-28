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


import re

def collapse_repetitive_text_in_line(line: str, max_repeat: int = 2) -> str:
    """
    Collapses word or phrase loops within a single line.
    E.g. 'Tab, tab, tab, tab, tab, D3, D3, D3, D3' -> 'Tab, Tab, D3, D3'
    """
    if not line:
        return line

    # 1. Single word loops e.g. "Tab, tab, tab, tab..." -> "Tab, Tab"
    pattern = r'(\b[\w\-\.%/]+\b)(?:[\s,•\-\|]+\1\b){' + str(max_repeat) + r',}'
    def _repl(match):
        word = match.group(1)
        return ", ".join([word] * max_repeat)

    cleaned = re.sub(pattern, _repl, line, flags=re.IGNORECASE)

    # 2. Short 2-3 word phrase loops e.g. "after food, after food, after food..." -> "after food, after food"
    phrase_pattern = r'(\b[\w\-\.%/]+(?:\s+[\w\-\.%/]+){1,2}\b)(?:[\s,•\-\|]+\1\b){' + str(max_repeat) + r',}'
    cleaned = re.sub(phrase_pattern, _repl, cleaned, flags=re.IGNORECASE)

    return cleaned.strip()


def text_to_lines(text, max_consecutive_repetitions: int = 3):
    """
    Splits VLM output into cleaned, non-empty lines.
    Collapses both within-line token repetition loops and consecutive line repetition loops.
    """
    if text is None:
        return []
    lines = []
    prev_norm = None
    consecutive_count = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            # A blank line breaks any run of consecutive repeats.
            prev_norm = None
            consecutive_count = 0
            continue

        # Collapse within-line token repetition loops (e.g. Tab, tab, tab... D3, D3, D3...)
        line = collapse_repetitive_text_in_line(line)

        norm = line.lower()
        if norm == prev_norm:
            consecutive_count += 1
        else:
            consecutive_count = 1
            prev_norm = norm
        if consecutive_count > max_consecutive_repetitions:
            continue
        lines.append(line)
    return lines


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
        "repetition_penalty": 1.15,
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