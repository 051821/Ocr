"""
model/mistral_printed.py

Runs PRINTED document images through Mistral OCR / Vision model via OpenRouter.
Replaces PaddleOCR in Stage 2 of the pipeline without PDF conversion.
"""
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config
from data.data_fetcher import download_image_bytes, item_key
from model.handwritten import text_to_lines

_session = None


def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=Retry(total=0, connect=0, read=0)
        )
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


from preprocessing.printed_preprocessing import preprocess_for_mistral


def _prepare_image_data_url(raw_bytes):
    """Applies deskew, shadow removal, CLAHE contrast enhancement, and unsharp sharpening,
    then converts preprocessed image bytes to a JPEG base64 Data URL."""
    try:
        raw_bytes = preprocess_for_mistral(raw_bytes)
    except Exception as e:
        print(f"[mistral_printed] Warning: Image preprocessing failed ({e}); using raw image bytes.")

    image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    image.thumbnail((2000, 2000))

    jpeg_buffer = BytesIO()
    quality = 85
    while True:
        jpeg_buffer.seek(0)
        jpeg_buffer.truncate(0)
        image.save(jpeg_buffer, format="JPEG", quality=quality, optimize=True)
        if jpeg_buffer.tell() <= config.MISTRAL_OCR_MAX_IMAGE_BYTES or quality <= 40:
            break
        quality -= 5

    img_base64 = base64.b64encode(jpeg_buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{img_base64}"


def _run_single_mistral_ocr(item):
    key = item_key(item)
    try:
        raw_bytes = item.get("raw_bytes") or download_image_bytes(item["file_id"])
        image_data_url = _prepare_image_data_url(raw_bytes)
    except Exception as e:
        return {
            "folder_name": item["folder_name"],
            "file_name": item["file_name"],
            "patient_id": item.get("patient_id"),
            "visit_id": item.get("visit_id"),
            "status": "failed",
            "text": None,
            "error": f"image encoding error: {e}",
        }

    payload = {
        "model": config.MISTRAL_OCR_DOWNSTREAM_MODEL,
        "temperature": 0,
        "repetition_penalty": 1.15,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": config.MISTRAL_OCR_PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    session = get_session()
    last_error = None
    for attempt in range(1, config.MISTRAL_OCR_MAX_RETRIES + 1):
        try:
            response = session.post(
                config.MISTRAL_OCR_URL,
                headers=headers,
                json=payload,
                timeout=config.MISTRAL_OCR_TIMEOUT,
            )

            if response.ok:
                result = response.json()
                content = (
                    result
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )
                return {
                    "folder_name": item["folder_name"],
                    "file_name": item["file_name"],
                    "patient_id": item.get("patient_id"),
                    "visit_id": item.get("visit_id"),
                    "status": "ok",
                    "text": content,
                    "error": None,
                }

            last_error = f"HTTP {response.status_code}: {response.text}"
            if response.status_code not in (429, 502, 503, 504):
                break

        except requests.Timeout:
            last_error = f"Request timed out after {config.MISTRAL_OCR_TIMEOUT}s (attempt {attempt}/{config.MISTRAL_OCR_MAX_RETRIES})"
        except requests.RequestException as e:
            last_error = f"Request error: {e}"

        if attempt < config.MISTRAL_OCR_MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))

    return {
        "folder_name": item["folder_name"],
        "file_name": item["file_name"],
        "patient_id": item.get("patient_id"),
        "visit_id": item.get("visit_id"),
        "status": "failed",
        "text": None,
        "error": last_error,
    }


def run_printed_ocr(printed_items):
    """printed_items: manifest items decided as printed by the CLIP filter stage.
    Runs each printed item through Mistral OCR via OpenRouter and writes extracted results to DB."""
    if not printed_items:
        print("[mistral_printed] nothing new to OCR.")
        return

    print(f"[mistral_printed] {len(printed_items)} image(s) to send to Mistral OCR via OpenRouter (workers={config.MISTRAL_OCR_CONCURRENCY})")
    start = time.monotonic()
    done = 0

    from model.db_writer import insert_extracted_document
    with ThreadPoolExecutor(max_workers=config.MISTRAL_OCR_CONCURRENCY) as pool:
        futures = {pool.submit(_run_single_mistral_ocr, item): item for item in printed_items}
        for future in as_completed(futures):
            r = future.result()
            done += 1
            if r["status"] == "ok":
                text_lines = text_to_lines(r["text"])
                insert_extracted_document(
                    imagename=r["file_name"],
                    text_lines=text_lines,
                    patient_id=r.get("patient_id"),
                    visit_id=r.get("visit_id"),
                )
                print(f"[mistral_printed] ({done}/{len(printed_items)}) {r['file_name']}: extracted {len(text_lines)} lines.")
            else:
                print(f"[mistral_printed] ({done}/{len(printed_items)}) FAILED {r['folder_name']}/{r['file_name']}: {r['error']}")

    elapsed = time.monotonic() - start
    print(f"[mistral_printed] finished {len(printed_items)} image(s) in {elapsed:.1f}s.")
