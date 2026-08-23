"""
data/data_fetcher.py

IMPORTANT: this module hands back RAW BYTES only, never a decoded array.
CLIP, PaddleOCR, and the handwritten EC2 model each need the image in a
different shape (BGR numpy tiles, resized BGR numpy, resized base64-JPEG)
-- decoding here would force one preprocessing choice on all three. Each
consumer decodes via its own module in preprocessing/.
"""
import os
import requests

import config

DRIVE_FILES_URL = config.DRIVE_FILES_URL
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS


def item_key(item):
    return f"{item['folder_name']}/{item['file_name']}"


def skip_key(item):
    return item["file_name"]


# ---------------------------------------------------------------------------
# SKIP-KEY COMPUTATION (constraint: never redo an image already in DB)
# ---------------------------------------------------------------------------
def load_skip_keys():
    """Union of everything already present in the database and pending linkages."""
    from model import db_writer
    return db_writer.load_skip_keys()


# ---------------------------------------------------------------------------
# DRIVE BACKEND
# ---------------------------------------------------------------------------
def _drive_get(url, params):
    resp = requests.get(url, params=params, timeout=30)
    ctype = resp.headers.get("Content-Type", "")
    if resp.status_code != 200 or "application/json" not in ctype:
        print(f"[drive] HTTP {resp.status_code} from {url}")
        print(f"[drive] Content-Type: {ctype}")
        print(f"[drive] Body (first 500 chars): {resp.text[:500]!r}")
        resp.raise_for_status()
        raise RuntimeError(
            "Drive API did not return JSON. This is usually caused by: "
            "(1) a proxy/firewall/antivirus intercepting requests.googleapis.com, "
            "(2) an invalid/restricted API key, or "
            "(3) the Drive API not being enabled on the key's Cloud project."
        )
    return resp.json()


def _list_drive_subfolders(root_folder_id, api_key):
    query = (
        f"'{root_folder_id}' in parents "
        "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    folders, page_token = [], None
    while True:
        params = {
            "q": query, "fields": "nextPageToken, files(id, name)", "pageSize": 1000,
            "key": api_key, "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
        }
        if page_token:
            params["pageToken"] = page_token
        data = _drive_get(DRIVE_FILES_URL, params)
        folders.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return folders


def _list_drive_images(folder_id, api_key):
    query = f"'{folder_id}' in parents and trashed = false"
    files, page_token = [], None
    while True:
        params = {
            "q": query, "fields": "nextPageToken, files(id, name, mimeType)", "pageSize": 1000,
            "key": api_key, "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
        }
        if page_token:
            params["pageToken"] = page_token
        data = _drive_get(DRIVE_FILES_URL, params)
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return [f for f in files if os.path.splitext(f["name"])[1].lower() in IMAGE_EXTENSIONS]


def download_image_bytes(file_id):
    """Authenticated Drive API download (alt=media) -- not the public
    uc?export=download link, which is unauthenticated, requires the file to
    be individually link-shared, and breaks whenever Google changes the
    confirm-token flow."""
    url = f"{DRIVE_FILES_URL}/{file_id}"
    resp = requests.get(
        url,
        params={"alt": "media", "key": config.DRIVE_API_KEY, "supportsAllDrives": "true"},
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"[drive] HTTP {resp.status_code} downloading {file_id}: {resp.text[:300]!r}")
        resp.raise_for_status()
    return resp.content


def build_manifest(skip_keys=frozenset(), limit=config.FETCH_LIMIT, max_raw_scan=config.MAX_RAW_SCAN):
    """One manifest entry per NEW image (already-done keys excluded), capped
    at `limit`. Stops scanning folders early once `limit` usable images are
    found, or once `max_raw_scan` total images have been looked at, so a
    small test batch doesn't require walking every folder on the drive."""
    if config.DATA_SOURCE != "drive":
        raise NotImplementedError(
            f"DATA_SOURCE={config.DATA_SOURCE!r} has no fetcher yet -- add a branch here "
            f"when you plug in a new source (local disk, S3, etc.)."
        )

    manifest = []
    raw_scanned = 0
    subfolders = _list_drive_subfolders(config.DRIVE_ROOT_FOLDER_ID, config.DRIVE_API_KEY)
    print(f"[drive] found {len(subfolders)} folder(s) under root.")

    for folder in subfolders:
        if limit is not None and len(manifest) >= limit:
            break
        if max_raw_scan is not None and raw_scanned >= max_raw_scan:
            print(f"[drive] hit MAX_RAW_SCAN={max_raw_scan}, stopping scan early.")
            break

        images = _list_drive_images(folder["id"], config.DRIVE_API_KEY)
        if images:
            print(f"[drive] {folder['name']}: {len(images)} image(s)")

        for img_file in images:
            if limit is not None and len(manifest) >= limit:
                break
            if max_raw_scan is not None and raw_scanned >= max_raw_scan:
                break
            raw_scanned += 1
            item = {
                "folder_id": folder["id"],
                "folder_name": folder["name"],
                "file_id": img_file["id"],
                "file_name": img_file["name"],
            }
            if skip_key(item) in skip_keys:
                continue
            manifest.append(item)

    print(f"[drive] scanned {raw_scanned} raw image(s) -> {len(manifest)} new usable image(s).")
    return manifest


def fetch_batch():
    """Public entry point used by main.py. Returns manifest items (no image
    bytes attached yet -- bytes are pulled lazily per-stage so a CLIP-only
    dry run doesn't pay for downloads it won't use)."""
    skip_keys = load_skip_keys()
    print(f"[data_fetcher] {len(skip_keys)} image(s) already processed, will be skipped.")
    manifest = build_manifest(skip_keys=skip_keys)
    return manifest


if __name__ == "__main__":
    batch = fetch_batch()
    print(f"[data_fetcher] {len(batch)} image(s) ready for stage 1 (CLIP classification).")
    for it in batch[:5]:
        print(f"  {item_key(it)}")
