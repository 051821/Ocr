"""
data/data_fetcher.py
Fetches medical documents from the PostgreSQL patientdocument table (Supabase storage URLs).
Constructs a manifest without downloading images to local disk.
"""
import os
import re
import sys
from urllib.parse import unquote, urlparse
import boto3

# Ensure project root is in sys.path when running as a standalone script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import requests
from sqlalchemy import create_engine, text

import config

if not config.DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from .env or config")

ENGINE = create_engine(config.DATABASE_URL, pool_pre_ping=True)
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS

# --- S3 support -------------------------------------------------------
# Authenticated client for private S3 objects (e.g. testdigiaarogyasaarathifiles).
# Credentials come from config.py / .env, or from an IAM role if those are None.
S3_CLIENT = boto3.client(
    "s3",
    aws_access_key_id=getattr(config, "AWS_ACCESS_KEY_ID", None),
    aws_secret_access_key=getattr(config, "AWS_SECRET_ACCESS_KEY", None),
    region_name=getattr(config, "AWS_REGION", None),
)

# Matches virtual-hosted-style (bucket.s3.region.amazonaws.com/key)
# and path-style (s3.region.amazonaws.com/bucket/key) S3 URLs.
_S3_URL_RE = re.compile(
    r"^https?://(?:(?P<bucket1>[^.]+)\.s3[.-][\w-]*\.amazonaws\.com/(?P<key1>.+)"
    r"|s3[.-][\w-]*\.amazonaws\.com/(?P<bucket2>[^/]+)/(?P<key2>.+))$"
)


def _parse_s3_url(file_url):
    """Return (bucket, key) if file_url is an S3 URL, else None."""
    if file_url.startswith("s3://"):
        without_scheme = file_url[len("s3://"):]
        bucket, _, key = without_scheme.partition("/")
        return bucket, unquote(key)

    m = _S3_URL_RE.match(file_url)
    if not m:
        return None
    bucket = m.group("bucket1") or m.group("bucket2")
    key = m.group("key1") or m.group("key2")
    return bucket, unquote(key)
# ------------------------------------------------------------------------


def item_key(item):
    folder = item.get("folder_name")
    return f"{folder}/{item['file_name']}" if folder else item["file_name"]


def skip_key(item):
    return item["file_name"]


def _file_name(file_path):
    return unquote(os.path.basename(urlparse(file_path).path))


def load_skip_keys():
    """Return image keys already present in the destination database."""
    from model import db_writer
    return db_writer.load_skip_keys()


def _list_database_images():
    """Query image documents for patients whose last name is OCR."""
    query = text(r"""
        SELECT
            pd.file_path,
            pd.patient_id,
            pd.visit_id
        FROM "patientdocument" pd
        JOIN "patient" p
            ON p.id = pd.patient_id
        WHERE pd.file_path ~* '\.(jpg|jpeg|png|webp|gif)$'
          AND pd.visit_id IS NOT NULL
          AND p.full_name ~* '(^|[[:space:]])ocr$'
        ORDER BY pd.file_path
    """)

    with ENGINE.connect() as conn:
        return conn.execute(query).mappings().all()


def download_image_bytes(file_url):
    """Fetch image bytes in memory. Uses an authenticated S3 GET for private
    S3 URLs, and a plain HTTP GET for everything else (e.g. public Supabase
    URLs). No file is saved locally to disk either way."""
    s3_target = _parse_s3_url(file_url)

    if s3_target:
        bucket, key = s3_target
        try:
            obj = S3_CLIENT.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except Exception as e:
            print(f"[s3] error downloading s3://{bucket}/{key}: {e}")
            raise

    resp = requests.get(file_url, timeout=60)
    if resp.status_code != 200:
        print(f"[database] HTTP {resp.status_code} downloading {file_url}: {resp.text[:300]!r}")
        resp.raise_for_status()
    return resp.content


def build_manifest(skip_keys=frozenset(), limit=config.FETCH_LIMIT, max_raw_scan=config.MAX_RAW_SCAN):
    """Build a manifest of new image records from patientdocument without downloading to disk."""
    rows = _list_database_images()
    manifest = []
    raw_scanned = 0

    print(f"[database] found {len(rows)} image record(s) matching criteria.")

    for row in rows:
        if max_raw_scan is not None and raw_scanned >= max_raw_scan:
            print(f"[database] hit MAX_RAW_SCAN={max_raw_scan}, stopping scan early.")
            break

        if limit is not None and len(manifest) >= limit:
            break

        raw_scanned += 1
        file_path = row["file_path"]
        file_name = _file_name(file_path)

        # Skip if image name or path already processed
        if file_name in skip_keys or file_path in skip_keys:
            continue

        patient_id = row["patient_id"]
        visit_id = row["visit_id"]

        manifest.append({
            "folder_id": None,
            "folder_name": str(patient_id) if patient_id else "",
            "file_id": file_path,          # URL used by download_image_bytes in memory
            "file_path": file_path,
            "file_name": file_name,
            "patient_id": patient_id,
            "visit_id": visit_id,
            "legacy_id": None,             # Null in destination DB
            "healthcase_id": None,         # Null in destination DB
        })

    print(f"[database] scanned {raw_scanned} raw image(s) -> {len(manifest)} new usable image(s).")
    return manifest


def fetch_batch():
    """Public entry point used by main.py."""
    skip_keys = load_skip_keys()
    print(f"[data_fetcher] {len(skip_keys)} image(s) already processed, will be skipped.")
    manifest = build_manifest(skip_keys=skip_keys)
    return manifest


if __name__ == "__main__":
    batch = fetch_batch()
    print(f"[data_fetcher] {len(batch)} image(s) ready for stage 1 (CLIP classification).")
    for it in batch[:5]:
        print(f"  {it['file_name']} | patient_id={it['patient_id']} | visit_id={it['visit_id']}")
