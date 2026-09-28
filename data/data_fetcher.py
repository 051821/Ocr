"""
data/data_fetcher.py
Fetches medical documents listed in the PostgreSQL patientdocument table.

The authoritative location of each document is patientdocument.storage_key,
the object path inside the Supabase Storage bucket (default
`digiswasthyafilescopy`). Objects are read through Supabase's S3-compatible
endpoint with the project's S3 access keys (same variables the API uses:
SUPABASE_URL, SUPABASE_BUCKET, SUPABASE_S3_ACCESS_KEY_ID,
SUPABASE_S3_SECRET_ACCESS_KEY).

The legacy patientdocument.file_path (old AWS S3 URL) is kept in the manifest
for reference / skip-key backward compatibility only and is NEVER used to
download anything. Image bytes are downloaded in memory only.
"""
import os
import posixpath
import sys

# Ensure project root is in sys.path when running as a standalone script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import boto3
from botocore.config import Config as BotoConfig
from sqlalchemy import create_engine, text

import config

if not config.DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from .env or config")

# --- Supabase Storage (S3-compatible) configuration -------------------
SUPABASE_URL = getattr(config, "SUPABASE_URL", None)
SUPABASE_BUCKET = getattr(config, "SUPABASE_BUCKET", None) or "digiswasthyafilescopy"
_S3_ACCESS_KEY_ID = getattr(config, "SUPABASE_S3_ACCESS_KEY_ID", None)
_S3_SECRET_ACCESS_KEY = getattr(config, "SUPABASE_S3_SECRET_ACCESS_KEY", None)
_S3_REGION = getattr(config, "SUPABASE_S3_REGION", None)
# Optional override; otherwise derived from SUPABASE_URL (…/storage/v1/s3).
_S3_ENDPOINT = getattr(config, "SUPABASE_S3_ENDPOINT", None) or (
    SUPABASE_URL.rstrip("/") + "/storage/v1/s3" if SUPABASE_URL else None
)

_missing = [
    name for name, val in {
        "SUPABASE_URL (or SUPABASE_S3_ENDPOINT)": _S3_ENDPOINT,
        "SUPABASE_S3_ACCESS_KEY_ID": _S3_ACCESS_KEY_ID,
        "SUPABASE_S3_SECRET_ACCESS_KEY": _S3_SECRET_ACCESS_KEY,
        "SUPABASE_S3_REGION": _S3_REGION,
    }.items() if not val
]
if _missing:
    raise RuntimeError("Missing from .env or config: " + ", ".join(_missing))

ENGINE = create_engine(config.DATABASE_URL, pool_pre_ping=True)
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS

S3_CLIENT = boto3.client(
    "s3",
    endpoint_url=_S3_ENDPOINT,
    aws_access_key_id=_S3_ACCESS_KEY_ID,
    aws_secret_access_key=_S3_SECRET_ACCESS_KEY,
    region_name=_S3_REGION,
    config=BotoConfig(s3={"addressing_style": "path"}, connect_timeout=10, read_timeout=30),  # required by Supabase
)
# ------------------------------------------------------------------------


def item_key(item):
    folder = item.get("folder_name")
    return f"{folder}/{item['file_name']}" if folder else item["file_name"]


def skip_key(item):
    return item["file_name"]


def _file_name(storage_key):
    """Derive the file name from a storage_key (an object path, not a URL,
    so it is used as-is with no URL-decoding)."""
    return posixpath.basename(storage_key)


def load_skip_keys():
    """Return image keys already present in the destination database."""
    from model import db_writer
    return db_writer.load_skip_keys()


def _list_database_images():
    """Query image documents that have a usable Supabase storage_key."""
    query = text(r"""
        SELECT
            pd.file_path,
            pd.storage_key,
            pd.patient_id,
            pd.visit_id
        FROM "patientdocument" pd
        JOIN "patient" p
            ON p.id = pd.patient_id
        WHERE pd.storage_key IS NOT NULL
          AND pd.storage_key ~* '\.(jpg|jpeg|png|webp|gif)$'
          AND pd.visit_id IS NOT NULL
        ORDER BY pd.storage_key
    """)

    with ENGINE.connect() as conn:
        return conn.execute(query).mappings().all()


def download_image_bytes(storage_key):
    """Download an image from Supabase Storage straight into memory.

    `storage_key` is the object path inside the bucket
    (patientdocument.storage_key). The legacy AWS `file_path` URL must not be
    passed here. No file is saved locally to disk.
    """
    if not storage_key:
        raise ValueError("download_image_bytes requires a non-empty storage_key")
    if storage_key.startswith(("http://", "https://", "s3://")):
        raise ValueError(
            f"download_image_bytes expects a Supabase storage_key, not a URL: {storage_key!r}. "
            "Use item['storage_key'], not item['file_path']."
        )

    try:
        obj = S3_CLIENT.get_object(Bucket=SUPABASE_BUCKET, Key=storage_key)
        data = obj["Body"].read()
    except Exception as e:
        print(f"[supabase] error downloading {SUPABASE_BUCKET}/{storage_key}: {e}")
        raise

    if not data:
        raise RuntimeError(f"Supabase returned no data for {SUPABASE_BUCKET}/{storage_key}")
    return data


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
        storage_key = row["storage_key"]
        file_path = row["file_path"]  # legacy AWS URL: reference only, never downloaded
        file_name = _file_name(storage_key)

        # Skip if the file name, storage_key, or legacy file_path was already processed
        if (
            file_name in skip_keys
            or storage_key in skip_keys
            or (file_path and file_path in skip_keys)
        ):
            continue

        patient_id = row["patient_id"]
        visit_id = row["visit_id"]

        manifest.append({
            "folder_id": None,
            "folder_name": str(patient_id) if patient_id else "",
            "file_id": storage_key,        # Supabase object path used by download_image_bytes
            "storage_key": storage_key,    # authoritative path in the Supabase bucket
            "file_path": file_path,        # legacy AWS URL, kept for reference only
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