"""
model/db_writer.py

Owns all Postgres access: the connection pool, the result.csv lookup
(person_id + imagename -> healthcase_id), the legacy_id -> patient_id and
healthcase_id -> visit_id resolution (cached in-memory to avoid N+1 queries),
the actual insert into document_extraction, and a small local pending-linkage
file for images whose OCR succeeded but whose patient/visit record doesn't
exist in the new db yet -- so OCR is never silently re-run just to retry a
DB lookup.

Dedup key is imagename ALONE (not legacy_id/healthcase_id, both of which can
legitimately repeat across many images for the same patient/visit). Matching
is exact full-string equality everywhere: the DB's UNIQUE(imagename)
constraint, the ON CONFLICT clause, and the in-Python skip-key set.
"""
import csv
import json
import os
import uuid

import psycopg2
import psycopg2.extras
import psycopg2.pool

import config

_pool = None
_csv_lookup = None          # (person_id, imagename) -> healthcase_id
_patient_cache = {}         # legacy_id -> patient_id (or None if confirmed missing)
_visit_cache = {}           # healthcase_id -> visit_id (or None if confirmed missing)


# ---------------------------------------------------------------------------
# POOL LIFECYCLE
# ---------------------------------------------------------------------------
def init_pool():
    global _pool
    if _pool is not None:
        return
    _pool = psycopg2.pool.ThreadedConnectionPool(
        config.DB_POOL_MIN,
        config.DB_POOL_MAX,
        dsn=config.DATABASE_URL,
    )
    print(f"[db] connection pool ready ({config.DB_POOL_MIN}-{config.DB_POOL_MAX})")


def close_pool():
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        print("[db] connection pool closed")


def _get_conn():
    if _pool is None:
        raise RuntimeError("db_writer.init_pool() must be called before use")
    return _pool.getconn()


def _put_conn(conn):
    _pool.putconn(conn)


# ---------------------------------------------------------------------------
# result.csv LOOKUP (person_id + imagename -> healthcase_id)
# ---------------------------------------------------------------------------
def load_result_csv():
    global _csv_lookup
    if _csv_lookup is not None:
        return _csv_lookup

    lookup = {}
    with open(config.RESULT_CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reportimgurl = row["reportimgurl"].strip()
            imagename = reportimgurl.split("/")[-1] if reportimgurl else ""
            key = (row["person_id"].strip(), imagename)
            lookup[key] = int(row["healthcase_id"].strip())

    _csv_lookup = lookup
    print(f"[db] loaded {len(lookup)} row(s) from {config.RESULT_CSV_PATH}")
    return lookup


def get_healthcase_id(legacy_id, imagename):
    """legacy_id (drive folder name) == person_id in result.csv."""
    lookup = load_result_csv()
    return lookup.get((str(legacy_id), imagename))


# ---------------------------------------------------------------------------
# NEW-DB LOOKUPS (patient / visit) -- cached to avoid one query per image
# ---------------------------------------------------------------------------
def resolve_patient_id(conn, legacy_id):
    if legacy_id in _patient_cache:
        return _patient_cache[legacy_id]
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM patient WHERE legacy_id = %s", (str(legacy_id),))
        row = cur.fetchone()
    patient_id = row[0] if row else None
    _patient_cache[legacy_id] = patient_id
    return patient_id


def resolve_visit_id(conn, healthcase_id):
    if healthcase_id in _visit_cache:
        return _visit_cache[healthcase_id]
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM visit WHERE legacy_id = %s", (str(healthcase_id),))
        row = cur.fetchone()
    visit_id = row[0] if row else None
    _visit_cache[healthcase_id] = visit_id
    return visit_id


# ---------------------------------------------------------------------------
# SKIP-KEY QUERY (replaces reading output.json/result.json)
# dedup key = imagename ONLY, exact full-string match
# ---------------------------------------------------------------------------
def load_skip_keys():
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT imagename FROM document_extraction")
            db_keys = {row[0] for row in cur.fetchall()}
    finally:
        _put_conn(conn)

    # images already OCR'd but stuck pending DB linkage are ALSO "done" from
    # the fetch stage's point of view -- must not be re-downloaded/re-OCR'd
    pending_keys = {p["imagename"] for p in _load_pending()}
    return db_keys | pending_keys


# ---------------------------------------------------------------------------
# PENDING-LINKAGE FILE (crash-safety net, not a general JSON output)
# ---------------------------------------------------------------------------
def _load_pending():
    if not os.path.exists(config.PENDING_LINKAGE_JSON):
        return []
    with open(config.PENDING_LINKAGE_JSON, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_pending(pending):
    with open(config.PENDING_LINKAGE_JSON, "w", encoding="utf-8") as f:
        json.dump(pending, f, indent=2, ensure_ascii=False)


def _add_pending(legacy_id, imagename, text_lines, healthcase_id):
    pending = _load_pending()
    pending.append({
        "legacy_id": legacy_id,
        "imagename": imagename,
        "text": text_lines,
        "healthcase_id": healthcase_id,
    })
    _save_pending(pending)


def retry_pending_linkage():
    """Call at the start of every run, before touching Drive at all. Retries
    the DB insert for every pending image -- if it now succeeds (patient/visit
    exists), it's removed from the pending file and never touches OCR again."""
    pending = _load_pending()
    if not pending:
        return

    print(f"[db] retrying {len(pending)} pending-linkage image(s)...")
    still_pending = []
    for p in pending:
        ok = _insert_row(p["legacy_id"], p["imagename"], p["text"], p["healthcase_id"])
        if not ok:
            still_pending.append(p)

    resolved = len(pending) - len(still_pending)
    if resolved:
        print(f"[db] resolved {resolved} previously-pending image(s).")
    _save_pending(still_pending)


# ---------------------------------------------------------------------------
# INSERT
# ---------------------------------------------------------------------------
def _insert_row(legacy_id, imagename, text_lines, healthcase_id):
    """Returns True if the row was inserted (or already existed, matched by
    imagename), False if skipped due to unresolved patient/visit. Raises on
    genuine DB errors so the caller can decide whether to isolate them."""
    conn = _get_conn()
    try:
        patient_id = resolve_patient_id(conn, legacy_id)
        if patient_id is None:
            print(f"[db] PENDING {legacy_id}/{imagename}: no patient found for legacy_id={legacy_id}")
            return False

        visit_id = resolve_visit_id(conn, healthcase_id)
        if visit_id is None:
            print(f"[db] PENDING {legacy_id}/{imagename}: no visit found for healthcase_id={healthcase_id}")
            return False

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_extraction
                    (id, healthcase_id, legacy_id, imagename, extracted_text, patient_id, visit_id, priority)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (imagename) DO NOTHING
                """,
                (
                    str(uuid.uuid4()),
                    int(healthcase_id),
                    int(legacy_id),
                    imagename,
                    psycopg2.extras.Json(text_lines) if text_lines else None,
                    patient_id,
                    visit_id,
                    config.DEFAULT_PRIORITY,
                ),
            )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)


def insert_extracted_document(legacy_id, imagename, text_lines):
    """Public entry point used by paddel.py / handwritten.py. Never raises --
    a genuine DB error for one image is logged and isolated so it doesn't
    kill the whole run; the image just stays eligible for retry next time."""
    key = f"{legacy_id}/{imagename}"

    healthcase_id = get_healthcase_id(legacy_id, imagename)
    if healthcase_id is None:
        print(f"[db] SKIP {key}: no healthcase_id match in result.csv")
        return False

    try:
        ok = _insert_row(legacy_id, imagename, text_lines, healthcase_id)
    except Exception as e:
        print(f"[db] ERROR inserting {key}: {e}")
        return False

    if not ok:
        _add_pending(legacy_id, imagename, text_lines, healthcase_id)
    return ok