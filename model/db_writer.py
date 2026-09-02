"""
model/db_writer.py
"""
import csv
import json
import os
import re
import uuid

import psycopg2
import psycopg2.extras
import psycopg2.pool

import config


def ensure_clean_text_column(conn):
    """Checks if the clean_text column exists in the document_extraction table,
    and adds it as JSONB if missing."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name = 'document_extraction' 
                      AND column_name = 'clean_text'
                )
                """
            )
            exists = cur.fetchone()[0]
            if not exists:
                print("[db] Adding missing column 'clean_text' (JSONB) to table 'document_extraction'...")
                cur.execute("ALTER TABLE document_extraction ADD COLUMN clean_text JSONB;")
                conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[db] Warning during schema check/update: {e}")


def split_text_lines(text_lines):
    """Splits raw text lines into PII/administrative/junk lines (extracted_text) and clean medical lines (clean_text)."""
    if not text_lines:
        return [], []

    if isinstance(text_lines, str):
        text_lines = text_lines.splitlines()

    pii_lines = []
    medical_lines = []

    # Standard PII patterns
    phone_pattern = r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b|\b\d{3,5}-\d{6,8}\b'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    url_pattern = r'\b(?:www\.|https?://)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    pincode_pattern = r'\b\d{6}\b|\b\d{3}\s*\d{3}\b'
    date_regex = r'\b\d{1,2}[-/\s.]\d{1,2}[-/\s.]\d{2,4}\b|\b\d{4}[-/\s.]\d{1,2}[-/\s.]\d{1,2}\b'

    honorific_pattern = r'\b(?i:mr|mrs|ms|miss|master|dr|prof|sr)\b\.?\s+(?!(?i:tb|tuberculosis|regimen|mono|poly|status)\b)[A-Z]'

    # NEW: relation-prefix patterns very common in Indian medical reports
    # S/O, D/O, W/O, C/O — with optional dots/spaces (s/o, s.o, s. o, S/O)
    relation_prefix_pattern = r'\b(?i:s|d|w|c)\s*[./]\s*o\b\.?'

    # Keywords that indicate the line is administrative or PII
    pii_keywords = [
        'patient name', 'guardian name', 'father name', 'husband name', 'wife name',
        'daughter name', 'son name', 'patient address', 'patient mobile', 'patient contact',
        'contact number', 'contact no', 'mobile no', 'phone no', 'phone number', 'email id',
        'ref. physician', 'referred by', 'reported by', 'tested by', 'date of collection',
        'date tested', 'date reported', 'date collected', 'hospital reg', 'registration no',
        'episode id', 'nikshay', 'establishment id', 'laboratory name', 'lab name',
        'signature', 'signed by', 'other contact'
    ]

    # Individual standalone keyword indicators
    # NEW: added 'name' and 'guardian' as bare keywords (word-boundary checked below)
    standalone_keywords = [
        'landmark', 'address', 'pincode', 'district', 'state', 'uhid', 'opd', 'ipd',
        'physician', 'referred', 'consultant', 'radiologist', 'name', 'guardian',
        'pathologist', 'clinic', 'hospital', 'laboratory', 'institute', 'college'
    ]

    # NEW: medical vocabulary used to keep the name-shape heuristic from
    # misfiring on short all-caps medical headings like "SPUTUM SMEAR" etc.
    medical_vocab = {
        'sputum', 'smear', 'afb', 'positive', 'negative', 'result', 'test',
        'report', 'sample', 'culture', 'sensitivity', 'resistance', 'tb',
        'mtb', 'rif', 'trace', 'detected', 'not', 'nil', 'normal', 'abnormal',
        'grade', 'scanty', 'reflex', 'panel', 'drug', 'regimen', 'status',
        'remarks', 'observation', 'findings', 'diagnosis', 'treatment',
        'dose', 'dosage', 'weight', 'height', 'age', 'gender', 'sex',
        'sample', 'specimen', 'collection', 'ward', 'bed', 'unit'
    }

    def looks_like_bare_name(s):
        """Fallback heuristic: a short line of 2-4 alphabetic tokens, each
        Title Case or ALL CAPS, none of which is a common medical word.
        Catches names printed with no label and no honorific."""
        tokens = s.split()
        if not (2 <= len(tokens) <= 4):
            return False
        for tok in tokens:
            core = re.sub(r'[^A-Za-z]', '', tok)
            if not core:
                return False
            if core.lower() in medical_vocab:
                return False
            # must be Title Case (Rekha) or ALL CAPS (REKHA), not lowercase/mixed
            if not (core.isupper() or (core[0].isupper() and core[1:].islower())):
                return False
        return True

    # NEW: a lightweight inline name-span pattern used only for redacting
    # embedded names inside otherwise-medical lines (not for classification).
    # Matches 2-3 consecutive Title-Case/ALL-CAPS word tokens.
    inline_name_span = r'\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){1,2}\b'

    for line in text_lines:
        line_str = str(line).strip()
        if not line_str:
            continue

        if line_str in ("```", "'''"):
            continue

        line_str = re.sub(r'\[ILLEGIBLE\]', '', line_str, flags=re.IGNORECASE).strip()
        if not line_str:
            continue

        line_lower = line_str.lower()
        is_pii = False

        # 1. Honorific + name (Mrs. Rekha, Dr. Nayeem)
        if re.search(honorific_pattern, line_str):
            is_pii = True

        # 2. Phone, email, url
        elif re.search(phone_pattern, line_str) or re.search(email_pattern, line_str) or re.search(url_pattern, line_lower):
            is_pii = True

        # 3. Pure digit / ID lines
        elif re.match(r'^\s*\d{5,}\s*$', line_str):
            is_pii = True

        # 4. Multi-word PII prefixes
        elif any(kw in line_lower for kw in pii_keywords):
            is_pii = True

        # 5. Patient demographics — now flags on 'patient' alone too
        elif 'patient' in line_lower:
            is_pii = True

        # 6. NEW: relation prefixes S/O, D/O, W/O, C/O
        elif re.search(relation_prefix_pattern, line_str):
            is_pii = True

        # 7. Standalone keywords (word-boundary matched)
        if not is_pii:
            for kw in standalone_keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', line_lower) or line_lower.startswith(kw):
                    is_pii = True
                    break

        # 8. Date lines
        if not is_pii:
            if line_lower.startswith('date') or re.search(date_regex, line_str):
                is_pii = True

        # 9. Pincode-bearing lines
        if not is_pii:
            if re.search(pincode_pattern, line_str):
                is_pii = True

        # 10. NEW: bare unlabeled name fallback (e.g. "Rekha Sharma" on its own line)
        if not is_pii:
            if looks_like_bare_name(line_str):
                is_pii = True

        if is_pii:
            pii_lines.append(line_str)
        else:
            # Redact inline phone/email as before, PLUS any embedded name-shaped
            # spans, so names inside otherwise-medical sentences don't leak.
            cleaned_line = re.sub(phone_pattern, '', line_str)
            cleaned_line = re.sub(email_pattern, '', cleaned_line)

            def _maybe_redact_name(m):
                span_text = m.group(0)
                tokens = [t for t in span_text.split() if re.sub(r'[^A-Za-z]', '', t).lower() not in medical_vocab]
                # only redact if none of the tokens are recognized medical vocab
                if len(tokens) == len(span_text.split()):
                    return ''
                return span_text

            cleaned_line = re.sub(inline_name_span, _maybe_redact_name, cleaned_line)
            cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()

            if cleaned_line:
                medical_lines.append(cleaned_line)

    return pii_lines, medical_lines

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

    # Check and add clean_text column if not exists
    conn = _pool.getconn()
    try:
        ensure_clean_text_column(conn)
    finally:
        _pool.putconn(conn)


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

        pii_lines, clean_lines = split_text_lines(text_lines)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_extraction
                    (id, healthcase_id, legacy_id, imagename, extracted_text, clean_text, patient_id, visit_id, priority)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (imagename) DO NOTHING
                """,
                (
                    str(uuid.uuid4()),
                    int(healthcase_id),
                    int(legacy_id),
                    imagename,
                    psycopg2.extras.Json(pii_lines) if pii_lines else None,
                    psycopg2.extras.Json(clean_lines) if clean_lines else None,
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