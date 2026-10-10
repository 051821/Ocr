"""SQLite-backed review and LLM cache with a legacy import shim name."""
import csv
import json
import logging
import re
import sqlite3
import sys
import threading
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from ..review_facts import PROMPT_VERSION, build_review_facts, compute_source_hash as _source_hash
    from ..review_facts import visit_hashes as _visit_hashes
    from ..core.config import flag
except ImportError:
    from review_facts import PROMPT_VERSION, build_review_facts, compute_source_hash as _source_hash
    from review_facts import visit_hashes as _visit_hashes
    from core.config import flag


logger = logging.getLogger(__name__)
FULL_RERUN_THRESHOLD = 4
ANALYSIS_TYPE = "retrospective_clinical_analysis"
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "cache.db"
_LEGACY_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "patient_ai_analysis.csv"
_SCHEMA_LOCK = threading.Lock()
_INITIALIZED_DBS = set()

_ENC_PREFIX = "enc:v1:"


def _get_fernet():
    key = os.getenv("CACHE_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("Cache encryption key missing: CACHE_ENCRYPTION_KEY environment variable is required when ENABLE_ENCRYPTED_CACHE is active.")
    try:
        from cryptography.fernet import Fernet
        return Fernet(key.encode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid CACHE_ENCRYPTION_KEY: failed to initialize encryption cipher ({type(exc).__name__}).") from None


def _encrypt_val(plaintext: str) -> str:
    if not flag("ENABLE_ENCRYPTED_CACHE"):
        return plaintext
    fernet = _get_fernet()
    token = fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")
    return f"{_ENC_PREFIX}{token}"


def _decrypt_val(stored_val: str) -> str | None:
    if not stored_val or not stored_val.startswith(_ENC_PREFIX):
        return stored_val
    if not flag("ENABLE_ENCRYPTED_CACHE"):
        logger.warning("Encrypted cache entry encountered while ENABLE_ENCRYPTED_CACHE is disabled; treating as miss.")
        return None
    token = stored_val[len(_ENC_PREFIX):]
    fernet = _get_fernet()
    try:
        return fernet.decrypt(token.encode("utf-8")).decode("utf-8")
    except Exception as exc:
        logger.warning("Cache entry decryption failed; treating as miss (%s).", type(exc).__name__)
        return None


def _calc_patient_hash(patient_id: str | None) -> str | None:
    if not patient_id:
        return None
    salt = os.getenv("AUDIT_HASH_SALT", "").encode("utf-8")
    return hmac.new(salt, str(patient_id).strip().encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def _connect(path=None):
    db_path = Path(path or _DEFAULT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    resolved = db_path.resolve()
    with _SCHEMA_LOCK:
        if resolved not in _INITIALIZED_DBS:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ai_review (
                    patient_id TEXT PRIMARY KEY,
                    source_hash TEXT,
                    prompt_version TEXT NOT NULL,
                    model TEXT,
                    review_md TEXT NOT NULL,
                    visit_count INTEGER NOT NULL DEFAULT 0,
                    visit_hashes TEXT NOT NULL DEFAULT '{}',
                    mismatch_count INTEGER NOT NULL DEFAULT 0,
                    generated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    patient_hash TEXT
                );
                CREATE TABLE IF NOT EXISTS chat_faq (
                    patient_id TEXT NOT NULL,
                    qkey TEXT NOT NULL,
                    question TEXT NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 1,
                    last_asked TEXT NOT NULL,
                    PRIMARY KEY (patient_id, qkey)
                );
                """
            )
            # Guarded column addition for existing databases
            try:
                conn.execute("ALTER TABLE llm_cache ADD COLUMN patient_hash TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists

            if resolved == _DEFAULT_DB_PATH.resolve():
                _migrate_legacy_csv(conn)

            # Best-effort secure file permissions (ignored on Windows failure)
            try:
                os.chmod(db_path, 0o600)
            except Exception:
                pass

            _INITIALIZED_DBS.add(resolved)
    return conn


def _review_markdown(raw):
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return str(parsed.get("raw_markdown", ""))
        return str(parsed)
    except (TypeError, ValueError):
        return str(raw or "")


def _migrate_legacy_csv(conn):
    source = _LEGACY_CSV_PATH
    backup = source.with_suffix(source.suffix + ".bak")
    if not source.exists() or backup.exists():
        return
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        # Windows' C long is narrower than Python's sys.maxsize; this still
        # supports fields up to the maximum representable C long.
        csv.field_size_limit(2**31 - 1)
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    with conn:
        for row in rows:
            patient_id = str(row.get("patient_id") or "").strip()
            if not patient_id:
                continue
            conn.execute(
                """INSERT OR IGNORE INTO ai_review
                (patient_id, source_hash, prompt_version, model, review_md,
                 visit_count, visit_hashes, mismatch_count, generated_at)
                VALUES (?, NULL, ?, ?, ?, ?, '{}', ?, ?)""",
                (
                    patient_id,
                    row.get("prompt_version") or PROMPT_VERSION,
                    row.get("model_name") or "",
                    _review_markdown(row.get("analysis")),
                    int(row.get("source_visit_count") or 0),
                    int(row.get("hash_mismatch_count") or 0),
                    row.get("generated_at") or datetime.now(timezone.utc).isoformat(),
                ),
            )
    source.rename(backup)
    logger.info("Migrated %s legacy AI review rows into SQLite", len(rows))


def compute_source_hash(patient_data, facts=None, hashes=None):
    facts = facts if facts is not None else build_review_facts(patient_data)
    hashes = hashes if hashes is not None else _visit_hashes(patient_data, facts)
    return _source_hash(patient_data.get("patient_id", ""), hashes)


def get_patient_row(patient_id, path=None):
    conn = _connect(path)
    try:
        row = conn.execute("SELECT * FROM ai_review WHERE patient_id = ?", (str(patient_id),)).fetchone()
        if not row:
            return None
        d = dict(row)
        if "review_md" in d:
            decrypted = _decrypt_val(d["review_md"])
            if decrypted is None and d["review_md"].startswith(_ENC_PREFIX):
                return None
            d["review_md"] = decrypted
        return d
    finally:
        conn.close()


def upsert_ai_analysis(
    patient_data,
    analysis,
    model_name="",
    model_version="",
    mismatch_count=0,
    path=None,
    source_hash=None,
    hashes=None,
):
    source_hash = source_hash or patient_data.get("_review_source_hash")
    if not source_hash:
        raise ValueError("Pass the precomputed source_hash when saving a review")
    hashes = hashes or patient_data.get("_review_visit_hashes") or {}
    markdown = _review_markdown(analysis) if isinstance(analysis, str) else str((analysis or {}).get("raw_markdown", ""))
    visit_count = len(patient_data.get("visits", []) or [])
    generated_at = datetime.now(timezone.utc).isoformat()
    stored_markdown = _encrypt_val(markdown)
    values = (
        str(patient_data.get("patient_id", "")), source_hash, PROMPT_VERSION,
        str(model_name or model_version or ""), stored_markdown, visit_count,
        json.dumps(hashes, sort_keys=True, separators=(",", ":")),
        int(mismatch_count), generated_at,
    )
    conn = _connect(path)
    try:
        with conn:
            conn.execute(
                """INSERT INTO ai_review
                (patient_id, source_hash, prompt_version, model, review_md,
                 visit_count, visit_hashes, mismatch_count, generated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    source_hash=excluded.source_hash,
                    prompt_version=excluded.prompt_version,
                    model=excluded.model,
                    review_md=excluded.review_md,
                    visit_count=excluded.visit_count,
                    visit_hashes=excluded.visit_hashes,
                    mismatch_count=excluded.mismatch_count,
                    generated_at=excluded.generated_at""",
                values,
            )
        row = conn.execute("SELECT * FROM ai_review WHERE patient_id = ?", (values[0],)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["review_md"] = markdown  # Return in-memory plaintext to caller
        return d
    finally:
        conn.close()


def get_llm_cache(key, path=None):
    conn = _connect(path)
    try:
        row = conn.execute("SELECT value, created_at FROM llm_cache WHERE key = ?", (str(key),)).fetchone()
        if not row:
            return None
        
        # Check TTL expiry if configured (> 0)
        try:
            ttl_days = int(os.getenv("CACHE_TTL_DAYS", "0").strip())
        except ValueError:
            ttl_days = 0
            
        if ttl_days > 0 and row["created_at"]:
            try:
                created_dt = datetime.fromisoformat(row["created_at"])
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - created_dt > timedelta(days=ttl_days):
                    return None
            except Exception as e:
                logger.warning("Failed to evaluate cache row timestamp: %s", e)

        return _decrypt_val(row["value"])
    finally:
        conn.close()


def set_llm_cache(key, value, path=None, patient_id=None):
    stored_val = _encrypt_val(str(value))
    p_hash = _calc_patient_hash(patient_id)
    conn = _connect(path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO llm_cache(key, value, created_at, patient_hash) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at, patient_hash=excluded.patient_hash",
                (str(key), stored_val, datetime.now(timezone.utc).isoformat(), p_hash),
            )
    finally:
        conn.close()


def purge_expired_cache(path=None) -> int:
    """Delete expired llm_cache and chat_faq entries based on CACHE_TTL_DAYS."""
    try:
        ttl_days = int(os.getenv("CACHE_TTL_DAYS", "0").strip())
    except ValueError:
        ttl_days = 0
    if ttl_days <= 0:
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).isoformat()
    conn = _connect(path)
    deleted = 0
    try:
        with conn:
            cur1 = conn.execute("DELETE FROM llm_cache WHERE created_at < ?", (cutoff,))
            cur2 = conn.execute("DELETE FROM chat_faq WHERE last_asked < ?", (cutoff,))
            deleted = cur1.rowcount + cur2.rowcount
        return deleted
    finally:
        conn.close()


def purge_patient(patient_id: str, path=None) -> dict[str, int]:
    """Delete all records for a patient across ai_review, chat_faq, and llm_cache."""
    if not patient_id:
        return {"ai_review": 0, "chat_faq": 0, "llm_cache": 0}
    pid = str(patient_id).strip()
    p_hash = _calc_patient_hash(pid)
    conn = _connect(path)
    try:
        with conn:
            c1 = conn.execute("DELETE FROM ai_review WHERE patient_id = ?", (pid,))
            c2 = conn.execute("DELETE FROM chat_faq WHERE patient_id = ?", (pid,))
            c3_count = 0
            if p_hash:
                c3 = conn.execute("DELETE FROM llm_cache WHERE patient_hash = ?", (p_hash,))
                c3_count = c3.rowcount
        return {"ai_review": c1.rowcount, "chat_faq": c2.rowcount, "llm_cache": c3_count}
    finally:
        conn.close()


def record_chat_question(patient_id, question, path=None):
    if not patient_id or not question:
        return
    qkey = re.sub(r"[^\w\s]|_", "", str(question).casefold())
    qkey = " ".join(qkey.split())
    if not qkey or len(qkey) > 150:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = _connect(path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO chat_faq(patient_id, qkey, question, hits, last_asked) "
                "VALUES (?, ?, ?, 1, ?) "
                "ON CONFLICT(patient_id, qkey) DO UPDATE SET "
                "hits = hits + 1, question = excluded.question, last_asked = excluded.last_asked",
                (str(patient_id), qkey, str(question).strip(), now_iso),
            )
            conn.execute(
                "DELETE FROM chat_faq WHERE patient_id = ? AND qkey NOT IN ("
                "SELECT qkey FROM chat_faq WHERE patient_id = ? ORDER BY hits DESC, last_asked DESC LIMIT 50)",
                (str(patient_id), str(patient_id)),
            )
    finally:
        conn.close()


def get_top_questions(patient_id, limit=5, min_hits=2, path=None):
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT question FROM chat_faq WHERE patient_id = ? AND hits >= ? ORDER BY hits DESC, last_asked DESC LIMIT ?",
            (str(patient_id), int(min_hits), int(limit)),
        ).fetchall()
        return [row["question"] for row in rows]
    finally:
        conn.close()

