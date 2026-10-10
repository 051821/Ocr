"""SQLite-backed review and LLM cache with a legacy import shim name."""
import csv
import json
import logging
import re
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

try:
    from ..review_facts import PROMPT_VERSION, build_review_facts, compute_source_hash as _source_hash
    from ..review_facts import visit_hashes as _visit_hashes
except ImportError:
    from review_facts import PROMPT_VERSION, build_review_facts, compute_source_hash as _source_hash
    from review_facts import visit_hashes as _visit_hashes


logger = logging.getLogger(__name__)
FULL_RERUN_THRESHOLD = 4
ANALYSIS_TYPE = "retrospective_clinical_analysis"
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "cache.db"
_LEGACY_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "patient_ai_analysis.csv"
_SCHEMA_LOCK = threading.Lock()


def _connect(path=None):
    db_path = Path(path or _DEFAULT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    with _SCHEMA_LOCK:
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
                created_at TEXT NOT NULL
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
    if db_path.resolve() == _DEFAULT_DB_PATH.resolve():
        _migrate_legacy_csv(conn)
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
        return dict(row) if row else None
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
    values = (
        str(patient_data.get("patient_id", "")), source_hash, PROMPT_VERSION,
        str(model_name or model_version or ""), markdown, visit_count,
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
        return dict(conn.execute("SELECT * FROM ai_review WHERE patient_id = ?", (values[0],)).fetchone())
    finally:
        conn.close()


def get_llm_cache(key, path=None):
    conn = _connect(path)
    try:
        row = conn.execute("SELECT value FROM llm_cache WHERE key = ?", (str(key),)).fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def set_llm_cache(key, value, path=None):
    conn = _connect(path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO llm_cache(key, value, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at",
                (str(key), str(value), datetime.now(timezone.utc).isoformat()),
            )
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

