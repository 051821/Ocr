"""Tests for Step 7: ENABLE_ENCRYPTED_CACHE and cache administration."""
import os
import sys
import sqlite3
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

# Block DB imports that hang
for _m in ("database", "core.db"):
    if _m not in sys.modules:
        _mock = MagicMock()
        _mock.db_pool = MagicMock()
        sys.modules[_m] = _mock

from storage.ai_analysis_csv import (
    get_llm_cache,
    set_llm_cache,
    get_patient_row,
    upsert_ai_analysis,
    purge_expired_cache,
    purge_patient,
    _connect,
    _INITIALIZED_DBS,
)

# Deterministic 32-byte Fernet key for testing
TEST_KEY = "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="  # urlsafe base64 of 32 bytes


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_cache.db"
    return str(db_file)


def test_encryption_roundtrip_and_no_raw_plaintext(temp_db, monkeypatch):
    """When ENABLE_ENCRYPTED_CACHE is ON, raw SQLite contains no plaintext; round-trip succeeds."""
    monkeypatch.setenv("ENABLE_ENCRYPTED_CACHE", "true")
    monkeypatch.setenv("CACHE_ENCRYPTION_KEY", TEST_KEY)
    monkeypatch.setenv("AUDIT_HASH_SALT", "test_audit_salt_1234567890123456")

    raw_secret_value = "DIAGNOSIS_CONFIDENTIAL_XYZ_12345"
    key = "test_key_1"
    set_llm_cache(key, raw_secret_value, path=temp_db, patient_id="P999")

    # Read back via API -> round-trip returns original plaintext
    retrieved = get_llm_cache(key, path=temp_db)
    assert retrieved == raw_secret_value

    # Verify directly in SQLite file that raw string is NOT in value column
    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT value, patient_hash FROM llm_cache WHERE key = ?", (key,)).fetchone()
    conn.close()
    
    assert row is not None
    stored_val, p_hash = row
    assert stored_val.startswith("enc:v1:")
    assert raw_secret_value not in stored_val
    assert p_hash is not None and len(p_hash) == 16


def test_wrong_key_fails_gracefully(temp_db, monkeypatch):
    """Wrong key fails decryption gracefully without leaking content or crashing."""
    monkeypatch.setenv("ENABLE_ENCRYPTED_CACHE", "true")
    monkeypatch.setenv("CACHE_ENCRYPTION_KEY", TEST_KEY)

    key = "test_key_wrong_key"
    val = "Secret data"
    set_llm_cache(key, val, path=temp_db)

    # Change to another valid Fernet key (different 32 bytes)
    wrong_key = "MDEyMzQ1Njc4OTAxMjM0NTY3ODkwMTIzNDU2Nzg5MDE="
    monkeypatch.setenv("CACHE_ENCRYPTION_KEY", wrong_key)

    # get_llm_cache should catch the decryption failure and treat as cache miss
    result = get_llm_cache(key, path=temp_db)
    assert result is None


def test_legacy_plaintext_readable_and_flag_off_miss(temp_db, monkeypatch):
    """Plaintext rows remain readable when flag ON; encrypted rows treat as misses when flag OFF."""
    # Insert raw legacy plaintext row
    conn = sqlite3.connect(temp_db)
    conn.execute("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TEXT NOT NULL, patient_hash TEXT)")
    conn.execute("INSERT INTO llm_cache (key, value, created_at) VALUES ('legacy_key', 'plain_legacy_val', ?)", (datetime.now(timezone.utc).isoformat(),))
    conn.commit()
    conn.close()

    # 1. With flag ON, reading plaintext row returns plaintext
    monkeypatch.setenv("ENABLE_ENCRYPTED_CACHE", "true")
    monkeypatch.setenv("CACHE_ENCRYPTION_KEY", TEST_KEY)
    assert get_llm_cache("legacy_key", path=temp_db) == "plain_legacy_val"

    # Encrypt a new key
    set_llm_cache("enc_key", "secret_content", path=temp_db)

    # 2. With flag OFF, reading encrypted row treats as cache miss (returns None, no crash)
    monkeypatch.setenv("ENABLE_ENCRYPTED_CACHE", "false")
    assert get_llm_cache("enc_key", path=temp_db) is None
    # legacy plaintext still returns normally
    assert get_llm_cache("legacy_key", path=temp_db) == "plain_legacy_val"


def test_ttl_expiry(temp_db, monkeypatch):
    """CACHE_TTL_DAYS > 0 expires older rows on read and via purge_expired_cache."""
    monkeypatch.setenv("ENABLE_ENCRYPTED_CACHE", "false")
    monkeypatch.setenv("CACHE_TTL_DAYS", "7")

    now = datetime.now(timezone.utc)
    old_time = (now - timedelta(days=10)).isoformat()
    fresh_time = (now - timedelta(days=2)).isoformat()

    conn = _connect(temp_db)
    conn.execute("INSERT INTO llm_cache (key, value, created_at) VALUES ('old_k', 'old_v', ?)", (old_time,))
    conn.execute("INSERT INTO llm_cache (key, value, created_at) VALUES ('fresh_k', 'fresh_v', ?)", (fresh_time,))
    conn.execute("INSERT INTO chat_faq (patient_id, qkey, question, hits, last_asked) VALUES ('P1', 'q_old', 'old?', 1, ?)", (old_time,))
    conn.execute("INSERT INTO chat_faq (patient_id, qkey, question, hits, last_asked) VALUES ('P1', 'q_new', 'new?', 1, ?)", (fresh_time,))
    conn.commit()
    conn.close()

    # Read old_k -> treated as miss due to TTL > 7 days
    assert get_llm_cache("old_k", path=temp_db) is None
    # Read fresh_k -> hits
    assert get_llm_cache("fresh_k", path=temp_db) == "fresh_v"

    # Purge expired
    deleted = purge_expired_cache(temp_db)
    assert deleted == 2  # 1 from llm_cache, 1 from chat_faq

    conn = _connect(temp_db)
    assert conn.execute("SELECT COUNT(*) FROM llm_cache WHERE key = 'old_k'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM llm_cache WHERE key = 'fresh_k'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chat_faq WHERE qkey = 'q_old'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chat_faq WHERE qkey = 'q_new'").fetchone()[0] == 1
    conn.close()


def test_purge_patient(temp_db, monkeypatch):
    """purge_patient removes all and only that patient's rows across tables."""
    monkeypatch.setenv("AUDIT_HASH_SALT", "salt_test_1234567890123456")
    monkeypatch.setenv("ENABLE_ENCRYPTED_CACHE", "false")

    # Set up records for Patient A and Patient B
    set_llm_cache("cache_A", "val_A", path=temp_db, patient_id="PATIENT_A")
    set_llm_cache("cache_B", "val_B", path=temp_db, patient_id="PATIENT_B")

    conn = _connect(temp_db)
    conn.execute("INSERT INTO ai_review (patient_id, prompt_version, review_md, generated_at) VALUES ('PATIENT_A', 'v1', 'rev_A', '2026-01-01')")
    conn.execute("INSERT INTO ai_review (patient_id, prompt_version, review_md, generated_at) VALUES ('PATIENT_B', 'v1', 'rev_B', '2026-01-01')")
    conn.execute("INSERT INTO chat_faq (patient_id, qkey, question, last_asked) VALUES ('PATIENT_A', 'qa', 'q?', '2026-01-01')")
    conn.execute("INSERT INTO chat_faq (patient_id, qkey, question, last_asked) VALUES ('PATIENT_B', 'qb', 'q?', '2026-01-01')")
    conn.commit()
    conn.close()

    # Purge Patient A
    counts = purge_patient("PATIENT_A", path=temp_db)
    assert counts["ai_review"] == 1
    assert counts["chat_faq"] == 1
    assert counts["llm_cache"] == 1

    # Patient B must still be intact
    conn = _connect(temp_db)
    assert conn.execute("SELECT COUNT(*) FROM ai_review WHERE patient_id = 'PATIENT_B'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM chat_faq WHERE patient_id = 'PATIENT_B'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM llm_cache WHERE key = 'cache_B'").fetchone()[0] == 1
    # Patient A rows are gone
    assert conn.execute("SELECT COUNT(*) FROM ai_review WHERE patient_id = 'PATIENT_A'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chat_faq WHERE patient_id = 'PATIENT_A'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM llm_cache WHERE key = 'cache_A'").fetchone()[0] == 0
    conn.close()


def test_schema_init_runs_once(temp_db):
    """_INITIALIZED_DBS ensures DDL and migration run only once per path."""
    import pathlib
    resolved = pathlib.Path(temp_db).resolve()
    _INITIALIZED_DBS.discard(resolved)
    
    conn1 = _connect(temp_db)
    conn1.close()
    assert resolved in _INITIALIZED_DBS
    
    # Second connection reuses initialization set without re-running DDL
    conn2 = _connect(temp_db)
    conn2.close()
    assert resolved in _INITIALIZED_DBS
