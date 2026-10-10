"""Smoke tests with all hardening flags turned ON (Item 1f).

Ensures:
1) Config imports and allowlist contains no ':free' model.
2) Fake llm_complete is used (zero network calls), chat path returns an answer.
3) Audit log output contains no raw patient ID and no question text.
"""
import os
import sys
import json
import logging
from unittest.mock import MagicMock, patch
import pytest

# Block DB imports that hang
for _m in ("database", "core.db"):
    if _m not in sys.modules:
        _mock = MagicMock()
        _mock.db_pool = MagicMock()
        sys.modules[_m] = _mock


class MockHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def flags_on_env(monkeypatch):
    # Set all hardening flags ON with fake secrets/configs
    monkeypatch.setenv("ENABLE_HARDENED_CONFIG", "true")
    monkeypatch.setenv("ENABLE_ANONYMIZED_AUDIT_LOGS", "true")
    monkeypatch.setenv("ENABLE_CHAT_OUTPUT_SANITIZER", "true")
    monkeypatch.setenv("ENABLE_STRICT_HTML_SANITIZATION", "true")
    monkeypatch.setenv("ENABLE_EXTERNAL_PROMPTS", "true")
    monkeypatch.setenv("ENABLE_PROMPT_INJECTION_SANDBOX", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-fake-test-key-12345")
    monkeypatch.setenv("AUDIT_HASH_SALT", "test_audit_hash_salt_1234567890123456")
    
    audit_logger = logging.getLogger("patient_chat.audit")
    handler = MockHandler()
    audit_logger.addHandler(handler)
    yield handler
    audit_logger.removeHandler(handler)


def test_hardening_flags_on_smoke(flags_on_env, monkeypatch):
    handler = flags_on_env
    
    # 1. Config check
    import importlib
    import core.config as cfg
    importlib.reload(cfg)
    assert not any(m.endswith(":free") for m in cfg.APPROVED_MODELS)
    assert not any(m.endswith(":free") for m in cfg.DEFAULT_MODELS)
    assert cfg.get_openrouter_api_key() == "sk-or-v1-fake-test-key-12345"

    # 2. Chat execution with synthetic data and mock LLM (ZERO network calls)
    from patient_chat import answer_patient_question

    fake_response = {
        "content": "### Answer\nPatient has documented normal blood pressure.\n\n### Evidence\n- Encounter #1 · 2026-01-01 · Visit 1 (Evidence)",
        "model": "google/gemini-2.5-flash",
        "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75}
    }

    patient_data = {
        "patient_id": "SYNTH_PATIENT_XYZ99",
        "visits": [{
            "visit_id": "1",
            "visit_date": "2026-01-01",
            "clean_text": "Blood pressure 120/80 mmHg."
        }]
    }
    raw_question = "What is the patient's blood pressure record?"

    with patch("patient_chat.llm_complete", return_value=fake_response) as mock_llm, \
         patch("patient_chat.get_llm_cache", return_value=None), \
         patch("patient_chat.set_llm_cache"):
        
        answer, evidence = answer_patient_question(
            patient_data=patient_data,
            patient_id="SYNTH_PATIENT_XYZ99",
            question=raw_question,
            api_key="sk-or-v1-fake-test-key-12345"
        )
        assert mock_llm.called
        assert "### Answer" in answer
        assert "normal blood pressure" in answer

    # 3. Audit log verification
    assert len(handler.records) >= 1
    for rec in handler.records:
        msg = rec.getMessage()
        # Must NOT contain raw patient ID or question text
        assert "SYNTH_PATIENT_XYZ99" not in msg
        assert raw_question not in msg
        
        # Check anonymized audit structure
        entry = json.loads(msg)
        assert entry.get("question") is None
        assert entry.get("patient_id") != "SYNTH_PATIENT_XYZ99"
        assert entry.get("question_length") == len(raw_question)
