import json
import logging
import os
import sys
from unittest.mock import MagicMock
import pytest

if "database" not in sys.modules:
    mock_db = MagicMock()
    mock_db.db_pool = MagicMock()
    sys.modules["database"] = mock_db

from patient_chat import answer_patient_question


class MockHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def capture_audit():
    audit_logger = logging.getLogger("patient_chat.audit")
    handler = MockHandler()
    audit_logger.addHandler(handler)
    yield handler
    audit_logger.removeHandler(handler)


def test_audit_logs_unmodified_when_flag_off(capture_audit, monkeypatch):
    """When ENABLE_ANONYMIZED_AUDIT_LOGS is False, raw patient_id and question are logged."""
    monkeypatch.setenv("ENABLE_ANONYMIZED_AUDIT_LOGS", "false")
    patient_data = {
        "patient_id": "synthetic-patient-123",
        "visits": [
            {
                "visit_id": "1",
                "visit_date": "2026-01-01",
                "chief_complaint": "Cough",
                "prescriptions": [],
                "lab_results": [],
            }
        ]
    }
    answer, _ = answer_patient_question(
        patient_data=patient_data,
        patient_id="synthetic-patient-123",
        question="What were the lab results?",
        api_key="mock",
    )
    assert len(capture_audit.records) >= 1
    last_record = json.loads(capture_audit.records[-1].getMessage())
    assert last_record["patient_id"] == "synthetic-patient-123"
    assert "What were the lab results?" in str(last_record["question"])


def test_audit_logs_anonymized_when_flag_on(capture_audit, monkeypatch):
    """When ENABLE_ANONYMIZED_AUDIT_LOGS is True, patient_id is hmac-hashed and question is null."""
    monkeypatch.setenv("ENABLE_ANONYMIZED_AUDIT_LOGS", "true")
    monkeypatch.setenv("AUDIT_HASH_SALT", "my_super_secret_audit_salt")

    patient_data = {
        "patient_id": "synthetic-patient-999",
        "visits": [
            {
                "visit_id": "1",
                "visit_date": "2026-01-01",
                "chief_complaint": "Headache",
                "prescriptions": [],
                "lab_results": [],
            }
        ]
    }
    answer, _ = answer_patient_question(
        patient_data=patient_data,
        patient_id="synthetic-patient-999",
        question="What medications were prescribed?",
        api_key="mock",
    )
    assert len(capture_audit.records) >= 1
    last_record = json.loads(capture_audit.records[-1].getMessage())
    # patient_id must be 16-hex hash
    assert last_record["patient_id"] != "synthetic-patient-999"
    assert len(last_record["patient_id"]) == 16
    assert all(c in "0123456789abcdef" for c in last_record["patient_id"])

    # question must be null, but question_length and token_count recorded
    assert last_record["question"] is None
    assert last_record["question_length"] == len("What medications were prescribed?")
    assert last_record["token_count"] == 4


def test_audit_logs_missing_salt_logs_unknown(capture_audit, monkeypatch):
    """When ENABLE_ANONYMIZED_AUDIT_LOGS is True but salt is missing, log 'unknown'."""
    monkeypatch.setenv("ENABLE_ANONYMIZED_AUDIT_LOGS", "true")
    monkeypatch.delenv("AUDIT_HASH_SALT", raising=False)

    patient_data = {
        "patient_id": "synthetic-patient-no-salt",
        "visits": [
            {
                "visit_id": "1",
                "visit_date": "2026-01-01",
                "chief_complaint": "Fever",
                "prescriptions": [],
                "lab_results": [],
            }
        ]
    }
    answer, _ = answer_patient_question(
        patient_data=patient_data,
        patient_id="synthetic-patient-no-salt",
        question="Any fever?",
        api_key="mock",
    )
    assert len(capture_audit.records) >= 1
    last_record = json.loads(capture_audit.records[-1].getMessage())
    assert last_record["patient_id"] == "unknown"
    assert last_record["question"] is None
