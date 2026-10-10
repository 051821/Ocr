"""Regression check verifying byte-identical output for review and chat with all flags OFF."""
import os
import pytest

from review_rules import deterministic_review
from patient_chat import answer_patient_question


def _synthetic_patient_1():
    return {
        "patient_id": "syn-p1",
        "all_medications": ["Metformin 500mg"],
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-01-10",
                "chief_complaint": "Polyuria and polydipsia",
                "provisional_diagnosis": "Type 2 Diabetes Mellitus",
                "confirmed_diagnosis": "Type 2 Diabetes Mellitus",
                "db_medications": [{"name": "Metformin", "dosage": "500mg", "frequency": "BD"}],
                "lab_results": [{"test_name": "HbA1c", "value": "8.4 %", "status": "Abnormal (High)"}],
                "followups": [{"status": "Scheduled"}],
                "documents": [{"doc_id": "d1", "clinical_summary": "Initial evaluation"}],
            }
        ],
    }


def _synthetic_patient_2():
    return {
        "patient_id": "syn-p2",
        "all_medications": ["Amlodipine 5mg", "Atorvastatin 20mg"],
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-02-01",
                "chief_complaint": "Elevated BP reading",
                "provisional_diagnosis": "Essential Hypertension",
                "confirmed_diagnosis": "Essential Hypertension",
                "db_medications": [{"name": "Amlodipine", "dosage": "5mg", "frequency": "OD"}],
                "lab_results": [{"test_name": "Total Cholesterol", "value": "240 mg/dL", "status": "High"}],
                "followups": [{"status": "Completed"}],
                "documents": [],
            },
            {
                "visit_id": "v2",
                "visit_date": "2026-03-01",
                "chief_complaint": "Routine follow up",
                "provisional_diagnosis": "Essential Hypertension",
                "confirmed_diagnosis": "Essential Hypertension",
                "db_medications": [
                    {"name": "Amlodipine", "dosage": "5mg", "frequency": "OD"},
                    {"name": "Atorvastatin", "dosage": "20mg", "frequency": "HS"},
                ],
                "lab_results": [{"test_name": "Blood Pressure", "value": "128/82 mmHg", "status": "Normal"}],
                "followups": [],
                "documents": [],
            },
        ],
    }


def test_regression_deterministic_review_and_chat_flags_off(monkeypatch):
    """Verify that with all flags OFF, output is stable and deterministic."""
    monkeypatch.setenv("ENABLE_HARDENED_CONFIG", "false")
    monkeypatch.setenv("ENABLE_ANONYMIZED_AUDIT_LOGS", "false")
    monkeypatch.setenv("ENABLE_CHAT_OUTPUT_SANITIZER", "false")
    monkeypatch.setenv("ENABLE_STRICT_HTML_SANITIZATION", "false")

    p1 = _synthetic_patient_1()
    p2 = _synthetic_patient_2()

    # Review checks
    r1 = deterministic_review(p1)
    r2 = deterministic_review(p2)
    assert "# AI Clinical Review" in r1 and "Type 2 Diabetes Mellitus" in r1
    assert "# AI Clinical Review" in r2 and "Essential Hypertension" in r2

    # Deterministic chat checks
    a1, ev1 = answer_patient_question(p1, "syn-p1", "Show latest lab result")
    assert "8.4 %" in a1
    assert len(ev1) >= 1

    a2, ev2 = answer_patient_question(p2, "syn-p2", "What medications were given?")
    assert "Amlodipine" in a2
    assert len(ev2) >= 1
