"""Tests for Step 3: ENABLE_CHAT_OUTPUT_SANITIZER and emergency warning."""
import pytest
from review_rules import sanitize_chat_answer
from patient_chat import answer_patient_question


def test_sanitize_chat_answer_removes_prescribing_imperatives():
    """Verify imperative prescribing instructions are stripped while keeping structure."""
    raw = (
        "### Answer\n"
        "The patient is taking Metformin 500 mg. You should increase the dose to 20 mg daily. "
        "Take with food. Discontinue the medicine if rash occurs.\n\n"
        "### Evidence\n"
        "- Visit 1: Metformin prescribed."
    )
    cleaned = sanitize_chat_answer(raw)
    assert "increase the dose to 20 mg" not in cleaned
    assert "Take with food" in cleaned
    assert "### Answer" in cleaned
    assert "### Evidence" in cleaned


def test_emergency_notice_prepended_when_sanitizer_on(monkeypatch):
    """When question indicates an emergency, prepend the emergency notice."""
    monkeypatch.setenv("ENABLE_CHAT_OUTPUT_SANITIZER", "true")
    # Mock LLM to return standard answer
    monkeypatch.setattr(
        "patient_chat.llm_complete",
        lambda *args, **kwargs: {
            "content": "### Answer\nDocumented vitals show stable pulse.\n\n### Evidence\n- Encounter 1",
            "model": "test-model",
            "usage": {},
        },
    )

    patient_data = {
        "patient_id": "p-emergency",
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-01-01",
                "chief_complaint": "Chest pain",
                "prescriptions": [],
                "lab_results": [],
            }
        ],
    }

    answer, _ = answer_patient_question(
        patient_data=patient_data,
        patient_id="p-emergency",
        question="Patient is having severe chest pain and difficulty breathing, what should I do?",
        api_key="mock",
    )
    assert "If this is an emergency, seek urgent medical care now." in answer


def test_emergency_notice_not_prepended_when_sanitizer_off(monkeypatch):
    """When ENABLE_CHAT_OUTPUT_SANITIZER is false, emergency notice is not prepended."""
    monkeypatch.setenv("ENABLE_CHAT_OUTPUT_SANITIZER", "false")
    monkeypatch.setattr(
        "patient_chat.llm_complete",
        lambda *args, **kwargs: {
            "content": "### Answer\nDocumented vitals show stable pulse.\n\n### Evidence\n- Encounter 1",
            "model": "test-model",
            "usage": {},
        },
    )

    patient_data = {
        "patient_id": "p-emergency-off",
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-01-01",
                "chief_complaint": "Chest pain",
                "prescriptions": [],
                "lab_results": [],
            }
        ],
    }

    answer, _ = answer_patient_question(
        patient_data=patient_data,
        patient_id="p-emergency-off",
        question="Patient is having severe chest pain, what should I do?",
        api_key="mock",
    )
    assert "If this is an emergency, seek urgent medical care now." not in answer
