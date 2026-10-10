import logging
from patient_chat import answer_patient_question, build_patient_index, route_question


def _patient():
    return {
        "patient_id": "patient-1",
        "all_medications": ["Metformin"],
        "visits": [
            {
                "visit_id": "v1", "visit_date": "2026-01-01", "chief_complaint": "Fatigue",
                "provisional_diagnosis": "Diabetes", "db_medications": [{"name": "Metformin"}],
                "lab_results": [{"test_name": "Glucose", "value": "162.99 mg/dL", "status": "Abnormal (High)"}],
                "documents": [{"doc_id": "d1", "extracted_text": ["Patient reports fatigue and increased thirst."]}],
            }
        ],
    }


def test_router_uses_whole_words_for_substring_cases():
    patient = _patient()
    assert route_question("What is the latest result?", patient) == "patient_record"
    assert route_question("Tell me a funny story", patient) == "out_of_scope"


def test_out_of_scope_and_simple_answers_do_not_call_llm(monkeypatch):
    patient = _patient()
    monkeypatch.setattr("patient_chat.llm_complete", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM called")))
    answer, _ = answer_patient_question(patient, "patient-1", "Tell me a funny story")
    assert "No record lookup" in answer
    answer, evidence = answer_patient_question(patient, "patient-1", "Show the latest lab result")
    assert "162.99 mg/dL" in answer
    assert evidence


def test_patient_isolation_and_local_index():
    patient = _patient()
    index = build_patient_index(patient)
    answer, evidence = answer_patient_question(patient, "other-patient", "Show labs", retrieval_index=index)
    assert "unavailable" in answer
    assert evidence == []


def test_stopped_medicine_question_does_not_return_current_medicines():
    answer, _ = answer_patient_question(
        _patient(), "patient-1", "which medicines were stopped after 2 visits?"
    )
    assert "explicitly documented as stopped" in answer


def test_index_covers_structured_patient_record_data():
    categories = {item["category"] for item in build_patient_index(_patient())}
    assert {"medications", "laboratory result"}.issubset(categories)


def test_retrieves_vitals_entry_for_visit_query(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda k, v: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    monkeypatch.setattr("patient_chat.llm_complete", lambda *args, **kwargs: {"content": "BP response"})
    patient = {
        "patient_id": "patient-1",
        "visits": [
            {"visit_id": "v1", "visit_date": "2026-01-01"},
            {
                "visit_id": "v2",
                "visit_date": "2026-02-01",
                "db_vitals_rows": [{"label": "BP", "value": "120/80 mmHg"}],
            },
        ],
    }
    answer, evidence = answer_patient_question(patient, "patient-1", "what was the BP in visit 2")
    assert any(item.get("category") == "vitals" for item in evidence)


def test_lab_interpretation_fallback_does_not_contain_hardcoded_values(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda k, v: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    monkeypatch.setattr("patient_chat.llm_complete", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("LLM unavailable")))
    patient = {
        "patient_id": "patient-1",
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-01-01",
                "lab_results": [{"test_name": "Hemoglobin", "value": "14.2 g/dL", "status": "Normal"}],
            }
        ],
    }
    answer, _ = answer_patient_question(patient, "patient-1", "What do the lab results indicate?")
    assert "7.6" not in answer
    assert "162.99" not in answer


def test_record_chat_question_and_top_questions_ordering_min_hits(tmp_path, monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda k, v: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    from storage.cache_db import get_top_questions, record_chat_question
    db_file = tmp_path / "cache.db"
    patient_id = "patient-test"
    for _ in range(3):
        record_chat_question(patient_id, "What are my vitals?", path=db_file)
    for _ in range(2):
        record_chat_question(patient_id, "What medications am I taking?", path=db_file)
    record_chat_question(patient_id, "When is my next visit?", path=db_file)

    top = get_top_questions(patient_id, limit=5, min_hits=2, path=db_file)
    assert top == ["What are my vitals?", "What medications am I taking?"]


def test_latest_vitals_deterministic_fast_path(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda k, v: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    monkeypatch.setattr("patient_chat.llm_complete", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))
    patient = {
        "patient_id": "patient-1",
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-01-01",
                "db_vitals_rows": [{"label": "BP", "value": "118/76 mmHg"}],
            }
        ],
    }
    meta = {}
    answer, evidence = answer_patient_question(patient, "patient-1", "what is the latest BP", meta=meta)
    assert "118/76 mmHg" in answer
    assert meta["path"] == "deterministic"
    assert meta["rule"] == "vitals_latest"


def test_llm_success_records_token_usage_in_meta(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda k, v: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    monkeypatch.setattr(
        "patient_chat.llm_complete",
        lambda *args, **kwargs: {
            "content": "### Answer\nNormal vitals",
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )
    patient = {
        "patient_id": "patient-1",
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-01-01",
                "db_vitals_rows": [{"label": "BP", "value": "120/80 mmHg"}],
            }
        ],
    }
    meta = {}
    answer, evidence = answer_patient_question(patient, "patient-1", "what are vitals over time", meta=meta)
    assert meta["total_tokens"] == 15
    assert meta["path"] == "llm"


def test_llm_failure_sets_fallback_path_and_not_cached(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    cached_writes = []
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda k, v: cached_writes.append((k, v)))
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    monkeypatch.setattr("patient_chat.llm_complete", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("LLM failed")))
    patient = {
        "patient_id": "patient-1",
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-01-01",
                "db_vitals_rows": [{"label": "BP", "value": "120/80 mmHg"}],
            }
        ],
    }
    meta = {}
    answer, evidence = answer_patient_question(patient, "patient-1", "what are vitals over time", meta=meta)
    assert meta["path"] == "fallback"
    assert cached_writes == []

def test_what_happen_in_visit1_deterministic_summary(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda k, v: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    monkeypatch.setattr("patient_chat.llm_complete", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))
    patient = {
        "patient_id": "patient-1",
        "all_medications": ["Metformin"],
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-01-01",
                "chief_complaint": "Persistent fatigue",
                "db_medications": [{"name": "Metformin", "dosage": "500mg"}],
                "lab_results": [{"test_name": "CBC", "value": "Normal", "status": "Normal"}],
            }
        ],
    }
    meta = {}
    answer, evidence = answer_patient_question(patient, "patient-1", "what happen in visit1", meta=meta)
    assert meta["path"] == "deterministic"
    assert meta["rule"] == "visit_summary"
    assert "Persistent fatigue" in answer
    assert "Metformin" in answer
    assert "Abnormal (High)" not in answer


def test_summarize_patient_medical_history_rule(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda k, v: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    monkeypatch.setattr("patient_chat.llm_complete", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM should not be called")))
    patient = {
        "patient_id": "patient-1",
        "all_medications": ["Metformin"],
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-01-01",
                "chief_complaint": "Fatigue",
                "db_medications": [{"name": "Metformin", "dosage": "500mg"}],
            }
        ],
    }
    meta = {}
    answer, evidence = answer_patient_question(patient, "patient-1", "summarize patient medical history", meta=meta)
    assert meta["rule"] == "history_summary"


def test_why_is_calcium_low_user_message_has_no_medication_shift(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda k, v: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    called_messages = []
    def fake_llm(messages, *args, **kwargs):
        called_messages.extend(messages)
        return {"content": "### Answer\nLow calcium clinical advice", "usage": {}}
    monkeypatch.setattr("patient_chat.llm_complete", fake_llm)

    patient = {
        "patient_id": "patient-1",
        "all_medications": ["Calcium"],
        "visits": [
            {
                "visit_id": "v1",
                "visit_date": "2026-01-01",
                "chief_complaint": "Fatigue",
                "db_medications": [{"name": "Calcium", "dosage": "500mg"}],
                "lab_results": [{"test_name": "Calcium", "value": "7.2 mg/dL", "status": "Abnormal (Low)"}],
            }
        ],
    }
    answer, evidence = answer_patient_question(patient, "patient-1", "why is calcium low")
    user_msg = next(m["content"] for m in called_messages if m["role"] == "user")
    assert "Medication Shift" not in user_msg

