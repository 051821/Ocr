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


def test_lab_tokens_come_from_this_patients_record():
    from patient_chat import _patient_lab_tokens
    patient = {
        "patient_id": "patient-1",
        "visits": [{"visit_id": "v1", "lab_results": [
            {"test_name": "Serum Creatinine", "value": "2.1 mg/dL", "status": "High"},
            {"test_name": "Serum Sodium", "value": "139 mmol/L", "status": "Normal"},
        ]}],
        "unlinked_documents": [{"doc_id": "u1", "lab_results": [{"test_name": "TSH", "value": "9.8", "status": "Abnormal (High)"}]}],
    }
    assert _patient_lab_tokens(patient) == {"creatinine": True, "sodium": False, "tsh": True}


def test_lab_question_uses_patient_labs_not_fixed_diabetes_terms(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda *a, **k: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    monkeypatch.setattr("patient_chat.llm_complete", lambda *a, **k: {"content": "### Answer\nok", "usage": {}})
    patient = {
        "patient_id": "patient-1",
        "visits": [{
            "visit_id": "v1", "visit_date": "2026-01-01", "chief_complaint": "Swelling",
            "lab_results": [{"test_name": "Serum Creatinine", "value": "2.1 mg/dL", "status": "High"}],
            "documents": [{"doc_id": "d1", "extracted_text": ["Old note: glucose and hba1c discussed with family."]}],
        }],
    }
    _, evidence = answer_patient_question(patient, "patient-1", "why is creatinine elevated")
    contents = " ".join(e["content"] for e in evidence)
    assert "Creatinine" in contents
    assert "hba1c discussed" not in contents


def test_cache_key_depends_on_conversation_history(monkeypatch):
    keys = []
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: keys.append(k))
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda *a, **k: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    monkeypatch.setattr("patient_chat.llm_complete", lambda *a, **k: {"content": "### Answer\nok", "usage": {}})
    patient = {
        "patient_id": "patient-1",
        "visits": [{"visit_id": "v1", "visit_date": "2026-01-01",
                    "db_medications": [{"name": "Metformin"}, {"name": "Amlodipine"}]}],
    }
    question = "what are its side effects"
    for history in (None, None,
                    [{"role": "user", "content": "tell me about metformin"}],
                    [{"role": "user", "content": "tell me about amlodipine"}]):
        answer_patient_question(patient, "patient-1", question, history=history, source_hash="h1")
    assert len(keys) == 4
    assert keys[0] == keys[1]
    assert len({keys[1], keys[2], keys[3]}) == 3


def test_rule_matched_evidence_survives_packing_budget(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda *a, **k: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    sent = []
    monkeypatch.setattr("patient_chat.llm_complete", lambda messages, *a, **k: sent.extend(messages) or {"content": "### Answer\nok", "usage": {}})
    # Twelve encounters whose summaries all mention the lab, so BM25 alone fills every slot with them.
    visits = [{"visit_id": f"v{i}", "visit_date": f"2026-01-{i:02d}", "chief_complaint": "Creatinine follow-up",
               "lab_results": [{"test_name": "Serum Creatinine", "value": f"{1 + i / 10:.1f} mg/dL", "status": "High"}]}
              for i in range(1, 13)]
    visits[0]["documents"] = [{"doc_id": "d1", "extracted_text": ["Nephrology impression: rising creatinine suggests early CKD."]}]
    patient = {"patient_id": "patient-1", "visits": visits}
    _, evidence = answer_patient_question(patient, "patient-1", "what does my creatinine indicate")
    user_msg = next(m["content"] for m in sent if m["role"] == "user")
    assert "Nephrology impression" in user_msg
    assert len(evidence) <= 8


def test_bm25_lab_boost_follows_patient_labs_not_fixed_list():
    from patient_chat import _bm25
    entries = [
        {"category": "laboratory result", "content": "Lab: TSH 9.8 uIU/mL High"},
        {"category": "document", "content": "Doc text: TSH repeat advised"},
    ]
    def lab_score(query, lab_terms):
        return next(s for s, e in _bm25(query, entries, limit=None, with_scores=True, lab_terms=lab_terms)
                    if e["category"] == "laboratory result")
    # Naming one of this patient's tests earns the lab-category boost (+4)...
    assert lab_score("tsh", {"tsh"}) - lab_score("tsh", ()) == 4.0
    # ...while a test this patient never had gets no boost of its own.
    assert not _bm25("hba1c", entries, limit=None, lab_terms={"tsh"})


def _vitals_patient():
    return {
        "patient_id": "patient-1",
        "visits": [
            {"visit_id": "v1", "visit_date": "2026-01-01", "db_vitals_rows": [{"label": "BP", "value": "150/95 mmHg"}]},
            {"visit_id": "v2", "visit_date": "2026-02-01"},
            {"visit_id": "v3", "visit_date": "2026-03-01", "db_vitals_rows": [{"label": "BP", "value": "128/82 mmHg"}]},
        ],
    }


def test_vitals_for_named_visit_not_latest(monkeypatch):
    monkeypatch.setattr("patient_chat.llm_complete", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
    for question in ("what was the BP in visit 1", "BP of visit 1", "visit 1 bp"):
        meta = {}
        answer, evidence = answer_patient_question(_vitals_patient(), "patient-1", question, meta=meta)
        assert "150/95" in answer and "128/82" not in answer, question
        assert meta["rule"] == "vitals_visit"
        assert evidence[0]["visit_index"] == 1


def test_vitals_for_visit_without_vitals_says_not_documented(monkeypatch):
    monkeypatch.setattr("patient_chat.llm_complete", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
    answer, evidence = answer_patient_question(_vitals_patient(), "patient-1", "what was the BP in visit 2")
    assert "No vitals are documented for Encounter #2" in answer
    assert "150/95" not in answer and "128/82" not in answer
    assert evidence == []


def test_vitals_relative_to_visit_skips_single_reading_fast_path(monkeypatch):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda *a, **k: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    monkeypatch.setattr("patient_chat.llm_complete", lambda *a, **k: {"content": "### Answer\nok", "usage": {}})
    meta = {}
    answer_patient_question(_vitals_patient(), "patient-1", "what was the BP after visit 1", meta=meta)
    assert meta["rule"] not in ("vitals_latest", "vitals_visit")


def test_router_uses_record_words_but_still_rejects_off_topic():
    patient = {
        "patient_id": "patient-1",
        "visits": [{"visit_id": "v1", "documents": [{"doc_id": "d1", "document_label": "Chest X-ray",
                    "extracted_text": ["The weather was cold; patient walked in for a story of cough."]}]}],
        "unlinked_documents": [{"doc_id": "u1", "document_label": "Renal ultrasound"}],
    }
    assert route_question("What did the ultrasound show?", patient) == "patient_record"
    assert route_question("Anything on the chest x-ray?", patient) == "patient_record"
    assert route_question("Which painkillers are safe?", patient) == "patient_record"
    # Words that only occur in raw OCR text don't open the router
    assert route_question("What is the weather today?", patient) == "out_of_scope"
    assert route_question("Tell me a funny story", patient) == "out_of_scope"


def _llm_patient():
    return {
        "patient_id": "patient-1",
        "visits": [{"visit_id": "v1", "visit_date": "2026-01-01", "chief_complaint": "Fatigue",
                    "provisional_diagnosis": "Hypothyroidism",
                    "db_medications": [{"name": "Levothyroxine", "dosage": "50mcg"}],
                    "lab_results": [{"test_name": "TSH", "value": "9.8 uIU/mL", "status": "High"}]}],
    }


def _stub_llm(monkeypatch, content):
    monkeypatch.setattr("patient_chat.get_llm_cache", lambda k: None)
    monkeypatch.setattr("patient_chat.set_llm_cache", lambda *a, **k: None)
    monkeypatch.setattr("patient_chat.audit_logger", logging.getLogger("test_null_logger"))
    sent = []
    monkeypatch.setattr("patient_chat.llm_complete", lambda messages, *a, **k: sent.extend(messages) or {"content": content, "usage": {}})
    return sent


def test_chat_prompt_asks_for_bullets_and_forbids_suggestions(monkeypatch):
    sent = _stub_llm(monkeypatch, "### Answer\n- ok")
    answer_patient_question(_llm_patient(), "patient-1", "what does my TSH indicate")
    system = next(m["content"] for m in sent if m["role"] == "system")
    assert "bullet points" in system
    assert "Never suggest, recommend or name a medicine" in system
    assert "paragraph" not in system.replace("no paragraphs", "")


def test_chat_answer_strips_medicine_and_test_suggestions(monkeypatch):
    _stub_llm(monkeypatch, "### Answer\n"
              "- TSH 9.8 uIU/mL is High at Encounter #1 (2026-01-01).\n"
              "- Consider repeating TSH in 6 weeks.\n"
              "- Levothyroxine dose may need review.\n"
              "- Hypothyroidism is documented since Encounter #1.")
    meta = {}
    answer, _ = answer_patient_question(_llm_patient(), "patient-1", "what does my TSH indicate", meta=meta)
    assert meta["path"] == "llm"
    assert "TSH 9.8 uIU/mL is High" in answer and "Hypothyroidism is documented" in answer
    assert "Consider repeating" not in answer and "may need review" not in answer


def test_chat_answer_of_only_suggestions_falls_back(monkeypatch):
    _stub_llm(monkeypatch, "### Answer\n- Consider repeating TSH.\n- Levothyroxine dose may need adjustment.")
    meta = {}
    answer, _ = answer_patient_question(_llm_patient(), "patient-1", "what does my TSH indicate", meta=meta)
    assert meta["path"] == "fallback"
    assert "Consider repeating" not in answer
