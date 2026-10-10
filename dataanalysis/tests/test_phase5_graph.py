from neo4j_client import _build_graph_payload


def test_compact_graph_payload_has_shared_concepts_and_no_ocr_chunks():
    payload = _build_graph_payload({
        "patient_id": "patient-1", "_review_source_hash": "source-1",
        "visits": [{
            "visit_id": "visit-1", "visit_date": "2026-02-03T10:00:00",
            "chief_complaint": "Fatigue", "confirmed_diagnosis": "Diabetes",
            "db_vitals_json": {"bp": "120/80", "pulse": "72"},
            "db_medications": [{"name": "Metformin", "dosage": "500 mg", "frequency": "daily"}],
            "lab_results": [{"test_name": "Glucose", "value": "123.4 mg/dL", "status": "High"}],
            "followups": [{"scheduled_date": "2026-03-01", "status": "Scheduled"}],
            "documents": [{"doc_id": "doc-1", "document_label": "Visit note", "extracted_text": ["x" * 400]}],
        }],
    })

    assert payload["visits"][0]["id"] == "patient-1:visit-1"
    assert payload["visits"][0]["bp_sys"] == 120
    assert payload["rx"] == [{"visit_id": "patient-1:visit-1", "name": "metformin", "dose": "500 mg", "freq": "daily", "dur": ""}]
    assert payload["dx"][0]["name"] == "diabetes"
    assert payload["labs"][0]["num"] == 123.4
    assert payload["docs"][0]["summary"] == "x" * 300
    assert "chunks" not in payload
