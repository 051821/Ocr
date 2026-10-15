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


def test_query_patient_graph_orders_by_recency_and_reports_truncation(monkeypatch, caplog):
    import neo4j_client

    calls = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run(self, cypher, **params):
            calls.append((cypher, params))
            if "HAS_UNLINKED_DOC" in cypher:
                return []
            # Pretend the database honoured ORDER BY and returned limit + 1 rows.
            return [{"visit": {"id": f"p1:v{i}", "date": f"2026-01-{30 - i:02d}"}, "relation": "LAB",
                     "labels": ["LabTest"], "node": {"name": f"test{i}"}, "relationship": {"val": str(i)}}
                    for i in range(params["limit"])]

    class FakeDriver:
        def session(self, **kwargs):
            return FakeSession()

    monkeypatch.setattr(neo4j_client, "get_driver", lambda: FakeDriver())
    with caplog.at_level("WARNING", logger="neo4j_client"):
        rows = neo4j_client.query_patient_graph("p1", limit=3)

    cypher, params = calls[0]
    assert "ORDER BY" in cypher and "v.date DESC" in cypher
    assert params["limit"] == 4  # one extra row to detect truncation
    assert [row["visit_id"] for row in rows] == ["p1:v0", "p1:v1", "p1:v2"]
    assert "truncated" in caplog.text
