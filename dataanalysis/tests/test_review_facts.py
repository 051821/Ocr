import csv
import json
from concurrent.futures import ThreadPoolExecutor

from review_facts import build_review_facts, compute_source_hash, visit_hashes
import review_facts
from storage import ai_analysis_csv


def _patient():
    return {
        "patient_id": "patient-1",
        "visits": [{
            "visit_id": "visit-1",
            "visit_date": "2026-01-01",
            "chief_complaint": "Headache",
            "lab_results": [{"test_name": "Glucose", "value": "100 mg/dL", "status": "Normal"}],
            "documents": [{"doc_id": 7}],
            "followups": [{"status": "Scheduled"}],
            "discrepancies": [],
        }],
        "_medication_journey_cache": [],
    }


def test_review_hash_stable_to_prompt_version_and_changes_with_clinical_fact(monkeypatch):
    patient = _patient()
    first_facts = build_review_facts(patient)
    first_hash = compute_source_hash(patient["patient_id"], visit_hashes(patient, first_facts))
    monkeypatch.setattr(review_facts, "PROMPT_VERSION", "different-prompt")
    assert first_hash == compute_source_hash(patient["patient_id"], visit_hashes(patient, first_facts))
    patient["visits"][0]["lab_results"][0]["value"] = "101 mg/dL"
    changed_facts = build_review_facts(patient)
    changed_hash = compute_source_hash(patient["patient_id"], visit_hashes(patient, changed_facts))
    assert first_hash != changed_hash
    assert first_hash == compute_source_hash("patient-1", visit_hashes(_patient()))


def test_concurrent_review_upserts_are_serialized(tmp_path):
    patient = _patient()
    facts = build_review_facts(patient)
    hashes = visit_hashes(patient, facts)
    source_hash = compute_source_hash(patient["patient_id"], hashes)

    def write(index):
        return ai_analysis_csv.upsert_ai_analysis(
            patient,
            f"review {index}",
            source_hash=source_hash,
            hashes=hashes,
            path=tmp_path / "reviews.db",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(8)))
    row = ai_analysis_csv.get_patient_row("patient-1", path=tmp_path / "reviews.db")
    assert row["source_hash"] == source_hash
    assert row["visit_hashes"] == json.dumps(hashes, sort_keys=True, separators=(",", ":"))


def test_legacy_csv_migrates_large_field_as_stale(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    csv_path = tmp_path / "patient_ai_analysis.csv"
    monkeypatch.setattr(ai_analysis_csv, "_DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(ai_analysis_csv, "_LEGACY_CSV_PATH", csv_path)
    long_review = "x" * 210_000
    fields = ["patient_id", "prompt_version", "model_name", "analysis", "source_visit_count", "hash_mismatch_count", "generated_at"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "patient_id": "patient-legacy",
            "prompt_version": "old-version",
            "model_name": "old-model",
            "analysis": json.dumps({"raw_markdown": long_review}),
            "source_visit_count": "3",
            "hash_mismatch_count": "0",
            "generated_at": "2025-01-01T00:00:00+00:00",
        })

    conn = ai_analysis_csv._connect()
    conn.close()
    row = ai_analysis_csv.get_patient_row("patient-legacy")
    assert row["source_hash"] is None
    assert row["review_md"] == long_review
    assert csv_path.with_suffix(".csv.bak").exists()
