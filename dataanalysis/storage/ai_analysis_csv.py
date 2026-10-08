"""Single-row-per-patient CSV cache for generated AI reviews."""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from Zai_analysis import PROMPT_VERSION, build_ai_review_context

_DEFAULT_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "patient_ai_analysis.csv"
FULL_RERUN_THRESHOLD = 4
ANALYSIS_TYPE = "retrospective_clinical_analysis"
FIELDS = ["id", "patient_id", "analysis_type", "model_name", "model_version", "prompt_version", "source_visit_count", "source_data_hash", "analysis", "generated_at", "hash_mismatch_count", "source_context"]


def compute_source_hash(patient_data):
    body = json.dumps(build_ai_review_context(patient_data), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_rows(path=None):
    path = Path(path or _DEFAULT_CSV_PATH)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_patient_row(patient_id, path=None):
    return next((row for row in _read_rows(path) if row.get("patient_id") == str(patient_id)), None)


def upsert_ai_analysis(patient_data, analysis, model_name="", model_version="", mismatch_count=0, path=None):
    path = Path(path or _DEFAULT_CSV_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _read_rows(path) if r.get("patient_id") != str(patient_data.get("patient_id"))]
    item = {
        "id": str(1 + max((int(r.get("id") or 0) for r in rows), default=0)),
        "patient_id": str(patient_data.get("patient_id", "")),
        "analysis_type": "retrospective_clinical_analysis", "model_name": model_name,
        "model_version": model_version, "prompt_version": PROMPT_VERSION,
        "source_visit_count": str(patient_data.get("total_visits", len(patient_data.get("visits", [])))),
        "source_data_hash": compute_source_hash(patient_data),
        "analysis": json.dumps(analysis, ensure_ascii=False),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hash_mismatch_count": str(mismatch_count),
        "source_context": json.dumps(build_ai_review_context(patient_data), ensure_ascii=False),
    }
    rows.append(item)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return item


def append_ai_analysis(patient_data, analysis, model_name="", model_version="", path=None):
    return upsert_ai_analysis(patient_data, analysis, model_name, model_version, path=path)
