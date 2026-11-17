"""Normalisation coverage: what the registries could not map, per record and in one report file.

Sections: medicines with no ingredients, medicines resolved only by the model, prescription lines whose route is unclear,
complaint phrases no symptom concept explains, diagnosis entries no registry condition explains, prescription lines
with an unreadable dose. Report: COVERAGE_REPORT (default logs/coverage_report.json; 'off' disables). Term text only,
never patient identifiers.
"""
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import clinical_terms
import symptom_map
from patient_chat import conditions_matching

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
# Visit reasons and history markers are not symptoms.
_NOT_A_SYMPTOM = re.compile(r"(?i)\b(?:follow[\s-]*up|f/u|report|review|check|routine|refill|k/?c/?o|kco|known\s+case|h/o)\b")
_NO_DX = re.compile(r"(?i)^\s*(?:none|nil|no\s+complaints?|null|-|na|n/a)?\s*$")


def record_coverage(patient_data, rec):
    """{section: [terms]} for one record (rec: medicine_profile.MedRecord)."""
    visits = patient_data.get("visits") or []
    complaints = []
    for _, part in clinical_terms.complaint_parts(visits):
        if _NOT_A_SYMPTOM.search(part) or conditions_matching(part):
            continue
        resolved = clinical_terms.complaint(patient_data, part) or {}
        if not symptom_map.classify(part) and not resolved.get("symptoms"):
            complaints.append(part.casefold())
    diagnoses = []
    for v in visits:
        for field in ("db_provisional_dx", "db_confirmed_dx"):
            text = str(v.get(field) or "").strip()
            if not text or _NO_DX.match(text):
                continue
            resolved = clinical_terms.diagnosis(patient_data, text)
            mapped = conditions_matching(text) or (resolved and any(p.get("condition") for p in resolved.get("parts") or []))
            if not mapped:
                diagnoses.append(text.casefold())
    lines = rec.prescribed()
    return {
        "medicines_unmapped": sorted({part for _, part in rec.unmapped()}),
        "medicines_model_only": sorted(set(rec.model_only())),
        "routes_unclear": sorted({l["name"] for l in lines if l["route"] == "unclear"}),
        "doses_unreadable": sorted({f"{l['name']} '{l['dosage']}'" for l in lines if l["dose"]["status"] == "unreadable"}),
        "complaints_unmapped": sorted(set(complaints)),
        "diagnoses_unmapped": sorted(set(diagnoses)),
    }


def report_path():
    value = os.getenv("COVERAGE_REPORT", os.getenv("UNMAPPED_BRANDS_REPORT"))
    if value is None:
        return Path(__file__).resolve().parent / "logs" / "coverage_report.json"
    return None if value.strip().casefold() in ("", "off", "false", "0", "none") else Path(value)


def write(coverage):
    """Add one record's coverage to the report: per section, each term with times seen and last seen. Never raises."""
    path = report_path()
    if not path or not any(coverage.values()):
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with _LOCK:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            for section, terms in coverage.items():
                bucket = data.setdefault(section, {})
                for term in terms:
                    entry = bucket.setdefault(term[:120], {"times_seen": 0})
                    entry["times_seen"] += 1
                    entry["last_seen"] = now
            data["updated"] = now
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
    except Exception as exc:
        logger.warning("coverage report not written: %s", exc)
