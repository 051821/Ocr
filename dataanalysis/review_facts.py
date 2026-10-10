"""Stable, compact clinical facts used to identify review source changes."""
import hashlib
import json
import re


PROMPT_VERSION = "clinical_ai_review_v7"


def _normalized_medication(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    text = re.sub(r"(?i)^(?:tab(?:let)?|cap(?:sule)?|inj(?:ection)?|syp(?:rup)?)\.?\s+", "", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|%)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def build_review_facts(patient_data):
    """Return stable clinical fields only; excludes prompts and display text."""
    journey_by_visit = {}
    for encounter in patient_data.get("_medication_journey_cache", []) or []:
        journey_by_visit[str(encounter.get("visit_id", ""))] = encounter

    facts = []
    for visit in patient_data.get("visits", []) or []:
        visit_id = str(visit.get("visit_id", ""))
        journey = journey_by_visit.get(visit_id, {})
        changes = []
        for change in journey.get("changes", []) or []:
            changes.append({
                "type": change.get("type"),
                "medication": _normalized_medication(change.get("medication") or change.get("text")),
                "details": re.sub(r"\s+", " ", str(change.get("text") or "")).strip(),
            })
        labs = []
        for lab in visit.get("lab_results", []) or []:
            labs.append({
                "name": re.sub(r"\s+", " ", str(lab.get("test_name") or lab.get("name") or "")).strip().casefold(),
                "value": lab.get("value"),
                "status": lab.get("status"),
            })
        followups = sorted(
            str(item.get("status") or "") for item in visit.get("followups", []) or []
        )
        doc_ids = sorted(
            str(doc.get("doc_id")) for doc in visit.get("documents", []) or []
            if doc.get("doc_id") is not None
        )
        facts.append({
            "visit_id": visit_id,
            "date": visit.get("visit_date"),
            "chief_complaint": visit.get("db_chief_complaint") or visit.get("chief_complaint"),
            "provisional_diagnosis": visit.get("db_provisional_dx") or visit.get("provisional_diagnosis"),
            "confirmed_diagnosis": visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis"),
            "medication_changes": changes,
            "labs": labs,
            "followup_status": followups,
            "discrepancy_count": len(visit.get("discrepancies", []) or []),
            "doc_ids": doc_ids,
        })
    return facts


def visit_hashes(patient_data, facts=None):
    facts = facts if facts is not None else build_review_facts(patient_data)
    return {str(fact["visit_id"]): _canonical_hash(fact) for fact in facts}


def compute_source_hash(patient_id, hashes):
    stable = {
        "patient_id": str(patient_id),
        "visit_hashes": sorted((str(vid), str(value)) for vid, value in hashes.items()),
    }
    return _canonical_hash(stable)
