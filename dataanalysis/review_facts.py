"""Stable, compact clinical facts used to identify review source changes."""
import hashlib
import json
import re
from pathlib import Path

from core.config import flag as _flag

PROMPT_VERSION = "clinical_ai_review_v25"  # v25: + quick tier (prompts/review_quick.md)

# When ENABLE_EXTERNAL_PROMPTS is on, swap to the file-derived hash so that
# any prompt edit automatically invalidates the review cache.
if _flag("ENABLE_EXTERNAL_PROMPTS"):
    try:
        from core.prompts import REVIEW_PROMPT_VERSION as _rpv
        PROMPT_VERSION = _rpv
    except Exception:
        pass  # keep the static fallback


def _guideline_version():
    """The guideline values version (and whether drafts run): a change regenerates cached reviews."""
    path = Path(__file__).resolve().parent / "data" / "reference" / "condition_registry.json"
    try:
        guidelines = json.loads(path.read_text(encoding="utf-8")).get("guidelines", {})
    except (OSError, ValueError):
        return ""
    version = guidelines.get("version", "")
    if not version:
        return ""
    # Verifying (or revoking) a section changes what the AI reads, so it changes the version too.
    verified = sorted(f"{name}:{rule.get('verified_by')}:{rule.get('verified_on')}" for name, rule in guidelines.items()
                      if isinstance(rule, dict) and rule.get("verified") and rule.get("verified_by") and rule.get("verified_on"))
    signature = "-verified-" + hashlib.sha1("|".join(verified).encode()).hexdigest()[:8] if verified else ""
    return f"+guidelines-{version}{signature}" + ("-draft" if _flag("ENABLE_DRAFT_GUIDELINES") else "")


PROMPT_VERSION += _guideline_version()


def _normalized_medication(value):
    text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    text = re.sub(r"(?i)^(?:tab(?:let)?|cap(?:sule)?|inj(?:ection)?|syp(?:rup)?)\.?\s+", "", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|%)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
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
                "reference": lab.get("reference"),
            })
        # The review reads prescriptions and vitals directly, so they are part of the source.
        medications = sorted(
            " ".join([_normalized_medication(m.get("name"))]
                     + [str(m.get(f) or "").strip() for f in ("dosage", "frequency", "duration")]).strip()
            for m in visit.get("db_medications", []) or []
        )
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
            "medications": medications,
            "vitals": visit.get("db_vitals_json") or {},
            "labs": labs,
            "followup_status": followups,
            "discrepancy_count": len(visit.get("discrepancies", []) or []),
            "doc_ids": doc_ids,
        })
    return facts


def unlinked_facts(patient_data):
    """Compact facts for documents without a visit_id (empty list when none)."""
    facts = []
    for doc in patient_data.get("unlinked_documents") or []:
        cs = doc.get("clinical_summary") or {}
        likely = doc.get("likely_visit") or {}
        facts.append({
            "doc_id": str(doc.get("doc_id")),
            "likely_visit": likely.get("visit_id"),
            "candidates": sorted(c.get("visit_id") for c in doc.get("candidate_visits") or []),
            "dx": cs.get("clinical_impression") or cs.get("provisional_diagnosis") or cs.get("confirmed_diagnosis"),
            "meds": sorted(doc.get("medications_found") or []),
            "labs": [[l.get("test_name"), l.get("value"), l.get("status")] for l in doc.get("lab_results") or []],
        })
    return facts


def visit_hashes(patient_data, facts=None):
    facts = facts if facts is not None else build_review_facts(patient_data)
    hashes = {str(fact["visit_id"]): _canonical_hash(fact) for fact in facts}
    unlinked = unlinked_facts(patient_data)
    if unlinked:
        hashes["unlinked_documents"] = _canonical_hash(unlinked)
    history = patient_data.get("health_history_present")
    if history:
        hashes["health_history"] = _canonical_hash(history)
    long_term = patient_data.get("long_term_conditions")
    if long_term:
        hashes["long_term_conditions"] = _canonical_hash(long_term)
    return hashes


def compute_source_hash(patient_id, hashes):
    stable = {
        "patient_id": str(patient_id),
        "visit_hashes": sorted((str(vid), str(value)) for vid, value in hashes.items()),
    }
    return _canonical_hash(stable)


def unlinked_signature(patient_data):
    """Short hash of documents not linked to a visit; empty when there are none.

    Kept out of the review source hash so existing review caches stay valid.
    """
    docs = patient_data.get("unlinked_documents") or []
    if not docs:
        return ""
    return _canonical_hash(unlinked_facts(patient_data))[:12]
