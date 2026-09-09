"""
Stage 15 (partial): conflicting-data detection.

Detects contradictory field values across documents belonging to the
same patient (e.g. age/sex mismatches) instead of silently picking one.
Generic by design — pass in any {field_name: [(value, source_document), ...]}
map you're able to extract from your documents; it flags any field with
more than one distinct value.

Extend this with whatever demographic/structured fields your
`patientdocument` table actually exposes (age, sex, name spelling, etc.)
by calling detect_field_conflicts() per field.
"""
from __future__ import annotations


def detect_field_conflicts(field_name: str, observations: list[tuple]) -> dict | None:
    """
    observations: list of (value, source_document_id) tuples.
    Returns a conflict dict if more than one distinct value is present,
    else None.
    """
    distinct = {}
    for value, source in observations:
        if value is None:
            continue
        distinct.setdefault(str(value), []).append(source)

    if len(distinct) <= 1:
        return None

    return {
        "field": field_name,
        "conflicting_values": [
            {"value": v, "source_documents": sources}
            for v, sources in distinct.items()
        ],
        "requires_verification": True,
    }


def detect_medication_uncertainty(timeline: dict) -> list[dict]:
    """Surfaces medication extractions flagged needs_verification=True."""
    flagged = []
    for visit in timeline["visits"]:
        for med in visit["medications"]:
            if med.get("needs_verification"):
                flagged.append({
                    "visit_id": visit["visit_id"],
                    "visit_date": visit["visit_date"],
                    "name_raw": med["name_raw"],
                    "source_text": med["source_text"],
                })
    return flagged

def detect_all_uncertain_findings(timeline: dict) -> dict:
    """
    Surfaces every needs_verification=True item across medications,
    diagnoses, and symptoms — this is where NER-sourced entities show up
    (they're always flagged needs_verification=True since they aren't
    dictionary-matched), alongside any low-confidence rule-based catches.

    Returns: {"medication": [...], "diagnosis": [...], "symptom": [...]}
    each item shaped {visit_id, visit_date, name, name_raw, source_text,
    confidence}.
    """
    buckets = {"medication": [], "diagnosis": [], "symptom": []}
    type_key = {
        "medication": "medications",
        "diagnosis": "diagnoses",
        "symptom": "symptoms",
    }

    for visit in timeline["visits"]:
        for event_type, key in type_key.items():
            for item in visit[key]:
                if item.get("needs_verification"):
                    buckets[event_type].append({
                        "visit_id": visit["visit_id"],
                        "visit_date": visit["visit_date"],
                        "name": item["name"],
                        "name_raw": item["name_raw"],
                        "confidence": item["confidence"],
                        "source_text": item["source_text"],
                    })

    return buckets