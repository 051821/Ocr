"""
Stage 10: Patient timeline construction.

Groups deduplicated events by patient -> visit (sorted chronologically by
visit_date), so multiple documents from one visit collapse into that one
visit, and multiple visits stay correctly ordered.
"""
from __future__ import annotations
from collections import defaultdict
from clinical_nlp.events import ClinicalEvent


def build_patient_timeline(events: list[ClinicalEvent]) -> dict:
    """
    Returns:
    {
        "patient_id": "...",
        "visits": [
            {
                "visit_id": "...",
                "visit_date": "2026-05-05",
                "diagnoses": [...],
                "symptoms": [...],
                "medications": [...],
                "laboratory": [...],
            },
            ...
        ]  # sorted by visit_date ascending; visits with no date go last
    }
    """
    if not events:
        return {"patient_id": None, "visits": []}

    patient_id = events[0].patient_id
    by_visit: dict[str, dict] = defaultdict(lambda: {
        "visit_id": None, "visit_date": None,
        "diagnoses": [], "symptoms": [], "medications": [], "laboratory": [],
    })

    type_key = {
        "diagnosis": "diagnoses",
        "symptom": "symptoms",
        "medication": "medications",
        "laboratory": "laboratory",
    }

    for e in events:
        bucket = by_visit[e.visit_id]
        bucket["visit_id"] = e.visit_id
        bucket["visit_date"] = e.visit_date
        bucket[type_key[e.event_type]].append(e.to_dict())

    visits = list(by_visit.values())
    visits.sort(key=lambda v: (v["visit_date"] is None, v["visit_date"] or ""))

    return {"patient_id": patient_id, "visits": visits}
