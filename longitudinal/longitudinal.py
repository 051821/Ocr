"""
Stage 5-15 orchestrator: runs clinical NLP extraction, deduplication,
timeline construction, and trend/longitudinal analysis for one patient.

This is the single function you call from app.py / pipeline_integration.py.
It does not touch the database or the LLM — it's pure, testable Python
over the visit records you pass in.
"""
from __future__ import annotations
import logging

from clinical_nlp.events import build_events_for_visit
from timeline.deduplicate import deduplicate_events
from timeline.patient_timeline import build_patient_timeline
from timeline.trends import (
    analyze_lab_trends,
    analyze_medication_longitudinal,
    analyze_diagnosis_longitudinal,
)
from longitudinal.conflicts import detect_medication_uncertainty

logger = logging.getLogger(__name__)


def run_longitudinal_analysis(patient_visits: list[dict]) -> dict:
    """
    patient_visits: list of visit dicts, each shaped like the docstring
    in clinical_nlp/events.py (patient_id, visit_id, visit_date,
    provisional_diagnosis, confirmed_diagnosis, documents=[...]).

    Returns a structured payload ready to hand to the LLM report
    generator, or to render directly in the UI:

    {
        "patient_id": ...,
        "observation_period": {"start": ..., "end": ...},
        "timeline": {...},                  # Stage 10 output
        "lab_trends": {...},                # Stage 11
        "medication_longitudinal": {...},   # Stage 12
        "diagnosis_longitudinal": {...},    # Stage 13
        "medication_uncertainty": [...],    # Stage 15 (partial)
        "total_events_extracted": int,
        "total_events_after_dedup": int,
    }
    """
    if not patient_visits:
        return {
            "patient_id": None, "observation_period": None, "timeline": {"visits": []},
            "lab_trends": {}, "medication_longitudinal": {}, "diagnosis_longitudinal": {},
            "medication_uncertainty": [], "total_events_extracted": 0,
            "total_events_after_dedup": 0,
        }

    all_events = []
    for visit in patient_visits:
        try:
            all_events.extend(build_events_for_visit(visit))
        except Exception:
            logger.exception("Event extraction failed for visit %s", visit.get("visit_id"))

    total_extracted = len(all_events)
    deduped = deduplicate_events(all_events)

    timeline = build_patient_timeline(deduped)
    dates = [v["visit_date"] for v in timeline["visits"] if v["visit_date"]]

    return {
        "patient_id": patient_visits[0]["patient_id"],
        "observation_period": {
            "start": min(dates) if dates else None,
            "end": max(dates) if dates else None,
        },
        "timeline": timeline,
        "lab_trends": analyze_lab_trends(timeline),
        "medication_longitudinal": analyze_medication_longitudinal(timeline),
        "diagnosis_longitudinal": analyze_diagnosis_longitudinal(timeline),
        "medication_uncertainty": detect_medication_uncertainty(timeline),
        "total_events_extracted": total_extracted,
        "total_events_after_dedup": len(deduped),
    }
