"""
Top-level entry point tying the whole extension together.

    OCR pipeline (existing)
        -> analysis.fetch_patient_history() / fetch_patient_documents()
        -> longitudinal.adapters
        -> longitudinal.longitudinal.run_longitudinal_analysis()   [Stages 5-15]
        -> longitudinal.report_generator.generate_longitudinal_report()  [Stage 16]

Use run_patient_pipeline() from app.py (or a script / notebook) exactly
like you currently call fetch_patient_history() + analyze_patient_with_ai().
"""
from __future__ import annotations
import logging

from dataanalysis.analysis import fetch_patient_history
from longitudinal.adapters import adapt_from_simple_history
from longitudinal.longitudinal import run_longitudinal_analysis
from longitudinal.report_generator import generate_longitudinal_report

logger = logging.getLogger(__name__)

try:
    # optional: only used if you've added fetch_patient_documents to analysis.py
    from dataanalysis.analysis import fetch_patient_documents
    from longitudinal.adapters import adapt_from_detailed_history
    _HAS_DETAILED_QUERY = True
except ImportError:
    _HAS_DETAILED_QUERY = False


def run_patient_pipeline(patient_id: str, use_document_level_query: bool = False,
                          generate_report: bool = True, model: str = "qwen3:4b") -> dict:
    """
    Runs the full longitudinal pipeline for one patient.

    Returns:
    {
        "patient_data": ...,        # raw DB fetch, for the existing UI/debug view
        "analysis": {...},          # structured output from run_longitudinal_analysis()
        "report": "..." or None,    # LLM-generated narrative report (Stage 16)
    }
    """
    if use_document_level_query and _HAS_DETAILED_QUERY:
        patient_data = fetch_patient_documents(patient_id)
        visits = adapt_from_detailed_history(patient_data)
    else:
        patient_data = fetch_patient_history(patient_id)
        visits = adapt_from_simple_history(patient_data)

    if patient_data["total_visits"] == 0:
        return {"patient_data": patient_data, "analysis": None, "report": None}

    analysis_payload = run_longitudinal_analysis(visits)

    report = None
    if generate_report:
        try:
            report = generate_longitudinal_report(analysis_payload, model=model)
        except RuntimeError:
            logger.exception("Report generation failed for patient %s", patient_id)
            report = None

    return {
        "patient_data": patient_data,
        "analysis": analysis_payload,
        "report": report,
    }


def run_patient_pipeline_single_visit(patient_id: str, visit_id: str, **kwargs) -> dict:
    """
    Convenience wrapper for testing one visit in isolation (spec
    requirement: "process one patient or one visit independently").
    """
    full = run_patient_pipeline(patient_id, generate_report=False)
    if full["analysis"] is None:
        return full

    filtered_visits = [
        v for v in full["analysis"]["timeline"]["visits"] if v["visit_id"] == str(visit_id)
    ]
    full["analysis"]["timeline"]["visits"] = filtered_visits
    return full
