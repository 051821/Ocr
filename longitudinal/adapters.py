"""
Adapter between your existing DB-query output (analysis.py) and the
visit/document shape clinical_nlp.events expects.

Two adapters are provided:

1. adapt_from_simple_history() — works with your CURRENT fetch_patient_history()
   with zero DB changes. Each visit's merged `medical_text` is wrapped as
   a single synthetic document (document_id=None), so labs/meds/symptoms
   extracted from it can't be traced back to one specific source document
   — only back to the visit. This is a reasonable starting point.

2. adapt_from_detailed_history() — works with the new
   fetch_patient_documents() added to analysis.py, which preserves real
   per-document IDs. Use this once you've adopted that query; it gives
   full document-level traceability as required by the spec ("every
   extracted item must retain patient_id, visit_id, document_id").
"""
from __future__ import annotations


def adapt_from_simple_history(patient_data: dict) -> list[dict]:
    patient_id = patient_data["patient_id"]
    visits = []
    for v in patient_data["visits"]:
        visits.append({
            "patient_id": patient_id,
            "visit_id": v["visit_id"],
            "visit_date": v["visit_date"],
            "provisional_diagnosis": v.get("provisional_diagnosis"),
            "confirmed_diagnosis": v.get("confirmed_diagnosis"),
            "documents": [{
                "document_id": None,
                "document_type": "merged_visit_text",
                "clean_text": v.get("medical_text") or "",
            }],
        })
    return visits


def adapt_from_detailed_history(patient_data: dict) -> list[dict]:
    patient_id = patient_data["patient_id"]
    visits = []
    for v in patient_data["visits"]:
        visits.append({
            "patient_id": patient_id,
            "visit_id": v["visit_id"],
            "visit_date": v["visit_date"],
            "provisional_diagnosis": v.get("provisional_diagnosis"),
            "confirmed_diagnosis": v.get("confirmed_diagnosis"),
            "documents": [
                {
                    "document_id": d["document_id"],
                    "document_type": d.get("document_type"),
                    "clean_text": d.get("clean_text") or "",
                }
                for d in v.get("documents", [])
            ],
        })
    return visits
