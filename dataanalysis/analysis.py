import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database import db_pool

def normalize_medical_text(text):
    if text is None:
        return ""

    if isinstance(text, list):
        return "\n".join(str(item) for item in text)

    return str(text)


def fetch_patient_history(patient_id):
    patient_id = str(patient_id).strip()
    conn = db_pool.getconn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    de.patient_id AS pid,
                    de.visit_id AS vid,
                    de.clean_text AS info,
                    v.created_at,
                    pe.provisional_diagnosis,
                    pe.confirmed_diagnosis
                FROM document_extraction de
                LEFT JOIN visit v
                    ON v.id = de.visit_id
                LEFT JOIN prescription pe
                    ON pe.visit_id = de.visit_id
                WHERE de.clean_text IS NOT NULL
                    AND de.patient_id = %s
                ORDER BY v.created_at ASC NULLS LAST
                """,
                (patient_id,)
            )

            rows = cur.fetchall()

        visits = {}

        for pid, vid, info, created_at, provisional_diagnosis, confirmed_diagnosis in rows:
            visit_id = str(vid)

            if visit_id not in visits:
                visits[visit_id] = {
                    "visit_id": visit_id,
                    "visit_date": created_at.isoformat() if created_at else None,
                    "provisional_diagnosis": None,
                    "confirmed_diagnosis": None,
                    "medical_texts": []
                }

            if provisional_diagnosis is not None:
                visits[visit_id]["provisional_diagnosis"] = provisional_diagnosis

            if confirmed_diagnosis is not None:
                visits[visit_id]["confirmed_diagnosis"] = confirmed_diagnosis

            if info:
                visits[visit_id]["medical_texts"].append(normalize_medical_text(info))

        formatted_visits = []

        for visit in visits.values():
            formatted_visits.append({
                "visit_id": visit["visit_id"],
                "visit_date": visit["visit_date"],
                "provisional_diagnosis": visit["provisional_diagnosis"],
                "confirmed_diagnosis": visit["confirmed_diagnosis"],
                "medical_text": "\n\n".join(visit["medical_texts"])
            })

        return {
            "patient_id": str(patient_id),
            "total_visits": len(formatted_visits),
            "visits": formatted_visits
        }

    finally:
        db_pool.putconn(conn)

def fetch_patient_documents(patient_id):
    """
    Same join as fetch_patient_history(), but keeps each document
    separate (document_id preserved) instead of merging clean_text
    across documents within a visit. Use this with
    longitudinal.adapters.adapt_from_detailed_history() for full
    document-level traceability in the clinical NLP layer.
    """
    patient_id = str(patient_id).strip()
    conn = db_pool.getconn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    de.patient_id AS pid,
                    de.visit_id AS vid,
                    de.id AS document_id,
                    de.document_type AS document_type,
                    de.clean_text AS info,
                    v.created_at,
                    pe.provisional_diagnosis,
                    pe.confirmed_diagnosis
                FROM document_extraction de
                LEFT JOIN visit v
                    ON v.id = de.visit_id
                LEFT JOIN prescription pe
                    ON pe.visit_id = de.visit_id
                WHERE de.clean_text IS NOT NULL
                    AND de.patient_id = %s
                ORDER BY v.created_at ASC NULLS LAST
                """,
                (patient_id,)
            )
            rows = cur.fetchall()

        visits = {}

        for (pid, vid, document_id, document_type, info, created_at,
             provisional_diagnosis, confirmed_diagnosis) in rows:
            visit_id = str(vid)

            if visit_id not in visits:
                visits[visit_id] = {
                    "visit_id": visit_id,
                    "visit_date": created_at.isoformat() if created_at else None,
                    "provisional_diagnosis": None,
                    "confirmed_diagnosis": None,
                    "documents": []
                }

            if provisional_diagnosis is not None:
                visits[visit_id]["provisional_diagnosis"] = provisional_diagnosis
            if confirmed_diagnosis is not None:
                visits[visit_id]["confirmed_diagnosis"] = confirmed_diagnosis

            if info:
                visits[visit_id]["documents"].append({
                    "document_id": str(document_id) if document_id is not None else None,
                    "document_type": document_type,
                    "clean_text": normalize_medical_text(info),
                })

        return {
            "patient_id": str(patient_id),
            "total_visits": len(visits),
            "visits": list(visits.values()),
        }

    finally:
        db_pool.putconn(conn)
