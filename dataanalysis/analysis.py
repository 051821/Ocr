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
                    v.chief_complaint
                FROM document_extraction de
                LEFT JOIN visit v
                    ON v.id = de.visit_id
                WHERE de.clean_text IS NOT NULL
                    AND de.patient_id = %s
                ORDER BY v.created_at ASC NULLS LAST
                """,
                (patient_id,)
            )

            rows = cur.fetchall()

        visits = {}

        for pid, vid, info, created_at, chief_complaint in rows:
            visit_id = str(vid)

            if visit_id not in visits:
                visits[visit_id] = {
                    "visit_id": visit_id,
                    "visit_date": (
                        created_at.isoformat()
                        if created_at
                        else None
                    ),
                    "chief_complaint": chief_complaint,
                    "medical_texts": []
                }

            if info:
                visits[visit_id]["medical_texts"].append(
                    normalize_medical_text(info)
                    )

        formatted_visits = []

        for visit in visits.values():
            formatted_visits.append({
                "visit_id": visit["visit_id"],
                "visit_date": visit["visit_date"],
                "chief_complaint": visit["chief_complaint"],
                "medical_text": "\n\n".join(
                    visit["medical_texts"]
                )
            })

        return {
            "patient_id": str(patient_id),
            "total_visits": len(formatted_visits),
            "visits": formatted_visits
        }

    finally:
        db_pool.putconn(conn)