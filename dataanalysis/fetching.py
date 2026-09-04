import re
import config
import psycopg2.pool


db_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=config.DB_POOL_MIN,
    maxconn=config.DB_POOL_MAX,
    dsn=config.DATABASE_URL
)

def fetch_batch(limit=2, offset=0):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            query = """
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

                ORDER BY de.visit_id

                LIMIT %s OFFSET %s
            """

            cur.execute(query, (limit, offset))

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

            return [
                dict(zip(columns, row))
                for row in rows
            ]

    finally:
        db_pool.putconn(conn)


import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.db_writer import split_text_lines


def clean_text(text):
    """Clean and filter text for AI medical analysis using the unified split_text_lines."""
    if not text:
        return []
    _, medical_lines = split_text_lines(text)
    return medical_lines


# ============================================================
# PREPARE DATA FOR AI ANALYSIS
# ============================================================

def prepare_medical_records(rows):

    medical_records = []

    for row in rows:

        cleaned_ocr_text = clean_text(row["info"])

        record = {
            "visit_id": row["vid"],

            "visit_date": (
                str(row["created_at"])
                if row["created_at"]
                else None
            ),

            "chief_complaint": row["chief_complaint"],

            "medical_document_text": cleaned_ocr_text
        }

        medical_records.append(record)

    return medical_records


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:
        # Fetch records
        rows = fetch_batch(limit=2)

        print("\nFetched Records:")
        print("=" * 70)

        # Clean and prepare records
        medical_records = prepare_medical_records(rows)

        for index, record in enumerate(medical_records, start=1):

            print(f"\nRECORD {index}")
            print("-" * 70)
            print(f"Visit ID: {record['visit_id']}")
            print(f"Visit Date: {record['visit_date']}")
            print(f"Chief Complaint: {record['chief_complaint']}")

            print("\nCleaned Medical OCR Text:")
            print(record["medical_document_text"])

            print("-" * 70)

    finally:
        # Close all connections when the script finishes
        db_pool.closeall()
