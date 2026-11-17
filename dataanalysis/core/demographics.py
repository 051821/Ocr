"""Patient demographics read: age and sex straight from the patient table, no record load."""
from core.db import db_pool


def fetch_demographics(patient_id):
    """(age_years, sex) for the patient, or None when no patient row has this id."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT age_years, sex FROM patient WHERE id = %s::uuid", (patient_id,))
            return cur.fetchone()
    finally:
        db_pool.putconn(conn)
