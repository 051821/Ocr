"""Doctor reads. A doctor is linked per visit (visit.assigned_doctor_user_id -> "user".id), never per patient."""
from core.db import db_pool


def _query(sql, params=()):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        db_pool.putconn(conn)


def fetch_doctors():
    """Doctors assigned to at least one visit, with their name and count of distinct patients, busiest first."""
    rows = _query(
        """
        SELECT u.id::text, NULLIF(trim(concat_ws(' ', u.first_name, u.last_name)), ''),
               count(DISTINCT v.patient_id)
        FROM "user" u
        JOIN visit v ON v.assigned_doctor_user_id = u.id
        WHERE u.role = 'DOCTOR' AND v.deleted_at IS NULL
        GROUP BY u.id
        ORDER BY 3 DESC, 1
        """
    )
    return [{"doctor_id": did, "name": name, "patients": n} for did, name, n in rows]


def fetch_doctor_patients(doctor_id, id_prefix="", limit=200):
    """Patients with 2+ completed, non-deleted visits assigned to this doctor, most recently seen first."""
    rows = _query(
        """
        SELECT v.patient_id::text, count(*), max(v.created_at)
        FROM visit v
        WHERE v.deleted_at IS NULL
          AND v.status = 'COMPLETED'
          AND v.assigned_doctor_user_id = %s
          AND v.patient_id::text LIKE %s
        GROUP BY v.patient_id
        HAVING count(*) >= 2
        ORDER BY 3 DESC
        LIMIT %s
        """,
        (doctor_id, f"{id_prefix}%", limit),
    )
    return [{"patient_id": pid, "visits": n, "last_visit": last.date().isoformat() if last else None}
            for pid, n, last in rows]
