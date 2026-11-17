"""Vital flags computed by the clinic app (visit_vital_flag), read as stored."""
from core.db import db_pool


def fetch_recent_vital_flags(patient_id, visits=5):
    """Flags on the patient's last `visits` completed visits, newest first; IMPLAUSIBLE readings are left out."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH recent AS (
                    SELECT id, created_at
                    FROM visit
                    WHERE patient_id = %s::uuid AND status = 'COMPLETED' AND deleted_at IS NULL
                    ORDER BY created_at DESC NULLS LAST
                    LIMIT %s
                )
                SELECT r.id::text, r.created_at, f.metric, f.code, f.severity, f.value
                FROM recent r
                JOIN visit_vital_flag f ON f.visit_id = r.id
                WHERE f.tier IS DISTINCT FROM 'IMPLAUSIBLE'
                ORDER BY r.created_at DESC NULLS LAST, f.metric, f.code
                """,
                (str(patient_id), visits),
            )
            rows = cur.fetchall()
    finally:
        db_pool.putconn(conn)
    return [{"visit_id": vid, "date": created_at.date().isoformat() if created_at else None,
             "metric": metric, "code": code, "severity": severity, "value": value}
            for vid, created_at, metric, code, severity, value in rows]
