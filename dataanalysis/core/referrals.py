"""Referrals out of the clinic for one patient: hospital referral cases plus doctor
recommendations that have no referral case yet. Patient is identified by ID only."""
from core.db import db_pool

_REFERRALS_SQL = """
SELECT
    'HOSPITAL_REFERRAL' AS referral_type,
    rc.id AS referral_id,
    rc.source_visit_id AS visit_id,
    rc.source_prescription_id AS prescription_id,
    rc.referred_on AS referral_date,
    rc.status AS referral_status,
    rc.claim_status,
    h.name AS hospital_name,
    hs.name AS specialty_name,
    rc.referral_reason,
    rc.notes,
    NULL::text AS referral_treatment_name,
    NULL::text AS referral_specialist_name,
    NULL::timestamp AS referral_resolved_at,
    c.name AS center_name,
    rc.expected_review_date,
    rc.closed_at,
    rc.closed_reason
FROM referral_case rc
LEFT JOIN hospital h ON h.id = rc.hospital_id
LEFT JOIN hospital_specialty hs ON hs.id = rc.referred_specialty_id
LEFT JOIN center c ON c.id = rc.center_id
WHERE rc.deleted_at IS NULL
  AND rc.patient_id = %(pid)s::uuid

UNION ALL

SELECT
    'DOCTOR_RECOMMENDATION' AS referral_type,
    NULL::uuid AS referral_id,
    v.id AS visit_id,
    pr.id AS prescription_id,
    v.completed_at AS referral_date,
    NULL::text AS referral_status,
    NULL::text AS claim_status,
    NULL::text AS hospital_name,
    NULL::text AS specialty_name,
    NULL::text AS referral_reason,
    NULL::text AS notes,
    pr.referral_treatment_name,
    pr.referral_specialist_name,
    pr.referral_resolved_at,
    c.name AS center_name,
    NULL::timestamp AS expected_review_date,
    NULL::timestamp AS closed_at,
    NULL::text AS closed_reason
FROM prescription pr
JOIN visit v ON v.id = pr.visit_id
LEFT JOIN center c ON c.id = v.center_id
WHERE pr.is_referral_recommended = TRUE
  AND pr.referral_case_id IS NULL
  AND v.completed_at IS NOT NULL
  AND pr.deleted_at IS NULL
  AND v.deleted_at IS NULL
  AND v.patient_id = %(pid)s::uuid

ORDER BY referral_date DESC
LIMIT 50
"""


def _clean(val):
    return val.strip() if isinstance(val, str) and val.strip() else None


def _summarise(row):
    """Adds the display fields: what the referral is for, where to, and whether it is still open.

    In prescription rows the columns are stored the other way round from their names:
    referral_treatment_name holds the specialty ("Psychiatrist") and
    referral_specialist_name holds the place ("Behavior Healthcare Center")."""
    if row["referral_type"] == "HOSPITAL_REFERRAL":
        row["referred_for"] = _clean(row["specialty_name"]) or _clean(row["referral_reason"])
        row["referred_to"] = _clean(row["hospital_name"])
        status = (row["referral_status"] or "").upper()
        row["is_open"] = row["closed_at"] is None and status not in ("CLOSED", "COMPLETED", "RESOLVED", "CANCELLED")
    else:
        row["referred_for"] = _clean(row["referral_treatment_name"])
        row["referred_to"] = _clean(row["referral_specialist_name"])
        row["is_open"] = row["referral_resolved_at"] is None
    return row


def fetch_referrals(patient_id):
    """Referral rows for the patient, newest first (empty list when there are none)."""
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(_REFERRALS_SQL, {"pid": patient_id})
            cols = [d[0] for d in cur.description]
            return [_summarise(dict(zip(cols, r))) for r in cur.fetchall()]
    finally:
        db_pool.putconn(conn)
