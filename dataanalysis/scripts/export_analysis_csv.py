#!/usr/bin/env python3
"""Export one CSV row per visit for a mixed set of patients, for offline analysis.

Patients are drawn in turn from three random pools (has a lab report, has a
long-term condition, 2+ diagnosed visits) and kept at most 2 per patient type,
so the export covers many body systems. Every column comes from the same
pipeline the dashboard uses. Lab results that failed validation are left out.

Usage:
  python scripts/export_analysis_csv.py                 # 12 patients -> ../output/patient_analysis.csv
  python scripts/export_analysis_csv.py --patients 15 --out path.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import clinical_snapshot
import patient_types
from core.db import db_pool
from lab_validation import UNVERIFIED
from services.patient_context import prepare_patient_data
from core.fetching import fetch_patient_history

COLUMNS = [
    "patient_id", "age_sex", "patient_type", "patient_type_reason", "exercise_fit", "exercise_reason",
    "visit_id", "visit_date", "chief_complaint", "provisional_diagnosis", "confirmed_diagnosis",
    "visit_cluster", "visit_exercise_fit", "visit_exercise_reason", "lab_tests_json",
    "long_term_conditions_json", "health_history_json", "clinical_snapshot_json",
]


# Three random pools, interleaved: a plain random sample rarely has a lab report
# or a long-term condition, so those get pools of their own.
_POOLS = {
    "labs": """
        SELECT de.patient_id::text FROM document_extraction de
        WHERE de.visit_id IS NOT NULL
          AND de.clean_text::text ~* '(haemoglobin|hemoglobin|hba1c|glucose|creatinine|tsh|cholesterol)'
        GROUP BY de.patient_id ORDER BY random() LIMIT %s""",
    "long_term": """
        SELECT pc.patient_id::text FROM patient_condition pc
        WHERE pc.deleted_at IS NULL
        GROUP BY pc.patient_id ORDER BY random() LIMIT %s""",
    "random": """
        SELECT v.patient_id::text FROM visit v JOIN prescription pe ON pe.visit_id = v.id
        WHERE v.patient_id IS NOT NULL
          AND COALESCE(pe.provisional_diagnosis, pe.confirmed_diagnosis, '') <> ''
        GROUP BY v.patient_id HAVING COUNT(DISTINCT v.id) >= 2
        ORDER BY random() LIMIT %s""",
}


def candidate_ids(limit):
    """Patient ids taken in turn from the lab, long-term-condition and random pools."""
    conn = db_pool.getconn()
    try:
        pools = []
        with conn.cursor() as cur:
            for sql in _POOLS.values():
                cur.execute(sql, (limit // len(_POOLS) + 1,))
                pools.append([r[0] for r in cur.fetchall()])
    finally:
        db_pool.putconn(conn)
    ids = []
    for group in zip(*pools):
        ids += [pid for pid in group if pid not in ids]
    return ids


def clean_labs(labs):
    """Labs that passed validation; implausible / unreadable values are left out."""
    return [{"test": lab.get("analyte") or lab.get("test_name"), "value": lab.get("value"),
             "reference": lab.get("reference"), "status": lab.get("status")}
            for lab in labs or [] if lab.get("status") != UNVERIFIED and lab.get("value") not in (None, "")]


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=lambda o: sorted(o) if isinstance(o, set) else str(o))


def patient_rows(patient_data):
    summary = patient_types.exercise_summary(patient_data)
    snapshot = clinical_snapshot.build_snapshot(patient_data) or {}
    snapshot.pop("dates", None)
    shared = {
        "patient_id": patient_data["patient_id"],
        "age_sex": patient_data.get("age_sex") or "",
        "patient_type": summary["patient_type"],
        "patient_type_reason": summary["type_reason"],
        "exercise_fit": summary["exercise"],
        "exercise_reason": summary["reason"],
        "long_term_conditions_json": _json(patient_data.get("long_term_conditions") or []),
        "health_history_json": _json(patient_data.get("health_history_present") or []),
        "clinical_snapshot_json": _json(snapshot),
    }
    clusters = {r["visit_id"]: r for r in patient_types.classify_visits(patient_data)}
    rows = []
    for v in patient_data.get("visits") or []:
        c = clusters.get(str(v.get("visit_id", ""))) or {}
        labs = clean_labs(v.get("lab_results"))
        rows.append({**shared,
                     "visit_id": v.get("visit_id"),
                     "visit_date": (v.get("visit_date") or "")[:10],
                     "chief_complaint": v.get("chief_complaint") or "",
                     "provisional_diagnosis": v.get("provisional_diagnosis") or "",
                     "confirmed_diagnosis": v.get("confirmed_diagnosis") or "",
                     "visit_cluster": c.get("cluster", ""),
                     "visit_exercise_fit": c.get("exercise", ""),
                     "visit_exercise_reason": c.get("exercise_reason", ""),
                     "lab_tests_json": _json(labs) if labs else ""})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patients", type=int, default=12)
    parser.add_argument("--scan", type=int, default=300, help="max random patients to look at")
    parser.add_argument("--out", default=str(PROJECT_DIR.parent / "output" / "patient_analysis.csv"))
    args = parser.parse_args()

    chosen, per_type, rows = [], {}, []
    candidates = candidate_ids(args.scan)
    for pid in candidates:
        if len(chosen) >= args.patients:
            break
        try:
            patient_data = prepare_patient_data(fetch_patient_history(pid))
        except Exception as exc:  # one bad record should not stop the export
            print(f"skip {pid[:8]}: {exc}", file=sys.stderr)
            continue
        ptype = patient_types.exercise_summary(patient_data)["patient_type"]
        # At most 2 per type, and a second only once 6 types are in.
        count = per_type.get(ptype, 0)
        if ptype == patient_types.UNCLASSIFIED or count >= 2 or (count and len(per_type) < 6):
            continue
        per_type[ptype] = count + 1
        chosen.append((pid, ptype))
        rows += patient_rows(patient_data)
        print(f"{len(chosen):>2}. {pid[:8]}  {ptype}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(chosen)} patients, {len(per_type)} types, {len(rows)} visits -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
