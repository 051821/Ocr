#!/usr/bin/env python3
"""Run the brief checks (brief_checks.py) over a random sample of patients and report the failure rate.

Each patient goes through the same pipeline as the dashboard (services.patient_context + insight_view.build_view).
The AI brief is the cached review when it is current; no new LLM review is generated unless --generate is given.
The report holds patient IDs and the checks that failed, never names; it is written to ../output/ by default.

Usage (from dataanalysis/):
  python scripts/audit_sample.py                       # 50 random patients
  python scripts/audit_sample.py --patients 100 --generate
  python scripts/audit_sample.py --ids <id> <id> ...   # specific patients
  python scripts/audit_sample.py --fail-over 0.05      # exit 1 when > 5% of patients have an error (for CI / cron)
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR, PROJECT_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import brief_checks  # noqa: E402
import insight_view  # noqa: E402
from core.fetching import fetch_patient_history  # noqa: E402
from services.patient_context import prepare_patient_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patients", type=int, default=50)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--generate", action="store_true", help="generate an AI review when none is current")
    parser.add_argument("--fail-over", type=float, default=None, help="exit 1 above this share of patients with errors")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.ids:
        ids = args.ids
    else:
        sys.path.insert(0, str(PROJECT_DIR / "scripts"))
        from export_analysis_csv import candidate_ids
        ids = candidate_ids(args.patients)[: args.patients]

    from services.review_service import get_or_generate_review, is_review_current
    rows, checked = [], 0
    for pid in ids:
        try:
            patient = prepare_patient_data(fetch_patient_history(pid))
            if not patient.get("visits"):
                continue
            review = get_or_generate_review(patient) if (args.generate or is_review_current(patient)) else None
            view = insight_view.build_view(patient, review)["patient"]
        except Exception as exc:  # one bad record never stops the audit
            rows.append({"patient_id": pid, "error": f"pipeline failed: {type(exc).__name__}"})
            continue
        checked += 1
        issues = view.get("brief_checks") or []
        rows.append({"patient_id": pid, "ai_review": bool(review), "summary": brief_checks.summarise(issues),
                     "issues": [{k: i[k] for k in ("check", "severity", "theme", "detail")} for i in issues]})

    with_errors = [r for r in rows if r.get("error") or (r.get("summary") or {}).get("errors")]
    by_check = {}
    for r in rows:
        for k, n in ((r.get("summary") or {}).get("by_check") or {}).items():
            by_check[k] = by_check.get(k, 0) + n
    rate = len(with_errors) / max(len(rows), 1)
    report = {"run_at": datetime.datetime.now().isoformat(timespec="seconds"), "patients": len(rows),
              "checked": checked, "patients_with_errors": len(with_errors), "error_rate": round(rate, 3),
              "by_check": by_check, "rows": rows}
    out = Path(args.out) if args.out else PROJECT_DIR.parent / "output" / f"brief_audit_{datetime.date.today()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"{checked} patients checked, {len(with_errors)} with errors ({rate:.0%}); by check: {by_check or 'none'}")
    print(f"report: {out}")
    if args.fail_over is not None and rate > args.fail_over:
        sys.exit(1)


if __name__ == "__main__":
    main()
