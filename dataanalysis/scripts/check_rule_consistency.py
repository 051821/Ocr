#!/usr/bin/env python3
"""Run every snapshot rule over patients, re-run each under equivalent inputs, and report inconsistent rules.

Usage:
  python scripts/check_rule_consistency.py --ids <id> <id> ...      # given patients (from the database)
  python scripts/check_rule_consistency.py --all [--limit N]        # every patient in the database
  python scripts/check_rule_consistency.py --pickle frozen.pkl      # {id: {"data": patient_data}} saved earlier
Writes JSON to stdout (or --out). Exit code 1 when any rule is inconsistent. Patients are labelled by a short hash.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import rule_consistency  # noqa: E402


def _label(pid):
    return hashlib.blake2b(str(pid).encode(), digest_size=4).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pickle")
    ap.add_argument("--out")
    args = ap.parse_args()
    patients = []
    if args.pickle:
        patients = [(_label(pid), item["data"]) for pid, item in pickle.load(open(args.pickle, "rb")).items()]
    else:
        from core.fetching import fetch_all_patient_ids, fetch_patient_history
        from services.patient_context import prepare_patient_data
        ids = args.ids or (fetch_all_patient_ids() if args.all else [])
        for pid in ids[:args.limit] if args.limit else ids:
            patients.append((_label(pid), prepare_patient_data(fetch_patient_history(pid))))
    report = rule_consistency.check_all(patients)
    report["patients_checked"] = len(patients)
    text = json.dumps(report, ensure_ascii=False, indent=1)
    Path(args.out).write_text(text, encoding="utf-8") if args.out else print(text)
    return 1 if report["inconsistent"] else 0


if __name__ == "__main__":
    sys.exit(main())
