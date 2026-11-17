#!/usr/bin/env python3
"""Freeze audited patients as golden regression fixtures (tests/golden/), or re-accept fixtures after a deliberate change.

Only freeze a patient after checking its outputs against the source record; the fixture then pins that output.

Usage (from dataanalysis/):
  python scripts/freeze_golden.py <patient_id> ...            # fetch, de-identify, record current outputs
  python scripts/freeze_golden.py --ids-file ids.txt
  python scripts/freeze_golden.py --update                    # recompute 'expected' for every fixture (after review)
  python scripts/freeze_golden.py --update tests/golden/gXXXX.json
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

# Fixture records never reach tracing or the unmapped-brands report.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.setdefault("UNMAPPED_BRANDS_REPORT", "off")
os.environ.setdefault("COVERAGE_REPORT", "off")

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR, PROJECT_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evals import golden  # noqa: E402

GOLDEN_DIR = PROJECT_DIR / "tests" / "golden"


def _write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def freeze(patient_id):
    import clinical_terms
    from core.fetching import fetch_patient_history
    record = fetch_patient_history(patient_id)
    terms = clinical_terms.resolve_terms(copy.deepcopy(record))  # the term resolutions the dashboard uses (cached)
    fixture = {"input": golden.deidentify(record), "terms": terms}
    fixture["expected"] = golden.outputs(fixture["input"], fixture["terms"])
    path = GOLDEN_DIR / f"{golden.fixture_id(patient_id)}.json"
    _write(path, fixture)
    return path


def update(paths):
    for path in paths:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        fixture["expected"] = golden.outputs(fixture["input"], fixture["terms"])
        _write(path, fixture)
        print("updated", path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="*", help="patient IDs (freeze) or fixture paths (--update)")
    parser.add_argument("--ids-file")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    if args.update:
        update([Path(t) for t in args.targets] or sorted(GOLDEN_DIR.glob("*.json")))
        return
    ids = list(args.targets) + (Path(args.ids_file).read_text().split() if args.ids_file else [])
    for pid in ids:
        try:
            print("frozen", freeze(pid).name)
        except Exception as exc:
            print(f"skip {golden.fixture_id(pid)}: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
