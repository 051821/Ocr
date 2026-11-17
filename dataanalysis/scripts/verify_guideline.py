"""Record a clinician's verification of a guideline section in condition_registry.json.

A verified section raises flags and reaches the AI review and brief; a draft one is shown to the doctor and in
/summary as information only. Verification needs the clinician's name and a date.

    python scripts/verify_guideline.py --list
    python scripts/verify_guideline.py --show diabetes_control      # the values being signed off
    python scripts/verify_guideline.py diabetes_control --by "Dr A. Rao" [--on 2026-11-06]
    python scripts/verify_guideline.py diabetes_control --revoke

Verifying or revoking changes the review prompt version, so cached AI reviews regenerate.
"""
import argparse
import datetime
import json
import re
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "data" / "reference" / "condition_registry.json"
_FIELDS = re.compile(r'"verified":\s*(?:true|false),\s*"verified_by":\s*(?:null|"(?:[^"\\]|\\.)*"),\s*'
                     r'"verified_on":\s*(?:null|"[^"]*")')


def _sections(data):
    return {k: v for k, v in data["guidelines"].items() if isinstance(v, dict)}


def _set(text, section, verified, by, on):
    """Rewrite only this section's three verification fields, keeping the file's layout."""
    start = re.search(r'\n    "' + re.escape(section) + r'":\s*\{', text)
    match = _FIELDS.search(text, start.end()) if start else None
    if not match:
        sys.exit(f"Could not find the verification fields of '{section}'.")
    fields = (f'"verified": {json.dumps(verified)}, "verified_by": {json.dumps(by)}, '
              f'"verified_on": {json.dumps(on)}')
    return text[:match.start()] + fields + text[match.end():]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("section", nargs="?")
    parser.add_argument("--by", help="clinician who checked the section's values against the source")
    parser.add_argument("--on", default=datetime.date.today().isoformat(), help="date verified (YYYY-MM-DD)")
    parser.add_argument("--revoke", action="store_true", help="return the section to draft")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show", metavar="SECTION")
    args = parser.parse_args()

    text = REGISTRY.read_text(encoding="utf-8")
    sections = _sections(json.loads(text))
    if args.list:
        for name, rule in sections.items():
            state = (f"verified by {rule['verified_by']} on {rule['verified_on']}"
                     if rule.get("verified") and rule.get("verified_by") and rule.get("verified_on") else "DRAFT")
            print(f"{name:18} {state:40} {rule.get('source') or ''}")
        return
    if args.show:
        if args.show not in sections:
            sys.exit(f"Unknown section '{args.show}'. Sections: {', '.join(sections)}")
        print(json.dumps(sections[args.show], indent=2, ensure_ascii=False))
        return
    if args.section not in sections:
        sys.exit(f"Unknown section '{args.section}'. Sections: {', '.join(sections)}")
    if args.revoke:
        text = _set(text, args.section, False, None, None)
    else:
        if not (args.by or "").strip():
            sys.exit("--by is required: the clinician who checked the values (see --show).")
        try:
            datetime.date.fromisoformat(args.on)
        except ValueError:
            sys.exit("--on must be a date, YYYY-MM-DD.")
        text = _set(text, args.section, True, args.by.strip(), args.on)
    rule = _sections(json.loads(text))[args.section]  # the edited file must still parse
    REGISTRY.write_text(text, encoding="utf-8")
    print(f"{args.section}: " + (f"verified by {rule['verified_by']} on {rule['verified_on']}" if rule["verified"]
                                 else "returned to draft"))


if __name__ == "__main__":
    main()
