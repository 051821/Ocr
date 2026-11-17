"""Data-quality layer: runs before every rule.

Corrections applied to the record in place (idempotent):
  - follow-up status re-checked against the actual visits (core.encounters.recheck_followups);
  - duplicate visit entries (same date, complaint, diagnosis and prescription) marked with "_duplicate_of".
Everything the snapshot reports as a record problem (implausible vitals, unit slips, weight / height entry errors,
OCR misfiles, unverified or unevaluated labs, document mismatches, follow-ups attended off-date) is a data-quality
item: level "info", category "data quality", kind "data", never a clinical finding (finalise()).
"""
import re

from core.encounters import recheck_followups, visit_date

LEVEL = "info"
CATEGORY = "data quality"
# Snapshot flags about the record rather than the patient, besides kind "data".
_RECORD_TITLES = re.compile(r"(?i)need checking against the source|lab result\(s\) not evaluated|read as lab results|"
                            r"not linked to a visit|entry error|attended off-date|duplicate visit|conflicting")


def _signature(v):
    meds = tuple(sorted(str(m.get("name") or "").strip().casefold() for m in v.get("db_medications") or []))
    text = lambda k: " ".join(str(v.get(k) or "").casefold().split())
    return (visit_date(v), text("db_chief_complaint"), text("db_provisional_dx"), text("db_confirmed_dx"), meds)


def duplicate_visits(visits):
    """[(duplicate encounter, original encounter)]: a later entry identical in date, complaint, diagnosis and medicines."""
    seen, dups = {}, []
    for i, v in enumerate(visits or [], 1):
        sig = _signature(v)
        if not sig[0] or not (sig[1] or sig[4]):
            continue
        if sig in seen:
            dups.append((i, seen[sig]))
            v["_duplicate_of"] = seen[sig]
        else:
            seen[sig] = i
    return dups


_DISCREPANCY = re.compile(r"(Document \d+):\s*(.+?) — structured record shows\s*(\d+(?:\.\d+)?).*?document shows\s*(\d+(?:\.\d+)?)")


def _tolerance():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent / "data" / "reference" / "clinical_thresholds.json"
    return json.loads(path.read_text(encoding="utf-8"))["conflicting_values"]


def conflicting_values(visits):
    """Move vitals the structured record and a document give differently out of db_vitals_rows into
    'conflicting_vitals' (so no rule or test suggestion reads them). Idempotent; returns [(encounter, label, record,
    document)]."""
    tol, found = _tolerance(), []
    for i, v in enumerate(visits or [], 1):
        for text in v.get("discrepancies") or []:
            m = _DISCREPANCY.match(str(text))
            if not m:
                continue
            label, rec, doc = m.group(2).strip(), float(m.group(3)), float(m.group(4))
            key = "SpO2" if label.casefold().startswith("spo") else label
            limit = tol["absolute"].get(key)
            if (abs(rec - doc) <= limit) if limit is not None else (max(rec, doc) and abs(rec - doc) / max(rec, doc) * 100 <= tol["tolerance_pct"]):
                continue
            rows = v.get("db_vitals_rows") or []
            moved = [r for r in rows if str(r.get("label") or "").casefold() == label.casefold()]
            if moved:
                v["db_vitals_rows"] = [r for r in rows if r not in moved]
                v.setdefault("conflicting_vitals", []).extend({**r, "document_value": doc} for r in moved)
            if any(str(r.get("label") or "").casefold() == label.casefold() for r in v.get("conflicting_vitals") or []):
                found.append((i, label, rec, doc))
    return found


# Vitals that are lab measurements (point-of-care glucose, Hb): read as the lab test they are, so the control,
# monitoring and guideline rules see them ("no glucose in the record" when the vitals hold 260 mg/dL is wrong).
_VITAL_LABS = {"fasting sugar": "Fasting Blood Sugar", "non-fasting sugar": "Random Blood Sugar",
               "sugar": "Random Blood Sugar", "hb": "Haemoglobin"}
VITALS_NOTE = "Recorded with the vitals at this visit (point-of-care reading)."


def vitals_as_labs(visits, age_sex=None):
    """Add each glucose / Hb vital as a validated lab result of its visit (marked '_from_vitals'), unless the visit's
    lab results already hold the same test with the same value. Idempotent; returns the number added."""
    import lab_validation
    sex, added = lab_validation.sex_from(age_sex), 0
    for v in visits or []:
        labs = v.get("lab_results") or []
        for row in v.get("db_vitals_rows") or []:
            name = _VITAL_LABS.get(str(row.get("label") or "").strip().casefold())
            if not name or row.get("value") in (None, "") or any(l.get("_from_vitals") == row.get("label") for l in labs):
                continue
            number = lab_validation.parse_value(row.get("value"))[1]
            if not number or number <= 0:
                continue  # an empty field saved as 0 is no reading
            key = lab_validation.identify(name)
            same = any(lab_validation.identify(l.get("test_name")) in (key, None)
                       and lab_validation.parse_value(l.get("value"))[1] == number for l in labs)
            if same:
                continue
            lab = lab_validation.validate_lab({"test_name": name, "value": str(row.get("value")), "reference": ""}, sex)
            note = " ".join(x for x in (lab.get("note"), VITALS_NOTE) if x)
            labs.append({**lab, "note": note, "source": "vitals", "_from_vitals": row.get("label")})
            added += 1
        if labs:
            v["lab_results"] = labs
    return added


# Advice written after a medicine on the same line ('Ct. T Eltroxin 50mcg, Review With Report'): not part of its name.
_TRAILING_ADVICE = re.compile(r"(?i)\s*[,;]\s*(?:review|report|advice|advised|follow[\s-]*up|f/u|continue\s+same)\b.*$")


def split_trailing_advice(visits):
    """Cut trailing advice off a prescription line's name (the original stays in 'raw_name'). Idempotent; returns the
    number of lines changed."""
    changed = 0
    for v in visits or []:
        for m in v.get("db_medications") or []:
            name = str(m.get("name") or "")
            cut = _TRAILING_ADVICE.sub("", name).strip()
            if cut and cut != name.strip():
                m.setdefault("raw_name", name)
                m["name"] = cut
                changed += 1
    return changed


def prepare(patient_data):
    """Apply the corrections; returns the visits."""
    visits = recheck_followups(patient_data.get("visits") or [])
    split_trailing_advice(visits)
    duplicate_visits(visits)
    conflicting_values(visits)
    vitals_as_labs(visits, patient_data.get("age_sex"))
    return visits


def finalise(flags):
    """Every record-quality flag becomes kind 'data', level 'info', category 'data quality'. Returns the flags."""
    for f in flags:
        if f.get("kind") == "data" or _RECORD_TITLES.search(str(f.get("title") or "")):
            f["kind"], f["level"], f["category"] = "data", LEVEL, CATEGORY
    return flags
