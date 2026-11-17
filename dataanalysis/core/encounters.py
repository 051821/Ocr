"""Encounter helpers shared by the snapshot rules and the review evidence.

Clinical days: a patient often sees two departments on one day, which the record stores as two encounters. Rules that
look for duplicate medicines, and visit counts shown to the doctor, work on clinical days (all encounters on one
date) so a same-day pair is neither missed as a duplicate nor counted as two visits. Encounter numbers (#n) are kept
as they are, so every citation still points at a real record entry.

Zero doses: a line written 0-0-0 WITH a duration is a prescription whose dosing was not entered (common for topicals and
SOS medicines): it is prescribed and current for its course, shown as "dose not specified". A 0-0-0 line with no duration
states nothing and is reported as unclear, never as an ongoing medicine.

Course window: a medicine is active from its prescription date to that date + its stated duration (inclusive); a line
with no duration is only known to be active on the day it was written.
"""
import re
from datetime import date as _date, timedelta

_ZERO_DOSE = re.compile(r"\s*0+(?:\.0+)?\s*(?:-\s*0+(?:\.0+)?\s*){2,3}")


def visit_date(visit):
    return str(visit.get("visit_date") or "")[:10]


def clinical_days(visits):
    """[(date, [encounter numbers])] in order; encounters without a date each stand alone."""
    days, index = [], {}
    for i, v in enumerate(visits, 1):
        date = visit_date(v)
        if date and date in index:
            days[index[date]][1].append(i)
        else:
            if date:
                index[date] = len(days)
            days.append((date, [i]))
    return days


def day_of_encounter(visits):
    """{encounter number: clinical-day number (1-based)}."""
    return {i: d for d, (_, encs) in enumerate(clinical_days(visits), 1) for i in encs}


def is_zero_dose(med):
    """True for a prescription line whose dosage is written 0-0-0 (or 0-0-0-0): no dose stated."""
    return bool(_ZERO_DOSE.fullmatch(str((med or {}).get("dosage") or "")))


# 'Ct. T Eltroxin', 'Cont. Amlo', 'Continue Metformin': the doctor continues the medicine (its dose is the one in use).
_CONTINUED = re.compile(r"(?i)^\s*(?:ct|cont|contd|continue)\b\.?")


def dose_not_specified(med):
    """0-0-0 with a stated duration, or written as continued: prescribed, dosing not entered (topical / SOS /
    'continue same')."""
    return is_zero_dose(med) and (duration_days(med) is not None or bool(_CONTINUED.match(str(med.get("name") or ""))))


def unclear_dose(med):
    """0-0-0 with no duration (and not written as continued): the line states neither a dose nor a course."""
    return is_zero_dose(med) and not dose_not_specified(med)


def taken_medications(visit):
    """The visit's prescription lines: every line except 0-0-0 lines with no duration (those state nothing)."""
    return [m for m in visit.get("db_medications") or [] if m.get("name") and not unclear_dose(m)]


def zero_dose_lines(visits):
    """['Vaseline 0-0-0 (#4)', ...]: 0-0-0 lines with no duration, for the record-quality notes."""
    return list(dict.fromkeys(f"{str(m.get('name')).strip()} {m.get('dosage')} (#{i})"
                              for i, v in enumerate(visits, 1) for m in v.get("db_medications") or []
                              if m.get("name") and unclear_dose(m)))


_DURATION = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(day|days|d|week|weeks|wk|wks|month|months|mo)\b")


def duration_days(med):
    """Days stated in a prescription line's duration ('5 days', '2 weeks', '1 month'), else None."""
    match = _DURATION.search(str((med or {}).get("duration") or ""))
    if not match:
        return None
    n, unit = float(match.group(1)), match.group(2).casefold()
    return n * (7 if unit.startswith("w") else 30 if unit.startswith("mo") else 1)


def course_end(start_iso, med):
    """ISO date the course written on `start_iso` runs to (start + duration), or None when no duration is stated."""
    days = duration_days(med)
    try:
        return (_date.fromisoformat(str(start_iso)[:10]) + timedelta(days=int(days))).isoformat() if days else None
    except ValueError:
        return None


ATTENDED_OFF_DATE = "ATTENDED_OFF_DATE"


def _tolerance_days():
    import json
    import os
    from pathlib import Path
    env = os.getenv("FOLLOWUP_TOLERANCE_DAYS")
    if env and env.strip().isdigit():
        return int(env)
    path = Path(__file__).resolve().parents[1] / "data" / "reference" / "clinical_thresholds.json"
    return json.loads(path.read_text(encoding="utf-8"))["followup_recheck"]["attended_within_days"]


def recheck_followups(visits, tolerance_days=None):
    """Never trust a source MISSED as-is: a follow-up marked MISSED becomes ATTENDED_OFF_DATE when the patient had any
    other visit within +/- tolerance_days of its due date. The source status is kept in source_status. In place and
    idempotent; returns the visits."""
    tol = _tolerance_days() if tolerance_days is None else tolerance_days
    dates = [(i, visit_date(v)) for i, v in enumerate(visits or [], 1)]
    # Due dates some follow-up (any encounter) records as completed: one appointment, not a miss.
    completed = {}
    for i, v in enumerate(visits or [], 1):
        for f in v.get("followups") or []:
            if "COMPLETED" in str(f.get("status") or "").upper() and f.get("scheduled_date"):
                completed.setdefault(str(f["scheduled_date"])[:10], i)
    for i, v in enumerate(visits or [], 1):
        for f in v.get("followups") or []:
            status = str(f.get("source_status") or f.get("status") or "").upper()
            if "MISSED" not in status:
                continue
            due = str(f.get("scheduled_date") or "")[:10]
            try:
                due_day = _date.fromisoformat(due)
            except ValueError:
                continue  # no usable due date: the source status stands
            # Only a later day attends it: another encounter on the booking day (a second department) is not the review.
            booked = visit_date(v)
            near = [(j, d) for j, d in dates if j != i and d and (not booked or d > booked)
                    and abs((_date.fromisoformat(d) - due_day).days) <= tol]
            f.setdefault("source_status", f.get("status"))
            if due in completed:
                f["status"] = ATTENDED_OFF_DATE
                f["status_label"] = f"attended: the same due date ({due}) is completed under #{completed[due]}"
            elif near:
                j, d = min(near, key=lambda x: abs((_date.fromisoformat(x[1]) - due_day).days))
                f["status"] = ATTENDED_OFF_DATE
                f["status_label"] = f"attended (off-date): visit #{j} on {d}, due {due}"
            else:
                f["status"] = f["source_status"]
                f["status_label"] = f"missed: no visit within {tol} days of {due}"
    return visits


# ── Prescriber specialization: one name per specialty (spelling variants merged) ─────────────────────────────────
def specialization_key(name):
    """What makes two spellings one specialty: letters only, British / American spelling folded (ae, oe -> e), and the
    -ics / -ic / -ists / -ist / -ians / -ian / -y / -s ending dropped. 'Orthopaedics', 'Orthopedic' -> 'orthoped';
    'Cardiology', 'Cardiologist' -> 'cardiolog'. Derived from the text; no specialty is listed."""
    words = re.findall(r"[a-z]+", str(name or "").casefold().replace("ae", "e").replace("oe", "e"))
    words = [w for w in words if w not in ("and", "of", "the", "dept", "department")]
    def stem(w):  # repeated: 'pediatrician' -> 'pediatric' -> 'pediatr', like 'pediatrics' -> 'pediatr'
        while True:
            s = re.sub(r"(?:ics|ic|ists|ist|ians|ian|y|s)$", "", w)
            if s == w or len(s) < 4:
                return w
            w = s
    return " ".join(stem(w) for w in words)


def unify_specializations(visits):
    """In place: every spelling of one specialty in this patient's record becomes its most frequent spelling (ties:
    the first seen), on each prescription line and on each visit's list, which is de-duplicated."""
    counts, first = {}, {}
    for v in visits or []:
        for m in v.get("db_medications") or []:
            s = (m.get("prescriber_specialization") or "").strip()
            if s:
                counts[s] = counts.get(s, 0) + 1
        for s in v.get("prescriber_specializations") or []:
            first.setdefault(s, len(first))
            counts.setdefault(s, 1)
    best = {}
    for s in sorted(counts, key=lambda s: (-counts[s], first.get(s, 1 << 30))):
        best.setdefault(specialization_key(s), s)
    name = lambda s: best.get(specialization_key(s), s) if s else s
    for v in visits or []:
        for m in v.get("db_medications") or []:
            if m.get("prescriber_specialization"):
                m["prescriber_specialization"] = name(m["prescriber_specialization"].strip())
        if v.get("prescriber_specializations"):
            v["prescriber_specializations"] = list(dict.fromkeys(name(s) for s in v["prescriber_specializations"]))
    return visits
