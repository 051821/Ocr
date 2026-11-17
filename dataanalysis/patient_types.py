"""Patient type: groups a patient by body system (Neurological, Musculoskeletal, ...).

Deterministic regex match over each visit's diagnoses and chief complaint; the
vocabulary and the exercise rule live in data/reference/patient_types.json.
A type found in a visit's diagnosis counts more than one found only in the
complaint, which ranks the matched types (per visit and per patient).

Exercise is decided from every matched cluster, not just the top one: each
cluster is 'benefit', 'neutral', 'avoid' or 'contraindicated'. 'avoid' (acute
fever/infection, injury) blocks exercise only at the latest visit, so a past
fever no longer counts; 'contraindicated' (a chronic condition where exercise
is unsafe) blocks it from anywhere in the record; otherwise any 'benefit'
cluster makes exercise beneficial.

The patient type (and a visit's cluster) is the type that decided exercise:
the top-ranked 'avoid' or 'contraindicated' type when one blocks it, the
top-ranked 'benefit' type when exercise is Yes, else the top-ranked type overall.
"""
import json
import re
from pathlib import Path

with (Path(__file__).resolve().parent / "data" / "reference" / "patient_types.json").open(encoding="utf-8") as _f:
    _REF = json.load(_f)

_NEGATION = re.compile(_REF["negation_pattern"], re.I)
TYPES = [(t["name"], re.compile(t["pattern"], re.I)) for t in _REF["types"]]
EXERCISE_RULE = {t["name"]: t.get("exercise", "neutral") for t in _REF["types"]}
UNCLASSIFIED = "Unclassified"
_DX_WEIGHT, _COMPLAINT_WEIGHT = 2, 1


def _text(*values):
    return _NEGATION.sub(" ", " ".join(str(v) for v in values if v))


def _diagnoses(visit):
    return (visit.get("confirmed_diagnosis"), visit.get("provisional_diagnosis"),
            visit.get("db_confirmed_dx"), visit.get("db_provisional_dx"))


def classify_texts(diagnoses, complaint=""):
    """Types matched in one visit -> {name: (weight, matched term)}, in registry order."""
    dx_text, cc_text = _text(*diagnoses), _text(complaint)
    found = {}
    for name, pattern in TYPES:
        hit = pattern.search(dx_text)
        if hit:
            found[name] = (_DX_WEIGHT, hit.group(0).lower())
        elif (hit := pattern.search(cc_text)):
            found[name] = (_COMPLAINT_WEIGHT, hit.group(0).lower())
    return found


def _by_rule(names, rule):
    return [n for n in names if EXERCISE_RULE.get(n) == rule]


def exercise_decision(current, chronic=None):
    """(Yes/No/—, reason, deciding type) from clusters, each list ranked best first.

    current: clusters at the visit being judged (their 'avoid' entries block exercise).
    chronic: clusters to look for a 'benefit' in (defaults to current).
    The deciding type is the top-ranked cluster behind the answer (UNCLASSIFIED if none).
    """
    chronic = list(current) if chronic is None else list(chronic)
    if not current and not chronic:
        return "—", "No cluster matched", UNCLASSIFIED
    avoid = _by_rule(current, "avoid")
    if avoid:
        return "No", f"Acute {', '.join(avoid)}: rest until recovered", avoid[0]
    contraindicated = _by_rule(chronic, "contraindicated")
    if contraindicated:
        return "No", f"Exercise not advised with {', '.join(contraindicated)}", contraindicated[0]
    benefit = _by_rule(chronic, "benefit")
    if benefit:
        return "Yes", f"Benefits: {', '.join(benefit)}", benefit[0]
    return ("No", "No cluster where exercise gives a specific benefit (not harmful either)",
            (chronic or list(current))[0])


def type_basis(name, chronic, where="at the latest visit"):
    """One line on why `name` was chosen as the type, given the ranked clusters it was picked from."""
    rule = EXERCISE_RULE.get(name)
    if name == UNCLASSIFIED:
        return "No body-system type matched the diagnoses or complaints"
    if rule == "avoid":
        return f"Acute {name} {where} makes exercise unsuitable until recovered"
    if rule == "contraindicated":
        return f"{name} is a condition where exercise is not advised"
    if rule == "benefit":
        others = [n for n in _by_rule(chronic, "benefit") if n != name]
        return (f"{name} benefits from exercise"
                + (f"; ranked above {', '.join(others)} by visits/diagnoses" if others else ""))
    return f"{name} is the most frequent type; exercise neither helps nor harms it"


def classify_visits(patient_data):
    """One row per visit: patient_id, visit_id, visit_date, cluster, also_matched, exercise.

    Matches are ranked by weight (diagnosis beats complaint); ties go to the
    earlier entry in patient_types.json. Exercise uses every cluster matched in
    that visit, and the visit's cluster is the one that decided it.
    """
    patient_id = str(patient_data.get("patient_id", ""))
    rows = []
    for v in patient_data.get("visits") or []:
        found = classify_texts(_diagnoses(v), v.get("chief_complaint"))
        ranked = sorted(found, key=lambda n: -found[n][0])
        exercise, reason, cluster = exercise_decision(ranked)
        rows.append({
            "patient_id": patient_id,
            "visit_id": str(v.get("visit_id", "")),
            "visit_date": (v.get("visit_date") or "")[:10],
            "cluster": cluster,
            "also_matched": [n for n in found if n != cluster],
            "terms": [term for _, term in found.values()],
            "exercise": exercise,
            "exercise_reason": reason,
            "type_reason": type_basis(cluster, ranked, where="at this visit"),
        })
    return rows


def _rank_patient(patient_data):
    """Matched types across all visits, highest score first: [{type, score, visits, terms}]."""
    scores = {}
    for v in patient_data.get("visits") or []:
        for name, (weight, term) in classify_texts(_diagnoses(v), v.get("chief_complaint")).items():
            entry = scores.setdefault(name, {"type": name, "score": 0, "visits": 0, "terms": []})
            entry["score"] += weight
            entry["visits"] += 1
            if term not in entry["terms"]:
                entry["terms"].append(term)
    return sorted(scores.values(), key=lambda e: (-e["score"], -e["visits"], e["type"]))


def _decide(patient_data, ranked):
    """(exercise, reason, patient type) for the whole record.

    A 'benefit' cluster anywhere in the record counts (chronic conditions persist);
    an 'avoid' cluster counts only at the latest visit (acute illness passes).
    """
    visits = patient_data.get("visits") or []
    latest = classify_texts(_diagnoses(visits[-1]), visits[-1].get("chief_complaint")) if visits else {}
    order = [t["type"] for t in ranked]
    current = sorted(latest, key=lambda n: order.index(n) if n in order else len(order))
    return exercise_decision(current, order)


def classify_patient(patient_data):
    """Types for a patient, patient type first: [{type, score, visits, terms, primary, exercise_rule}].

    The patient type ('primary') is the type that decided exercise, not simply the
    most frequent one; the rest follow in score order.
    """
    ranked = _rank_patient(patient_data)
    _, _, primary = _decide(patient_data, ranked)
    ranked.sort(key=lambda e: e["type"] != primary)
    for entry in ranked:
        entry["primary"] = entry["type"] == primary
        entry["exercise_rule"] = EXERCISE_RULE.get(entry["type"], "neutral")
    return ranked


def exercise_summary(patient_data):
    """Patient-level answer: {exercise: Yes/No/—, reason, patient_type, type_reason}."""
    ranked = _rank_patient(patient_data)
    exercise, reason, patient_type = _decide(patient_data, ranked)
    type_reason = type_basis(patient_type, [t["type"] for t in ranked])
    entry = next((t for t in ranked if t["type"] == patient_type), None)
    if entry:
        type_reason += (f" (seen in {entry['visits']} visit{'s' if entry['visits'] != 1 else ''}: "
                        f"{', '.join(entry['terms'])})")
    return {"exercise": exercise, "reason": reason, "patient_type": patient_type, "type_reason": type_reason}
