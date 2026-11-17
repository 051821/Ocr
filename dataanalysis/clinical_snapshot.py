"""30-second clinical snapshot: highlighted concerns and a condition timeline.

Deterministic rules over the fetched record (no LLM, no tokens). Each flag states
what the record shows, the evidence, and why it matters clinically. It never
suggests questions, treatment or medication changes; interpretation is left to
the clinician.
"""
import re
from collections import Counter
from difflib import SequenceMatcher

from datetime import date as _date_cls

import json
import os
from pathlib import Path

import clinical_terms
import coverage
import data_quality
import lab_validation
import medicine_profile
import snapshot_view
import symptom_map
from core import config
from core.encounters import (clinical_days, day_of_encounter, duration_days, recheck_followups, taken_medications,
                             zero_dose_lines)
from core.tracing import traced
from medication_journey import compute_medication_journey
from patient_chat import (REGISTRY, _CONDITION_LABS, _PAIN_MEDS, _med_display, build_condition_trend,
                          complaint_index, conditions_matching, long_term_conditions, long_term_labels)

with (Path(__file__).resolve().parent / "data" / "reference" / "clinical_thresholds.json").open(encoding="utf-8") as _f:
    THRESHOLDS = json.load(_f)
_VITALS = THRESHOLDS["vital_plausibility"]

_NSAID = REGISTRY["nsaid_pattern"]
_NSAID_CLASS = "NSAID (pain / anti-inflammatory)"
_STEROID_CLASS = "Systemic corticosteroid"
_CONDITION_MEDS_HTN = REGISTRY["condition_meds"]["Hypertension"]
_CONTROL_LABS = REGISTRY["control_labs"]
_TOPICAL = r"(?i)\b(?:gel|cream|oint(?:ment)?|spray|lotion|patch)\b"
# "verify": a finding that would be high if an unclear fact (a route not written) were as feared; ranked after high.
_LEVEL_ORDER = {"high": 0, "verify": 1, "medium": 2, "info": 3, "data_quality": 4}
# Diseases that stay active once diagnosed, whether or not the latest visit was about them.
_CHRONIC_CONDITIONS = (set(REGISTRY["chronic_treatment_conditions"]) | set(REGISTRY["cardiometabolic_conditions"])
                       | set(_CONTROL_LABS) | set(REGISTRY["long_standing_conditions"]))


def active_conditions(patient_data, trend):
    """Long-term conditions first (the patient-level list and health history, and chronic diseases diagnosed at
    any visit), then the problems recorded at the latest visit. 'Active' must not drop asthma, diabetes or
    hypertension because the latest visit was for weakness or a rash."""
    chronic = [name for text, _ in long_term_labels(patient_data) for name in (conditions_matching(text) or [text[:60]])]
    chronic += [r["condition"] for r in trend["conditions"] if r["condition"] in _CHRONIC_CONDITIONS]
    current = [r["condition"] for r in trend["conditions"] if r["status"].startswith("ongoing")]
    return list(dict.fromkeys(chronic + current))


def _out_of_range(lab, direction="any"):
    status = str(lab.get("status") or "").casefold()
    high = "high" in status or "elevated" in status
    low = "low" in status or "decreased" in status
    if direction == "high":
        return high
    if direction == "low":
        return low
    return high or low or "abnormal" in status


def _join(names):
    names = [n for n in names if n]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] if names else ""


def _listed(items, n=4):
    """'a; b; c (+2 more)': the first n distinct items, and how many more there are, so a cut list never reads as
    the whole list (concern lines and the AI review count the hidden ones)."""
    items = list(dict.fromkeys(i for i in items if i))
    return "; ".join(items[:n]) + (f" (+{len(items) - n} more)" if len(items) > n else "")


def _grouped_readings(readings):
    """[(text, encounter, date, note)] -> one entry per identical reading with every encounter it was recorded at:
    'Temperature 35 °F (#1, #2, #5) — likely °C entered as °F (≈ 95.0 °F)'. A single encounter keeps its date."""
    groups = {}
    for r in readings:
        if isinstance(r, str):  # already written out (a document reading: '35 (#4 Document 2)')
            groups.setdefault((r, None), None)
            continue
        text, i, d, note = r
        groups.setdefault((text, note), []).append((i, d))
    return [text if at is None else
            f"{text} ({'#%d, %s' % at[0] if len(at) == 1 else ', '.join(f'#{i}' for i, _ in at)}){note}"
            for (text, note), at in groups.items()]


def _trim_to_match(phrase, pattern):
    """'swelling on both eyes lower back pain' -> 'swelling on both eyes' (run-together source text)."""
    match = re.search(pattern, phrase, re.I)
    if not match:
        return phrase
    end = re.match(r"\w*", phrase[match.end():]).end() + match.end()
    return phrase[:end].strip()


def _num(value):
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


def _within(number, bounds):
    return bounds[0] <= number <= bounds[1]


def vital_implausible(label, value):
    """A reading an outpatient could not have (35.7 °F, 94 °F, BP 12/80): a unit or entry error, not a finding.
    Ranges: clinical_thresholds.json vital_plausibility."""
    label, value = str(label or ""), str(value or "")
    number = _num(value)
    if label == "Temperature":
        rule = _VITALS["Temperature"]
        celsius = re.search(r"([\d.]+)\s*°?\s*C\b", value)
        if celsius:
            return not _within(float(celsius.group(1)), rule["celsius"])
        if number is None:
            return False
        # °F when written or when the number can only be Fahrenheit; a bare 35-42 is a Celsius reading.
        if re.search(r"°?\s*F\b", value) or number > 50:
            return not _within(number, rule["fahrenheit"])
        return not _within(number, rule["celsius"])
    if label == "BP":
        rule = _VITALS["BP"]
        pressure = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", value)
        if not pressure:
            return False
        systolic, diastolic = int(pressure.group(1)), int(pressure.group(2))
        return not (_within(systolic, rule["systolic"]) and _within(diastolic, rule["diastolic"])
                    and (systolic > diastolic or not rule.get("systolic_above_diastolic")))
    if number is None:
        return False
    key = "SpO2" if label.startswith("SpO") else "Pulse" if label in ("Pulse", "Heart Rate") else label
    return key in _VITALS and not _within(number, _VITALS[key])


def _fahrenheit_written(value):
    """The number of a temperature written in °F (or a bare number above 50), else None."""
    value = str(value or "")
    match = re.search(r"([\d.]+)\s*°?\s*F\b", value)
    if match:
        return float(match.group(1))
    number = _num(value)
    return number if number and number > 50 and not re.search(r"°?\s*C\b", value) else None


def _celsius_note(label, value):
    """' — likely °C entered as °F (≈ 98.2 °F)' for a °F reading of 34-42.5, which is a Celsius value."""
    if label != "Temperature":
        return ""
    value = str(value or "")
    match = re.search(r"([\d.]+)\s*°?\s*F\b", value)
    number = float(match.group(1)) if match else None
    if number is not None and 34 <= number <= 42.5:
        return f" — likely °C entered as °F (≈ {number * 9 / 5 + 32:.1f} °F)"
    return ""


def _low_temp(value):
    """A °F reading just below the plausible range (93-95 °F): hypothermia range or, far more often, a measuring error."""
    number = _fahrenheit_written(value)
    rule = _VITALS["Temperature"]
    return number is not None and rule["low_band_fahrenheit_from"] <= number < rule["fahrenheit"][0]


_DURATION = re.compile(r"(?i)(?<!age\s)(?<!age:\s)(\d+(?:\.\d+)?)(?:\s*(?:-|–|to)\s*(\d+(?:\.\d+)?))?\s*"
                       r"(years?|yrs?|yr|y|months?|mths?|mo|weeks?|wks?|days?)\b(?!\s*old)")
_DAYS_PER = {"y": 365, "m": 30, "w": 7, "d": 1}


def _stated_duration_days(text):
    """Longest duration written in a complaint ('since 8-10yr' -> 3650), in days; None when there is none."""
    found = [float(m.group(2) or m.group(1)) * _DAYS_PER[m.group(3)[0].casefold()] for m in _DURATION.finditer(str(text or ""))]
    return max(found) if found else None


_MAX_DOSE = [(re.compile(e["pattern"], re.I), e["drug"], e["max_mg"]) for e in REGISTRY["max_daily_dose"]]
# Combination products ("Zerodol-P", "Combiflam") carry two strengths; the written number is not one drug's dose.
_COMBINATION = re.compile(r"(?i)[+&]|\bplus\b|[-\s](?:p|sp|mr|th|cr)\b")


def daily_dose_over_max(med, info=None):
    """'Ezact 90 1-0-1 = 180 mg/day (usual max 120 mg/day etoricoxib)' when the written regimen exceeds the
    usual adult maximum; None when it doesn't or can't be worked out (no strength, SOS, combination, topical).

    The ingredient, strength and form come from clinical_terms when resolved (any brand), else from the name;
    the maximum always comes from the registry."""
    name = str(med.get("name") or "")
    if info and info["generic"]:
        if info["combination"] or info["route"] != "oral" or len(info["generic"]) != 1:
            return None
        identities = [name, info["generic"][0]]
    elif re.search(_TOPICAL, name) or _COMBINATION.search(name):
        return None
    else:
        identities = [name]  # unresolved: the curated brand / ingredient patterns only
    rule = next(((drug, limit) for identity in identities for pattern, drug, limit in _MAX_DOSE
                 if pattern.search(identity)), None)
    regimen = re.fullmatch(r"\s*(\d(?:\.\d)?|1/2|½)\s*-\s*(\d(?:\.\d)?|1/2|½)\s*-\s*(\d(?:\.\d)?|1/2|½)"
                           r"(?:\s*-\s*(\d(?:\.\d)?|1/2|½))?\s*", str(med.get("dosage") or ""))
    written = re.search(r"(\d+(?:\.\d+)?)\s*(?:mg)?\b", " ".join([name, str(med.get("strength") or "")]))
    strength = (info or {}).get("strength_mg") or (float(written.group(1)) if written else None)
    if not rule or not regimen or not strength:
        return None
    count = sum(0.5 if x in ("1/2", "½") else float(x) for x in regimen.groups() if x)
    daily = count * float(strength)
    if daily <= rule[1]:
        return None
    return (f"{_med_display(name)} {med.get('dosage')} = {daily:g} mg/day "
            f"(usual max {rule[1]:g} mg/day {rule[0]})")


def _summary(doc):
    """A document's extracted clinical summary; {} when missing or stored as plain text."""
    summary = doc.get("clinical_summary")
    return summary if isinstance(summary, dict) else {}


def _date(visit):
    return str(visit.get("visit_date") or "")[:10]


def _is_high(status):
    low = str(status or "").casefold()
    return "high" in low or "elevated" in low


def _vital_series(visits, label):
    """[(encounter, date, value)] for one vital, one reading per date."""
    seen, series = set(), []
    for idx, v in enumerate(visits, 1):
        for row in v.get("db_vitals_rows") or []:
            if row.get("label") == label and _date(v) not in seen:
                seen.add(_date(v))
                series.append((idx, _date(v), str(row.get("value") or "")))
    return series


def _lab_rows(patient_data):
    """[(lab, source, date)] for every result of the patient. A report not linked to a visit takes the date of the
    visit it most likely belongs to (as in clinical_insights), else None."""
    rows = []
    for idx, v in enumerate(patient_data.get("visits") or [], 1):
        rows += [(lab, f"Visit #{idx}", _date(v) or None) for lab in v.get("lab_results") or []]
    for doc in patient_data.get("unlinked_documents") or []:
        likely = str((doc.get("likely_visit") or {}).get("visit_date") or "")[:10] or None
        rows += [(lab, "undated report", likely) for lab in doc.get("lab_results") or []]
    # Derived values (eAG from HbA1c) are never interpreted; the same report linked to two encounters (same test and
    # value within 14 days) counts once.
    kept, seen = [], []
    for lab, src, date in rows:
        if lab_validation.is_derived(lab.get("test_name")):
            continue
        key = (lab_validation.identify(lab.get("test_name")) or str(lab.get("test_name") or "").casefold(),
               lab_validation.parse_value(lab.get("value"))[1])
        if key[1] is not None and any(k == key and (d == date or abs(_days_between(d, date)) <= 14) for k, d in seen
                                      if d and date):
            continue
        seen.append((key, date))
        kept.append((lab, src, date))
    return kept


def _dx_category(text):
    """Clinical category of a diagnosis text after normalising spelling and synonyms: the registry conditions it
    names, else the body systems of the symptoms it names (symptom_map); empty when neither registry knows it."""
    found = {"condition:" + c for c in conditions_matching(text)}
    return found or {"system:" + s for s in symptom_map.systems(text)}


def _clinically_different(a, b):
    """True only when both diagnoses map to registry categories and share none. Unknown wording is not compared
    by spelling alone ('Gastritis' / 'gastritis.' / 'acid peptic' are not a change)."""
    ca, cb = _dx_category(a), _dx_category(b)
    return bool(ca and cb and not ca & cb)


def _overdue_days(due, visits):
    """Days from a follow-up due date to the first visit after it; None when no visit followed it."""
    try:
        due_d = _date_cls.fromisoformat(str(due)[:10])
    except ValueError:
        return None
    after = sorted(d for d in (_date(v) for v in visits) if d and d > str(due)[:10])
    if not after:
        return None
    ref = after[0]
    try:
        return max(0, (_date_cls.fromisoformat(ref) - due_d).days)
    except (TypeError, ValueError):
        return None


_EVIDENCE_ITEM = re.compile(r"([^;()·]+?)\s*\((#\d+(?:[^()]*?))\)")


def _evidence_items(flag):
    """[(phrase, {visits})] from a flag's main finding ('swelling on both legs (#1, #2); puffiness over face (#18)')."""
    main = str(flag.get("evidence") or "").split(" · ")[0]
    out = []
    for phrase, refs in _EVIDENCE_ITEM.findall(main):
        encs = {int(n) for n in re.findall(r"#(\d+)", refs)}
        for a, b in re.findall(r"#(\d+)\s*[–-]\s*#?(\d+)", refs):
            encs.update(range(int(a), int(b) + 1))
        out.append((phrase.strip(" ,;:"), encs))
    return out


def fold_symptom_flags(flags):
    """A symptom-only flag (every item a recorded symptom) that cites the same symptom at the same visits as a problem
    flag is folded into it: the visits and evidence merge, the higher level is kept, the symptom flag goes. A symptom no
    problem flag explains keeps its own flag."""
    def concept(phrase):
        return set(symptom_map.tags(phrase)) or {phrase.casefold()}

    folded = []
    for s in flags:
        if s.get("kind") != "risk":
            continue
        items = _evidence_items(s)
        if not items or not all(symptom_map.tags(p) for p, _ in items):
            continue
        best = None
        for p in flags:
            if p is s or p in folded or p.get("kind") == "data":
                continue
            theirs = _evidence_items(p)
            if len(theirs) + (" · " in str(p.get("evidence"))) <= len(items):
                continue  # a problem flag says more than the symptom: more items, or a condition / medicine beside it
            if all(any(concept(a) & concept(b) and ea & eb for b, eb in theirs) for a, ea in items):
                if best is None or _LEVEL_ORDER[p["level"]] < _LEVEL_ORDER[best["level"]]:
                    best = p
        if best is None:
            continue
        merged = [(b, set(eb)) for b, eb in _evidence_items(best)]
        for a, ea in items:
            for b, eb in merged:
                if concept(a) & concept(b) and ea & eb:
                    eb.update(ea)
        main, *rest = str(best["evidence"]).split(" · ")
        best["evidence"] = " · ".join(["; ".join(f"{b} ({_encs(sorted(eb))})" for b, eb in merged)] + rest)
        if _LEVEL_ORDER[s["level"]] < _LEVEL_ORDER[best["level"]]:
            best["level"] = s["level"]
        best.setdefault("folded", []).append(s["title"])
        folded.append(s)
    flags[:] = [f for f in flags if f not in folded]
    return flags


# Flags about two or more medicines taken together: an interaction, two of one class, one ingredient twice.
_COMBO_TITLE = re.compile(r"(?i)^oral [\w ]+? with (?:an? )?[\w /-]*?(?:\barb\b|\bace\b|inhibitor|blocker|diuretic|antagonist|"
                          r"anticoagulant|statin)|same-class medicines|same ingredient in two|two (?:oral )?nsaids")


def tag_combo_time(flags, rec, latest):
    """Current / past for a medicine-combination flag: current only when every medicine in it is on the latest
    prescription or its course (date + duration) still runs on the latest visit date; else past, as of the last date of
    the medicine that stopped. Medicines are read from the finding and its prescription status, not from an 'also on
    the latest prescription' note (another medicine) or a route note."""
    lines = [l for l in rec.prescribed() if l.get("start")]
    for f in flags:
        if f.get("kind") != "medication" or not _COMBO_TITLE.search(str(f.get("title") or "")):
            continue
        text = " · ".join(seg for seg in str(f.get("evidence") or "").split(" · ")
                          if not re.match(r"(?i)\s*(?:also on the latest prescription|route not written)", seg))
        names = {}
        for l in lines:
            if re.search(rf"(?<![\w-]){re.escape(l['name'])}(?![\w-])", text, re.I):
                names.setdefault(l["name"].casefold(), []).append(l)
        if len(names) < 2 and not re.search(r"(?i)\btwice\b", text):
            continue
        stopped = {}
        for key, ls in names.items():
            same = [l for l in lines if any(rec.same(l, x) for x in ls)] or ls
            last = max(same, key=lambda l: (l["start"], l["end"] or ""))
            on_latest = any(l["start"] == latest for l in same)
            running = bool(last["end"] and last["start"] <= latest <= last["end"])
            if not (on_latest or running):
                stopped[last["name"]] = last["start"]
        f["combo"] = True
        if stopped:
            f["time"], f["as_of"] = "past", max(stopped.values())
            f["stopped"] = sorted(stopped)
        else:
            f["time"], f["as_of"] = "current", latest
    return flags


def tag_current(flags, visits, current_medicines, rec):
    """Tag each flag 'current' or 'past' with the date it is current as of ('as_of').
    Current: it cites a visit on the latest clinical day, a date on or after it, names a medicine on the latest
    prescription (or a course still running), or it has no date at all (a gap that is still open). Else past."""
    latest = _date(visits[-1]) if visits else ""
    latest_encs = {i for i, v in enumerate(visits, 1) if _date(v) == latest}
    names = set()
    for c in current_medicines:
        names.add((_first_word(c["name"]) or c["name"]).casefold())
    for row in rec.lines:
        for l in row:
            if any(c["name"] == l["name"] for c in current_medicines):
                names.update(str(g).casefold() for g in l.get("ingredients") or [])
    names.discard("")
    for f in flags:
        text = f"{f.get('title', '')} {f.get('evidence', '')}"
        low = text.casefold()
        encs = {int(n) for n in re.findall(r"#(\d+)", text)}
        for a, b in re.findall(r"#(\d+)\s*[–-]\s*#?(\d+)", text):
            encs.update(range(int(a), int(b) + 1))
        dates = sorted(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
        on_med = any(re.search(rf"\b{re.escape(n)}\b", low) for n in names)
        current = bool(encs & latest_encs) or (dates and dates[-1] >= latest) or on_med or (not encs and not dates) \
            or str(f.get("title", "")).startswith("No ")  # a gap in the record stays open
        f["time"] = "current" if current else "past"
        last_enc = max(encs) if encs else None
        cands = [d for d in dates] + ([_date(visits[last_enc - 1])] if last_enc and last_enc <= len(visits) else [])
        f["as_of"] = latest if current else (max(c for c in cands if c) if any(cands) else "")
        # Values only from undated reports (no latest-visit encounter or date): current, but never given a visit date.
        dated_now = bool(encs & latest_encs) or bool(dates and dates[-1] >= latest)
        if current and not dated_now and re.search(r"(?i)\bundated\b", text):
            f["as_of"], f["undated"] = "", True
    return flags


def _named_lines(text, rec):
    """Record medicine lines a flag's text names (by product word or ingredient)."""
    low = str(text).casefold()
    out = []
    for row in rec.lines:
        for l in row:
            words = {(_first_word(l["name"]) or "").casefold()} | {str(g).casefold() for g in l.get("ingredients") or []}
            if any(w and len(w) >= 4 and re.search(rf"\b{re.escape(w)}\b", low) for w in words):
                out.append(l)
    return out


def add_related_medicines(flags, rec, current_lines):
    """A flag about an oral medicine also names the same-group medicines by another route (topical, local, injected)
    on the latest prescription, so the whole exposure is in one place (registry therapeutic groups)."""
    for f in flags:
        if f.get("kind") == "data":
            continue
        text = f"{f.get('title', '')} {f.get('evidence', '')}"
        oral = [l for l in _named_lines(text, rec) if l.get("route") == "oral"]
        grp = {g for l in oral for g in l.get("groups") or []}
        if not grp:
            continue
        low = text.casefold()
        extra = []
        for l in current_lines:
            word = (_first_word(l["name"]) or l["name"]).casefold()
            if l.get("route") != "oral" and set(l.get("groups") or []) & grp and word not in low \
                    and not any(e[0] == " ".join(str(l["name"]).split()) for e in extra):
                extra.append((" ".join(str(l["name"]).split()), l.get("route") or "route unclear"))
        if extra:
            f["evidence"] = f"{f.get('evidence', '')} · also on the latest prescription: " + \
                ", ".join(f"{n} ({r})" for n, r in extra[:3])
            f["related_medicines"] = [n for n, _ in extra]
    return flags


def overlapping_ingredients(visits, rec):
    """[(date, encounters, group, products, shared ingredients)]: two different products in one therapeutic group on one
    clinical day that share an ingredient (e.g. a combination product plus the single ingredient)."""
    out = []
    for day, encs in clinical_days(visits):
        lines = [l for i in encs for l in rec.lines[i - 1] if l.get("prescribed")]
        for a_i, a in enumerate(lines):
            for b in lines[a_i + 1:]:
                shared = set(a.get("ingredients") or []) & set(b.get("ingredients") or [])
                grp = set(a.get("groups") or []) & set(b.get("groups") or [])
                if shared and grp and set(a["ingredients"]) != set(b["ingredients"]) and a["name"] != b["name"]:
                    # The encounters that wrote the two products, not every encounter of the day.
                    at = sorted({l.get("i") for l in (a, b) if l.get("i")}) or encs
                    out.append((day, at, sorted(grp)[0], (a["name"], b["name"]), sorted(shared)))
    return out


def _group_days(rec, group, route):
    """Distinct days covered by courses of a therapeutic group by one route (start to course end)."""
    from datetime import timedelta
    days = set()
    for l in rec.prescribed():
        if group in (l.get("groups") or []) and l.get("route") == route and l.get("start"):
            try:
                a = _date_cls.fromisoformat(l["start"])
                b = _date_cls.fromisoformat(l.get("end") or l["start"])
            except ValueError:
                continue
            days.update(a + timedelta(n) for n in range(max(0, (b - a).days) + 1))
    return len(days)


def _group_problem(group):
    spec = (medicine_profile.REF.get("groups") or {}).get(group) or {}
    return spec.get("problem") or spec.get("class") or group


def _title_groups(title):
    """Registry groups a flag title names by one of the group's terms, in title order."""
    low = str(title).casefold()
    found = []
    for g, spec in (medicine_profile.REF.get("groups") or {}).items():
        pos = [m.start() for t in spec.get("terms") or [g] for m in re.finditer(rf"\b{re.escape(t)}\b", low)]
        if pos:
            found.append((min(pos), g))
    return [g for _, g in sorted(found)]


def link_flags(flags, rec):
    """Give each flag a 'problem' so the flags about one clinical problem (dose, duration, gastritis, same-day use of
    one medicine group) are read as one finding. The problem is the registry group the title names (first term in
    the title), else the group of the medicines the title (then the evidence) names, the commonest across the flags
    winning; else the registry condition the title names."""
    named = []
    for f in flags:
        groups = _title_groups(f.get("title", "")) or \
            [g for l in _named_lines(f.get("title", ""), rec) for g in l.get("groups") or []] or \
            [g for l in _named_lines(f.get("evidence", ""), rec) for g in l.get("groups") or []]
        named.append(groups)
    count = Counter(g for groups in named for g in set(groups))
    for f, groups in zip(flags, named):
        if f.get("kind") == "data" or f.get("problem"):
            continue
        if groups:
            first = _title_groups(f.get("title", ""))
            g = first[0] if first else max(dict.fromkeys(groups), key=lambda x: count[x])
            f["problem"] = _group_problem(g)
        else:
            cond = conditions_matching(f.get("title", ""))
            if cond:
                f["problem"] = cond[0]
    return flags


def _same_diagnosis_group(a, b):
    """True when two diagnosis texts name members of one registry diagnosis_refinements group (LSC and eczema)."""
    a, b = str(a or "").casefold(), str(b or "").casefold()
    for group in REGISTRY.get("diagnosis_refinements") or []:
        hit = lambda text: any(re.search(rf"\b{re.escape(term)}\b", text) for term in group)
        if hit(a) and hit(b):
            return True
    return False


def _first_word(name):
    # 'Ct. T. Amlo' / 'Cont. Tab Aten' continue a medicine: the same medicine as 'T. Amlo'.
    clean = re.sub(r"(?i)^(?:(?:ct|cont|contd|continue)\b\.?\s*)?(?:(?:ta|tb|tab(?:let)?|cap(?:sule)?|syp|syr(?:up)?|inj(?:ection)?)\.?\s+|[tc]\.\s*)?",
                   "", str(name or ""))
    match = re.match(r"[a-z]+", clean.casefold())
    return match.group() if match and len(match.group()) >= 3 else ""


def _implausible(lab):
    """Value far outside its reference range: likely an OCR misread (row number, unit slip)."""
    value = _num(lab.get("value"))
    limits = re.search(r"([\d.]+)\s*[-–—]\s*([\d.]+)", str(lab.get("reference") or ""))
    if value is None or not limits:
        return False
    low, high = float(limits.group(1)), float(limits.group(2))
    return (low > 0 and value < low / 3) or (high > 0 and value > high * 10)


def _encs(numbers):
    numbers = sorted(set(numbers))
    return ", ".join(f"#{n}" for n in numbers[:6]) + (" …" if len(numbers) > 6 else "")


def _days(first_visit, last_visit):
    try:
        return (_date_cls.fromisoformat(_date(last_visit)) - _date_cls.fromisoformat(_date(first_visit))).days
    except ValueError:
        return None


def _course_patterns(visits, trend, meds_by_visit, record_dx, flag, supportive=lambda m: False):
    """Patterns only visible across visits: a problem that keeps returning while treatment keeps changing,
    visits coming closer together, a growing medicine load, and how the recorded diagnosis evolved."""
    regimen = [{_first_word(m) for m in meds} - {""} for meds in meds_by_visit]
    # Antihistamines and emollients relieve symptoms; they are not counted as treatments tried for control.
    treatment = [{_first_word(m) for m in meds if not supportive(m)} - {""} for meds in meds_by_visit]
    # Chronic diseases are expected at every visit; their control is judged by their markers instead.
    chronic = set(_CONTROL_LABS) | set(REGISTRY["cardiometabolic_conditions"])
    for r in trend["conditions"]:
        encs = r["encounters"]
        if r["condition"] in chronic or len(encs) < 3 or not r["status"].startswith("ongoing"):
            continue
        tried = set().union(*(treatment[i - 1] for i in encs))
        changes = sum(1 for a, b in zip(encs, encs[1:]) if treatment[a - 1] != treatment[b - 1])
        days = _days(visits[encs[0] - 1], visits[encs[-1] - 1])
        pace = f", about every {round(days / (len(encs) - 1))} days" if days else ""
        flag("medium" if len(encs) >= 4 else "info", f"{r['condition']} not settling",
             f"At {len(encs)} of {len(visits)} visits{pace}, still recorded at the latest; "
             f"{len(tried)} different treatments (antihistamines and emollients not counted), regimen changed at {changes} of {len(encs) - 1} return visits",
             "Repeated returns while treatment keeps changing: no regimen so far has controlled it.", kind="pattern")

    # Visit spacing: first half of the record against the second half.
    gaps = [g for g in (_days(a, b) for a, b in zip(visits, visits[1:])) if g is not None and g > 0]
    if len(gaps) >= 4:
        half = len(gaps) // 2
        early, recent = sum(gaps[:half]) / half, sum(gaps[half:]) / (len(gaps) - half)
        if recent <= 0.6 * early:
            flag("medium", "Visits becoming more frequent", f"Every ~{round(early)} days early on → ~{round(recent)} days recently",
                 "Shorter gaps usually mean symptoms are returning sooner.", kind="pattern")
        elif recent >= 1.7 * early:
            flag("info", "Visits spacing out", f"Every ~{round(early)} days early on → ~{round(recent)} days recently",
                 "Consistent with the problem settling.", kind="pattern")

    # Medicine load: first visit against the latest.
    # Counted per clinical day: two same-day encounters are one day's medicines.
    days = clinical_days(visits)
    counts = [len(set().union(*(regimen[i - 1] for i in encs))) for _, encs in days]
    if len(counts) >= 3 and counts[-1] >= counts[0] + 2:
        flag("info", "Medicine load rising", f"{counts[0]} medicines on {days[0][0]} → {counts[-1]} on {days[-1][0]} "
             f"(peak {max(counts)})", "Escalating treatment; also raises interaction and adherence burden.", kind="medication")

    # Diagnosis changed for the same complaint / site: consecutive visits with the same complaint words but
    # diagnoses that are neither the same nor one refinement family.
    complaint_words = lambda i: {w for w in re.findall(r"[a-z]{5,}", str(visits[i - 1].get("db_chief_complaint") or "").casefold())
                                 if w not in {"follow", "visit", "since", "days", "report", "check", "known", "patient",
                                              "lower", "upper", "pain", "both", "left", "right", "general", "weakness"}}
    switched = []
    dated = sorted(i for i, dx in record_dx.items() if dx.strip(" ?.") and not _GARBLED.search(dx))
    for a, b in zip(dated, dated[1:]):
        da, db = record_dx[a].strip(" ?."), record_dx[b].strip(" ?.")
        shared = complaint_words(a) & complaint_words(b)
        chronic_dx = set(REGISTRY.get("chronic_treatment_conditions") or [])
        if _date(visits[a - 1]) == _date(visits[b - 1]) or set(conditions_matching(da) + conditions_matching(db)) & chronic_dx:
            continue
        if (shared and SequenceMatcher(None, da.casefold(), db.casefold()).ratio() < 0.8
                and da.casefold() not in db.casefold() and db.casefold() not in da.casefold()
                and not _same_diagnosis_group(da, db) and _clinically_different(da, db)):
            switched.append(f"'{da[:30]}' (#{a}) → '{db[:30]}' (#{b}) for '{' '.join(sorted(shared))[:30]}'")
    if switched:
        flag("medium", "Diagnosis changed for the same complaint", _listed(switched, 3),
             "The same complaint was given a different diagnosis at a later visit.", kind="pattern")

    # Diagnosis wording over time ("Eczema #1–#7 → LSC #8"), ignoring OCR noise.
    runs = []
    for i in sorted(record_dx):
        dx = record_dx[i].strip(" ?.")
        if not dx or _GARBLED.search(dx):
            continue
        if runs and SequenceMatcher(None, runs[-1][0].casefold(), dx.casefold()).ratio() >= 0.8:
            runs[-1][2] = i
        else:
            runs.append([dx, i, i])
    same_problem = any({a for _, a, _ in runs} <= set(r["encounters"]) for r in trend["conditions"])
    if 2 <= len(runs) <= 5 and len(visits) >= 3 and same_problem:
        flag("info", "Diagnosis evolution", " → ".join(f"{dx[:30]} (#{a}" + (f"–#{b})" if b != a else ")") for dx, a, b in runs),
             kind="pattern")


# OCR noise read as text: doses, words fused with digits, letter-digit codes ("G6A2B1C0").
_GARBLED = re.compile(r"(?i)\d\s*(?:mg|mcg|ml|gm)|[a-z]{3,}\d|\d[a-z]{3,}|\b(?=[a-z]*\d)(?=\d*[a-z])[a-z0-9]{6,}\b")
# A medicine line the OCR filed as a lab result ("LOBET GM CREAM 10", "TAB CITRIZINE 5").
_MEDICINE_LINE = re.compile(r"(?i)\b(?:tab|tablet|cap|capsule|cream|gel|oint(?:ment)?|lotion|syp|syrup|inj|drops?|"
                            r"vaseline|vasaline|soap|powder)\b")


def is_medicine_line(lab, prescribed_words=()):
    """A 'Not Evaluated' lab row that is really a prescription line (dose form, or a prescribed medicine's name)."""
    if str(lab.get("status") or "").casefold() != "not evaluated":
        return False
    name = str(lab.get("test_name") or lab.get("test") or "")
    return bool(_MEDICINE_LINE.search(name)) or _first_word(name) in set(prescribed_words) - {""}


# ── Guideline context (ICMR and national programmes) ────────────────────────
# What the record shows against Indian guideline cut-offs, personalised by age, sex and pregnancy. Every value comes
# from REGISTRY["guidelines"]. Every section produces rows for the doctor and /summary, each marked verified or draft.
# Only a verified section (verified + verified_by + verified_on, set with scripts/verify_guideline.py) raises flags and
# reaches the AI review and brief; ENABLE_DRAFT_GUIDELINES treats drafts as verified for testing.
_GUIDELINES = REGISTRY.get("guidelines") or {}
GUIDELINE_VERSION = _GUIDELINES.get("version", "")
_VISIT_DX = ("db_provisional_dx", "db_confirmed_dx")


def _guideline(section):
    return _GUIDELINES.get(section)


def guideline_verified(rule):
    """Verified by a named clinician on a date; a bare verified: true does not count."""
    return bool(rule and rule.get("verified") and rule.get("verified_by") and rule.get("verified_on"))


def guideline_in_use(rule):
    """Whether a section's findings may raise flags and reach the AI: verified, or drafts switched on."""
    return bool(rule) and (guideline_verified(rule) or config.flag("ENABLE_DRAFT_GUIDELINES"))


def _verification(rule):
    ok = guideline_verified(rule)
    return {"verified": ok, "verified_by": rule.get("verified_by") if ok else None,
            "verified_on": rule.get("verified_on") if ok else None}


def guideline_sections():
    """[{section, source, verified, verified_by, verified_on}] for every guideline section."""
    return [{"section": name, "source": rule.get("source") or "", **_verification(rule)}
            for name, rule in _GUIDELINES.items() if isinstance(rule, dict)]


def guideline_rows_in_use(rows):
    """The rows that may raise flags or reach the AI review / brief (see guideline_in_use)."""
    return [r for r in rows or [] if r.get("verified") or config.flag("ENABLE_DRAFT_GUIDELINES")]


def _patient_profile(age_sex, visits):
    """Age, sex and a pregnancy recorded within the last pregnancy_days of the record: what selects the cut-offs."""
    age = re.search(r"(\d{1,3})\s*(?:y|yrs?|years?)\b", str(age_sex or ""), re.I)
    sex = lab_validation.sex_from(age_sex)
    hb = _GUIDELINES.get("haemoglobin") or {}
    latest = _date(visits[-1])
    pregnant = sex == "f" and hb.get("pregnancy_pattern") and any(
        re.search(hb["pregnancy_pattern"], " ".join(str(v.get(k) or "") for k in _VISIT_DX + ("db_chief_complaint",)), re.I)
        and _days_between(_date(v), latest) <= hb.get("pregnancy_days", 280) for v in visits)
    return {"age": int(age.group(1)) if age else None, "sex": {"f": "female", "m": "male"}.get(sex),
            "pregnancy": "recorded" if pregnant else None}


def _lab_number(lab, unit):
    """A result as a number in `unit` (HbA1c mmol/mol converted to %); None when unverified, unreadable or in
    another unit."""
    if str(lab.get("status") or "").startswith("Unverified"):
        return None
    _, number, _, written = lab_validation.parse_value(lab.get("value"))
    if number is None:
        return None
    if unit == "%" and written == "mmol/mol":
        return round(number / 10.929 + 2.15, 1)
    return number if written in (None, unit) else None


def _latest_result(labs, keys):
    """(lab, date) of the newest result for any catalog key in `keys`: a dated result first, else an undated one."""
    rows = [(lab, date) for lab, _, date in labs if lab_validation.identify(lab.get("test_name")) in keys
            and not str(lab.get("status") or "").startswith("Unverified")]
    dated = [r for r in rows if r[1]]
    return max(dated, key=lambda r: r[1]) if dated else (rows[0] if rows else None)


def _monitoring_state(labs, keys, every_days, today):
    """Why a monitoring test is due ('none in record', 'last 2025-06-02, overdue', ...); None when it is current."""
    hit = _latest_result(labs, keys)
    if not hit:
        return "none in record"
    if not hit[1]:
        return "only in an undated report"
    return f"last {hit[1]}, overdue" if _days_between(hit[1], today) > every_days else None


def _regimen(visits, condition, treats):
    """'Metformin 500 1-0-1: same regimen since 2026-01-15' for a condition's medicines at the last visit that
    prescribes them; visits without them in between do not break the run."""
    # Signature per visit: the medicine's first word and its dose digits ("Ta Telmisartan 1-0-0" and
    # "Telmisartan 1 - 0 - 0" are one regimen), so the run is not cut by spelling or formatting.
    def sig(m):
        name = _med_display(m.get("name"))
        return (_first_word(name) or name.casefold(), re.sub(r"[^\d/.½]", "", str(m.get("dosage") or "")))
    shown = {}
    sigs = []
    for v in visits:
        entry = set()
        for m in taken_medications(v):
            if treats(_med_display(m.get("name")), condition):
                entry.add(sig(m))
                shown[sig(m)] = f"{_med_display(m.get('name'))} {str(m.get('dosage') or '').strip()}".strip()
        sigs.append(frozenset(entry))
    last = max((i for i, s in enumerate(sigs) if s), default=None)
    if last is None:
        return f"no {condition} medicine identified in the prescriptions"
    # Each current drug's own unchanged run: back from the last prescription while every visit that prescribes this
    # condition's medicines still has this drug at this dose (another drug added and dropped does not end it).
    runs = []
    for k in sorted(sigs[last], key=lambda k: shown[k]):
        start = last
        for i in range(last - 1, -1, -1):
            if not sigs[i]:
                continue
            if k not in sigs[i]:
                break
            start = i
        runs.append(f"{shown[k]}: same regimen since {_date(visits[start])}")
    text = "; ".join(runs)
    return text + (f", last prescribed {_date(visits[last])}" if _date(visits[last]) != _date(visits[-1]) else "")


def _hb_group(rule, profile):
    """The haemoglobin cut-off group for this patient; an unknown age counts as adult, an unknown sex has none."""
    age, sex = profile["age"], (profile["sex"] or "")[:1] or None
    for g in rule["groups"]:
        if g.get("pregnant") and not profile["pregnancy"]:
            continue
        if g.get("sex") and g["sex"] != sex:
            continue
        if age is None and g.get("max_age") is not None:
            continue
        if age is not None and not (g.get("min_age", 0) <= age <= g.get("max_age", 200)):
            continue
        return g
    return None


def guideline_context(patient_data, labs, present, profile, bp, treats, med_is, flag):
    """Rows of what the record shows against the guideline cut-offs, for the snapshot and the AI evidence:
    {topic, kind, status, finding, source} (+ condition / marker). The ones a clinician should act on are also
    flagged, with their source. Returns (rows, conditions whose monitoring the rows cover)."""
    visits = patient_data.get("visits") or []
    today = _date(visits[-1])  # the latest visit is the reference date, not the day the page is opened
    span = f"{len(visits)} visit{'s' if len(visits) != 1 else ''}, {_date(visits[0])} to {today}"
    rows, monitored, due = [], set(), {}

    def row(section, topic, kind, status, finding, source, **extra):
        rows.append({"topic": topic, "kind": kind, "status": status, "finding": finding, "source": source, **extra,
                     "section": section, **_verification(_guideline(section))})

    def gflag(section, *args, **kwargs):
        # A draft section is information for the doctor only: it raises no flag.
        if guideline_in_use(_guideline(section)):
            flag(*args, **kwargs)

    # Diabetes: latest HbA1c against the target; old, undated or missing results are said as such.
    dm = _guideline("diabetes_control")
    if dm and present([dm["condition"]]):
        cond, label, target = dm["condition"], dm["label"], f"<{dm['target_below']:g} {dm['unit']}"
        if guideline_in_use(dm):
            monitored.add(cond)
        hit = _latest_result(labs, [dm["analyte"]])
        value = _lab_number(hit[0], dm["unit"]) if hit else None
        regimen = _regimen(visits, cond, treats)
        if hit and value is not None and hit[1]:
            above = value >= dm["target_below"]
            age_days = _days_between(hit[1], today)
            if age_days > dm["recheck_days_above_target" if above else "recheck_days_at_target"]:
                status = "status unknown"
                finding = f"last {label} {value:g} {dm['unit']} ({hit[1]}), {age_days} days before the latest visit; current control unknown"
                due.setdefault(cond, []).append(f"{label}: last {hit[1]}, overdue")
            else:
                status = "above target" if above else "at target"
                finding = f"{label} {value:g} {dm['unit']} ({hit[1]}) {status} ({target}); {regimen}"
        elif hit and value is not None:
            status, finding = "on record, date unknown", f"{label} {value:g} {dm['unit']} from an undated report; cannot show it is recent"
            due.setdefault(cond, []).append(f"{label}: no dated result")
        elif hit:
            status, finding = "not readable", f"{label} '{hit[0].get('value')}' could not be read against the target"
        else:
            glucose = _latest_result(labs, dm["glucose_analytes"])
            if glucose:
                status = "glucose only"
                finding = (f"no {label}; latest {glucose[0].get('analyte') or glucose[0].get('test_name')} "
                           f"{glucose[0].get('value')} ({glucose[1] or 'undated report'}); control is not judged from glucose alone")
            else:
                status, finding = "not monitored", f"no {label} or glucose result across {span}"
            due.setdefault(cond, []).append(f"{label}: none in record")
        row("diabetes_control", f"{cond} control", "condition", status, finding, dm["source"], condition=cond, marker=label, target=target)

    # Hypertension: latest structured BP against the target for this patient's conditions.
    bpg = _guideline("bp_targets")
    if bpg and present([bpg["condition"]]):
        lower = present(REGISTRY["bp_lower_target_conditions"])
        limit = bpg["lower"] if lower else bpg["default"]
        target = f"<{limit[0]}/{limit[1]}" + (f" (with {_join(lower)})" if lower else "")
        if not bp:
            status, finding = "not monitored", f"no BP reading across {span}"
        else:
            above = [r for r in bp if r[2][0] >= limit[0] or r[2][1] >= limit[1]]
            (_, date, (s, d)) = bp[-1]
            recent = bp[-3:]
            if len(bp) <= 1:
                status = "single reading"
            elif bp[-1] in above:
                status = "above target"
            elif any(r in above for r in recent):
                status = "not consistently at target"  # latest below, but one of the last three at or above
            else:
                status = "at target"
            finding = (f"latest BP {s}/{d} ({date}), target {target}; {len(above)} of {len(bp)} reading(s) at or above it; "
                       + _regimen(visits, bpg["condition"], treats))
        row("bp_targets", f"{bpg['condition']} control", "condition", status, finding, bpg["source"], condition=bpg["condition"], target=target)

    # Monitoring tests by interval, from all of the patient's results (visits without labs are normal).
    mon = _guideline("monitoring")
    for cond, items in (mon["conditions"].items() if mon else ()):
        if not present([cond]):
            continue
        if guideline_in_use(mon):
            monitored.add(cond)
        for item in items:
            state = _monitoring_state(labs, item["keys"], item["every_days"], today)
            if state:
                due.setdefault(cond, []).append(f"{item['label']}: {state}")
    for cond, items in due.items():
        section = "monitoring" if mon else "diabetes_control"
        source = _guideline(section)["source"]
        row(section, f"{cond} monitoring", "followup", "due", "; ".join(items), source, condition=cond)
        gflag(section, "medium" if any("none in record" in i for i in items) else "info", f"{cond}: monitoring due",
             "; ".join(items), f"Per {source}; based on all results in the record up to {today}.", kind="followup", source=source)

    # Haemoglobin against the anaemia cut-off for this patient's age, sex and pregnancy.
    hb = _guideline("haemoglobin")
    hit = _latest_result(labs, ["haemoglobin"]) if hb else None
    value = _lab_number(hit[0], "g/dl") if hit else None
    if value is not None:
        group = _hb_group(hb, profile)
        when = hit[1] or "undated report"
        if group is None:
            if _out_of_range(hit[0], "low") or present(["Anaemia"]):
                why = "sex not recorded" if not profile["sex"] else "no cut-off for this age"
                row("haemoglobin", "Haemoglobin", "lab", "not banded", f"Hb {value:g} g/dL ({when}); {why}, so the report range is used",
                    hb["source"])
        elif value < group["below"]:
            band = next(label for lower, label in group["bands"] if value >= lower)
            row("haemoglobin", "Haemoglobin", "lab", band,
                f"Hb {value:g} g/dL ({when}): {band} for a {group['group']} (anaemia below {group['below']:g})", hb["source"])
        elif _out_of_range(hit[0], "low") or present(["Anaemia"]):
            row("haemoglobin", "Haemoglobin", "lab", "above anaemia cut-off",
                f"Hb {value:g} g/dL ({when}) is at or above the {group['below']:g} cut-off for a {group['group']}", hb["source"])

    # Results read against guideline bands, including results the report gave no range for.
    ctx = _guideline("lab_context")
    for key, rule in (ctx["analytes"].items() if ctx else ()):
        if present(rule.get("skip_conditions", [])):
            continue
        hit = _latest_result(labs, [key])
        value = _lab_number(hit[0], rule["unit"]) if hit else None
        if value is None:
            continue
        band = next((label for low, high, label in rule["bands"]
                     if (low is None or value >= low) and (high is None or value < high)), None)
        unevaluated = str(hit[0].get("status") or "").casefold() == "not evaluated"
        if band or unevaluated:
            name = hit[0].get("analyte") or hit[0].get("test_name")
            row("lab_context", name, "lab", band or "within guideline cut-offs",
                f"{name} {hit[0].get('value')} ({hit[1] or 'undated report'}): {band or 'within guideline cut-offs'}"
                + ("; the report gave no reference range" if unevaluated else ""), rule["source"])

    # Antibiotics at a visit whose only recorded illness is usually viral.
    ab = _guideline("antibiotics")
    if ab:
        pattern = next(c["pattern"] for c in REGISTRY["medicine_classes"] if c["class"] == "Antibiotic")
        courses, viral = [], []
        for i, v in enumerate(visits, 1):
            given = [_med_display(m.get("name")) for m in taken_medications(v)
                     if med_is(_med_display(m.get("name")), "Antibiotic", pattern, systemic=True)]
            if not given:
                continue
            courses.append((i, _date(v), given))
            dx = " ".join(str(v.get(k) or "") for k in _VISIT_DX)
            syndromes = [s["name"] for s in ab["usually_viral"] if re.search(s["pattern"], dx, re.I)]
            if syndromes and not re.search(ab["bacterial_markers"], f"{dx} {v.get('db_chief_complaint') or ''}", re.I):
                viral.append(f"{', '.join(given)} at #{i} ({_join(syndromes)})")
        if viral:
            row("antibiotics", "Antibiotic use", "medication", "usually viral illness", "; ".join(viral), ab["source"])
            gflag("antibiotics", "medium", "Antibiotic for a usually viral illness", _listed(viral, 3),
                 f"{ab['source']}: these illnesses are usually viral and antibiotics are not routinely indicated.",
                 kind="medication", source=ab["source"])
        dated = [c for c in courses if c[1]]
        for start in range(len(dated)):
            window = [c for c in dated[start:] if _days_between(dated[start][1], c[1]) <= ab["repeat_within_days"]]
            if len(window) >= ab["repeat_courses"]:
                evidence = "; ".join(f"{', '.join(g)} at #{i} ({d})" for i, d, g in window[:4])
                row("antibiotics", "Antibiotic use", "medication", "repeated courses",
                    f"{len(window)} antibiotic courses within {ab['repeat_within_days']} days: {evidence}", ab["source"])
                gflag("antibiotics", "medium", f"{len(window)} antibiotic courses within {ab['repeat_within_days']} days", evidence,
                     "Repeated courses add to antimicrobial resistance and side effects.", kind="medication", source=ab["source"])
                break

    # A cough of two weeks or more with no TB test anywhere in the record.
    tb = _guideline("tb_screening")
    if tb:
        # Each complaint part keeps its own duration ("fever 2 days, cough since 3 weeks").
        long_cough = {}
        for i, part in clinical_terms.complaint_parts(visits):
            days = _stated_duration_days(part) or (clinical_terms.complaint(patient_data, part) or {}).get("duration_days") or 0
            if re.search(tb["cough_pattern"], part, re.I) and days >= tb["min_days"]:
                long_cough.setdefault(part.casefold(), set()).add(i)
        record = " ".join([str(lab.get("test_name") or "") for lab, _, _ in labs]
                          + [str(v.get(k) or "") for v in visits for k in _VISIT_DX]
                          + [str(n) for v in visits for n in v.get("db_instructions") or []]
                          + [str(t) for v in visits for d in v.get("documents") or [] for t in _summary(d).get("tests_advised") or []])
        if long_cough and not re.search(tb["tested_pattern"], record, re.I):
            evidence = "; ".join(f"{p} ({_encs(sorted(e))})" for p, e in list(long_cough.items())[:2])
            row("tb_screening", "TB screening", "followup", "no TB test", f"cough of {tb['min_days']}+ days: {evidence}; no sputum or TB test in the record",
                tb["source"])
            gflag("tb_screening", "medium", f"Cough for {tb['min_days']}+ days with no TB test in the record", evidence,
                 f"{tb['source']}: a cough of two weeks or more is presumptive TB.", kind="followup", source=tb["source"])
    return rows, monitored


# An instruction phrase that is advice or a condition to watch for, not a symptom the patient has.
_ADVICE = re.compile(r"(?i)\b(?:if|in case|avoid|sos|return|come back|watch|diet|review|continue|take|apply|advised?|stop|"
                     r"check|monitor|follow)\b")


@traced(name="snapshot.build_snapshot")
def build_snapshot(patient_data):
    visits = data_quality.prepare(patient_data)  # data-quality corrections run before every rule
    if not visits:
        return None
    trend = build_condition_trend(patient_data)
    conditions = {r["condition"]: r for r in trend["conditions"]}
    # Long-term list / health history count as present even when no visit names them.
    long_term = long_term_conditions(patient_data)
    present = lambda names: [n for n in names if n in conditions or n in long_term]
    complaints = complaint_index(visits)
    parts = clinical_terms.complaint_parts(visits)

    # Symptom tags recorded at each visit (all complaint parts), for excluded_by.
    visit_tags = {}
    for i, part in parts:
        visit_tags.setdefault(i, set()).update(symptom_map.tags(part))

    # Instruction lines split into phrases (for symptom_hits): a symptom can be written only there.
    instruction_phrases = {}
    for idx, v in enumerate(visits, 1):
        lines = list(v.get("db_instructions") or []) + ([v["instructions"]] if isinstance(v.get("instructions"), str) else [])
        for line in lines:
            for part in re.split(r"[,;.\n]", str(line or "")):
                phrase = " ".join(part.strip(" -:").lower().split())
                if 4 <= len(phrase) <= 60 and not _ADVICE.search(phrase):
                    instruction_phrases.setdefault(phrase, set()).add(idx)

    def symptom_hits(pattern, tags=(), encounters=None, trim=True):
        """{phrase: visits} for complaints matching a registry pattern, or the symptom tags clinical_terms
        resolved for a complaint the pattern does not cover ('pedal oedema', 'sob on exertion')."""
        hits = {}
        wanted = set(tags or ())
        for phrase, encs in complaints.items():
            encs = [i for i in encs if encounters is None or i in encounters]
            if not encs:
                continue
            # The symptom registry first: a phrase is matched by what it is (tags), not by its words. A concept explained
            # another way at the same visit (excluded_by: swelling after a fall) is not read as this concept there.
            for hit in symptom_map.classify(phrase):
                if set(hit["tags"]) & wanted:
                    kept = [i for i in encs if not set(hit["excluded_by"]) & visit_tags.get(i, set())]
                    if kept:
                        hits.setdefault(hit["text"] if trim else phrase, set()).update(kept)
            # A rule's own pattern only reads the words the registry does not explain.
            rest = symptom_map.remainder(phrase)
            if pattern and re.search(pattern, rest, re.I):
                hits.setdefault(_trim_to_match(rest.strip(), pattern) if trim else phrase, set()).update(encs)
        # The instructions field: a symptom the doctor wrote there ('Puffiness over face') counts, read only through the
        # symptom registry (tags), never a rule's own word pattern, so advice ('low salt diet') is not a symptom.
        for phrase, encs in instruction_phrases.items():
            encs = [i for i in encs if encounters is None or i in encounters]
            for hit in symptom_map.classify(phrase) if encs else []:
                if set(hit["tags"]) & wanted:
                    kept = [i for i in encs if not set(hit["excluded_by"]) & visit_tags.get(i, set())]
                    if kept:
                        hits.setdefault(hit["text"] if trim else phrase, set()).update(kept)
        # A bare symptom word with no site ('swelling') at a visit soon after the same symptom was recorded with its site
        # ('swelling on both legs') continues that complaint.
        if wanted:
            window = THRESHOLDS["cause_timing"]["symptom_window_days"]
            for phrase, encs in complaints.items():
                words = phrase.split()
                if len(words) > 2 or symptom_map.classify(phrase):
                    continue
                for hit_phrase, hit_encs in hits.items():
                    if not set(words) <= set(hit_phrase.casefold().split()):
                        continue
                    for i in encs:
                        if (encounters is None or i in encounters) and i not in hit_encs and any(
                                e < i and _date(visits[e - 1]) and _date(visits[i - 1])
                                and 0 <= _days_between(_date(visits[e - 1]), _date(visits[i - 1])) <= window
                                for e in list(hit_encs)):
                            hit_encs.add(i)
        seen = {i for e in hits.values() for i in e}
        for i, part in parts:
            info = clinical_terms.complaint(patient_data, part)
            # The model's tags only for a part the registry does not classify at all.
            if (i not in seen and (encounters is None or i in encounters) and info and not symptom_map.classify(part)
                    and set(info["symptoms"]) & set(tags or ())):
                hits.setdefault(part.casefold(), set()).add(i)
        return hits

    # 0-0-0 lines state no dose: they are not counted as given (core.encounters).
    meds_by_visit = [[_med_display(m.get("name")) for m in taken_medications(v)] for v in visits]
    distinct_meds = list(dict.fromkeys(m for meds in meds_by_visit for m in meds))
    # A registry drug pattern matches the name as written or, for any brand, its resolved generic names.
    generics = {m: (clinical_terms.medicine(patient_data, m) or {}).get("generic") or [] for m in distinct_meds}
    meds_matching = lambda pattern: [m for m in distinct_meds
                                     if re.match(pattern, m, re.I) or any(re.match(pattern, g, re.I) for g in generics[m])]

    def med_is(m, med_class, pattern, systemic=False):
        """True when a medicine is of a class: as clinical_terms resolved it, else by the registry pattern."""
        info = clinical_terms.medicine(patient_data, m)
        if info:
            return med_class in info["classes"] and not (systemic and info["route"] in ("topical", "drops", "inhaled"))
        return bool(re.search(rf"\b(?:{pattern})", m, re.I)) and not (systemic and re.search(_TOPICAL, m))

    def treats(m, condition):
        """True when a medicine is a specific treatment for a condition (resolved, else registry)."""
        info = clinical_terms.medicine(patient_data, m)
        if info and info["treats"]:
            return condition in info["treats"]
        pattern = REGISTRY["condition_meds"].get(condition)
        return bool(pattern and re.search(rf"\b(?:{pattern})", m, re.I))

    # The latest visit that has any prescription: what the patient is on now, as far as the record shows.
    last_rx = max((i for i, meds in enumerate(meds_by_visit, 1) if meds), default=0)

    def rx_status(names):
        """' · Ezact 90 on the latest prescription (#23)' or ' · not on the latest prescription (last #13, date)'."""
        key = lambda m: _first_word(m) or m.casefold()
        wanted = {key(n) for n in names}
        latest_day = _date(visits[last_rx - 1]) if last_rx else None
        latest = {key(m) for i, meds in enumerate(meds_by_visit, 1) if latest_day and _date(visits[i - 1]) == latest_day
                  for m in meds}
        unique = list({key(n)[:5]: n for n in reversed(names)}.values())[::-1]
        on = [n for n in unique if key(n) in latest]
        off = [n for n in unique if key(n) not in latest]
        if on and off:
            # Part of the finding only: the finding as a whole is not current.
            return (f" · not all on the latest prescription ({' + '.join(on)} on it, #{last_rx}; "
                    f"{' + '.join(off)} not)")
        if on:
            return f" · {' + '.join(on)} on the latest prescription (#{last_rx})"
        last = max((i for i, meds in enumerate(meds_by_visit, 1) if any(key(m) in wanted for m in meds)), default=None)
        if not last:
            return ""
        # A course written earlier that still runs on the latest visit date is current, not "not on" it.
        line = next((rec.line(last, m) for m in meds_by_visit[last - 1] if key(m) in wanted), None)
        if line and line["end"] and line["end"] >= _date(visits[-1]):
            return f" · course from #{last} ({_date(visits[last - 1])}) still running at the latest visit (to {line['end']})"
        return f" · not on the latest prescription (last #{last}, {_date(visits[last - 1])})"

    def regimen(condition, row):
        """(text, on_it_now) for a condition's medicines: 'on X' when they are on the latest prescription,
        'X last prescribed at #n (date), not on the latest prescription' when only earlier, else none recorded.
        The trend's main_medicines are the most-prescribed over the whole record, so they never say 'on' alone."""
        main = {_first_word(m) for m in (row or {}).get("main_medicines") or []} - {""}
        own = lambda m: treats(m, condition) or _first_word(m) in main
        rx = [i for i, meds in enumerate(meds_by_visit, 1) if any(own(m) for m in meds)]
        # When the condition's medicines were first prescribed: a result is "despite" them only if dated after this.
        regimen.first_date = _date(visits[rx[0] - 1]) if rx else None
        # Every encounter on the latest prescription day counts (two departments on one day are one prescription day).
        latest_day = _date(visits[last_rx - 1]) if last_rx else None
        current = list(dict.fromkeys(m for i, meds in enumerate(meds_by_visit, 1)
                                     if latest_day and _date(visits[i - 1]) == latest_day for m in meds if own(m)))
        if current:
            return "on " + " + ".join(current) + ("" if last_rx == len(visits) else f" (latest prescription #{last_rx})"), True
        if rx:
            names = list(dict.fromkeys(m for m in meds_by_visit[rx[-1] - 1] if own(m)))
            return (f"{' + '.join(names)} last prescribed at #{rx[-1]} ({_date(visits[rx[-1] - 1])}), "
                    "not on the latest prescription"), False
        return "no related medicine recorded", False

    docs = [d for v in visits for d in v.get("documents") or []] + list(patient_data.get("unlinked_documents") or [])
    age_sex = patient_data.get("age_sex") or next((_summary(d).get("age_sex") for d in docs
                                                   if _summary(d).get("age_sex")), None)
    # Every prescription line with its ingredients, route and course window; products the registry cannot map are
    # reported (UNMAPPED_BRANDS_REPORT), never skipped silently.
    rec = medicine_profile.record(patient_data)
    # Normalisation coverage: unmapped / ambiguous medicines, routes, doses, complaints and diagnoses (logged, counted).
    record_coverage = coverage.record_coverage(patient_data, rec)
    coverage.write(record_coverage)
    labs = _lab_rows(patient_data)
    covered = set()  # (test name, value) already reported by a condition-level flag
    flags = []
    # [(encounter, date, (systolic, diastolic))], structured readings that are physiologically possible.
    bp = [(i, d, tuple(int(x) for x in re.findall(r"\d+", val)[:2])) for i, d, val in _vital_series(visits, "BP")
          if len(re.findall(r"\d+", val)) >= 2 and not vital_implausible("BP", val)]
    # Every visit's own reading (the trend above keeps one per date; two visits on one day each have theirs).
    bp_at = {}
    for i, v in enumerate(visits, 1):
        for row in v.get("db_vitals_rows") or []:
            numbers = re.findall(r"\d+", str(row.get("value") or ""))
            if row.get("label") == "BP" and len(numbers) >= 2 and not vital_implausible("BP", row.get("value")):
                bp_at.setdefault(i, (int(numbers[0]), int(numbers[1])))

    # kind routes a flag in the AI review: lab (listed with the lab results), condition, pattern (course across
    # visits), risk, medication, followup or data (record quality).
    # source: the guideline behind a guideline-context flag (the AI reads those as guideline rows instead).
    def flag(level, title, evidence, significance="", kind="risk", source=None):
        flags.append({"level": level, "title": title, "evidence": evidence, "significance": significance, "kind": kind,
                      **({"source": source} if source else {})})

    def lab_key(lab):
        return (str(lab.get("test_name") or "").strip().casefold(), str(lab.get("value")))

    # Lab results the clinician must not miss: critical values, values that can't be trusted, and values
    # that nothing could evaluate. Statuses come from lab_validation.
    def lab_line(lab, src):
        return f"{lab.get('analyte') or lab.get('test_name')} {lab.get('value')} ({src})"

    critical = [(l, s) for l, s, _ in labs if str(l.get("status") or "").startswith("Critical")]
    if critical:
        covered.update(lab_key(l) for l, _ in critical)
        flag("high", "Critical lab value" + ("s" if len(critical) > 1 else ""),
             "; ".join(f"{lab_line(l, s)} {str(l.get('status')).split('(')[-1].rstrip(')')}" for l, s in critical[:4]),
             "Beyond critical limits; needs prompt clinical review.", kind="lab")
    unverified = [(l, s) for l, s, _ in labs if str(l.get("status") or "").startswith("Unverified")]
    if unverified:
        flag("medium", f"{len(unverified)} lab result(s) need checking against the source document",
             "; ".join(lab_line(l, s) for l, s in unverified[:4]) + (" …" if len(unverified) > 4 else ""),
             "Possible OCR misread or unreadable entry; not used in any interpretation until confirmed.", kind="lab")
    unevaluated = [(l, s) for l, s, _ in labs if str(l.get("status") or "").casefold() == "not evaluated"]
    rx_words = {_first_word(m) for m in distinct_meds} - {""}
    medicine_lines = [(l, s) for l, s in unevaluated if is_medicine_line(l, rx_words)]
    if medicine_lines:
        unevaluated = [x for x in unevaluated if x not in medicine_lines]
        flag("info", f"{len(medicine_lines)} medicine line(s) read as lab results",
             "OCR filed prescription lines under lab results; ignored in the analysis", kind="data")
    if unevaluated:
        flag("info", f"{len(unevaluated)} lab result(s) not evaluated",
             "; ".join(lab_line(l, s) for l, s in unevaluated[:4]) + (" …" if len(unevaluated) > 4 else ""),
             "No reference range in the report and none on file for this test.", kind="lab")

    # Guideline context: the record against Indian guideline cut-offs for this patient's age, sex and conditions.
    profile = _patient_profile(age_sex, visits)
    guideline, guideline_monitored = guideline_context(patient_data, labs, present, profile, bp, treats, med_is, flag)
    # A result in a band suggesting a condition the record does not note (registry undiagnosed_lab_bands): "probable".
    for rule in REGISTRY.get("undiagnosed_lab_bands") or []:
        treated = any(rule.get("treated_by_class") in l["classes"] for l in rec.prescribed())
        if present([rule["condition"]]) or treated:
            continue
        hit = _latest_result(labs, [rule["analyte"]])
        value = _lab_number(hit[0], rule["unit"]) if hit else None
        band = next((b for b in rule["bands"] if value is not None and value >= b["from"]
                     and (b["to"] is None or value < b["to"])), None)
        if band:
            when = hit[1] or "undated report"
            flag(band["level"], f"{band['title']} (HbA1c {value:g} {rule['unit']})" if rule["analyte"] == "hba1c"
                 else f"{band['title']} ({rule['label']} {value:g} {rule['unit']})",
                 f"{rule['label']} {value:g} {rule['unit']} ({when}) · no {rule['condition'].lower()} diagnosis, complaint, "
                 f"long-term entry or treating medicine in the record", band["meaning"], kind="condition")

    # A diabetic whose latest HbA1c is at target is not "out of range" because the report prints a non-diabetic range.
    at_target = {r["condition"] for r in guideline_rows_in_use(guideline) if r.get("marker") and r["status"] == "at target"}

    # Per-condition control markers, for every documented condition that has one.
    abnormal_control = {}
    for name, row in conditions.items():
        rule = _CONTROL_LABS.get(name)
        related_pattern = rule["pattern"] if rule else _CONDITION_LABS.get(name)
        if not related_pattern:
            continue
        meds, on_therapy = regimen(name, row)
        therapy_from = regimen.first_date
        related = [(l, s) for l, s, _ in labs if re.search(related_pattern, str(l.get("test_name") or ""), re.I)]
        out = [(l, s) for l, s in related if rule and _out_of_range(l, rule["direction"]) and not _implausible(l)]
        if name in at_target:
            out = [(l, s) for l, s in out
                   if lab_validation.identify(l.get("test_name")) != _GUIDELINES["diabetes_control"]["analyte"]]
        if out:
            abnormal_control[name] = out
            covered.update(lab_key(l) for l, _ in out)
            values = "; ".join(f"{re.sub(r'(?i)glycosylated haemoglobin', '', str(l.get('test_name'))).strip()} "
                               f"{l.get('value')} ({s})" for l, s in out[:3])
            # "despite current therapy" needs every shown result dated after the therapy started.
            dated = [next((d for x, _, d in labs if x is lab), None) for lab, _ in out[:3]]
            if on_therapy and therapy_from and dated and all(d and d > therapy_from for d in dated):
                timing = " despite current therapy."
            elif on_therapy and therapy_from and any(d and d <= therapy_from for d in dated):
                # Medicines already there at the first recorded visit were not "started" then: only first recorded.
                verb = "were first recorded" if therapy_from == _date(visits[0]) else "started"
                timing = f"; the medicines {verb} on or after this result ({therapy_from})."
            elif on_therapy:
                timing = "; result undated, so it cannot be placed against the treatment start."
            else:
                timing = "."
            flag(rule["level"], f"{name}: {rule['label']} out of range", f"{values} · {meds}",
                 "Values outside the reference range" + timing,
                 kind="condition")
        elif not related and name not in guideline_monitored:  # else the guideline monitoring row covers it
            label = rule["label"] if rule else "related monitoring tests"
            if name == "Hypertension" and bp:
                # BP readings are the control marker; only the related lab tests are missing.
                flag("info", f"{name}: no related lab tests in the record",
                     f"BP recorded at {len(bp)} visit(s) · {meds}",
                     "Control is shown by the BP readings; kidney function, electrolytes and lipids are not in the record.",
                     kind="followup")
                continue
            flag("medium" if rule else "info", f"{name} without {label} in the record",
                 f"Followed {row['first_seen']} to {row['last_seen']} ({row['visits']} visits) · {meds}",
                 "Control cannot be assessed from the record.", kind="followup")

    # Condition-linked symptoms from the chief complaints.
    for name in list(conditions) + sorted(long_term - set(conditions)):
        for rule in REGISTRY["condition_symptoms"].get(name, []):
            hits = symptom_hits(rule["pattern"], rule.get("tags"))
            if rule.get("requires_raised_bp"):
                # Not "raised BP" symptoms at a visit whose own reading is below 140/90.
                hits = {p: {i for i in e if i not in bp_at or bp_at[i][0] >= 140 or bp_at[i][1] >= 90}
                        for p, e in hits.items()}
                hits = {p: e for p, e in hits.items() if e}
            if hits:
                # "Consistent with hyperglycaemia" needs an out-of-range glucose / HbA1c; weakness alone is not.
                if rule.get("requires_abnormal_control") and name not in abnormal_control:
                    continue
                if any(f["title"] == rule["title"] for f in flags):  # a rule shared by several conditions fires once
                    continue
                flag(rule["level"], rule["title"], "; ".join(f"{p} ({_encs(sorted(e))})" for p, e in hits.items()),
                     rule["significance"])

    # Symptom combinations (registry symptom_combinations): a symptom concept with a condition or another symptom.
    for combo in REGISTRY.get("symptom_combinations") or []:
        hits = symptom_hits(None, combo["tags"])
        if not hits:
            continue
        conditions_with = present(combo.get("with_conditions") or [])
        partners = symptom_hits(None, combo.get("with_tags")) if combo.get("with_tags") else {}
        if conditions_with or partners:
            flag(combo["level"], combo["title"],
                 "; ".join(f"{p} ({_encs(sorted(e))})" for p, e in list(hits.items()) + list(partners.items()))
                 + (f" · {_join(conditions_with)}" if conditions_with else ""), combo["significance"])

    # Symptoms that can be cardiac, in a patient with cardiovascular risk (a condition, or age).
    red = REGISTRY["red_flag_symptoms"]
    age = re.search(r"(\d{1,3})\s*(?:y|yrs?|years?)\b", str(patient_data.get("age_sex") or ""), re.I)
    # Adult cut-offs (BMI, pulse, low BP) do not apply under 18; an unknown age is treated as adult.
    adult = not (age and int(age.group(1)) < 18)
    risk = present(red["risk_conditions"]) + ([f"age {age.group(1)}"] if age and int(age.group(1)) >= red["risk_age"] else [])
    red_hits = symptom_hits(red["pattern"], red.get("tags")) if risk else {}
    if red_hits:
        flag(red["level"], red["title"], "; ".join(f"{p} ({_encs(sorted(e))})" for p, e in red_hits.items())
             + f" · {_join(risk[:3])}",
             f"With {_join(risk[:3])}, these symptoms can have a cardiac cause.")

    # Symptoms the complaint says have lasted months or years: the record's own span understates the course.
    record_days = _days(visits[0], visits[-1]) or 0
    long_standing = []
    pregnancy = (_GUIDELINES.get("haemoglobin") or {}).get("pregnancy_pattern")
    for i, part in parts:
        if pregnancy and re.search(pregnancy, part, re.I):
            continue  # '4 month pregnancy' is how far along she is, not a symptom lasting months
        # The written duration ("since 8-10yr"), else the one clinical_terms read ("for two winters").
        days = _stated_duration_days(part) or (clinical_terms.complaint(patient_data, part) or {}).get("duration_days")
        if days and days >= 90 and days > record_days:
            long_standing.append(f"{part} (#{i})")
    if long_standing:
        flag("medium", "Long-standing symptom" + ("s" if len(long_standing) > 1 else ""),
             _listed(long_standing, 3),
             "Duration stated in the complaint is longer than the period this record covers.", kind="pattern")

    # An ulcer, lump, bleeding or weight loss that has lasted weeks (stated duration, or seen at visits weeks apart).
    persistent = REGISTRY["persistent_red_flags"]
    if not age or int(age.group(1)) >= persistent["min_age"]:
        lasting = []
        for i, part in parts:
            info = clinical_terms.complaint(patient_data, part) or {}
            if not (re.search(persistent["pattern"], part, re.I) or set(info.get("symptoms") or ()) & set(persistent["tags"])):
                continue
            days = _stated_duration_days(part) or info.get("duration_days") or 0
            later = [j for j, other in parts if j > i and other.casefold() == part.casefold()]
            span = _days(visits[i - 1], visits[later[-1] - 1]) if later else 0
            if max(days, span or 0) >= persistent["min_days"]:
                lasting.append(f"{part} (#{i})")
        if lasting:
            flag(persistent["level"], persistent["title"], _listed(lasting, 3),
                 persistent["significance"])

    # Blood pressure trend
    htn = conditions.get("Hypertension")
    known_htn = bool(htn) or "Hypertension" in long_term
    # One BP target for the flag title, its evidence and the timeline row (130/80 with diabetes / CKD / CAD / HF).
    lower_target = present(REGISTRY["bp_lower_target_conditions"])
    target = (130, 80) if lower_target else (140, 90)
    # "at/above" because either number counts: 128/94 is above 140/90 on the diastolic.
    target_label = f"at/above {target[0]}/{target[1]} target"
    raised_rule = f"systolic ≥{target[0]} or diastolic ≥{target[1]}"
    raised = [i for i, _, (s, dia) in bp if s >= target[0] or dia >= target[1]]
    # BP targets apply to patients the targets are for: hypertension, or a condition with a BP target (diabetes, CKD,
    # CAD, heart failure). Raised readings in anyone else are not judged against a target.
    bp_patient = known_htn or bool(lower_target)
    antihypertensive = lambda m: treats(m, "Hypertension") or med_is(m, "Antihypertensive", _CONDITION_MEDS_HTN)
    if bp and bp_patient:
        last = bp[-1]
        recent, earlier = bp[-3:], bp[:-3]
        evidence = "Last " + (f"{len(recent)} readings: " if len(recent) > 1 else "reading: ") + ", ".join(
            f"{s_}/{d_} ({d})" for _, d, (s_, d_) in recent)
        bp_trend = None
        if earlier:
            mean_recent = sum(r[2][0] for r in recent) / len(recent)
            mean_earlier = sum(r[2][0] for r in earlier) / len(earlier)
            bp_trend = ("lower" if mean_recent <= mean_earlier - 5 else "higher" if mean_recent >= mean_earlier + 5
                     else "similar")
            evidence += (f" · {bp_trend} {'to' if bp_trend == 'similar' else 'than'} the {len(earlier)} earlier reading(s) "
                         f"(mean systolic {mean_earlier:.0f} → {mean_recent:.0f})")
        if raised:
            evidence += f" · {raised_rule} at {_encs(raised)}"
        # The last change in BP treatment (per clinical day with a prescription) and the readings since it.
        # Names kept in prescription order (a list, not a set), so the text is the same on every run.
        day_sets = [(date, {_first_word(m) or m for i in encs for m in meds_by_visit[i - 1] if antihypertensive(m)},
                     list(dict.fromkeys(m for i in encs for m in meds_by_visit[i - 1] if antihypertensive(m))))
                    for date, encs in clinical_days(visits) if any(meds_by_visit[i - 1] for i in encs)]
        change = None
        for (pd, prev, prev_names), (cd, cur, cur_names) in zip(day_sets, day_sets[1:]):
            if cur != prev:
                added = [n for n in cur_names if (_first_word(n) or n) not in prev]
                # A medicine whose last course still runs on this day was not stopped (core.encounters.course_end).
                stopped = [n for n in prev_names if (_first_word(n) or n) not in cur
                           and not any(l["name"] == n and rec.running_on(l, cd) for row in rec.lines for l in row)]
                change = (cd, "; ".join(filter(None, [f"{' + '.join(dict.fromkeys(added))} added" if added else "",
                                                       f"{' + '.join(dict.fromkeys(stopped))} not on the latest prescription" if stopped else ""])))
        if change and change[1]:
            since = [f"{s_}/{d_}" for _, d, (s_, d_) in bp if d and d >= change[0]]
            evidence += f" · last treatment change {change[0]}: {change[1]}" + (
                f"; readings since: {', '.join(since[-3:])}" if since else "; no reading since")
        if htn:
            meds, _ = regimen("Hypertension", htn)
            if meds != "no related medicine recorded":
                evidence += " · " + meds
        latest_raised = last[2][0] >= target[0] or last[2][1] >= target[1]
        if latest_raised:
            significance = (("Recent readings are lower than earlier ones, but the latest is not" if bp_trend == "lower"
                             else "Not") + " at the usual target" + (f" with {_join(lower_target)}." if lower_target else "."))
            flag("medium", f"BP {target_label} at latest visit", evidence, significance)
        elif raised and bp_trend == "lower":
            flag("info", "BP improved; earlier readings raised", evidence)
        elif raised:
            flag("medium", f"BP {target_label} at {len(raised)} of {len(bp)} readings", evidence,
                 f"Latest reading below target ({target[0]}/{target[1]}), but not consistently controlled.")

    if bp:  # low readings and dizziness on BP-lowering medicine apply to every patient
        # Low readings, and dizziness / weakness while BP is not raised on BP-lowering medicine.
        # Low means a low systolic; an isolated low diastolic with a normal systolic (138/51) is not low BP.
        low = [(i, d, r) for i, d, r in bp if r[0] < 90 or (r[1] < 60 and r[0] < 100)]
        if low and adult:
            flag("medium", "Low BP reading" + ("s" if len(low) > 1 else ""),
                 _listed([f"{s}/{dia} ({d}, #{i})" for i, d, (s, dia) in low], 3),
                 "Systolic below 90 (or below 100 with diastolic under 60).")
        antihypertensives = [(i, m) for i, meds in enumerate(meds_by_visit, 1) for m in meds
                             if treats(m, "Hypertension") or med_is(m, "Antihypertensive", _CONDITION_MEDS_HTN)]
        symptom_pattern = REGISTRY["low_bp_symptoms"]
        episodes, on_meds = [], []
        for i, _, (s, dia) in bp:
            on = [m for j, m in antihypertensives if j == i]
            phrases = list(symptom_hits(symptom_pattern, REGISTRY["low_bp_symptom_tags"], encounters={i}))
            if on and phrases and s < 120 and dia < 80:
                episodes.append(f"{', '.join(phrases[:3])} at #{i} (BP {s}/{dia})")
                on_meds += on
        if episodes:
            flag("medium", "Dizziness / weakness with BP not raised, on BP-lowering medicine",
                 "; ".join(episodes[-3:]) + f" · on {', '.join(dict.fromkeys(on_meds))}",
                 "The readings do not support raised BP as the cause; low BP or a medicine effect "
                 "(e.g. postural drop) is possible.", kind="medication")

    # Weight readings that go A -> B -> about A within the window: B is an entry error, not a change
    # (clinical_thresholds.json weight_entry_error). Errors are left out of BMI and every weight-change rule.
    we = THRESHOLDS["weight_entry_error"]
    all_weights = [(i, _date(v), _num(r.get("value"))) for i, v in enumerate(visits, 1) for r in v.get("db_vitals_rows") or []
                   if r.get("label") == "Weight" and _num(r.get("value")) and not vital_implausible("Weight", r.get("value"))]
    entry_errors = set()
    for k in range(1, len(all_weights) - 1):
        (_, da, a), (j, db, b), (_, dc, c) = all_weights[k - 1], all_weights[k], all_weights[k + 1]
        if (0 <= _days_between(da, dc) <= we["window_days"] and abs(b - a) / a * 100 >= we["min_excursion_pct"]
                and abs(c - a) / a * 100 <= we["return_within_pct"]):
            entry_errors.add((j, db, b))
    if entry_errors:
        flag("info", "Weight entry error", "; ".join(
            f"{b:g} kg (#{j}, {d}) between {all_weights[all_weights.index((j, d, b)) - 1][2]:g} and "
            f"{all_weights[all_weights.index((j, d, b)) + 1][2]:g} kg within {we['window_days']} days" for j, d, b in sorted(entry_errors)),
             "The weight returns to its earlier value: a recording error, not a clinical change. Not used for BMI or weight change.",
             kind="data")

    # Weight / BMI
    weights = [(i, d, _num(v)) for i, d, v in _vital_series(visits, "Weight")
               if _num(v) and not vital_implausible("Weight", v) and (i, d, _num(v)) not in entry_errors]
    heights = [_num(v) for _, _, v in _vital_series(visits, "Height") if _num(v) and not vital_implausible("Height", v)]
    if weights and heights:
        height = Counter(round(h) for h in heights).most_common(1)[0][0]
        if adult and 120 <= height <= 220:
            bmi = weights[-1][2] / (height / 100) ** 2
            cardiometabolic = present(REGISTRY["cardiometabolic_conditions"])
            # Population cut-offs (clinical_thresholds.json bmi; BMI_POPULATION, default Asian Indian), with the WHO
            # category always shown alongside.
            bmi_rules = THRESHOLDS["bmi"]
            population = os.getenv("BMI_POPULATION", bmi_rules["default_population"]).strip().casefold()
            cut = bmi_rules["populations"].get(population) or bmi_rules["populations"][bmi_rules["default_population"]]
            who = bmi_rules["populations"]["who"]

            def category(rule):
                return ("obese" if bmi >= rule["obese_from"] else "overweight" if bmi >= rule["overweight_from"]
                        else "underweight" if bmi < rule["underweight_below"] else "normal range")
            category_used, category_who = category(cut), category(who)
            alongside = "" if cut is who else f"; WHO: {category_who}"
            label = f"{category_used}, {cut['label']} cut-off{alongside}"
            if category_used in ("obese", "overweight"):
                flag("medium" if category_used == "obese" else "info", f"BMI {bmi:.1f} ({label})",
                     f"Weight {weights[0][2]:g} → {weights[-1][2]:g} kg, height ~{height} cm",
                     f"Adds cardiometabolic risk alongside {_join(cardiometabolic)}." if cardiometabolic else "")
            elif category_used == "underweight":
                flag("medium", f"BMI {bmi:.1f} ({label})", f"Weight {weights[-1][2]:g} kg, height ~{height} cm",
                     "Low body weight can reflect undernutrition or an underlying illness.")
            guideline.append({"topic": "BMI", "kind": "measure", "status": category_used,
                              "finding": f"BMI {bmi:.1f} ({weights[-1][1]}): {category_used} ({cut['label']} cut-off: "
                                         f"overweight from {cut['overweight_from']:g}, obese from {cut['obese_from']:g})"
                                         + (f"; WHO category: {category_who}" if alongside else ""),
                              "source": cut["source"], "section": "bmi", "verified": True, "verified_by": "configuration",
                              "verified_on": None})
        he = THRESHOLDS["height_entry_error"]
        grown = age and int(age.group(1)) >= he["adult_from_age"] or not age
        if grown and max(heights) - min(heights) > he["max_variation_cm"]:
            flag("info", f"Height varies by {max(heights) - min(heights):g} cm (entry error)",
                 f"Height recorded between {min(heights):g} and {max(heights):g} cm; BMI uses the most frequent, ~{height} cm",
                 "An adult's height does not change: a recording error, not a clinical change.", kind="data")

    # Weight readings that jump more than a body can change in the time between them (every reading, same-day too).
    raw_weights = [w for w in all_weights if w not in entry_errors]
    jumps = []
    for (i, d1, w1), (j, d2, w2) in zip(raw_weights, raw_weights[1:]):
        gap = _days_between(d1, d2)
        if 0 <= gap <= 90 and abs(w2 - w1) / max(w1, w2) >= 0.15:
            jumps.append(f"{w1:g} kg (#{i}) → {w2:g} kg (#{j}, {gap} days later)")
    if jumps:
        flag("medium", "Weight readings inconsistent", _listed(jumps, 3),
             "Too large a change for the time between readings: likely an entry error, so weight and BMI are unreliable.",
             kind="data")

    # Real weight change: 5% or more within 30 days (15% or more is the entry-error rule above).
    changes = []
    for (i, d1, w1), (j, d2, w2) in zip(raw_weights, raw_weights[1:]):
        gap = _days_between(d1, d2)
        pct = abs(w2 - w1) / max(w1, w2)
        if 0 < gap <= 30 and 0.05 <= pct < 0.15:
            changes.append(f"{w1:g} kg (#{i}, {d1}) → {w2:g} kg (#{j}, {d2}): {'+' if w2 > w1 else '−'}{pct * 100:.0f}% in {gap} days")
    if changes:
        flag("medium", "Weight change of 5% or more within 30 days", _listed(changes, 3),
             "A change this fast is clinically significant if the readings are right.")

    # Pulse
    fast = [(i, _num(v)) for i, _, v in _vital_series(visits, "Pulse")
            if (_num(v) or 0) > 100 and not vital_implausible("Pulse", v)]
    if fast and adult:  # a child's normal pulse is above 100
        flag("info", "Pulse above 100", ", ".join(f"{p:g} bpm at #{i}" for i, p in fast))

    # ── Medicine rules on ingredients and routes (medicine_profile) ──────────────────────────────────────
    # Every class rule reads the classes of each prescription LINE, which come from its active ingredients (a brand is
    # expanded first). A rule that needs an oral / systemic route runs on lines whose route is known; when it would
    # only fire by counting lines whose route is unclear (a name with no form, no template) as oral, it is reported
    # for checking: level high becomes "verify" and the evidence names the line(s) with no route.
    rules = REGISTRY.get("drug_rules") or {}
    day_list = clinical_days(visits)
    class_patterns = [(re.compile(e["pattern"], re.I), e["class"]) for e in REGISTRY["medicine_classes"]]

    # Conditions in this record that an antiplatelet treats (registry condition_meds), e.g. coronary artery disease.
    antiplatelet_for = [c for c, pat in REGISTRY["condition_meds"].items()
                        if present([c]) and re.search(rf"\b(?:{pat})", "aspirin clopidogrel", re.I)]

    def classes_of(m, i=None):
        line = rec.line(i, m) if i else next((l for row in rec.lines for l in row if l["name"] == m), None)
        if line:
            found = set(line["classes"])
        else:
            info = clinical_terms.medicine(patient_data, m)
            found = set(info["classes"]) if info and info.get("classes") else {c for pat, c in class_patterns if pat.search(m)}
        # Aspirin in a patient with a condition it is the antiplatelet treatment for is that treatment, not a pain
        # NSAID: the NSAID risk rules (worsens CAD / BP / kidneys) do not read it.
        if antiplatelet_for and {"Antiplatelet", _NSAID_CLASS} <= found:
            found.discard(_NSAID_CLASS)
        return found

    def ingredients_of(m, i=None):
        line = rec.line(i, m) if i else next((l for row in rec.lines for l in row if l["name"] == m), None)
        return set(line["ingredients"]) if line else set(generics.get(m) or [])

    def same_drug(a, b, ia=None, ib=None):
        ga, gb = ingredients_of(a, ia), ingredients_of(b, ib)
        if ga and gb:
            return ga == gb  # Calcium + D3 and plain D3 are two products (vitamin D twice), not one
        wa, wb = _first_word(a), _first_word(b)
        return bool(wa and wb and (wa[:4] == wb[:4] or SequenceMatcher(None, wa, wb).ratio() >= 0.75))

    def matches(m, pattern):
        return bool(pattern and (re.search(pattern, m, re.I) or any(re.search(pattern, g, re.I) for g in generics.get(m, []))
                                 or any(re.search(pattern, g, re.I) for g in ingredients_of(m))))

    def where(date, encs):
        return f"{date or 'date unknown'} ({_encs(encs)})"

    def systemic_at(mode):
        """(m, i) -> True when line m at visit i is systemic; 'possible' also counts lines with an unclear route."""
        allowed = {"yes"} if mode == "definite" else {"yes", "unclear"}
        return lambda m, i: rec.systemic(i, m) in allowed

    unclear_lines = [l for row in rec.lines for l in row if l["route"] == "unclear" and l["prescribed"]]

    def route_rule(rule):
        """Run a rule that needs an oral route; report it for checking if only unclear-route lines make it fire."""
        before = len(flags)
        rule(systemic_at("definite"), flag)
        if len(flags) > before:
            return
        probe = []
        rule(systemic_at("possible"), lambda level, title, evidence, significance="", kind="risk", source=None:
             probe.append((level, title, evidence, significance, kind)))
        for level, title, evidence, significance, kind in probe:
            named = list(dict.fromkeys(f"{l['name']} (#{l['i']})" for l in unclear_lines
                                       if re.search(rf"\b{re.escape(l['name'])}\b", evidence, re.I)))
            flag("verify" if level == "high" else level, f"{title} (route unclear)",
                 evidence + f" · route not written for {_join(named[:3]) or 'a medicine'}; counted as oral only for this check",
                 (significance + " " if significance else "") + "Fires only if the line(s) with no written form are oral: "
                 "check the form prescribed.", kind=kind)

    oral_nsaid_definite = lambda m, i: _NSAID_CLASS in classes_of(m, i) and rec.systemic(i, m) == "yes"
    topical_nsaid = lambda m, i: _NSAID_CLASS in classes_of(m, i) and rec.systemic(i, m) == "no"
    nsaid_visits = [i for i, meds in enumerate(meds_by_visit, 1) if any(oral_nsaid_definite(m, i) for m in meds)]
    nsaid_risk = present(REGISTRY["nsaid_risk_conditions"])
    topical = sorted({m for i, meds in enumerate(meds_by_visit, 1) for m in meds if topical_nsaid(m, i)})

    # Repeated oral NSAIDs with a condition they can worsen (topical gels counted separately)
    def repeated_nsaids(systemic, emit):
        oral = lambda m, i: _NSAID_CLASS in classes_of(m, i) and systemic(m, i)
        visits_on = [i for i, meds in enumerate(meds_by_visit, 1) if any(oral(m, i) for m in meds)]
        if len(visits_on) >= 2 and nsaid_risk:
            names = [m for m, _ in Counter(m for i, meds in enumerate(meds_by_visit, 1) for m in meds
                                           if oral(m, i)).most_common()]
            emit("high" if len(nsaid_risk) >= 2 else "medium", f"Repeated oral NSAIDs with {_join(nsaid_risk)}",
                 f"{', '.join(names[:3])} at {len(visits_on)} visits on "
                 f"{len({day_of_encounter(visits)[i] for i in visits_on})} days ({_encs(visits_on)})"
                 + (f" · plus topical {', '.join(topical[:2])}" if topical else "") + rx_status(names),
                 "NSAIDs can raise BP, worsen fluid retention and affect kidney function.", kind="medication")
        elif visits_on and nsaid_risk:
            names = list(dict.fromkeys(m for i in visits_on for m in meds_by_visit[i - 1] if oral(m, i)))
            emit("medium", f"Oral NSAID with {_join(nsaid_risk)}", f"{', '.join(names[:3])} at {_encs(visits_on)}",
                 "NSAIDs can raise BP, worsen fluid retention and affect kidney function.", kind="medication")
    route_rule(repeated_nsaids)
    if not nsaid_visits and topical and nsaid_risk and not any("NSAID" in f["title"] for f in flags):
        flag("info", "Topical NSAIDs in record", ", ".join(topical[:3]), kind="medication")

    # ── Duplication and interactions, per clinical day (all encounters on one date are one regimen) ──────────
    def duplicates(systemic):
        out = []
        for date, encs in day_list:
            for cls in rules.get("duplicate_classes", []):
                members = [(m, i) for i in encs for m in meds_by_visit[i - 1]
                           if cls in classes_of(m, i) and (cls != _NSAID_CLASS or systemic(m, i))]
                drugs = []
                for m, i in members:
                    if not any(same_drug(m, d, i, di) for d, di in drugs):
                        drugs.append((m, i))
                # Two products of the class, or one product prescribed again at a second encounter that day.
                repeated = [d for d, di in drugs if len({i for m, i in members if same_drug(m, d, i, di)}) >= 2]
                if len(drugs) >= 2 or repeated:
                    names = [d for d, _ in drugs] if len(drugs) >= 2 else [f"{repeated[0]} twice"]
                    out.append((cls, names, date, encs))
        return out

    def dual_nsaids(systemic, emit):
        found = [d for d in duplicates(systemic) if d[0] == _NSAID_CLASS]
        if found:
            emit("high", "Two oral NSAIDs on the same day",
                 f"On {len(found)} day(s): "
                 + "; ".join(f"{' + '.join(d[1][:3])} on {where(d[2], d[3])}" for d in found[:3])
                 + (f" (+{len(found) - 3} more days)" if len(found) > 3 else "")
                 + rx_status([n for d in found for n in d[1] if not n.endswith(" twice")]),
                 "Two NSAIDs together add stomach-bleeding, kidney and BP risk without added pain relief.", kind="medication")
    # Days with two oral NSAIDs for certain stay high; days that need an unclear-route line to count are "verify".
    definite_days = {d[2] for d in duplicates(systemic_at("definite")) if d[0] == _NSAID_CLASS}
    dual_nsaids(systemic_at("definite"), flag)
    possible_days = [d for d in duplicates(systemic_at("possible")) if d[0] == _NSAID_CLASS and d[2] not in definite_days]
    if possible_days:
        named = list(dict.fromkeys(f"{l['name']} (#{l['i']})" for l in unclear_lines if l["date"] in {d[2] for d in possible_days}
                                   and _NSAID_CLASS in l["classes"]))
        flag("verify", "Two NSAIDs on the same day (route unclear)",
             f"On {len(possible_days)} day(s): "
             + "; ".join(f"{' + '.join(d[1][:3])} on {where(d[2], d[3])}" for d in possible_days[:3])
             + f" · route not written for {_join(named[:3])}",
             "If the line(s) with no written form are tablets, this is two oral NSAIDs on one day; a gel would not be. "
             "Check the form prescribed.", kind="medication")
    other_dups = [d for d in duplicates(systemic_at("definite")) if d[0] != _NSAID_CLASS]
    # Calcium / vitamin D twice, or a PPI with a plain antacid, carry little risk: listed last and at info level.
    minor = lambda d: d[0] == "Calcium / vitamin D supplement" or (
        d[0].startswith("Acid suppressant") and any(re.search(r"(?i)antacid|gelusil|digene|sucral", n) for n in d[1]))
    other_dups.sort(key=minor)
    if other_dups:
        # One part per class, so a class with few days (calcium twice) is never hidden behind another's list.
        by_class = {}
        for d in other_dups:
            by_class.setdefault(d[0], []).append(d)
        dup_parts = [f"{' + '.join(ds[0][1][:3])} ({cls.split(' (')[0]}) on {where(ds[0][2], ds[0][3])}"
                     + (f" and {len(ds) - 1} more day(s)" if len(ds) > 1 else "") for cls, ds in by_class.items()]
        flag("info" if all(minor(d) for d in other_dups) else "medium", "Same-class medicines on the same day",
             f"On {len({d[2] for d in other_dups})} day(s): " + "; ".join(dup_parts),
             "Two medicines of one class on one day, often from two departments: a duplicate, not added benefit.",
             kind="medication")

    # Cumulative oral NSAID exposure (every oral NSAID ingredient after brand expansion): stated durations added, one
    # value per clinical day. Lines with no written route are counted apart, as possible additional days.
    def nsaid_exposure(systemic):
        total, rx_days, names = 0.0, [], []
        for date, encs in day_list:
            lines = [(m, i) for i in encs for m in taken_medications(visits[i - 1])
                     if _NSAID_CLASS in classes_of(_med_display(m.get("name")), i) and systemic(_med_display(m.get("name")), i)]
            if lines:
                rx_days.append((date, encs))
                total += max((duration_days(m) or 0 for m, _ in lines), default=0)
                names += [_med_display(m.get("name")) for m, _ in lines]
        return total, rx_days, list(dict.fromkeys(names))
    nsaid_total, nsaid_rx_days, nsaid_names = nsaid_exposure(systemic_at("definite"))
    possible_total, _, possible_names = nsaid_exposure(systemic_at("possible"))
    extra = possible_total - nsaid_total
    unclear_named = [n for n in possible_names if n not in nsaid_names] or \
        list(dict.fromkeys(l["name"] for l in unclear_lines if _NSAID_CLASS in l["classes"]))
    extra_note = (f" · possible additional {extra:g} days where the route is not written ({', '.join(unclear_named[:3])})"
                  if extra > 0 else "")
    limit = rules.get("nsaid_cumulative_days", 30)
    if nsaid_total >= limit:
        flag("medium", f"Oral NSAIDs for about {nsaid_total:g} days in total",
             f"Stated durations added over {len(nsaid_rx_days)} prescription day(s): "
             + _encs([i for _, encs in nsaid_rx_days for i in encs]) + f" ({', '.join(nsaid_names[:8])})" + extra_note,
             "Longer cumulative NSAID use raises stomach-bleeding, kidney and BP risk.", kind="medication")
    elif possible_total >= limit:
        flag("medium", f"Oral NSAIDs for about {possible_total:g} days in total (route unclear)",
             f"{nsaid_total:g} days confirmed oral ({', '.join(nsaid_names[:8]) or 'none'})" + extra_note,
             "Reaches the cumulative limit only if the lines with no written form are tablets: check the form prescribed.",
             kind="medication")

    # Oral azole antifungal with a calcium-channel blocker (same day, or a CCB prescribed in the days before).
    is_azole_line = lambda m, i: matches(m, rules.get("systemic_azole")) or "azole" in medicine_profile.groups(ingredients_of(m, i)) \
        and any(g in ingredients_of(m, i) for g in ("fluconazole", "itraconazole", "ketoconazole", "voriconazole", "posaconazole"))
    is_ccb = lambda m: matches(m, rules.get("ccb"))
    ccb_days = [(date, [m for i in encs for m in meds_by_visit[i - 1] if is_ccb(m)]) for date, encs in day_list]
    ccb_days = [d for d in ccb_days if d[1]]

    def azole_days_for(systemic):
        days = [(date, encs, [m for i in encs for m in meds_by_visit[i - 1] if is_azole_line(m, i) and systemic(m, i)])
                for date, encs in day_list]
        return [d for d in days if d[2]]

    def azole_ccb(systemic, emit):
        azole_days, combos = azole_days_for(systemic), []
        for date, encs, azoles in azole_days:
            recent = [c for c in ccb_days if c[0] and date and 0 <= _days_between(c[0], date) <= rules.get("ccb_lookback_days", 30)]
            if recent:
                when = "same day" if recent[-1][0] == date else recent[-1][0]
                combos.append(f"{' + '.join(dict.fromkeys(azoles))} on {where(date, encs)} with "
                              f"{' + '.join(dict.fromkeys(recent[-1][1]))} ({when})")
        if combos:
            emit("medium", "Oral azole antifungal with a calcium-channel blocker",
                 f"On {len(combos)} day(s): " + "; ".join(combos[:3]) + (f" (+{len(combos) - 3} more days)" if len(combos) > 3 else "")
                 + rx_status([m for _, _, ms in azole_days for m in ms] + [m for _, ms in ccb_days for m in ms]),
                 "Azole antifungals raise calcium-channel-blocker levels: ankle swelling, dizziness, low BP.", kind="medication")
    route_rule(azole_ccb)

    # Repeated / long oral azole with no liver test anywhere in the record.
    lab_keys = {lab_validation.identify(l.get("test_name")) for l, _, _ in labs}
    long_rule = rules.get("azole_long_term") or {}

    def azole_liver(systemic, emit):
        azole_days = azole_days_for(systemic)
        total = sum(max((duration_days(m) or 0 for i in encs for m in taken_medications(visits[i - 1])
                         if is_azole_line(_med_display(m.get("name")), i) and systemic(_med_display(m.get("name")), i)),
                        default=0) for _, encs, _ in azole_days)
        if (azole_days and (len(azole_days) >= long_rule.get("min_days", 2) or total >= long_rule.get("min_duration_days", 14))
                and not lab_keys & set(rules.get("liver_tests") or [])):
            names = list(dict.fromkeys(m for _, _, ms in azole_days for m in ms))
            emit("medium", "Repeated oral azole antifungal without liver tests",
                 f"{', '.join(names[:2])} on {len(azole_days)} day(s) ({_encs([i for _, e, _ in azole_days for i in e])})"
                 + (f", about {total:g} days stated" if total else "") + " · no liver test in the record" + rx_status(names),
                 "Oral azoles can affect the liver; no liver function result is in the record.", kind="followup")
    route_rule(azole_liver)

    # Expected investigations that the record lacks (registry expected_investigations): stated as a record gap.
    visit_text = {i: " ".join(str(v.get(k) or "") for k in ("db_chief_complaint", "chief_complaint", "db_provisional_dx",
                                                              "db_confirmed_dx", "provisional_diagnosis", "confirmed_diagnosis"))
                  for i, v in enumerate(visits, 1)}
    for rule in REGISTRY.get("expected_investigations") or []:
        if lab_keys & set(rule["tests"]):
            continue
        if rule.get("requires_conditions") and not present(rule["requires_conditions"]):  # [] = any patient
            continue
        tagged = {i for e in symptom_hits(None, rule.get("symptom_tags")).values() for i in e} if rule.get("symptom_tags") else set()
        hit_days = [(date, encs) for date, encs in day_list
                    if any(re.search(rule["pattern"], visit_text[i], re.I) or i in tagged for i in encs)]
        temps = []
        if rule.get("temp_f_at_least"):
            temps = [(i, row.get("value")) for i, v in enumerate(visits, 1) for row in v.get("db_vitals_rows") or []
                     if row.get("label") == "Temperature" and not vital_implausible("Temperature", row.get("value"))
                     and (_fahrenheit_written(row.get("value")) or 0) >= rule["temp_f_at_least"]]
        if len(hit_days) >= rule.get("min_days", 1) or temps:
            found = []
            if hit_days:
                found.append(f"{rule['finding']} on {len(hit_days)} day(s) ({_encs([i for _, e in hit_days for i in e])})")
            found += [f"temperature {t} at #{i}" for i, t in temps[:2]]
            flag(rule.get("level", "info"), f"No {rule['test_label']} in the record ({rule['finding']})",
                 "; ".join(found) + f" · no {rule['test_label']} result in the record",
                 rule.get("why", ""), kind="followup")

    # Oral NSAID with an ARB / ACE inhibitor in a patient with diabetes (same clinical day).
    def nsaid_arb(systemic, emit):
        if not present(["Type 2 diabetes mellitus"]):
            return
        both = []
        for date, encs in day_list:
            meds = [(m, i) for i in encs for m in meds_by_visit[i - 1]]
            nsaids = list(dict.fromkeys(m for m, i in meds if _NSAID_CLASS in classes_of(m, i) and systemic(m, i)))
            arbs = list(dict.fromkeys(m for m, _ in meds if matches(m, rules.get("arb_acei"))))
            if nsaids and arbs:
                diuretics = [m for m, _ in meds if matches(m, rules.get("diuretic"))]
                both.append((date, encs, nsaids, arbs, diuretics))
        if both:
            triple = any(b[4] for b in both)
            emit("high" if triple else "medium", "Oral NSAID with ARB / ACE inhibitor in diabetes",
                 f"On {len(both)} day(s): "
                 + "; ".join(f"{' + '.join(b[2][:2])} with {' + '.join(b[3][:2])}" + (f" + {b[4][0]}" if b[4] else "")
                             + f" on {where(b[0], b[1])}" for b in both[:3])
                 + (f" (+{len(both) - 3} more days)" if len(both) > 3 else "")
                 + rx_status([n for b in both for n in b[2] + b[3]]),
                 "An NSAID with an ARB / ACE inhibitor lowers kidney blood flow, and diabetes adds kidney risk"
                 + ("; with a diuretic as well this is the high-risk 'triple whammy'." if triple else "."),
                 kind="medication")
    route_rule(nsaid_arb)

    # Oral NSAIDs on a day when gastritis / acidity is recorded (diagnosis or complaint).
    gastric = re.compile(r"(?i)gastrit|acidity|\bapd\b|gerd|reflux|dyspeps|peptic|epigastr")

    def nsaid_gastritis(systemic, emit):
        oral = lambda m, i: _NSAID_CLASS in classes_of(m, i) and systemic(m, i)
        days = [(date, encs) for date, encs in day_list
                if any(oral(m, i) for i in encs for m in meds_by_visit[i - 1])
                and any(gastric.search(" ".join(str(visits[i - 1].get(k) or "") for k in
                        ("db_chief_complaint", "db_provisional_dx", "db_confirmed_dx"))) for i in encs)]
        if days:
            names = list(dict.fromkeys(m for _, encs in days for i in encs for m in meds_by_visit[i - 1] if oral(m, i)))
            emit("medium", "Oral NSAIDs with gastritis / acidity",
                 f"On {len(days)} day(s): {', '.join(names[:3])} on "
                 + ", ".join(where(d, e) for d, e in days[:3]) + (" …" if len(days) > 3 else "") + rx_status(names),
                 "NSAIDs irritate the stomach lining and can worsen gastritis or cause bleeding.", kind="medication")
    route_rule(nsaid_gastritis)

    # Any product containing a corticosteroid (topical or systemic, by its ingredients) at a visit whose complaint or
    # diagnosis is fungal (registry steroid_fungal), at any visit.
    sf = REGISTRY.get("steroid_fungal") or {"pattern": "fung|tinea|ringworm|dermatophyt|candid", "level": "medium"}
    koh = [l for l, _, _ in labs if re.search(r"(?i)\bkoh\b|potassium hydroxide|fungal (?:smear|culture|scraping)",
                                              str(l.get("test_name") or ""))]
    steroid_classes = {_STEROID_CLASS, "Topical corticosteroid"}
    has_steroid = lambda m, i: "corticosteroid" in medicine_profile.groups(ingredients_of(m, i)) or bool(classes_of(m, i) & steroid_classes)
    fungal_steroid = [i for i, v in enumerate(visits, 1)
                      if re.search(sf["pattern"], " ".join(str(v.get(k) or "") for k in
                                   ("db_chief_complaint", "chief_complaint", "db_provisional_dx", "db_confirmed_dx")), re.I)
                      and any(has_steroid(m, i) for m in meds_by_visit[i - 1])]
    if fungal_steroid:
        names = list(dict.fromkeys(m for i in fungal_steroid for m in meds_by_visit[i - 1] if has_steroid(m, i)))
        steroids_in = {m: [g for g in ingredients_of(m) if "corticosteroid" in medicine_profile.groups([g])] for m in names}
        shown = [f"{m} ({', '.join(s)})" if s and s[0] not in m.casefold() else m for m, s in steroids_in.items()]
        flag(sf.get("level", "medium"),
             "Steroid for a fungal complaint" + ("" if koh else " with no KOH test"),
             f"{', '.join(shown[:3])} at {_encs(fungal_steroid)} · "
             + (f"KOH / fungal test in the record: {koh[0].get('test_name')} {koh[0].get('value')}" if koh
                else "no KOH / fungal test in the record"),
             "Steroids can worsen a fungal infection" + ("." if koh else "; a KOH test confirms or rules it out."),
             kind="medication")

    # Daily dose above the usual adult maximum (strength written on the medicine x morning-noon-night count).
    over = []
    for i, v in enumerate(visits, 1):
        for m in taken_medications(v):
            line = daily_dose_over_max(m, clinical_terms.medicine(patient_data, m.get("name")))
            if line and line not in [o for o, _, _ in over]:
                over.append((line, i, _med_display(m.get("name"))))
    if over:
        flag("high", "Daily dose above the usual maximum",
             "; ".join(f"{line} (#{i})" for line, i, _ in over[:3]) + rx_status([n for _, _, n in over]),
             "Exceeds the usual adult maximum daily dose; check the prescription as written.", kind="medication")

    # Systemic steroids with a condition or reading they can worsen (any diagnosis; the recurrent-skin rule
    # below covers repeated courses).
    def oral_steroid(systemic, emit):
        is_sys = lambda m, i: "corticosteroid" in medicine_profile.groups(ingredients_of(m, i)) and systemic(m, i) \
            or (_STEROID_CLASS in classes_of(m, i) and rec.systemic(i, m) == "yes")
        steroid_visits = [i for i, meds in enumerate(meds_by_visit, 1) if any(is_sys(m, i) for m in meds)]
        if not steroid_visits:
            return
        names = list(dict.fromkeys(m for i in steroid_visits for m in meds_by_visit[i - 1] if is_sys(m, i)))
        raised_at = [f"BP {bp_at[i][0]}/{bp_at[i][1]} at #{i}" for i in steroid_visits
                     if i in bp_at and (bp_at[i][0] >= 140 or bp_at[i][1] >= 90)]
        worsened = present(["Hypertension", "Type 2 diabetes mellitus"])
        if raised_at or worsened:
            effects = []
            if raised_at or "Hypertension" in worsened:
                effects.append("raise BP")
            if "Type 2 diabetes mellitus" in worsened:
                effects.append("raise blood glucose")
            emit("medium", "Oral steroid with " + _join(raised_at[:1] + worsened),
                 f"{', '.join(names[:2])} at {_encs(steroid_visits)}"
                 + (f" · {'; '.join(raised_at[:2])}" if raised_at else "") + rx_status(names),
                 f"Systemic steroids can {_join(effects)}.", kind="medication")
    route_rule(oral_steroid)

    # A chronic condition whose own medicines stop appearing on later prescriptions (unless the latest note says
    # to continue the previous medicines).
    continued = re.search(r"(?i)\bcontinue|\bcont\.|same\s+(?:treatment|medicines?)|as\s+before",
                          " ".join(str(n) for n in visits[-1].get("db_instructions") or []))
    for name in REGISTRY["chronic_treatment_conditions"]:
        pattern = REGISTRY["condition_meds"].get(name)
        if not pattern or not present([name]) or continued:
            continue
        own = lambda m, name=name: treats(m, name)
        rx = [i for i, meds in enumerate(meds_by_visit, 1) if any(own(m) for m in meds)]
        if not rx or rx[-1] == len(visits):
            continue
        # The last course runs from its date to date + duration: visits inside it are still on it.
        ends = [l["end"] for m in meds_by_visit[rx[-1] - 1] if own(m) for l in [rec.line(rx[-1], m)] if l and l["end"]]
        course_to = max(ends) if ends else _date(visits[rx[-1] - 1])
        if course_to >= _date(visits[-1]):
            continue
        later = [i for i in range(rx[-1] + 1, len(visits) + 1) if meds_by_visit[i - 1] and _date(visits[i - 1]) > course_to]
        days = _days(visits[rx[-1] - 1], visits[-1]) or 0
        if len(later) >= 2 or (later and days >= 60):
            names = list(dict.fromkeys(m for i in rx for m in meds_by_visit[i - 1] if own(m)))
            since = ""
            if name == "Hypertension":
                stop = _date(visits[rx[-1] - 1])
                after = [f"{s_}/{d_} ({d})" for _, d, (s_, d_) in bp if d and d > stop]
                since = (f" · BP since then: {', '.join(after[-4:])}" + (f" (+{len(after) - 4} earlier)" if len(after) > 4 else "")
                         if after else " · no BP reading recorded since")
            flag("medium", f"No {name} medicine prescribed since #{rx[-1]}",
                 f"{', '.join(names[:3])} last at #{rx[-1]} ({_date(visits[rx[-1] - 1])}); "
                 f"{len(later)} later visit(s) with prescriptions, up to {_date(visits[-1])}" + since,
                 "Treatment may continue from an earlier prescription or elsewhere; it is not visible in this record.",
                 kind="medication")

    # A chronic-disease medicine with no matching condition anywhere in the record (diagnoses, long-term list,
    # health history): the indication is undocumented.
    unmatched = {}
    required = REGISTRY["indication_required"]
    for i, meds in enumerate(meds_by_visit, 1):
        for m in meds:
            if any(re.search(required["except_generics"], x, re.I) for x in [m] + generics[m]):
                continue
            for name in required["conditions"]:
                if not present([name]) and (treats(m, name) or med_is(m, REGISTRY["class_treats"].get(name, ""), r"(?!)")):
                    entry = unmatched.setdefault(m, {"conditions": [], "encounters": []})
                    entry["conditions"] += [name] if name not in entry["conditions"] else []
                    entry["encounters"] += [i] if i not in entry["encounters"] else []
    if unmatched:
        flag("medium", "Medicine for a condition not recorded",
             "; ".join(f"{m} ({_join([c.lower() if c != 'Type 2 diabetes mellitus' else 'diabetes' for c in e['conditions']])})"
                       f" at {_encs(e['encounters'])}" for m, e in list(unmatched.items())[:3]),
             "No matching diagnosis, long-term condition or history is recorded; the indication is not documented.",
             kind="medication")

    # Kidney function
    renal_meds = meds_matching(REGISTRY["renal_caution_meds"])
    kidney_risk = present(REGISTRY["kidney_risk_conditions"])
    for lab, src, _ in labs:
        if re.search(r"creatinine", str(lab.get("test_name") or ""), re.I) and lab_key(lab) not in covered:
            value = _num(lab.get("value"))
            limits = re.search(r"([\d.]+)\s*[-–—]\s*([\d.]+)", str(lab.get("reference") or ""))
            upper = float(limits.group(2)) if limits else None
            context = f" · on {', '.join(renal_meds[:3])}" if renal_meds else ""
            context += " · NSAIDs in record" if len(nsaid_visits) >= 2 else ""
            if _is_high(lab.get("status")):
                flag("high", "Creatinine raised", f"{lab.get('value')} (ref {lab.get('reference')}, {src}){context}")
            elif value and upper and value >= 0.95 * upper:
                flag("medium", "Creatinine at upper limit of normal",
                     f"{lab.get('value')} (ref {lab.get('reference')}, {src}){context}",
                     f"Borderline renal marker with {_join(kidney_risk)}." if kidney_risk else "")

    # Symptoms that are known effects of medicines in the record
    for link in REGISTRY["symptom_med_links"]:
        # A resolved tag (leg_swelling) already carries the site; the registry pattern needs it written.
        site_ok = lambda p: not link["site"] or re.search(link["site"], p, re.I) or not re.search(link["symptom"], p, re.I)
        hits = {p: e for p, e in symptom_hits(link["symptom"], link.get("tags")).items() if site_ok(p)}
        if not hits:
            continue
        window = THRESHOLDS["cause_timing"]["symptom_window_days"]

        def given_before(med, i):
            """A course of the medicine overlapping the window before visit i with the symptom (date + duration)."""
            d = _date(visits[i - 1])
            return bool(d) and any(l["start"] <= d and _days_between(l["end"] or l["start"], d) <= window
                                   for row in rec.lines for l in row
                                   if l["name"] == med and l["prescribed"] and l["start"])
        # A medicine not taken in the window before the symptom cannot explain it: only the symptom visits a linked
        # medicine came before are shown with it (shoulder pain months before a statin is not "on a statin").
        linked = []
        for m in link["meds"]:
            names = [n for n in meds_matching(m["pattern"]) if any(given_before(n, i) for e in hits.values() for i in e)]
            if names:
                linked.append((names, m["note"]))
        if link["only_if_med"] and not linked:
            continue
        names = [n for group, _ in linked for n in group]
        if link["only_if_med"]:  # the finding IS the medicine effect; a symptom that is a finding itself keeps every visit
            hits = {p: [i for i in e if any(given_before(n, i) for n in names)] for p, e in hits.items()}
            hits = {p: e for p, e in hits.items() if e}
        flag("medium", link["title"], "; ".join(f"{p} ({_encs(sorted(e))})" for p, e in hits.items())
             + (f" · on {', '.join(names[:3])}" if names else ""),
             " ".join(note for _, note in linked), kind="medication" if names else "risk")  # no medicine linked: a symptom

    record_dx = {i: str(v.get("db_provisional_dx") or v.get("db_confirmed_dx") or v.get("provisional_diagnosis") or "").strip()
                 for i, v in enumerate(visits, 1)}

    # Sensory symptoms in hands / feet, whatever the documented condition.
    neuro = REGISTRY["neuropathic_symptoms"]
    skin_dx = next(c["pattern"] for c in REGISTRY["condition_map"] if c["name"] == "Skin condition")
    hits = {}
    for p, encs in symptom_hits(neuro["pattern"], neuro.get("tags"), trim=False).items():
        # Burning at a visit diagnosed only as a skin condition is the rash itself, not a nerve symptom.
        encs = [i for i in sorted(encs) if not re.search(skin_dx, record_dx.get(i, ""), re.I)]
        if encs and not re.search(neuro["exclude"], p, re.I):
            hits[p] = encs
    if hits:
        treated = [m for m in distinct_meds if med_is(m, "Nerve vitamin (B12 / neurotropic)", neuro["treatment_meds"])
                   or med_is(m, "Neuropathic pain medicine", neuro["treatment_meds"])]
        causes = present(neuro["cause_conditions"])
        flag(neuro["level"] if causes or treated else "info", neuro["title"],
             "; ".join(f"{re.sub(r'^patient ', '', p)} ({_encs(e)})" for p, e in hits.items())
             + (f" · already treated with {', '.join(treated[:3])}" if treated else ""),
             f"{_join(causes)} in the record can cause peripheral neuropathy." if causes
             else "Can reflect peripheral neuropathy.")

    # A skin diagnosis that keeps returning while steroids are repeated.
    skin = conditions.get("Skin condition")
    if skin and skin["visits"] >= 2:
        steroids = REGISTRY["steroids"]
        oral, topical = {}, {}
        for i, meds in enumerate(meds_by_visit, 1):
            for m in (n.title() if n.isupper() else n for n in meds):
                if re.search(_TOPICAL, m):
                    if re.search(steroids["topical_potent"], m, re.I):
                        topical.setdefault(m, []).append(i)
                elif re.search(steroids["systemic"], m, re.I):
                    oral.setdefault(m, []).append(i)
        if oral or any(len(e) >= 2 for e in topical.values()):
            dxs = list({record_dx[i].strip(" ?.").casefold(): record_dx[i].strip(" ?.")
                        for i in reversed(skin["encounters"]) if record_dx.get(i)}.values())[::-1]
            name = re.search(rf"\w*(?:{skin_dx})\w*", " ".join(dxs), re.I)
            name = name.group(0).lower() if name and len(name.group(0)) > 4 else "skin condition"
            used = [f"oral {m} at {_encs(e)}" for m, e in oral.items()] + [f"topical {m} at {_encs(e)}" for m, e in topical.items()]
            notes = []
            if re.search(r"(?i)tinea|fung", " ".join(dxs)):
                notes.append("Steroids can worsen a fungal (tinea) infection.")
            if oral and present(["Hypertension", "Type 2 diabetes mellitus"]):
                notes.append(f"Oral steroids can worsen {_join(present(['Hypertension', 'Type 2 diabetes mellitus']))}.")
            legs = any(re.search(r"\bleg|foot|feet|ankle|shin", p) for p, e in complaints.items() if set(e) & set(skin["encounters"]))
            venous = [i for i, dxt in record_dx.items() if re.search(REGISTRY["venous_pattern"], dxt, re.I)]
            if legs and venous:
                notes.append(f"Leg-predominant with {record_dx[venous[0]]} (#{venous[0]}): venous (stasis) dermatitis is a differential.")
            # Response to each oral course: was the condition back at a later visit?
            oral_encs = sorted({i for e in oral.values() for i in e})
            # One course per clinical day (two same-day encounters are one course).
            day_no = day_of_encounter(visits)
            courses = sorted({day_no[i] for i in oral_encs})
            relapsed_days = sorted({day_no[i] for i in oral_encs if any(day_no[j] > day_no[i] for j in skin["encounters"])})
            relapsed = relapsed_days
            if relapsed and len(courses) >= 2:
                notes.insert(0, f"Returned after {len(relapsed)} of {len(courses)} oral steroid courses: "
                                "control appears steroid-dependent.")
            elif relapsed:
                notes.insert(0, "Returned after the oral steroid course.")
            long_topical = {m: e for m, e in topical.items() if len(e) >= 3}
            if long_topical:
                m, e = max(long_topical.items(), key=lambda kv: len(kv[1]))
                days = _days(visits[e[0] - 1], visits[e[-1] - 1])
                notes.append(f"Potent topical steroid ({m}) at {len(e)} visits" + (f" over {days} days" if days else "")
                             + "; prolonged use risks skin thinning.")
            flag("medium", f"Recurrent {name} with repeated steroids",
                 f"{' / '.join(dxs[:2]) or 'Skin condition'} · " + "; ".join(used[:3]),
                 " ".join(notes), kind="medication")

    _course_patterns(visits, trend, meds_by_visit, record_dx, flag,
                     supportive=lambda m: bool(classes_of(m) & {"Antihistamine", "Soothing / emollient lotion"}))

    # ── Anomalies from extracted documents ───────────────────────────────
    doc_items = [(i, f"#{i} {d.get('document_label', 'Document')}", d)
                 for i, v in enumerate(visits, 1) for d in v.get("documents") or []]
    doc_items += [(None, f"{d.get('document_label', 'Document')} (not linked)", d)
                  for d in patient_data.get("unlinked_documents") or []]
    # Results ahead of (or without) a matching diagnosis, for every condition with a control marker.
    for cond, rule in _CONTROL_LABS.items():
        first = conditions[cond]["encounters"][0] if cond in conditions else None
        first_date = _date(visits[first - 1]) if first else None
        early = []
        for enc, where, d in doc_items:
            for lab in d.get("lab_results") or []:
                name = str(lab.get("test_name") or "")
                if (not re.search(rule["pattern"], name, re.I) or not _out_of_range(lab, rule["direction"])
                        or _implausible(lab) or (rule["direction"] == "any" and re.search(r"(?i)\bhdl", name))):
                    continue
                before = bool(enc and first and (_date(visits[enc - 1]) < first_date if first_date else enc < first))
                if (enc and (first is None or before)) or (enc is None and first is None):
                    covered.add(lab_key(lab))
                    then = f"; record diagnosis then: {record_dx.get(enc) or 'none'}" if enc else ""
                    early.append(f"{name} {lab.get('value')} ({lab.get('status')}) at {where}{then}")
        if early:
            flag(rule["level"],
                 f"{cond} pattern result before it was documented" if first else f"{cond} pattern result with no matching diagnosis",
                 "; ".join(list(dict.fromkeys(early))[:2])
                 + (f" · {cond} first documented #{first}" if first else f" · no {cond} diagnosis in the record"),
                 "Document finding not reflected in the structured diagnosis.", kind="lab")

    # Abnormal results as risk-factor points: one line per clinical problem, not every value.
    abnormal = {}
    for enc, where, d in doc_items:
        for lab in d.get("lab_results") or []:
            name, status = str(lab.get("test_name") or "").strip(), str(lab.get("status") or "")
            if not re.search(r"(?i)abnormal|high|low", status):
                continue
            high = _is_high(status)
            # Already reported above, creatinine has its own concern, and high HDL is protective.
            if lab_key(lab) in covered or re.search(r"(?i)creatinine", name) or (high and re.search(r"(?i)\bhdl", name)):
                continue
            entry = abnormal.setdefault((name.casefold(), str(lab.get("value"))), {
                "name": name, "value": lab.get("value"), "high": high, "odd": _implausible(lab), "where": []})
            src = f"#{enc}" if enc else "undated report"
            if src not in entry["where"]:
                entry["where"].append(src)

    def _point(entry):
        tag = ("High" if entry["high"] else "Low") + (", check OCR" if entry["odd"] else "")
        return f"{entry['name']} {entry['value']} ({tag}, {', '.join(entry['where'])})"

    lipid_dx = any(re.search(r"(?i)lipid|cholesterol", c) for c in conditions)
    cardiometabolic = present(REGISTRY["cardiometabolic_conditions"])
    risk_rules = (
        ("Dyslipidaemia" + ("" if lipid_dx else " (no lipid diagnosis recorded)"),
         lambda e: e["high"] and re.search(r"(?i)triglycer|\bldl|vldl|cholest", e["name"]),
         f"Adds cardiovascular risk alongside {_join(cardiometabolic)}." if cardiometabolic else ""),
        ("Low haemoglobin / PCV (possible anaemia)",
         lambda e: not e["high"] and re.search(r"(?i)h(?:a)?emoglobin|\bhb\b|pcv|\brbc|hct|haematocrit", e["name"]), ""),
        ("Raised white cell count",
         lambda e: e["high"] and re.search(r"(?i)\btlc\b|wbc|leu[ck]ocyte", e["name"]), "May reflect infection or inflammation."),
        ("Low white cell count", lambda e: not e["high"] and re.search(r"(?i)\btlc\b|wbc|leu[ck]ocyte", e["name"]), ""),
        ("Abnormal platelet count", lambda e: re.search(r"(?i)platelet", e["name"]), ""),
        ("Kidney / electrolyte abnormality",
         lambda e: (e["high"] and re.search(r"(?i)urea|\bbun\b|uric", e["name"])) or re.search(r"(?i)sodium|potassium|chloride", e["name"]), ""),
        ("Liver test abnormality",
         lambda e: re.search(r"(?i)bilirubin|sgot|sgpt|\bast\b|\balt\b|alkaline|ggt|albumin|total protein", e["name"]), ""),
        ("Thyroid test abnormal", lambda e: re.search(r"(?i)\btsh\b|\bt3\b|\bt4\b|thyro", e["name"]), ""),
        ("Glucose / HbA1c out of range", lambda e: re.search(r"(?i)hba1c|glyco|glucose|sugar", e["name"]), ""),
    )
    used = set()
    for title, rule, significance in risk_rules:
        hits = [k for k, e in abnormal.items() if k not in used and rule(e)]
        if not hits:
            continue
        used.update(hits)
        entries = [abnormal[k] for k in hits]
        all_odd = all(e["odd"] for e in entries)
        flag("info" if all_odd else "medium", title + (" (check OCR)" if all_odd else ""),
             "; ".join(_point(e) for e in entries),
             "Values far outside the range; likely an OCR misread." if all_odd else significance, kind="lab")
    rest = [abnormal[k] for k in abnormal if k not in used]
    if rest:
        flag("info", "Other abnormal results", "; ".join(_point(e) for e in rest), kind="lab")

    # Record vs document mismatches (diagnosis and vitals).
    mismatches, vital_mismatches = [], []
    mismatched_vitals = set()  # (where, label): reported here, so not again as an out-of-range vital
    for i, v in enumerate(visits, 1):
        for d in v.get("documents") or []:
            cs = _summary(d)
            doc_dx = str(cs.get("provisional_diagnosis") or cs.get("confirmed_diagnosis") or "").strip()
            rec_dx = record_dx.get(i, "")
            # Doses, letters fused with digits or letter-digit codes mean the OCR grabbed a garbled line, not a diagnosis.
            if (doc_dx and rec_dx and not _GARBLED.search(doc_dx) and doc_dx.casefold() not in rec_dx.casefold()
                    and rec_dx.casefold() not in doc_dx.casefold() and not _same_diagnosis_group(doc_dx, rec_dx)
                    and not re.search(r"(?i)\b(?:bp|pulse|spo2|sign|signature|weight|height)\b", doc_dx)):
                mismatches.append(f"#{i}: document '{doc_dx[:40]}' vs record '{rec_dx[:40]}'")
        for text in dict.fromkeys(v.get("discrepancies") or []):
            m = re.match(r"(Document \d+):\s*(.+?) — structured record shows\s*(\d+(?:\.\d+)?)?.*?document shows\s*(\d+(?:\.\d+)?)?", text)
            if m:
                mismatched_vitals.add((f"#{i} {m.group(1)}", m.group(2)))
                record_value, doc_value = (float(x) if x else None for x in m.group(3, 4))
                # Readings taken minutes apart differ a little; only a real disagreement is worth a look.
                if record_value and doc_value and abs(record_value - doc_value) / max(record_value, doc_value) < 0.15:
                    continue
                if any(str(r.get("label")) == m.group(2) for r in v.get("conflicting_vitals") or []):
                    continue  # reported as a conflicting value
                vital_mismatches.append(f"#{i}: " + re.sub(r"^Document \d+:\s*", "", text).rstrip("."))
    if mismatches:
        flag("medium", f"Diagnosis differs between document and record ({len(mismatches)})",
             "; ".join(mismatches[:3]) + (" …" if len(mismatches) > 3 else ""),
             "The scanned document and the structured record name different problems.", kind="data")
    # A long-term list entry the visits contradict: 'Hyperthyroidism' on the list while every visit writes
    # hypothyroidism (or the reverse: hypo-/hyper-tension, -glycaemia, -kalaemia ...).
    visit_dx = {i: " ".join(str(v.get(k) or "") for k in ("db_provisional_dx", "db_confirmed_dx", "provisional_diagnosis",
                                                           "confirmed_diagnosis", "db_chief_complaint")).casefold()
                for i, v in enumerate(visits, 1)}
    list_conflicts = []
    for c in patient_data.get("long_term_conditions") or []:
        m = re.match(r"(?i)(hyper|hypo)(\w{4,})", str(c.get("name") or "").strip())
        if not m:
            continue
        own, other, stem = m.group(1).casefold(), "hypo" if m.group(1).casefold() == "hyper" else "hyper", m.group(2)[:5].casefold()
        said_own = [i for i, t in visit_dx.items() if re.search(rf"\b{own}\s*-?\s*{stem}", t)]
        said_other = [i for i, t in visit_dx.items() if re.search(rf"\b{other}\s*-?\s*{stem}", t)]
        if said_other and not said_own:
            list_conflicts.append(f"long-term list '{c['name']}' vs visits '{other}{m.group(2).casefold()}' ({_encs(said_other)})")
    if list_conflicts:
        flag("medium", "Long-term condition list conflicts with the visit diagnoses", _listed(list_conflicts, 3),
             "The patient's condition list and the visit diagnoses name opposite conditions; check which is right.",
             kind="data")
    if vital_mismatches:
        flag("info", f"Vitals differ between document and record ({len(vital_mismatches)})",
             "; ".join(vital_mismatches[:3]) + (" …" if len(vital_mismatches) > 3 else ""), kind="data")

    # Out-of-range vitals recorded in documents.
    odd_vitals, implausible_vitals, low_temps = [], [], []
    for enc, where, d in doc_items:
        for row in _summary(d).get("vitals") or []:
            label, value = str(row.get("label") or ""), str(row.get("value") or "")
            celsius = re.search(r"([\d.]+)\s*°C", value)
            celsius = float(celsius.group(1)) if celsius else None
            pressure = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", value)
            number = _num(value)
            odd = ((label == "Temperature" and celsius and not 35.0 <= celsius < 38.0)
                   or (label == "BP" and pressure and (int(pressure.group(1)) >= 140 or int(pressure.group(2)) >= 90))
                   or (label.startswith("SpO") and number and number < 94)
                   or (label in ("Pulse", "Heart Rate") and number and number > 100))
            if label == "Temperature" and _low_temp(value):
                low_temps.append(f"{value} ({where})")
                continue
            # Readings an outpatient walking in would not have: a recording or OCR error, not a finding.
            implausible = vital_implausible(label, value)
            if odd and (where, label) not in mismatched_vitals:
                (implausible_vitals if implausible else odd_vitals).append(f"{label} {value} ({where})")
    if odd_vitals:
        flag("medium", "Out-of-range vitals in documents", _listed(odd_vitals),
             "Could be a true finding or an OCR/unit error; verify.", kind="data")
    if implausible_vitals:
        flag("info", "Implausible vital reading", _listed(implausible_vitals, 3),
             "Not physiologically likely in an outpatient; probably a recording or unit error.", kind="data")
    # The same check on the structured record (35.7 °F is a Celsius value in the wrong unit).
    impossible = [(f"{row.get('label')} {row.get('value')}", i, _date(v), _celsius_note(row.get("label"), row.get("value")))
                  for i, v in enumerate(visits, 1)
                  for row in v.get("db_vitals_rows") or [] if vital_implausible(row.get("label"), row.get("value"))
                  # 93-95 °F has its own flag below.
                  and not (row.get("label") == "Temperature" and _low_temp(row.get("value")))]
    low_temps += [(str(row.get('value')), i, _date(v), "") for i, v in enumerate(visits, 1)
                  for row in v.get("db_vitals_rows") or [] if row.get("label") == "Temperature" and _low_temp(row.get("value"))]
    if low_temps:
        flag("medium", "Temperature below 95 °F", _listed(_grouped_readings(low_temps)),
             "Below 95 °F in a walking outpatient is almost always a measuring or entry error: data quality, not a finding.",
             kind="data")
    if impossible:
        flag("medium", "Impossible vital reading in the structured record", _listed(_grouped_readings(impossible)),
             "Not physiologically possible as recorded; likely a unit or entry error (e.g. °C entered as °F). "
             "Not used in the analysis.", kind="data")

    # Symptoms and medicines that appear only in documents.
    record_text = " ".join(str(v.get("db_chief_complaint") or v.get("chief_complaint") or "") for v in visits).casefold()
    doc_symptoms = {}
    for enc, where, d in doc_items:
        for sym in _summary(d).get("symptoms") or []:
            words = [w for w in re.findall(r"[a-z]{4,}", sym.casefold()) if w not in ("pain", "since", "days")]
            if words and not any(w in record_text for w in words):
                doc_symptoms.setdefault(sym[:50], where)
    if doc_symptoms:
        flag("info", "Symptoms only in documents", _listed([f"{s} ({w})" for s, w in doc_symptoms.items()], 3),
             kind="data")
    rx_names = {_first_word(_med_display(m.get("name"))) for v in visits for m in v.get("db_medications") or []}
    doc_only = {}
    for enc, where, d in doc_items:
        for med in d.get("medications_found") or []:
            word = _first_word(med)
            # Fuzzy match so spelling variants (Cetirizine / citrizine, Vaseline / vasaline) count as prescribed.
            if word and not any(SequenceMatcher(None, word, rx).ratio() >= 0.75 for rx in rx_names):
                doc_only.setdefault(med, where)
    if doc_only:
        flag("info", "Medicines only in documents", _listed([f"{m} ({w})" for m, w in doc_only.items()]),
             kind="data")

    # Follow-up adherence
    missed = [(i, str(f.get("scheduled_date") or "")[:10]) for i, v in enumerate(visits, 1)
              for f in v.get("followups") or [] if "MISSED" in str(f.get("status", "")).upper()]
    if missed:
        # One appointment per due date: two same-day encounters booking the same review date are one missed review.
        by_due = {}
        for i, d in missed:
            by_due.setdefault(d, []).append(i)
        latest_day = _date(visits[-1])
        latest_missed = any(_date(visits[i - 1]) == latest_day for i, _ in missed)
        flag("high" if latest_missed else "medium",
             f"{len(by_due)} missed follow-up(s)" + (" incl. the latest" if latest_missed else ""),
             ", ".join(f"due {d}, {_overdue_days(d, visits)} days overdue (after {_encs(e)})" if _overdue_days(d, visits) is not None
                       else f"due {d}, no visit recorded since (after {_encs(e)})" for d, e in by_due.items()),
             "Latest scheduled review did not happen." if latest_missed else "", kind="followup")

    # Follow-ups the source marked MISSED although a visit happened near the due date (core.encounters.recheck_followups).
    off_date = [f["status_label"].split(": ", 1)[1] for v in visits for f in v.get("followups") or []
                if f.get("status") == "ATTENDED_OFF_DATE"]
    if off_date:
        flag("info", f"{len(off_date)} follow-up(s) marked missed by the source but attended off-date",
             "; ".join(off_date[:3]), "Not counted as missed: the patient was seen within the tolerance window.", kind="data")

    conflicts = [f"{r.get('label')} {r.get('value')} vs document {r.get('document_value'):g} (#{i})"
                 for i, v in enumerate(visits, 1) for r in v.get("conflicting_vitals") or []]
    if conflicts:
        flag("info", f"{len(conflicts)} conflicting value(s): record and document disagree", "; ".join(conflicts[:4]),
             "Not used in any finding or test suggestion until the source is checked.", kind="data")
    dups = [(i, v["_duplicate_of"]) for i, v in enumerate(visits, 1) if v.get("_duplicate_of")]
    if dups:
        flag("info", f"{len(dups)} duplicate visit entr{'y' if len(dups) == 1 else 'ies'}",
             "; ".join(f"#{i} repeats #{j}" for i, j in dups[:4]),
             "Same date, complaint, diagnosis and prescription entered twice.", kind="data")

    # Chronic pain and record gaps
    pain = next((r for name, r in conditions.items() if name.startswith("Musculoskeletal") or "pain" in name.casefold()), None)
    if pain and pain["visits"] >= 5:
        flag("info", "Recurrent musculoskeletal pain", f"{pain['visits']} visits, {pain['status']}")
    unlinked = patient_data.get("unlinked_documents") or []
    if unlinked:
        flag("info", f"{len(unlinked)} report(s) not linked to a visit", "Lab dates unknown", kind="data")

    from clinical_insights import plain_wording
    for f in flags:
        for k in ("title", "evidence", "significance"):
            if f.get(k):
                f[k] = plain_wording(f[k])
    data_quality.finalise(flags)  # record problems: kind data, level data_quality, never findings
    flags.sort(key=lambda f: _LEVEL_ORDER[f["level"]])

    timeline = [{"label": r["condition"], "encounters": set(r["encounters"])} for r in trend["conditions"][:5]]
    if raised:  # no alert row when every reading is below target
        timeline.append({"label": f"BP {target_label}", "encounters": set(raised), "kind": "alert"}) if bp_patient else None
    if missed:
        timeline.append({"label": "Missed follow-up", "encounters": {i for i, _ in missed}, "kind": "alert"})
    abnormal_report_encs = {enc for enc, _, d in doc_items if enc for lab in d.get("lab_results") or []
                            if re.search(r"(?i)abnormal|high|low", str(lab.get("status") or ""))}
    if abnormal_report_encs:
        timeline.append({"label": "Abnormal report results", "encounters": abnormal_report_encs, "kind": "alert"})

    latest_day = _date(visits[-1])
    current_medicines = []
    for l in reversed([l for l in rec.prescribed() if l["start"]]):
        on = l["start"] == latest_day or (l["end"] and l["start"] <= latest_day <= l["end"])
        if on and not any(rec.same(l, c["_line"]) for c in current_medicines):
            renewed = l["start"] == latest_day
            current_medicines.append({"name": l["name"], "since": l["start"], "until": l["end"], "_line": l,
                                      **({"specialization": l["specialization"]} if l.get("specialization") else {}),
                                      # On the latest prescription, or an earlier course still running then (not renewed).
                                      "status": "current" if renewed else "course running, not renewed",
                                      "note": "; ".join(x for x in (
                                          "dose not specified" if l["dose_not_specified"] else "",
                                          "" if renewed else f"course running, not renewed: to ~{l['end']}") if x)})
    current_medicines = [{k: v for k, v in c.items() if k != "_line"} for c in reversed(current_medicines)]
    current_lines = [l for l in rec.prescribed() if l["start"] and (l["start"] == latest_day or
                                                                     (l["end"] and l["start"] <= latest_day <= l["end"]))]
    overlaps = overlapping_ingredients(visits, rec)
    if overlaps:
        flag("medium", "Same ingredient in two products on one day",
             _listed([f"{a} + {b}: both contain {', '.join(sh)} ({d}, {_encs(e)})" for d, e, _, (a, b), sh in overlaps], 3),
             "Two products in the same therapeutic group share an ingredient; the total dose of that ingredient adds up.",
             kind="medication")
    # A symptom linked to a medicine risk (registry symptom_medicine_risks).
    for rule in REGISTRY.get("symptom_medicine_risks") or []:
        hits = symptom_hits(None, rule["tags"])
        days = _group_days(rec, rule["med_group"], rule.get("route", "oral"))
        conds = present(rule.get("with_conditions") or [])
        if hits and days >= rule.get("min_days", 1) and conds and not (lab_keys & set(rule.get("tests") or [])):
            flag(rule["level"], rule["title"],
                 "; ".join(f"{p} ({_encs(sorted(e))})" for p, e in hits.items())
                 + f" · {rule['med_group'].upper() if len(rule['med_group']) <= 6 else rule['med_group']} for about {days} days"
                 + f" · {_join(conds)} · no {rule['tests'][0]} result in the record", rule["significance"], kind="medication")
            flags[-1]["problem"] = _group_problem(rule["med_group"])
    fold_symptom_flags(flags)  # a symptom another flag explains is part of that flag
    tag_current(flags, visits, current_medicines, rec)  # before the related-medicine note, which names current medicines
    tag_combo_time(flags, rec, latest_day)  # medicines taken together: current only while all of them are
    add_related_medicines(flags, rec, current_lines)
    link_flags(flags, rec)
    flags.sort(key=lambda f: (_LEVEL_ORDER[f["level"]], f.get("time") != "current"))

    snapshot = {
        "age_sex": age_sex,
        "encounters": len(visits),
        "span": trend["span"],
        "active": active_conditions(patient_data, trend),
        "active_meds": len(current_medicines),
        # Medicines on the latest date: prescribed then, or an earlier course (date + duration) still running.
        "current_medicines": current_medicines,
        "coverage": {k: len(v) for k, v in record_coverage.items()},
        "flags": flags,
        # Verified rows first; each row says whether a clinician has verified its section.
        "guideline": {"version": GUIDELINE_VERSION, "profile": profile,
                      "rows": sorted(guideline, key=lambda r: not r["verified"]),
                      "sections": guideline_sections()},
        "timeline": timeline,
        # Medicine classes per visit (NSAIDs, same-class duplicates) for the grid under the visit timeline.
        "medicine_timeline": snapshot_view.medicine_timeline(rec, visits),
        "dates": [_date(v) for v in visits],
        # Per visit: specializations of the doctors who wrote its prescriptions (no names); [] when unknown.
        "specialties": [list(v.get("prescriber_specializations") or []) for v in visits],
    }
    # Status tiles, one short line per concern, record errors last (snapshot_view).
    return snapshot_view.decorate(snapshot, patient_data, labs)


# ── Clinical course: concise timeline of diagnosis and medicine shifts ──────
_NOT_MEDICINE = r"(?i)\b(?:diet|salt|exercise|walk|advice|counsel)"
_SUPPLEMENTS = r"(?i)complex|calc|calcu|cal\+|d3|vit|methyc|methoc|neurob|shell|calsh"
# Condition-specific medicines (pain medicines excluded) are tracked individually in the course.
_CHRONIC = dict(REGISTRY["condition_meds"])


def _med_key(name):
    clean = re.sub(r"(?i)^(?:ta|tab(?:let)?|cap(?:sule)?|syp|syr(?:up)?|inj(?:ection)?)\.?\s+", "", str(name or "")).strip()
    return re.sub(r"[^a-z0-9]", "", clean.casefold())[:4]


def _short_lab(lab):
    """'Serum Triglycerides 162.99 High' -> 'Triglycerides 162.99 ↑' for one-line timelines."""
    name = re.sub(r"(?i)^serum\s+", "", str(lab.get("test_name") or ""))[:22]
    arrow = "↑" if "high" in str(lab.get("status") or "").casefold() else "↓"
    return f"{name} {lab.get('value')} {arrow}"


def _month(iso):
    try:
        return _date_cls.fromisoformat(iso).strftime("%b %Y")
    except ValueError:
        return iso or "Undated"


def _days_between(a, b):
    try:
        return (_date_cls.fromisoformat(b) - _date_cls.fromisoformat(a)).days
    except ValueError:
        return 0


def _med_group(name, patient_conditions=()):
    """The patient's own conditions are tried first, so shared drugs (metoprolol, calcium)
    attach to what this patient actually has."""
    if re.search(_NOT_MEDICINE, name):
        return None
    for condition in patient_conditions:
        if condition in _CHRONIC and re.match(_CHRONIC[condition], name, re.I):
            return condition
    if re.search(_SUPPLEMENTS, name):
        return "supplement"
    for condition, pattern in _CHRONIC.items():
        if re.match(pattern, name, re.I):
            return condition
    if re.match(_PAIN_MEDS, name, re.I):
        return "pain"
    return "other"


def _age_sex(patient_data):
    """The patient table's age / sex; the first document that states it when the table has none."""
    if patient_data.get("age_sex"):
        return patient_data["age_sex"]
    docs = [d for v in patient_data.get("visits") or [] for d in v.get("documents") or []]
    docs += list(patient_data.get("unlinked_documents") or [])
    return next((_summary(d).get("age_sex") for d in docs
                 if _summary(d).get("age_sex")), None)


@traced(name="snapshot.build_clinical_course")
def build_clinical_course(patient_data):
    """Concise, patient-focused course: headline, what stayed consistent, diagnosis
    and medicine shifts, and a month-by-month timeline. Deterministic, no LLM."""
    visits = recheck_followups(patient_data.get("visits") or [])
    if not visits:
        return None
    dates = [_date(v) for v in visits]
    latest = dates[-1]
    trend = build_condition_trend(patient_data)
    patient_conditions = [r["condition"] for r in trend["conditions"]]

    # Medicines per visit, keyed so spelling variants merge (Glimperide / GLIMIPRIDE).
    forms, seen = {}, {}
    for idx, v in enumerate(visits, 1):
        for m in v.get("db_medications") or []:
            name = _med_display(m.get("name"))
            key = _med_key(name)
            if not key or _med_group(name, patient_conditions) is None:
                continue
            forms.setdefault(key, Counter())[name] += 1
            if idx not in seen.setdefault(key, []):
                seen[key].append(idx)
    display = {k: c.most_common(1)[0][0] for k, c in forms.items()}
    group = {k: _med_group(display[k], patient_conditions) for k in display}

    # Consistent: ongoing conditions and the medicines given at most of their visits.
    consistent = []
    for r in trend["conditions"]:
        if not r["status"].startswith("ongoing"):
            continue
        encs = set(r["encounters"])
        regimen = [f"{display[k]} ({len(encs & set(seen[k]))}/{len(encs)})" for k in display
                   if group[k] == r["condition"] and len(encs & set(seen[k])) >= max(2, len(encs) / 2)]
        consistent.append(f"{r['condition']} since {_month(r['first_seen'])} · {len(encs)} visits"
                          + (f" · {', '.join(regimen)}" if regimen else ""))
    bp = _vital_series(visits, "BP")
    if bp:
        consistent.append(f"BP {bp[0][2].split()[0]} ({_month(bp[0][1])}) → {bp[-1][2].split()[0]} ({_month(bp[-1][1])})")

    # Diagnosis shifts.
    dx_events = []
    for r in trend["conditions"]:
        first = r["encounters"][0]
        dx_events.append((first, f"#{first} {_month(r['first_seen'])}: {r['condition']} "
                                 + ("at first visit" if first == 1 else "first documented")))
        if not r["status"].startswith("ongoing"):
            last = r["encounters"][-1]
            dx_events.append((last, f"#{last} {_month(r['last_seen'])}: {r['condition']} last noted"))
    confirmed_seen = set()
    for idx, v in enumerate(visits, 1):
        conf = str(v.get("db_confirmed_dx") or v.get("confirmed_diagnosis") or "").strip()
        if conf and conf.casefold() not in confirmed_seen:
            confirmed_seen.add(conf.casefold())
            dx_events.append((idx, f"#{idx} {_month(dates[idx - 1])}: {conf} confirmed"))
    dx_events = [text for _, text in sorted(dx_events)][:7]

    # Medicine shifts: chronic medicines individually; pain medicines and supplements grouped.
    dose_changes = {}
    journey = patient_data.get("_medication_journey_cache") or compute_medication_journey(patient_data)
    for enc in journey:
        for ch in enc.get("changes", []):
            key = _med_key(_med_display(ch.get("medication")))
            if ch.get("type") == "dose_changed" and group.get(key) in _CHRONIC:
                dose_changes.setdefault(key, set()).add(enc["visit_index"])
    med_events = []
    for key in sorted(display, key=lambda k: seen[k][0]):
        if group[key] not in _CHRONIC:
            continue
        first, last = seen[key][0], seen[key][-1]
        line = f"{display[key]}: " + ("from first visit" if first == 1 else f"added #{first} ({_month(dates[first - 1])})")
        if _days_between(dates[last - 1], latest) > 30:
            line += f", last prescribed #{last} ({_month(dates[last - 1])})"
        if dose_changes.get(key):
            line += f"; dose/frequency changed at {_encs(sorted(dose_changes[key]))}"
        med_events.append(line)
    for label, kind in (("Pain medicines", "pain"), ("Supplements", "supplement")):
        keys = [k for k in display if group[k] == kind]
        if keys:
            visits_with = sorted({i for k in keys for i in seen[k]})
            names = [display[k] for k in sorted(keys, key=lambda k: -len(seen[k]))][:4]
            med_events.append(f"{label}: {', '.join(names)} · {len(visits_with)} visits (#{visits_with[0]}–#{visits_with[-1]})")

    # Month-by-month timeline.
    complaints = complaint_index(visits)
    missed = [(i, str(f.get("scheduled_date") or "")[:10]) for i, v in enumerate(visits, 1)
              for f in v.get("followups") or [] if "MISSED" in str(f.get("status", "")).upper()]
    months = []
    for idx, d in enumerate(dates, 1):
        if not months or months[-1][0] != d[:7]:
            months.append((d[:7], []))
        months[-1][1].append(idx)
    timeline = []
    for month, encs in months:
        parts = []
        new_dx = [r["condition"] for r in trend["conditions"] if r["encounters"][0] in encs]
        if new_dx:
            parts.append("New: " + ", ".join(new_dx))
        started = [display[k] for k in display if seen[k][0] in encs and seen[k][0] > 1 and group[k] in _CHRONIC]
        if started:
            parts.append("Started: " + ", ".join(started))
        # Run-together source text ("swelling on both eyes lower back pain") is cut after the eye term.
        new_complaints = list(dict.fromkeys(re.sub(r"\b(eyes?)\b.*", r"\1", p)
                                            for p, e in sorted(complaints.items(), key=lambda kv: -len(kv[1])) if e[0] in encs))[:3]
        if new_complaints:
            parts.append("Complaints: " + ", ".join(new_complaints))
        month_bp = [val.split()[0] for i, _, val in bp if i in encs]
        if month_bp:
            parts.append("BP " + (month_bp[0] if len(month_bp) == 1 else f"{month_bp[0]} → {month_bp[-1]}"))
        reports = [_short_lab(lab) for i in encs for d in visits[i - 1].get("documents") or []
                   for lab in d.get("lab_results") or []
                   if re.search(r"(?i)abnormal|high|low", str(lab.get("status") or ""))
                   and not re.search(r"(?i)\bhdl", str(lab.get("test_name")))]
        if reports:
            parts.append("Reports: " + ", ".join(list(dict.fromkeys(reports))[:3]))
        month_missed = [day for _, day in missed if day[:7] == month]
        if month_missed:
            parts.append("Missed follow-up " + ", ".join(month_missed))
        timeline.append({"month": _month(f"{month}-01"),
                         "encounters": f"#{encs[0]}" + (f"–#{encs[-1]}" if len(encs) > 1 else ""),
                         "text": " · ".join(parts) or "Follow-up, no change documented"})
    for _, day in missed:
        if day[:7] > months[-1][0]:
            timeline.append({"month": _month(f"{day[:7]}-01"), "encounters": "—", "text": f"Missed follow-up {day}"})

    # One-sentence headline.
    age_sex = _age_sex(patient_data)
    active = active_conditions(patient_data, trend)
    first_seen = {r["condition"]: r["first_seen"] for r in trend["conditions"]}
    ongoing = [f"{c} (since {_month(first_seen[c])})" if c in first_seen else c for c in active]
    other = [r["condition"] for r in trend["conditions"] if r["condition"] not in active]
    headline = (f"{age_sex + ' · ' if age_sex else ''}{len(visits)} visits, {_month(dates[0])} to {_month(latest)}. "
                + (f"Ongoing {' and '.join(ongoing)}." if ongoing else "No ongoing chronic condition documented.")
                + (f" Intermittent {', '.join(other)}." if other else ""))
    return {"headline": headline, "consistent": consistent, "diagnosis_shifts": dx_events,
            "medicine_shifts": med_events, "timeline": timeline}
