import json
import logging
import re
from collections import Counter
import datetime
from difflib import SequenceMatcher
from pathlib import Path

import clinical_terms
from core.encounters import recheck_followups, course_end, dose_not_specified, clinical_days, day_of_encounter, duration_days, taken_medications, zero_dose_lines
from core.tracing import traced
import lab_validation
import medicine_profile
from clinical_snapshot import build_snapshot, guideline_rows_in_use, vital_implausible
from patient_chat import (REGISTRY, _CONDITION_IGNORE, _CONDITION_LABS, _CONDITION_MAP, _med_display,
                          conditions_matching, drop_superseded, long_term_labels)

with (Path(__file__).resolve().parent / "data" / "reference" / "lab_significance.json").open(encoding="utf-8") as _f:
    _SIGNIFICANCE = json.load(_f)["analytes"]
_CONTROL_LABS = REGISTRY["control_labs"]
_CONDITION_MEDS = REGISTRY["condition_meds"]
# Condition -> catalog key of the marker a guideline control row judges (diabetes -> hba1c).
_GUIDELINE_MARKERS = {g["condition"]: g["analyte"] for g in [(REGISTRY.get("guidelines") or {}).get("diabetes_control")] if g}

# "known diagnosis": K/C/O ("known case of") in a complaint, or the patient's long-term list / health history: an
# established diagnosis, ranked above a working (provisional) one. "known history": H/O ("history of") only.
STATUS_ORDER = ("confirmed", "known diagnosis", "working diagnosis", "known history", "suspected", "document only")
_KNOWN = ("known diagnosis", "known history")
_CHRONIC = set(REGISTRY.get("chronic_treatment_conditions") or []) | set(REGISTRY.get("long_standing_conditions") or [])
_KCO = re.compile(r"(?i)\b(?:k/?c/?o|kco|known\s+case\s+of)\b")
_SUSPECTED = re.compile(r"\?|\b(?:r/o|rule\s*out|suspected|probable|possible|query|likely|under\s+evaluation)\b", re.I)
_HISTORY = re.compile(r"(?i)\b(?:k/?c/?o|kco|known\s+case\s+of|h/o|history\s+of)\b")
# 'K/C/O', 'k / c / o', 'H/O' as one token, so splitting a complaint on '/' never leaves 'o-htn-dm'.
_HISTORY_SLASHED = re.compile(r"(?i)\bk\s*/\s*c\s*/\s*o\b|\bh\s*/\s*o\b")
_DX_SPLIT = re.compile(r"[,;/+&\n]|\bwith\b|\band\b", re.I)
# A diagnosis abbreviation ("AKI", "ACD"); dosing and "nothing abnormal" shorthand are not diagnoses.
_DX_ABBREVIATION = re.compile(r"(?!(?:NAD|WNL|OK|SOS|OD|BD|TDS|TID|QID|HS|NA|NIL|NR)$)[A-Z]{2,6}")
_DX_FILLER = re.compile(r"(?i)\b(?:k/?c/?o|kco|known|case|of|h/o|history|with|and|on|the)\b|[^a-z\s]")
# OCR noise read as a diagnosis: doses, words fused with digits, letter-digit codes ("G6A2B1C0").
_GARBLED = re.compile(r"(?i)\d\s*(?:mg|mcg|ml|gm)|[a-z]{3,}\d|\d[a-z]{3,}|\b(?=[a-z]*\d)(?=\d*[a-z])[a-z0-9]{6,}\b")
_STANDARD_REF = re.compile(r"\s*\(standard adult range; report gave none\)\s*", re.I)
_ROUTINE_RESOLUTIONS = {"", "VISIT_COMPLETED", "TREATMENT_GOING_ON", "TREATMENT_COMPLETED", "TAKING_MEDICINE_ON_TIME"}
# Follow-up call outcomes (followup_schedule.resolution) in clinical terms.
_FOLLOWUP_OUTCOMES = {
    "CONTACT_DETAILS_MISSING": "patient not reached (no contact details)",
    "CALL_NOT_RECEIVED": "patient not reached (call not answered)",
    "CALL_NOT_CONNECTED": "patient not reached (call not connected)",
    "DID_NOT_TAKE_MEDICINE": "patient reported not taking the medicine",
    "MEDICINE_TAKEN_NO_IMPROVEMENT": "medicine taken, no improvement reported",
    "LAMA": "left against medical advice",
    "CLOSE_CASE_WITHOUT_TREATMENT": "case closed without treatment",
    "PASSED_AWAY": "recorded as passed away",
}
_MAX_ABNORMAL = 10
logger = logging.getLogger(__name__)


def _date(visit):
    return str(visit.get("visit_date") or "")[:10]


def _encs(numbers):
    numbers = sorted(set(numbers))
    if not numbers:
        return ""
    if len(numbers) > 1 and numbers[-1] - numbers[0] == len(numbers) - 1:
        return f"#{numbers[0]}–{numbers[-1]}"
    return ", ".join(f"#{n}" for n in numbers[:6]) + (" …" if len(numbers) > 6 else "")


def _tidy(text):
    """Record free text in sentence case without changing abbreviations ('URTI', 'LSC')."""
    # A line break in the record separates items ('White discharge\nBackache'): it becomes ', ', never a run-on.
    text = re.sub(r"\s*[,;]?\s*[\r\n]+\s*[,;]?\s*", ", ", str(text or "").strip())
    text = re.sub(r"\s+", " ", text).strip(" .,-:;")
    if text.isupper() and len(text) > 5 or text.islower():
        text = text[:1].upper() + text[1:].lower()
    return text


def _summary(doc):
    """A document's extracted clinical summary; {} when missing or stored as plain text."""
    summary = doc.get("clinical_summary") if isinstance(doc, dict) else None
    return summary if isinstance(summary, dict) else {}


def _age_sex(visits, unlinked):
    """Age / sex from the most recent document that states it."""
    docs = [d for v in visits for d in v.get("documents") or []] + list(unlinked)
    return next((_summary(d)["age_sex"] for d in reversed(docs) if _summary(d).get("age_sex")), None)


def _join(items):
    items = [str(i) for i in items if i]
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else (items[0] if items else "")


# ── Conditions ──────────────────────────────────────────────────────────────

def _split_diagnosis(text):
    text = str(text or "").strip()
    if not text or _CONDITION_IGNORE.match(text) or text.casefold() in {"none", "null"}:
        return []
    found = []
    for part in _DX_SPLIT.split(text):
        part = part.strip(" .-:")
        if not part or _CONDITION_IGNORE.match(part):
            continue
        suspected = bool(_SUSPECTED.search(part))
        written = _tidy(re.sub(r"\?", "", part))
        rest, mapped = part, []
        for pattern, name in _CONDITION_MAP:
            if re.search(pattern, part, re.I):
                mapped.append(name)
                rest = re.sub(rf"\w*(?:{pattern})\w*", " ", rest, flags=re.I)
        mapped = drop_superseded(mapped)
        if len(mapped) == 1:
            # One registry condition: the whole phrase is its label ("Iatrogenic hyperthyroidism").
            found.append((mapped[0], written, suspected))
            continue
        found += [(name, "", suspected) for name in mapped]
        # Free text the registry doesn't know ("Herpes zoster", or "vertigo" in "HTN T2DM vertigo") is kept,
        # and so is a bare diagnosis abbreviation written in capitals ("AKI", "ACD").
        leftover = " ".join(_DX_FILLER.sub(" ", rest.casefold()).split())
        if len(re.sub(r"[^a-z]", "", leftover)) >= 4 or (not mapped and _DX_ABBREVIATION.fullmatch(written)):
            found.append((None, written if not mapped else _tidy(leftover), suspected))
    return found


_DX_QUALIFIERS = ("recalcitrant", "chronic", "acute", "recurrent", "resistant", "extensive", "persistent", "refractory",
                  "mild", "moderate", "severe", "bilateral", "left", "right", "known", "case")


def _condition_key(canonical, label):
    return canonical or re.sub(r"[^a-z0-9]", "", label.casefold())


def _diagnosis_parts(text, patient_data=None):
    """[(canonical, label, suspected, meaning)] for one diagnosis field.

    Uses the entry as clinical_terms resolved it (split, abbreviation meaning, registry condition, or "not a
    diagnosis"); the registry patterns when it is unresolved.
    """
    info = clinical_terms.diagnosis(patient_data, text) if patient_data is not None else None
    if info is None:
        return [(canonical, label, suspected, None) for canonical, label, suspected in _split_diagnosis(text)]
    if not info["is_diagnosis"]:
        return []
    parts = [p for p in info["parts"] if not _CONDITION_IGNORE.match(p["text"])]
    whole_suspected = len(parts) == 1 and bool(_SUSPECTED.search(str(text)))
    kept = drop_superseded([p["condition"] for p in parts if p["condition"]])
    return [(p["condition"], _tidy(re.sub(r"\?", "", p["text"])), whole_suspected or bool(_SUSPECTED.search(p["text"])),
             p["meaning"]) for p in parts if not p["condition"] or p["condition"] in kept]


def _collect_conditions(visits, unlinked, long_term=(), patient_data=None):
    conditions = {}

    def add(canonical, label, status, idx, date, meaning=None):
        if not label and not canonical:
            return None
        key = _condition_key(canonical, label)
        if not canonical and key not in conditions:
            # Free-text spellings of one diagnosis ("Tinea" / "Timea") share an entry.
            key = next((k for k, c in conditions.items() if not c["canonical"]
                        and SequenceMatcher(None, k, key).ratio() >= 0.8), key)
        c = conditions.setdefault(key, {"canonical": canonical, "labels": Counter(),
                                        "statuses": set(), "encounters": [], "dates": []})
        if label:
            # Case variants count as one spelling; the most frequent spelling names the condition.
            c["labels"][next((x for x in c["labels"] if x.casefold() == label.casefold()), label)] += 1
        c["statuses"].add(status)
        if idx and idx not in c["encounters"]:
            c["encounters"].append(idx)
        if date:
            c["dates"].append(date)
        if meaning and not canonical:
            c.setdefault("meaning", meaning)
        return key

    for idx, v in enumerate(visits, 1):
        date = _date(v)
        for canonical, label, suspected, meaning in _diagnosis_parts(
                v.get("db_confirmed_dx") or v.get("confirmed_diagnosis"), patient_data):
            add(canonical, label, "suspected" if suspected else "confirmed", idx, date, meaning)
        for canonical, label, suspected, meaning in _diagnosis_parts(
                v.get("db_provisional_dx") or v.get("provisional_diagnosis"), patient_data):
            add(canonical, label, "suspected" if suspected else "working diagnosis", idx, date, meaning)
        complaint = str(v.get("db_chief_complaint") or v.get("chief_complaint") or "")
        history = _HISTORY.search(complaint)
        if history:
            # Only registry conditions count as history; the rest of a complaint is symptoms.
            # K/C/O makes a CHRONIC condition an established diagnosis; anything else after it in the complaint
            # ("K/C/O DM, blurred vision, low back pain") is a presenting symptom, kept as history only.
            kco = bool(_KCO.search(complaint))
            for name in conditions_matching(complaint[history.start():]):
                add(name, "", "known diagnosis" if kco and name in _CHRONIC else "known history", idx, date)

    sources = [(idx, _date(v), d) for idx, v in enumerate(visits, 1) for d in v.get("documents") or []]
    sources += [(None, ((d.get("likely_visit") or {}).get("visit_date") or "")[:10], d) for d in unlinked]
    for idx, date, doc in sources:
        cs = _summary(doc)
        for field in ("confirmed_diagnosis", "provisional_diagnosis"):
            text = str(cs.get(field) or "")
            if not text or _GARBLED.search(text) or re.search(r"(?i)\b(?:bp|pulse|spo2|weight|height)\b", text):
                continue
            for canonical, label, suspected in _split_diagnosis(text):
                key = _condition_key(canonical, label)
                if key not in conditions:
                    add(canonical, label, "document only", idx, date)

    # Conditions held at patient level (long-term list, health history) are known history even when no
    # visit names them; a condition the visits also record is marked long-term.
    for text, source in long_term:
        for canonical, label, _, meaning in _diagnosis_parts(text, patient_data) or [(None, _tidy(text), False, None)]:
            key = add(canonical, label, "known diagnosis", None, "", meaning)
            if key:
                conditions[key].setdefault("sources", []).append(source)

    # A free-text diagnosis whose words, without qualifiers ("recalcitrant", "chronic"), are another condition's
    # recorded name is that condition: merged before visits are counted.
    def core(label):
        words = [w for w in re.findall(r"[a-z]{3,}", label.casefold())
                 if not any(SequenceMatcher(None, w, q).ratio() >= 0.8 for q in _DX_QUALIFIERS)]
        return set(words)
    for key in [k for k, c in conditions.items() if not c["canonical"]]:
        mine = set().union(*(core(label) for label in conditions[key]["labels"])) if conditions[key]["labels"] else core(key)
        target = next((k for k, c in conditions.items() if k != key and mine and any(
            core(label) and core(label) <= mine for label in list(c["labels"]) + [c["canonical"] or ""])), None)
        if target:
            src, dst = conditions.pop(key), conditions[target]
            dst["labels"].update(src["labels"])
            dst["statuses"] |= src["statuses"]
            dst["encounters"] = sorted(set(dst["encounters"]) | set(src["encounters"]))
            dst["dates"] += src["dates"]

    rows = []
    for c in conditions.values():
        status = next(s for s in STATUS_ORDER if s in c["statuses"])
        labels = [label for label, _ in c["labels"].most_common()]
        name = c["canonical"] or labels[0]
        dates = sorted(d for d in c["dates"] if d)
        rows.append({
            "name": name,
            "canonical": c["canonical"],
            "recorded_as": _distinct(r for r in labels if not _same_name(r, name, c["canonical"]))[:3],
            "status": status,
            "visits": len(c["encounters"]),
            "encounters": _encs(c["encounters"]),
            "first_seen": dates[0] if dates else "",
            "last_seen": dates[-1] if dates else "",
            **({"span": _span_text(dates[0], dates[-1])} if len(dates) > 1 and dates[0] != dates[-1] else {}),
            **({"long_term": " / ".join(dict.fromkeys(c["sources"]))} if c.get("sources") else {}),
            **({"meaning": c["meaning"]} if c.get("meaning") and c["meaning"].casefold() not in name.casefold() else {}),
        })
    rows.sort(key=lambda r: (STATUS_ORDER.index(r["status"]), -r["visits"], r["first_seen"]))
    return rows


_PATTERN_OF = {name: pattern for pattern, name in _CONDITION_MAP}


def _same_name(label, name, canonical):
    """True when a recorded label is only an abbreviation of the registry name.

    'Htn' adds nothing to 'Hypertension', so the registry name is shown; 'LSC' does add to a group name
    such as 'Skin condition' (a group name is one its own registry pattern does not match).
    """
    if not canonical:
        return SequenceMatcher(None, label.casefold(), name.casefold()).ratio() >= 0.8
    is_group = not re.search(_PATTERN_OF.get(canonical, r"(?!)"), canonical, re.I)
    return len(re.sub(r"[^a-z]", "", label.casefold())) <= 5 and not is_group


def _distinct(labels):
    """Labels without near-duplicate spellings ('Tinea' / 'Timea'), first spelling kept."""
    kept = []
    for label in labels:
        if not any(SequenceMatcher(None, label.casefold(), k.casefold()).ratio() >= 0.8 for k in kept):
            kept.append(label)
    return kept


def condition_label(condition):
    """Display name: the record's own words, or the registry name when the record only abbreviates it."""
    spellings = condition["recorded_as"]
    # A spelling another one already contains ('Thyroid' in 'Hypothyroidism', 'Diabetes' in 'Diabetes Mellitus') adds nothing.
    spellings = [s for s in spellings if not any(s != o and s.casefold() in o.casefold() for o in spellings)]
    label = " / ".join(spellings) or condition["name"]
    # An abbreviation the record uses, with its meaning ("SN loss (sensorineural hearing loss)").
    return f"{label} ({condition['meaning'].lower()})" if condition.get("meaning") else label


# ── Medicines ───────────────────────────────────────────────────────────────

def _med_word(name):
    words = re.findall(r"[a-z]{3,}", str(name or "").casefold())
    return words[0] if words else ""


# Advice and instructions entered on the prescription ("Salt restricted diet", "Review With Report").
_NOT_A_MEDICINE = re.compile(r"(?i)\b(?:diet|salt|exercise|walk(?:ing)?|advice|advised|counsel\w*|review|report|"
                             r"monitor\w*|follow[\s-]*up|plenty|fluids?\s+intake|rest)\b")
_MEDICINE_CLASSES = [(re.compile(e["pattern"], re.I), e["class"]) for e in REGISTRY["medicine_classes"]]
_TOPICAL_FORM = re.compile(r"(?i)\b(?:gel|cream|oint(?:ment)?|spray|lotion|patch)\b")


def medicine_class(name, info=None):
    """What a medicine is ('NSAID (pain / anti-inflammatory)'): as clinical_terms resolved it, else from the
    registry patterns; '' when unknown."""
    name = str(name or "")
    classes = (info or {}).get("classes") or list(dict.fromkeys(c for pattern, c in _MEDICINE_CLASSES
                                                                if pattern.search(name)))
    if _TOPICAL_FORM.search(name) or (info or {}).get("route") == "topical":
        classes = ["Topical NSAID (pain gel)" if c.startswith("NSAID") else c for c in classes]
    return " + ".join(classes[:3])


def _prescribed_for(med_encounters, condition, visits):
    """A medicine of the right class is for this condition only when the record ties them: prescribed at an encounter
    that records the condition, or the condition is known / long-term / chronic (registry), or the encounter
    has no diagnosis of its own. Clotrimazole written at a gynaecology encounter is not 'for' urticaria recorded at the
    dermatology encounter the same day."""
    # Known / long-term, or a chronic condition (registry) whose medicine is written at any visit.
    if condition["status"] in _KNOWN or condition.get("long_term") or condition.get("canonical") in _CHRONIC:
        return True
    at = set(_encounter_numbers(condition.get("encounters")))
    if not at or at & set(med_encounters):
        return True
    has_dx = lambda i: any(str(visits[i - 1].get(k) or "").strip() for k in
                           ("db_provisional_dx", "db_confirmed_dx", "provisional_diagnosis", "confirmed_diagnosis"))
    return any(not has_dx(i) for i in med_encounters if 0 < i <= len(visits))


def _collect_medicines(visits, conditions, patient_data=None):
    latest = max((_date(v) for v in visits), default="")
    canonical = [c for c in conditions if c["canonical"] in _CONDITION_MEDS]
    meds = {}
    for idx, v in enumerate(visits, 1):
        for m in taken_medications(v):  # 0-0-0 lines state no dose: never an ongoing medicine
            display = _med_display(m.get("name"))
            word = _med_word(display)
            info = clinical_terms.medicine(patient_data, display)
            if not word or (info["is_medicine"] is False if info else _NOT_A_MEDICINE.search(display)):
                continue
            # Spelling variants of one medicine (Cetrizine / Citrizine) share an entry.
            key = next((k for k in meds if SequenceMatcher(None, k, word).ratio() >= 0.8), word)
            regimen = " ".join(str(m.get(f) or "").strip() for f in ("dosage", "frequency", "duration")).strip()
            if dose_not_specified(m):  # 0-0-0 with a duration: prescribed, dosing not entered
                regimen = " ".join(["dose not specified", str(m.get("duration") or "").strip()]).strip()
            entry = meds.setdefault(key, {"name": _tidy(display), "encounters": [], "first": _date(v), "last": "",
                                          "regimen": "", "info": info, "end": None})
            entry["name"] = _tidy(display)
            if idx not in entry["encounters"]:
                entry["encounters"].append(idx)
            entry["last"] = _date(v)
            entry["end"] = course_end(_date(v), m)  # the latest course's end (None: no duration stated)
            entry["regimen"] = regimen or entry["regimen"]
    record_first = min((_date(v) for v in visits if _date(v)), default="")
    # Pre-existing: a condition the record calls known (K/C/O, KCO, "known case of", long-term list) -> its first date.
    preexisting = {condition_label(c): c["first_seen"] for c in conditions if c["status"] in _KNOWN or c.get("long_term")}
    rows = []
    for entry in meds.values():
        treats = set((entry["info"] or {}).get("treats") or ())
        used_for = [c for c in conditions if c["canonical"] and (
            c["canonical"] in treats
            or (c in canonical and re.search(rf"\b(?:{_CONDITION_MEDS[c['canonical']]})", entry["name"], re.I)))
            and _prescribed_for(entry["encounters"], c, visits)]
        # One label per condition: 'Diabetes Mellitus' (long-term list) and 'Diabetes' (a visit) are one condition.
        used_for = list(dict.fromkeys(condition_label(next(x for x in used_for if x["canonical"] == c["canonical"]))
                                      for c in used_for))
        rows.append({
            "name": entry["name"],
            "class": medicine_class(entry["name"], entry["info"]),
            "regimen": entry["regimen"],
            # Current: on the latest prescription, or an earlier course (date + duration) still running on that date.
            "at_latest_visit": entry["last"] == latest or bool(entry["end"] and entry["end"] >= latest),
            **({"course_running_to": entry["end"]} if entry["last"] != latest and entry["end"] and entry["end"] >= latest else {}),
            # On the latest prescription; else a course still running on the latest visit date (not renewed); else not.
            "status": ("current" if entry["last"] == latest else "course running, not renewed"
                       if entry["end"] and entry["end"] >= latest else "not on latest prescription"),
            "course_end": entry["end"],
            "visits": len(entry["encounters"]),
            "encounters": _encs(entry["encounters"]),
            "first": entry["first"],
            "last": entry["last"],
            "for": used_for,
            # Seen at the first recorded visit, or with a known (pre-existing) condition from its first mention: when it
            # was started is not in the record, so it is only ever "first recorded on <date>".
            "first_recorded": bool(entry["first"] and (entry["first"] == record_first or any(
                preexisting.get(c) and entry["first"] <= preexisting[c] for c in used_for))),
        })
    rows.sort(key=lambda r: (r["at_latest_visit"], bool(r["for"]), r["last"], r["visits"]), reverse=True)
    return rows


def start_phrase(m):
    """'first recorded on 2026-05-19' for a medicine already in use when the record begins, else 'started 2026-06-01'."""
    return f"first recorded on {m['first']}" if m.get("first_recorded") else f"started {m['first']}"


_STARTED = re.compile(r"(?i)\b(?:was\s+|were\s+)?(?:started|initiated|commenced|begun)\b(\s+(?:on|at|in|from)\s+)?"
                      r"(\d{4}-\d{2}-\d{2}|#\d+)?")


_STOP_WORDING = re.compile(
    r"(?i)\b(?:(?:was|were|has\s+been|have\s+been|is|are)\s+)?(?:stopped|discontinued|withdrawn|ceased|not\s+continued|"
    r"no\s+longer\s+(?:on\s+it|prescribed|given|taken))\b")
_COUNT_S = re.compile(r"\b(\d+|one|no)(\s+(?:[\w-]+\s+){0,2}?)([\w-]+?)\(s\)")


def _plural(m):
    n, mid, word = m.groups()
    one = n in ("1", "one")
    if one:
        return f"{n}{mid}{word}"
    return f"{n}{mid}{word[:-1]}ies" if re.search(r"[^aeiou]y$", word) else f"{n}{mid}{word}{'es' if re.search(r'(?:s|x|ch|sh)$', word) else 's'}"


def plain_wording(text):
    """Doctor-facing wording: a medicine is never 'stopped' (the record only shows it is not on the latest
    prescription) and a count agrees with its noun ('1 missed follow-up', '3 visits', not '(s)')."""
    if isinstance(text, list):
        return [plain_wording(t) for t in text]
    text = re.sub(r"(?i)\bNot continued:", "Not on the latest prescription:", str(text))
    text = _STOP_WORDING.sub("not on the latest prescription", text)
    text = re.sub(r"(?i)\bnot on the latest prescription(?:\s+not on the latest prescription)+", "not on the latest prescription", text)
    return _COUNT_S.sub(_plural, text)


def use_first_recorded_wording(lines, evidence):
    """A medicine first seen at the first recorded visit, or with a known (K/C/O) condition, is 'first recorded on
    <date>', never 'started on <date>'."""
    meds = [(m, _med_word(m["name"])) for m in evidence["medicines"] if m.get("first_recorded") and _med_word(m["name"])]
    if not meds:
        return lines
    out = []
    for line in lines:
        parts = re.split(r"(;\s+(?![^()]*\)))", line)
        for k, part in enumerate(parts):
            named = [m for m, word in meds if re.search(rf"\b{re.escape(word[:5])}", part, re.I)]
            if named and _STARTED.search(part):
                first = min(m["first"] for m in named)
                parts[k] = _STARTED.sub(f"first recorded on {first}", part, count=1)
        out.append("".join(parts))
    return out


# ── Labs ────────────────────────────────────────────────────────────────────

def _direction(status):
    low = str(status or "").casefold()
    if "reactive" in low and "non" not in low:
        return "reactive"
    if "high" in low:
        return "high"
    if "low" in low:
        return "low"
    return None


def _status_label(status):
    status = str(status or "")
    direction = _direction(status)
    if status.startswith("Critical"):
        return f"Critical {direction}"
    return {"high": "High", "low": "Low", "reactive": "Reactive"}.get(direction, status)


def _unit(value, reference):
    """Unit as written with the value, else from the reference text; '' when neither states one."""
    _, number, _, unit = lab_validation.parse_value(value)
    if number is not None and unit and unit != "?":
        return lab_validation._unit_label(unit)
    tail = re.sub(r"^[\s\d.,<>≤≥=\-–—]+", "", _STANDARD_REF.sub("", str(reference or "")))
    unit = lab_validation.parse_unit(tail) if tail else None
    return lab_validation._unit_label(unit) if unit and unit != "?" else ""


def _value_text(value, unit):
    value = str(value or "").strip()
    if unit and not re.search(r"[a-z%µ×/]", re.sub(r"^[<>≤≥=]*\s*[\d.,]+", "", value), re.I):
        return f"{value} {unit}"
    return value


def _reference(lab):
    ref = str(lab.get("reference") or "").strip()
    if not ref or ref.casefold() in {"not specified", "none", "n/a", "-"}:
        return "", "none given"
    if lab.get("reference_source") == "standard" or _STANDARD_REF.search(ref):
        return _STANDARD_REF.sub("", ref), "standard adult range; report gave none"
    return ref, "report"


def _collect_labs(visits, unlinked):
    rows = []
    for idx, v in enumerate(visits, 1):
        rows += [(lab, idx, _date(v)) for lab in v.get("lab_results") or []]
    for doc in unlinked:
        likely = (doc.get("likely_visit") or {}).get("visit_date") or ""
        rows += [(lab, None, str(likely)[:10]) for lab in doc.get("lab_results") or []]
    results, unverified, not_evaluated = {}, [], 0
    for lab, idx, date in rows:
        status = str(lab.get("status") or "")
        name = str(lab.get("test_name") or "").strip()
        key = lab_validation.identify(name) or re.sub(r"[^a-z0-9]", "", name.casefold())
        if not name or not key:
            continue
        if status.startswith("Unverified"):
            item = {"test": lab_validation.clean_test_name(name), "value": str(lab.get("value") or ""),
                    "date": date, "note": str(lab.get("note") or "")[:140]}
            if item not in unverified:
                unverified.append(item)
            continue
        if status.startswith("Derived") or lab_validation.is_derived(name):
            continue  # calculated from another result (eAG from HbA1c): never interpreted or listed as history
        if not status or status.casefold() == "not evaluated":
            not_evaluated += 1
            continue
        comparator, number, _, _ = lab_validation.parse_value(lab.get("value"))
        ident = (key, number if number is not None else str(lab.get("value")).casefold(), date)
        # The same report linked to two encounters (same test and value within 14 days) is one result.
        entry = results.get(ident) or next(
            (e for (k, n, d), e in results.items() if k == key and number is not None and n == number
             and _within_days(d, date, 14)), None)
        if entry is None:
            reference, source = _reference(lab)
            unit = _unit(lab.get("value"), lab.get("reference"))
            entry = results[ident] = {
                "key": key, "test": lab_validation.clean_test_name(name),
                "test_as_written": name, "value": _value_text(lab.get("value"), unit), "number": number,
                "unit": unit, "reference": reference, "reference_source": source,
                "status": _status_label(status), "direction": _direction(status),
                "critical": status.startswith("Critical"), "date": date, "encounters": [],
                "as_written": _as_written(lab.get("note")),
                "linked_to_visit": idx is not None,
            }
        if idx:
            entry["encounters"].append(idx)
    return list(results.values()), unverified, not_evaluated


def _span_text(first, last):
    """'135 days (about 4 months)' between two ISO dates; '' when either is missing."""
    try:
        days = (datetime.date.fromisoformat(last) - datetime.date.fromisoformat(first)).days
    except (TypeError, ValueError):
        return ""
    if days < 14:
        return f"{days} days"
    if days < 60:
        return f"{days} days (about {round(days / 7)} weeks)"
    return f"{days} days (about {round(days / 30.4)} months)"


def _condition_words(c):
    return set(re.findall(r"[a-z]{3,}", " ".join([c["name"], *c.get("recorded_as", [])]).casefold()))


def _encounter_text(v):
    parts = [v.get("db_chief_complaint") or v.get("chief_complaint"), v.get("db_provisional_dx") or v.get("provisional_diagnosis"),
             v.get("db_confirmed_dx") or v.get("confirmed_diagnosis")]
    parts += [f"{m.get('name')} {m.get('dosage') or ''}" for m in v.get("db_medications") or [] if m.get("name")]
    parts += [f"document discrepancy {d}" for d in v.get("discrepancies") or []]
    parts += [str(a) for a in v.get("db_instructions") or []]  # symptoms the doctor wrote there ('Chest pain, DOE')
    parts += [f"{l.get('test_name')} {l.get('value')}" for l in v.get("lab_results") or []]
    parts += [f"{r.get('label')} {r.get('value')}" for r in v.get("db_vitals_rows") or []]
    parts += [str((d.get("clinical_summary") or {}).get("provisional_diagnosis") or "") for d in v.get("documents") or []
              if isinstance(d.get("clinical_summary"), dict)]
    parts += [f"{f.get('status')} {f.get('resolution') or ''} {f.get('scheduled_date') or ''}" for f in v.get("followups") or []]
    return " ".join(str(x) for x in parts if x).casefold()


def _within_days(a, b, days):
    try:
        return abs((datetime.date.fromisoformat(a) - datetime.date.fromisoformat(b)).days) <= days
    except (TypeError, ValueError):
        return a == b


def _as_written(note):
    """The report's own wording, which lab_validation keeps in the note when it re-read a value."""
    match = re.match(r"Written as '(.{1,80}?)'", str(note or ""))
    return " ".join(match.group(1).split()) if match else ""


def _significance(entry):
    spec = _SIGNIFICANCE.get(entry["key"]) or {}
    return spec.get(entry["direction"] or "", "")


def _is_adverse(entry):
    concern = (_SIGNIFICANCE.get(entry["key"]) or {}).get("concern")
    return entry["critical"] or not concern or entry["direction"] in concern


def _short(entry, with_status=True):
    when = entry["date"] or "date unknown"
    status = f", {entry['status']}" if with_status and entry["direction"] else ""
    return f"{entry['value']} ({when}{status})"


def _abnormal_groups(labs, conditions, medicines):
    """One group per test: its latest out-of-range result plus every other result of that test."""
    by_key = {}
    for entry in labs:
        by_key.setdefault(entry["key"], []).append(entry)
    groups, non_adverse = [], []
    latest_rx = max((m["last"] for m in medicines if m["last"]), default="")
    for key, series in by_key.items():
        series.sort(key=lambda e: e["date"] or "")
        out = [e for e in series if e["direction"]]
        if not out:
            continue
        adverse = [e for e in out if _is_adverse(e)]
        if not adverse:
            non_adverse.append(out[-1])
            continue
        latest = adverse[-1]
        name = latest["test_as_written"]
        # A result is the control marker of a recorded condition (HbA1c for diabetes) only per control_labs,
        # and only in that condition's adverse direction; condition_labs links are context for the model.
        controls = {condition_label(c) for c in conditions if _controls(c["canonical"], name, latest["direction"])}
        context = {condition_label(c) for c in conditions if c["canonical"]
                   and re.search(_CONDITION_LABS.get(c["canonical"], r"(?!)"), name, re.I)} - controls
        # Same-day results of related tests that were in range ("PCV low, haemoglobin normal").
        family = [p for p in _CONDITION_LABS.values() if re.search(p, name, re.I)]
        in_range = list(dict.fromkeys(
            f"{e['test']} {e['value']}" for e in labs
            if e["key"] != key and not e["direction"] and e["date"] == latest["date"]
            and any(re.search(p, e["test_as_written"], re.I) for p in family)))[:2]
        groups.append({
            **{k: v for k, v in latest.items() if k not in ("number", "linked_to_visit")},
            "in_range_context": in_range,
            "encounters": _encs(latest["encounters"]),
            "significance": _significance(latest),
            "earlier": [_short(e) for e in series if e is not latest and (e["date"] or "") <= (latest["date"] or "")][-3:],
            "later": [_short(e) for e in series if (e["date"] or "") > (latest["date"] or "")][:3],
            "times_abnormal": len(adverse),
            "related_conditions": sorted(controls),
            # Only medicines on the latest prescription count as current ("despite X"); earlier ones are kept apart.
            "related_medicines": [m["name"] for m in medicines if set(m["for"]) & controls and m["last"] == latest_rx][:4],
            "earlier_related_medicines": [f"{m['name']} (last {m['last']})" for m in medicines
                                          if set(m["for"]) & controls and m["last"] != latest_rx][:3],
            # Timing: a result can be "despite" a medicine only if the medicine started before the result's date.
            "medicines_started_before_result": [m["name"] for m in medicines if set(m["for"]) & controls
                                                and m["last"] == latest_rx and latest["date"] and m["first"]
                                                and m["first"] < latest["date"]][:4],
            "medicines_started_on_or_after_result": [f"{m['name']} ({start_phrase(m)})" for m in medicines
                                                     if set(m["for"]) & controls and latest["date"] and m["first"]
                                                     and m["first"] >= latest["date"]][:4],
            "also_relevant_to": sorted(context),
        })
    groups.sort(key=lambda g: (g["critical"], bool(g["related_conditions"]), g["date"] or ""), reverse=True)
    return groups, non_adverse


def _controls(canonical, test_name, direction):
    rule = _CONTROL_LABS.get(canonical or "")
    return bool(rule and re.search(rule["pattern"], test_name, re.I)
                and (rule["direction"] == "any" or rule["direction"] == direction))


def _possible_conditions(groups, conditions):
    """Conditions the abnormal results point to (registry control markers) that the record never diagnoses."""
    present = {c["canonical"] for c in conditions if c["canonical"]}
    found = []
    for condition, rule in _CONTROL_LABS.items():
        if condition in present:
            continue
        basis = [g for g in groups if _controls(condition, g["test_as_written"], g["direction"])]
        if basis:
            found.append({
                "condition": condition,
                "basis": [f"{g['test']} {g['value']} ({g['status']}, {g['date'] or 'date unknown'})" for g in basis[:3]],
                "single_result": all(g["times_abnormal"] == 1 for g in basis),
                "note": "Not diagnosed in the record; the result(s) raise the possibility only.",
            })
    return found


# ── Follow-up ───────────────────────────────────────────────────────────────

_NOT_REACHED = {"CONTACT_DETAILS_MISSING", "CALL_NOT_RECEIVED", "CALL_NOT_CONNECTED"}


def _followup_gaps(visits, labs, groups):
    items, not_reached = [], []
    for idx, v in enumerate(visits, 1):
        for f in v.get("followups") or []:
            status = str(f.get("status") or "").upper()
            resolution = str(f.get("resolution") or "").upper()
            when = str(f.get("scheduled_date") or "")[:10]
            if status == "PENDING":
                items.append(f"Follow-up scheduled {when} after #{idx} is pending.")
            elif resolution in _NOT_REACHED:
                not_reached.append(f"{when} after #{idx} ({_FOLLOWUP_OUTCOMES[resolution].split('(')[-1].rstrip(')')})")
            elif resolution not in _ROUTINE_RESOLUTIONS:
                outcome = _FOLLOWUP_OUTCOMES.get(resolution, resolution.replace("_", " ").lower())
                items.append(f"Follow-up call {when} after #{idx}: {outcome}.")
    if not_reached:
        # A call that did not reach the patient is not a missed visit; missed visits are the MISSED follow-ups.
        items.append("Follow-up call(s) where the patient was not reached (not missed visits): "
                     + ", ".join(not_reached) + ".")
    seen = set()
    lab_dates = [(e["key"], e["date"] or "") for e in labs]
    for idx, v in enumerate(visits, 1):
        missing = []
        for doc in v.get("documents") or []:
            for test in _summary(doc).get("tests_advised") or []:
                key = lab_validation.identify(test)
                if not key or key in seen:
                    continue
                seen.add(key)
                panel = key.startswith("panel_")
                if not any(d >= _date(v) and (panel or k == key) for k, d in lab_dates):
                    missing.append(_tidy(test))
        if missing:
            items.append(f"Tests advised at #{idx} ({_date(v)}) with no result in the record: {', '.join(missing)}.")
    not_repeated = [f"{g['test']} ({g['date']})" for g in groups if not g["later"] and g["date"]]
    if not_repeated:
        items.append("No repeat result after the out-of-range value for: " + ", ".join(not_repeated[:6]) + ".")
    return items


# ── Evidence pack ───────────────────────────────────────────────────────────

# Visit reasons that are not symptoms ("report check", "follow up", "refill").
_ADMIN_COMPLAINT = re.compile(r"(?i)\b(?:repor\w*|follow[\s-]*up|review|check[\s-]*up|routine|refill|repeat|f/u)\b")


# Status notes written in the complaint, not symptoms: pregnancy (how far along: registry pattern), LMP / EDD dates.
_STATUS_NOTE = re.compile(r"(?i)\b(?:lmp|edd)\b|\b\d+(?:\.\d+)?\s*(?:wks?|weeks?|months?)\s+(?:of\s+)?(?:pregnan\w*|gestation)|"
                          + ((REGISTRY.get("guidelines") or {}).get("haemoglobin") or {}).get("pregnancy_pattern", r"(?!x)x"))


# An empty field written as text ('Not recorded', 'Nill', 'NA', '-'): no complaint.
_PLACEHOLDER = re.compile(r"(?i)\s*(?:not\s+(?:recorded|mentioned|available|given)|nil+|none|n\s*/?\s*a|no\s+complaints?|-+|\.+)\s*")


def _complaint_phrases(text):
    """Symptom phrases of one chief complaint, without known-history parts ('K/C/O DM') or visit reasons."""
    phrases = []
    text = _HISTORY_SLASHED.sub(lambda m: "kco" if "c" in m.group(0).casefold() else "history of", str(text or ""))
    for part in re.split(r"[,;/+\n]|\band\b", text, flags=re.I):
        part = part.strip(" .-:").casefold()
        if (not part or _HISTORY.search(part) or _ADMIN_COMPLAINT.search(part) or _STATUS_NOTE.search(part)
                or _PLACEHOLDER.fullmatch(part)):
            continue
        if re.search(r"\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|iu|units?|ml)\b", part):
            continue  # 'Thyroid 25mcg' notes a medicine and its dose, not a symptom
        rest = part
        for pattern, _ in _CONDITION_MAP:
            rest = re.sub(rf"\w*(?:{pattern})\w*", " ", rest, flags=re.I)
        # A bare condition abbreviation ('dm', 'htn') is history, not a symptom; 'back pain' is a symptom.
        if len(re.sub(r"[^a-z]", "", part)) > 5 or len(re.sub(r"[^a-z]", "", rest)) >= 3:
            phrases.append(" ".join(part.split()))
    return phrases


def _problem_key(phrase):
    """A registry concept for a complaint or diagnosis phrase: the condition (condition_map) first, then the symptom
    concept (symptom_map), else the phrase itself."""
    import symptom_map
    found = conditions_matching(phrase)
    if found:
        return found[0]
    hits = symptom_map.classify(phrase)
    return hits[0]["symptom"] if len(hits) == 1 and hits[0]["text"].casefold() == phrase else phrase


def _recurring_complaints(visits):
    """Problems raised at two or more visits, in the complaint OR the diagnosis, most frequent first. Phrases are grouped
    by registry concept ('lower back pain' and 'Low back ache' are one), spelling variants merged."""
    seen, label = {}, {}
    for idx, v in enumerate(visits, 1):
        phrases = _complaint_phrases(v.get("db_chief_complaint") or v.get("chief_complaint"))
        for field in ("db_provisional_dx", "db_confirmed_dx", "provisional_diagnosis", "confirmed_diagnosis"):
            # One phrase per condition: 'HT with DM with IHD' is three, so IHD at one visit never recurs with HT.
            phrases += [p.strip().casefold() for p in _DX_SPLIT.split(str(v.get(field) or ""))
                        if len(p.strip()) >= 2 and not _CONDITION_IGNORE.match(p.strip())]
        for phrase in dict.fromkeys(phrases):
            key = _problem_key(phrase)
            # Spelling variants, or one wording inside the other ('gradual weight gain' / 'weight gain'), are one problem.
            def inside(a, b):  # a is a whole-word part of a longer b
                return len(a) >= 8 and len(a) < len(b) and bool(re.search(rf"\b{re.escape(a.casefold())}\b", b.casefold()))
            key = next((k for k in seen if SequenceMatcher(None, k.casefold(), key.casefold()).ratio() >= 0.8
                        or inside(k, key) or inside(key, k)), key)
            label.setdefault(key, Counter())[phrase] += 1  # named by the record's most frequent wording
            if idx not in seen.setdefault(key, []):
                seen[key].append(idx)
    def name(k, e):
        # The record's wording when one wording covers every visit; different complaints of one concept ('shoulder
        # pain', 'knee pain') are named by the concept, so no visit is cited for a complaint it does not hold.
        phrase, count = label[k].most_common(1)[0]
        # One visit's duration ('since 2 days') is not the recurring problem's: #2 may say 'since 4 days'.
        phrase = re.sub(r"(?i)\s*\b(?:since|for|from|x)\s*(?:last\s+|past\s+)?\d+(?:\.\d+)?\s*-?\s*\w+.*$", "", phrase).strip() or phrase
        # A wording every other one contains ('weight gain' in 'gradual weight gain') names them all.
        core = min(label[k], key=len)
        if len(label[k]) > 1 and all(re.search(rf"\b{re.escape(core)}\b", p) for p in label[k]):
            return core
        same = all(SequenceMatcher(None, phrase, p).ratio() >= 0.8 for p in label[k])  # spelling variants of one wording
        if same or count >= len(e) or k.casefold() in label[k]:
            return phrase
        return re.sub(r"\s*\([^)]*\)", "", k).casefold()
    rows = [{"complaint": name(k, e), "visits": len(e), "encounters": _encs(sorted(e)),
             "encounter_numbers": sorted(e)}
            for k, e in seen.items() if len(e) >= 2]
    return sorted(rows, key=lambda r: -r["visits"])[:5]


def _timeline(visits):
    rows = []
    for idx, v in enumerate(visits, 1):
        rows.append({
            "enc": f"#{idx}",
            "date": _date(v),
            "complaint": _tidy(v.get("db_chief_complaint") or v.get("chief_complaint"))[:90],
            "diagnosis": _tidy(v.get("db_confirmed_dx") or v.get("confirmed_diagnosis")
                               or v.get("db_provisional_dx") or v.get("provisional_diagnosis"))[:70],
            "documents": len(v.get("documents") or []),
        })
    return rows if len(rows) <= 14 else rows[:2] + rows[-12:]


def _latest_vitals(visits):
    for v in reversed(visits):
        rows = v.get("db_vitals_rows") or []
        if rows:
            return {"date": _date(v), **{
                str(r.get("label")): str(r.get("value")) + (" (impossible value; entry/unit error, do not interpret)"
                                                             if vital_implausible(r.get("label"), r.get("value")) else "")
                for r in rows if r.get("value")}}
    return {}


@traced(name="snapshot.build_insight_evidence")
def build_insight_evidence(patient_data):
    """Structured, patient-specific facts for the review; cached on patient_data."""
    cached = patient_data.get("_insight_evidence")
    if cached is not None:
        return cached
    visits = recheck_followups(patient_data.get("visits") or [])
    unlinked = patient_data.get("unlinked_documents") or []
    conditions = _collect_conditions(visits, unlinked, long_term_labels(patient_data), patient_data)
    medicines = _collect_medicines(visits, conditions, patient_data)
    labs, unverified, not_evaluated = _collect_labs(visits, unlinked)
    groups, non_adverse = _abnormal_groups(labs, conditions, medicines)
    # A medicine may be named as a possible cause of a result only if taken within its plausible window before the
    # test (registry drug_lab_effects, per drug class); undated results get none.
    rec = medicine_profile.record(patient_data)
    for g in groups:
        in_window = medicine_profile.effect_window_medicines(rec, g["key"], g["date"], g["direction"])
        for effect in medicine_profile.REF["drug_lab_effects"]:
            term = effect.get("text_term")
            if term and g["key"] in effect["labs"] and not any(w["effect"] == effect["effect"] for w in in_window):
                text = re.sub(rf"(?i),?\s*\b{re.escape(term)}\w*(?:\s+use)?\b", "", g["significance"] or "")
                g["significance"] = re.sub(r",\s*or\b", " or", re.sub(r"\s{2,}", " ", text)).strip()
        g["medicines_in_effect_window"] = [
            f"{w['name']} ({w['effect']}; taken up to {w['last_taken']}, window {w['window_days']} days)"
            for w in medicine_profile.effect_window_medicines(rec, g["key"], g["date"], g["direction"])]
    try:
        snapshot = build_snapshot(patient_data) or {}
    except Exception:  # the review must not fail on a snapshot rule; it only loses those flags
        snapshot = {}
    flags = snapshot.get("flags") or []
    guideline = snapshot.get("guideline") or {}
    # Draft (unverified) guideline rows are for the doctor's screen only; the AI and the brief read verified ones.
    guideline = {**guideline, "rows": guideline_rows_in_use(guideline.get("rows"))}
    # A guideline control row (diabetes) states whether its marker is due, so followup_gaps does not repeat it.
    guided = {_GUIDELINE_MARKERS.get(r["condition"]) for r in guideline.get("rows") or [] if r.get("marker")}
    dates = sorted(_date(v) for v in visits if _date(v))
    discrepancies = list(dict.fromkeys(d for v in visits for d in v.get("discrepancies") or []))
    # Free text the clinician wrote at the visit (advice, but also impressions such as a suspected condition).
    notes = list(dict.fromkeys(_tidy(a) for v in visits for a in v.get("db_instructions") or [] if str(a).strip()))
    evidence = {
        "patient": {
            "age_sex": patient_data.get("age_sex") or _age_sex(visits, unlinked) or "not recorded",
            "encounters": len(visits),
            # All encounters on one date are one clinical day (two departments on the same day).
            "clinical_days": len(clinical_days(visits)),
            # The record's span, computed here: the model copies it and never works a duration out itself.
            "span": _span_text(dates[0], dates[-1]) if len(dates) > 1 else "",
            "first_visit": dates[0] if dates else "",
            "latest_visit": dates[-1] if dates else "",
        },
        # A bare health-history item that is already a condition above ("Diabetes") is not listed twice; one that adds
        # a detail ("Diabetes: since 2019") stays.
        "health_history": [f"{h['item']}: {h['detail']}" if h.get("detail") else h["item"]
                           for h in patient_data.get("health_history_present") or []
                           if h.get("detail") or not any(_label_in(h["item"], _condition_words(c)) for c in conditions)],
        # What each encounter holds, casefolded: a citation is kept only where the cited encounter contains the claim.
        "encounter_text": {str(i): _encounter_text(v) for i, v in enumerate(visits, 1)},
        "encounter_dates": {str(i): _date(v) for i, v in enumerate(visits, 1)},
        # Diagnosis fields only, for points that state a diagnosis.
        "encounter_dx": {str(i): " ".join(str(v.get(k) or "") for k in ("db_provisional_dx", "db_confirmed_dx",
                                                                        "provisional_diagnosis", "confirmed_diagnosis")).casefold()
                         for i, v in enumerate(visits, 1)},
        "conditions": conditions,
        "possible_conditions": _possible_conditions(groups, conditions),
        "medicines": medicines[:15],
        # Every medicine name, for citation checks (the list above is capped for the prompt).
        "medicine_names": [m["name"] for m in medicines],
        "abnormal_labs": groups,
        "outside_range_not_adverse": [f"{e['test']} {_short(e)}" for e in non_adverse],
        "unverified_labs": unverified[:6],
        "lab_counts": {"results": len(labs), "out_of_range": len(groups), "unverified": len(unverified),
                       "not_evaluated": not_evaluated},
        "latest_vitals": _latest_vitals(visits),
        # Flags with a source are guideline findings: the model reads them once, as guideline_context rows.
        "rule_flags": [{k: f.get(k) for k in ("level", "kind", "title", "evidence", "significance", "category", "time", "as_of", "undated", "combo", "stopped", "problem")
                        if f.get(k)}
                       for f in flags if f.get("kind") != "lab" and not f.get("source")],
        "guideline_context": ({"profile": {k: v for k, v in guideline["profile"].items() if v is not None},
                               "rows": guideline["rows"]} if guideline.get("rows") else {}),
        "followup_gaps": _followup_gaps(visits, labs, [g for g in groups if g["key"] not in guided]),
        "clinician_notes": notes[:6],
        "record_checks": discrepancies[:4] + ([f"{len(unlinked)} report(s) not linked to a visit"] if unlinked else [])
                         + ([f"No dose stated (0-0-0), not counted as taken: {', '.join(zero_dose_lines(visits)[:4])}"]
                            if zero_dose_lines(visits) else []),
        "recurring_complaints": _recurring_complaints(visits),
        "timeline": _timeline(visits),
    }
    patient_data["_insight_evidence"] = evidence
    return evidence


def evidence_for_prompt(evidence):
    """The evidence pack without internal keys, as compact JSON for the model."""
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if k not in ("key", "canonical", "direction", "test_as_written") and v not in ("", [], None, {})}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    shown = dict(evidence)
    shown.pop("encounter_text", None)
    shown.pop("encounter_dates", None)
    shown.pop("encounter_dx", None)
    shown.pop("_metrics", None)
    # Conditions under the record's own words ("Gastritis", not the registry group "GERD / acidity").
    shown["conditions"] = [{**c, "name": _short_label(c), "recorded_as": [r for r in c["recorded_as"] if r != _short_label(c)],
                            **({"registry_group": c["name"]} if _short_label(c) != c["name"] else {})}
                           for c in evidence["conditions"]]
    return json.dumps(clean(shown), ensure_ascii=False, separators=(",", ":"))


# ── Markdown ────────────────────────────────────────────────────────────────

def _lab_line(g):
    ref = f"reference {g['reference']}" + (f" ({g['reference_source']})" if g["reference_source"] != "report" else "") \
        if g["reference"] else "no reference range given"
    when = g["date"] or "date unknown"
    where = f", {g['encounters']}" if g["encounters"] else ("" if g["date"] else ", report not linked to a visit")
    line = f"- **{g['test']} {g['value']}** ({g['status']}; {ref}; {when}{where})"
    if g["significance"]:
        line += f": {g['significance']}"
    if g.get("as_written") and g["direction"] == "reactive":
        line += ("" if line.endswith(".") else ".") + f" Report text: '{g['as_written']}'."
    extra = []
    if g["earlier"]:
        extra.append("Earlier: " + "; ".join(g["earlier"]))
    if g["later"]:
        extra.append("Later: " + "; ".join(g["later"]))
    if g["related_conditions"]:
        extra.append("Recorded condition: " + _join(g["related_conditions"])
                     + (f", with {_join(g['related_medicines'])} on the latest prescription" if g["related_medicines"] else "")
                     + (f"; earlier {_join(g['earlier_related_medicines'])}" if g.get("earlier_related_medicines") else ""))
    if extra:
        line += (" " if line.endswith(".") else ". ") + ". ".join(extra) + "."
    return line


def render_abnormal_results(evidence):
    groups = evidence["abnormal_labs"]
    counts = evidence["lab_counts"]
    gaps = [f for f in evidence["rule_flags"] if f.get("kind") == "followup" and str(f["title"]).startswith("No ")
            and "in the record (" in f["title"]]
    def gap_line(f):
        match = re.match(r"^No (.+) in the record \((.+)\)$", f["title"])
        test, finding = match.groups() if match else (f["title"], "")
        return f"- **Not in the record:** {test}" + (f" (for {finding})" if finding else "") \
            + f" — {f['evidence'].split(' · ')[0]}."
    gap_lines = [gap_line(f) for f in gaps]
    if not counts["results"] and not evidence["unverified_labs"]:
        return ["- No laboratory results are available in the record."] + gap_lines
    lines = [_lab_line(g) for g in groups[:_MAX_ABNORMAL]]
    rest = groups[_MAX_ABNORMAL:]
    if rest:
        lines.append("- **Also out of range:** " + "; ".join(f"{g['test']} {g['value']} ({g['status']}, {g['date'] or 'date unknown'})"
                                                              for g in rest) + ".")
    if not groups:
        lines.append(f"- All {counts['results']} evaluated laboratory result(s) are within their reference ranges.")
    if evidence["outside_range_not_adverse"]:
        lines.append("- **Outside the printed range, usually without clinical significance in this direction:** "
                     + "; ".join(evidence["outside_range_not_adverse"]) + ".")
    if evidence["unverified_labs"]:
        lines.append("- **Not interpreted (check the source document):** "
                     + "; ".join(f"{u['test']} '{u['value']}'" + (f" ({u['note'].rstrip('.')})" if u["note"] else "")
                                 for u in evidence["unverified_labs"][:4]) + ".")
    return lines + gap_lines


def _encounter_numbers(text):
    return [int(n) for n in re.findall(r"#(\d+)", _expand_ranges(text or ""))]


# Words too common in records to tie a point to an encounter.
_CITE_STOP = {"follow", "report", "reports", "visits", "latest", "prescription", "recorded", "record", "medicine",
              "medicines", "result", "results", "patient", "encounter", "between", "within", "without", "general"}


def cite_points(lines, evidence, limit=3):
    """Cite, for each point, only visits that contain what it states.

    - A point that states a rule flag cites exactly that flag's visits.
    - Otherwise: the visits whose record holds the medicines, conditions and lab results the point names.
    - A lab result from an undated or unlinked report gets no visit number: '(undated report)'.
    - Citations the model wrote are kept only where the cited visit holds something the point names.
    """
    flags = evidence["rule_flags"]
    texts = {int(k): v for k, v in (evidence.get("encounter_text") or {}).items()}
    dates = {int(k): v for k, v in (evidence.get("encounter_dates") or {}).items()}

    diagnosis_point = [False]

    def same_day_only_holders(encs, words):
        """From two visits on one date, keep only the one(s) holding what the point names."""
        out = []
        for n in encs:
            pair = [m for m in encs if m != n and dates.get(m) and dates.get(m) == dates.get(n)]
            if pair and words and not holds(n, words) and any(holds(m, words) for m in pair):
                continue
            out.append(n)
        return out

    def terms(point):
        low, words = point.casefold(), set(re.findall(r"[a-z]{3,}", point.casefold()))
        found, undated = [], False
        for g in evidence["abnormal_labs"]:
            number = _first_number(g["value"])
            if g["test"].casefold() in low or (number and re.search(rf"(?<![\d.]){re.escape(number)}(?!\d)", point)):
                if _encounter_numbers(g["encounters"]):
                    found.append(g["test"].casefold())
                else:
                    undated = True
        for name in evidence.get("medicine_names") or [m["name"] for m in evidence["medicines"]]:
            word = re.sub(r"[^a-z]", "", name.split()[0].casefold())[:5]
            if len(word) >= 4 and re.search(rf"\b{word}", low):
                found.append(word)
        for c in evidence["conditions"]:
            for label in [c["name"], *c.get("recorded_as", [])]:
                if label and _label_in(label, words):
                    found += [w for w in re.findall(r"[a-z]{4,}", label.casefold())][:2]
        # A possible (undiagnosed) condition rests on its lab results: those are the facts to cite.
        for pc in evidence.get("possible_conditions") or []:
            if _label_in(pc["condition"].split(" / ")[0], words):
                found += [re.sub(r"[^a-z ]", "", b.split(" ")[0].casefold()) for b in pc["basis"]]
        # Complaint / diagnosis words held by any encounter (every encounter, not only the shortened timeline).
        found += [w for w in re.findall(r"[a-z]{6,}", low) if w not in _CITE_STOP
                  and any(re.search(rf"\b{re.escape(w[:6])}", t) for t in texts.values())][:3]
        # A two-word name an encounter's record holds as written ('rbc count', 'serum sodium'): a result that is not
        # abnormal (unverified, in range) is still a fact of that encounter.
        tokens = re.findall(r"[a-z][a-z0-9]{1,}", low)
        found += [f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if len(a + b) >= 6
                  and a not in _CITE_STOP and b not in _CITE_STOP
                  and any(re.search(rf"\b{re.escape(a)} {re.escape(b)}\b", t) for t in texts.values())][:2]
        return list(dict.fromkeys(found)), undated

    def stem(w):
        # Short words (and medicine stems, already cut to 5) by their first 5 letters; longer words need more of
        # themselves, so 'complaint' never matches 'completed'.
        return w[:5] if len(w) <= 6 else w[:len(w) - 2]

    dx_texts = {int(k): v for k, v in (evidence.get("encounter_dx") or {}).items()}

    def held(n, words):
        # A point that states a diagnosis is held only by an encounter whose diagnosis fields hold it.
        text = dx_texts.get(n, "") if diagnosis_point[0] and dx_texts else texts.get(n, "")
        return sum(1 for w in words if re.search(rf"\b{re.escape(stem(w))}", text))

    def holds(n, words):
        return held(n, words) > 0

    def best_holders(words):
        """The visits holding the most of the point's facts (latest last)."""
        scores = {n: held(n, words) for n in texts} if words else {}
        top = max(scores.values(), default=0)
        return sorted(n for n, s in scores.items() if top and s == top)

    # Words that only name an undated lab result (they do not tie the point to any encounter).
    _lab_words = {g["test"].casefold() for g in evidence["abnormal_labs"] if not _encounter_numbers(g["encounters"])}
    out = []
    for line in lines:
        prefix = re.match(r"^(\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*[^*]+\*\*:?\s*)?)", line).group(0)
        points = []
        for point in re.split(r";\s+(?![^()]*\))", line[len(prefix):]):
            if not point.strip():
                continue
            end = re.search(r"[.\s]*$", point).start()
            body, tail = point[:end], point[end:]
            flag = _point_flag(body, flags)
            words, undated = terms(body)
            diagnosis_point[0] = bool(re.search(r"(?i)\bdiagnos[ie]s|\bdiagnosed\b", body))
            # '#1–3' -> '#1, #2, #3': each encounter is checked (and removed) on its own; written back as a range after.
            had_range, body = _expand_ranges(body) != body, _expand_ranges(body)
            written = [int(n) for n in re.findall(r"#(\d+)", body)]
            plain = re.sub(r"\((?:not (?:all )?)?on the latest prescription[^)]*\)*|(?:not (?:all )?)?on the latest prescription", "", body)
            if flag and re.search(r"(?i)missed follow-up", flag["title"]):
                # Every missed follow-up is cited.
                pairs = re.findall(r"(\d{4}-\d{2}-\d{2})[^()]*\(after ((?:#\d+(?:,\s*)?)+)\)", str(flag.get("evidence")))
                named = sorted(set([int(n) for d, ns in pairs if d in body for n in re.findall(r"\d+", ns)]
                                   or _encounter_numbers(flag.get("evidence"))))
                # Numbers written inside the point ("after #6") that are right are kept in place.
                keep = [n for n in written if n in named]
                cite = [n for n in named if n not in keep]
            elif re.search(r"(?i)\blatest\b|\bcurrent(?:ly)?\b|\bat present\b", plain) and texts:
                # A claim about the latest visit cites the latest encounter (on the latest date, the one holding it).
                last_date = dates.get(max(texts))
                latest = [n for n in texts if dates.get(n) == last_date]
                keep, cite = [], [max([n for n in latest if holds(n, words)] or [max(texts)])]
            elif undated and not re.findall(r"\d+/\d+", body) and not any(w for w in words if w not in _lab_words):
                # A result from an undated / unlinked report: no encounter holds it.
                keep, cite = [], []
            elif flag and (_flag_score(body, flag) >= 0.65 or not words or flag.get("kind") == "data"
                           or any(holds(n, words) for n in _encounter_numbers(flag.get("evidence")))):
                # A point that clearly states a rule flag cites that flag's encounters (the rule computed them); a weaker
                # match only those of them that also hold what the point names.
                flag_encs = set(_encounter_numbers(flag.get("evidence")))
                # A record check (kind data: a document mismatch, an impossible reading) is where the rule found it.
                strong = _flag_score(body, flag) >= 0.65 or flag.get("kind") == "data"
                allowed = flag_encs if strong else {n for n in flag_encs if not words or holds(n, words)}
                keep = [n for n in written if n in allowed]
                cite = sorted(allowed)[:limit] if not keep else []
            else:
                # A value the point states (7.6, 150/93, 11500) pins the fact to the encounters that hold that value.
                # Counts and durations ('20 visits over 135 days') are not values an encounter holds.
                values = [m.group(0) for m in re.finditer(r"\d(?:\.\d)?-\d(?:\.\d)?-\d(?:\.\d)?|\d+/\d+|\d+\.\d+|(?<![#\d])\d{3,}(?![\d-])", body)
                          if not re.fullmatch(r"(?:19|20)\d\d", m.group(0))
                          and not re.match(r"\s*(?:days?|weeks?|months?|years?|visits?|encounters?|clinical)\b", body[m.end():])]
                if re.search(r"(?i)\bno\s+dose|dose\s+not\s+(?:stated|specified|written)|without\s+(?:a\s+)?dose", body):
                    values.append("0-0-0")
                holders = sorted(n for n in texts if any(re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", texts[n]) for v in values))
                holders = [n for n in holders if not words or holds(n, words)] or holders
                if holders:
                    keep = [n for n in written if n in holders]
                    cite = [] if keep else holders[-limit:]
                else:
                    # A condition the point names is held at its own encounters, however the record wrote it ('T2dm').
                    point_words = set(re.findall(r"[a-z]{3,}", body.casefold()))
                    named = {n for c in evidence["conditions"]
                             if any(_label_in(label, point_words) for label in [c["name"], *c.get("recorded_as", [])] if label)
                             for n in _encounter_numbers(c.get("encounters"))}
                    keep = [n for n in written if holds(n, words) or n in named]
                    cite = [] if keep else best_holders(words)[-limit:]
            keep, cite = same_day_only_holders(keep, words), same_day_only_holders(cite, words)
            # A list of values (readings, results) held at different encounters: each value cites its own.
            listed = [v for v in dict.fromkeys(re.findall(r"\b\d{2,3}/\d{2,3}\b|\b\d+\.\d+\b", body))]
            own = {v: sorted(n for n in texts if re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", texts[n])) for v in listed}
            if len(listed) >= 2 and all(own.values()) and len({tuple(e) for e in own.values()}) > 1:
                body = re.sub(r"\s*\((?:#\d+(?:,\s*)?)+\)\s*$", "", body)  # the list's own citations replace it
                for v, encs in own.items():
                    tag = f" ({', '.join(f'#{n}' for n in encs[-2:])})"
                    m_ = re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])(?:\s*\([^()#]*\d{{4}}-\d{{2}}-\d{{2}}[^()#]*\))?(?!\s*\(#)", body)
                    if m_:
                        body = body[:m_.end()] + tag + body[m_.end():]
                keep, cite = [], []
                written = [n for n in written if n not in {x for e in own.values() for x in e}]
            metrics = evidence.setdefault("_metrics", {})
            removed, added = sorted(set(written) - set(keep)), sorted(set(cite) - set(written))
            if written and (removed or added):
                metrics["citations_corrected"] = metrics.get("citations_corrected", 0) + 1
                entry = {"point": body[:120], "removed": [f"#{n}" for n in removed], "added": [f"#{n}" for n in added]}
                metrics.setdefault("citation_log", []).append(entry)
                logger.info("citation corrected: %s", entry)
            # Remove citations the encounter does not support; a "(#a, #b)" group left empty disappears.
            for n in set(written) - set(keep):
                # The citation goes with its own words ('at visit #3', 'on #2'), never leaving 'noted at'.
                body = re.sub(rf"(?i)(?:\b(?:at|on|in|from)\s+)?(?:\b(?:visit|encounter)\s+)?#\s*{n}(?!\d)\s*,?\s*", "", body)
            body = re.sub(r"\(\s*,?\s*\)", "", body).replace("( ", "(").replace(", )", ")").rstrip(" ,")
            if cite:
                group = re.search(r"\s*\((#\d+(?:,\s*#\d+)*)\)$", body)
                if group:  # one citation group at the end, not two
                    cite = sorted(set(cite) | {int(n) for n in re.findall(r"\d+", group.group(1))})
                    body = body[:group.start()]
                body += f" ({', '.join(f'#{n}' for n in cite)})"
            if undated and "undated" not in body.casefold():
                body += " (undated report)"
            points.append((_compress_ranges(body) if had_range else body) + tail)
        text = (prefix + "; ".join(points)).replace(".; ", "; ")
        text = re.sub(r"(\w)\(s\)", r"\1s", text).replace(" incl. ", " including ").replace("at/above", "at or above")
        out.append(text)
    return out


def complete_flag_points(lines, evidence):
    """Add to a point the facts its flag carries but the point leaves out: how many days, whether the medicine is on
    the latest prescription, and the BP readings recorded after BP medicine stopped."""
    flags = evidence["rule_flags"]
    out = []
    for line in lines:
        prefix = re.match(r"^(\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*[^*]+\*\*:?\s*)?)", line).group(0)
        points = []
        for point in re.split(r";\s+(?![^()]*\))", line[len(prefix):]):
            if not point.strip():
                continue
            end = re.search(r"[.\s]*$", point).start()
            body, tail = point[:end], point[end:]
            f = _point_flag(body, flags)
            ev = str((f or {}).get("evidence") or "")
            low = body.casefold()
            days = re.match(r"On (\d+) day\(s\)", ev)
            if days and int(days.group(1)) > 1 and not re.search(r"\b\d+\s+days?\b", low):
                body += f" on {days.group(1)} days"
            status = re.search(r"(not all on the latest prescription \([^)]*\)|not on the latest prescription \(last #\d+, [\d-]+\)|on the latest prescription \(#\d+\))", ev)
            if status and "latest prescription" not in low:
                body += f" ({status.group(1)})"
            since = re.search(r"BP since then: ([^·]+)|no BP reading recorded since", ev)
            if since and not re.search(r"\b\d{2,3}/\d{2,3}\b", body) and "no bp reading" not in low:
                body += f" (BP since then: {since.group(1).strip()})" if since.group(1) else " (no BP reading recorded since)"
            points.append(body + tail)
        out.append(prefix + "; ".join(points))
    return out


_RECORD_ISSUE = re.compile(r"(?i)impossible|implausible|entry error|unit error|°\s*c entered|entered as °|misread|\bocr\b|"
                           r"differs between document|document\s*(?:and|vs\.?)\s*record|discrepanc|data quality|"
                           r"check (?:the )?source|not physiologically|measuring error|recording error|\bverif|"
                           r"\bno\b[^;]{0,40}\bin the record\b|\bwithout\b[^;]{0,40}\b(?:test|testing|haemoglobin|hemoglobin|"
                           r"blood count|hba1c|glucose|b12|liver|lft)\b")


_TEMP_VALUE = re.compile(r"(?i)(\d{2,3}(?:\.\d+)?)\s*°\s*([FC])\b|\btemp\w*\D{0,15}?(\d{2,3}(?:\.\d+)?)")


def _implausible_temperature(point):
    """True when a point quotes a temperature outside the plausible outpatient range (clinical_thresholds.json)."""
    for m in _TEMP_VALUE.finditer(point):
        value = f"{m.group(1)} °{m.group(2).upper()}" if m.group(1) else m.group(3)
        if vital_implausible("Temperature", value):
            return True
    return False


_VITAL_MENTIONS = [
    ("BP", re.compile(r"(?i)\b(?:bp|blood\s+pressure)\D{0,12}?(\d{1,3}\s*/\s*\d{1,3})")),
    ("SpO2", re.compile(r"(?i)\bspo\S{0,2}\D{0,8}?(\d{1,3}(?:\.\d+)?)\s*%?")),
    ("Pulse", re.compile(r"(?i)(?:\bpulse\D{0,8}?(\d{1,3})|(\d{1,3})\s*bpm\b)")),
    ("Weight", re.compile(r"(?i)(\d{1,4}(?:\.\d+)?)\s*kg\b")),
    ("Height", re.compile(r"(?i)(\d{1,4}(?:\.\d+)?)\s*cm\b")),
]


def _implausible_vital(point):
    """A point quoting any vital outside its plausibility range (clinical_thresholds.json)."""
    if _implausible_temperature(point):
        return True
    for label, pattern in _VITAL_MENTIONS:
        for m in pattern.finditer(point):
            value = next(g for g in m.groups() if g)
            if vital_implausible(label, value + (" mmHg" if label == "BP" else "")):
                return True
    return False


def route_record_issues(lines):
    """Key findings hold clinical findings only; record / entry errors (impossible readings, unit errors, document
    mismatches) are moved to Gaps to check."""
    def theme(line):
        m = re.match(r"^\s*-\s*\*\*([^*:]+):?\*\*", line)
        return m.group(1).strip().casefold() if m else None
    moved, out = [], []
    for line in lines:
        if theme(line) == "key findings":
            prefix = re.match(r"^(\s*-\s*\*\*[^*]+\*\*:?\s*)", line).group(0)
            points = re.split(r";\s+(?![^()]*\))", line[len(prefix):])
            issue = lambda x: _RECORD_ISSUE.search(x) or _implausible_vital(x)
            keep = [x for x in points if x.strip() and not issue(x)]
            moved += [x.strip().rstrip(".") for x in points if x.strip() and issue(x)]
            if keep:
                out.append(prefix + "; ".join(keep))
        else:
            out.append(line)
    if moved:
        at = next((i for i, l in enumerate(out) if theme(l) == "gaps to check"), None)
        if at is None:
            out.append(_bullet("Gaps to check", moved))
        else:
            out[at] = out[at].rstrip().rstrip(".") + "; " + "; ".join(moved) + "."
    return out


def dedupe_gaps(lines, evidence):
    """Gaps to check names each issue once: points that state the same flag, or share most of their words, are one
    point (the fuller one is kept)."""
    out = []
    for line in lines:
        theme = re.match(r"^(\s*-\s*\*\*[^*]+\*\*:?\s*)", line)
        if not theme:
            out.append(line)
            continue
        # Gaps: one point per flag or per issue. Other themes: only near-identical wording is merged, so two different
        # results that share a flag ("HbA1c 7.6 %", "blood sugar 156.92") both stay.
        gaps = bool(re.match(r"^\s*-\s*\*\*Gaps to check", line))
        prefix = theme.group(0)
        kept = []
        for point in [x.strip().rstrip(".") for x in re.split(r";\s+(?![^()]*\))", line[len(prefix):]) if x.strip()]:
            flag = _point_flag(point, evidence["rule_flags"])
            words = set(re.findall(r"[a-z]{4,}", point.casefold()))
            twin = next((k for k in kept if (gaps and flag and _point_flag(k, evidence["rule_flags"]) is flag) or (
                words and len(words & set(re.findall(r"[a-z]{4,}", k.casefold()))) / len(words) >= 0.6)), None)
            if twin is None:
                kept.append(point)
            elif len(point) > len(twin):
                kept[kept.index(twin)] = point
        out.append(prefix + "; ".join(kept) + ("." if gaps or line.rstrip().endswith(".") else ""))
    return out


def use_record_terms(lines, evidence):
    """Name each condition in the record's own words: the registry group ('GERD / acidity', 'GERD') becomes what the
    record wrote ('Gastritis')."""
    out = []
    for line in lines:
        for c in evidence["conditions"]:
            term = _short_label(c)
            if term == c["name"] or not c.get("recorded_as"):
                continue
            names = [c["name"]] + [seg.strip() for seg in c["name"].split(" / ")[:1]
                                   if len(seg.strip()) >= 3 and seg.strip().casefold() not in term.casefold()]
            for name in names:
                line = re.sub(rf"(?i)(?<![\w/]){re.escape(name)}(?![\w])", term, line)
        out.append(line)
    return out


def render_evidence(evidence, insights=()):
    """The visits the brief's claims rest on: every '#n' it cites, plus the abnormal results' visits; the
    latest two only when nothing is cited."""
    cited, cited_for = set(), {}
    for line in insights:
        theme = re.match(r"^\s*-\s*\*\*([^*:]+)", line)
        for n in re.findall(r"#(\d+)", _expand_ranges(line)):
            cited.add(int(n))
            if theme:
                cited_for.setdefault(int(n), []).append(theme.group(1).strip())
    for g in evidence["abnormal_labs"]:
        cited.update(_encounter_numbers(g["encounters"]))
    rows = {int(r["enc"][1:]): r for r in evidence["timeline"]}
    if not cited & set(rows):
        cited.update(sorted(rows)[-2:])
    lines = []
    for n in sorted(cited & set(rows))[:12]:
        r = rows.get(n)
        if not r:
            continue
        parts = [r["complaint"] and f"complaint '{r['complaint']}'", r["diagnosis"] and f"diagnosis '{r['diagnosis']}'",
                 r["documents"] and f"{r['documents']} document(s)"]
        themes = list(dict.fromkeys(cited_for.get(n, [])))
        parts.append(themes and f"cited for {', '.join(themes)}")
        lines.append(f"- {r['enc']} {r['date'] or 'date not recorded'}: " + "; ".join(p for p in parts if p) + ".")
    return lines or ["- No visit source references are available."]


def _flag_line(f):
    text = f"- **{f['title']}:** {f['evidence']}"
    return text + (f". {f['significance']}" if f.get("significance") else ".")


def record_overview(evidence):
    """Who, over what period, and each condition by how the record supports it; always from the record."""
    p = evidence["patient"]
    conditions = evidence["conditions"]
    by_status = {}
    for c in conditions:
        notes = [f"{c['visits']} visits" if c["visits"] > 1 else "",
                 "long-term" if c.get("long_term") and c["status"] not in _KNOWN else ""]
        notes = ", ".join(n for n in notes if n)
        by_status.setdefault(c["status"], []).append(condition_label(c) + (f" ({notes})" if notes else ""))
    who = "" if p["age_sex"] == "not recorded" else f"{p['age_sex']}; "
    span = f" from {p['first_visit']} to {p['latest_visit']}" if p["first_visit"] and p["first_visit"] != p["latest_visit"] \
        else (f" on {p['latest_visit']}" if p["latest_visit"] else "")
    days = p.get("clinical_days") or p["encounters"]
    overall = [f"{who}{p['encounters']} visit(s)" + (f" on {days} clinical days" if days < p["encounters"] else "")
               + f"{span}."]
    names = {"confirmed": "Confirmed diagnoses", "working diagnosis": "Working (provisional) diagnoses",
             "known diagnosis": "Known diagnoses (K/C/O or long-term list)",
             "known history": "Known history", "suspected": "Suspected (marked uncertain)",
             "document only": "Mentioned only in documents"}
    for status in STATUS_ORDER:
        if by_status.get(status):
            overall.append(f"{names[status]}: {_join(by_status[status][:8])}.")
    if not conditions:
        overall.append("No diagnosis is recorded.")
    if evidence["possible_conditions"]:
        overall.append("Possible but not diagnosed: " + _join(pc["condition"] for pc in evidence["possible_conditions"]) + ".")
    return " ".join(overall)


_STATUS_CLAIM = re.compile(r"(?i)\b(?:confirmed|working|provisional|established)\s+diagnos[ie]s\s+(?:of\s+|is\s+|was\s+)?"
                           r"([a-z][a-z0-9 '\-/]{2,40})")


_NEW_CLAIM = re.compile(r"(?i)\b(?:newly\s+(?:recorded|diagnosed|detected|documented)|new(?:ly)?[\s-]+onset|new\s+diagnosis(?:\s+of)?)\b")


_FINE = re.compile(r"(?i)\b(?:at|within|on|below)\s+(?:the\s+)?(?:usual\s+)?(?:target|goal|normal(?:\s+range)?|range)\b|"
                   r"\b(?:well[\s-]+)?controlled\b|\bnormal\b|\bimproved\b|\bresolved\b")
_NOT_FINE_FLAG = re.compile(r"(?i)at/above|at or above|above|out of range|raised|not consistently|uncontrolled|"
                            r"high|low|below 95|missed|not settling")


def replace_contradicted_points(lines, evidence):
    """A point that calls something fine (at target, controlled, normal) while the rule flag it states says otherwise
    is replaced by the flag, rewritten (the rule decides; the model only rewords)."""
    flags = [f for f in evidence["rule_flags"] if f.get("level") in ("high", "verify", "medium") and _NOT_FINE_FLAG.search(f["title"])]
    out = []
    for line in lines:
        prefix = re.match(r"^(\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*[^*]+\*\*:?\s*)?)", line).group(0)
        points = []
        for pt in [x for x in re.split(r";\s+(?![^()]*\))", line[len(prefix):]) if x.strip()]:
            f = _point_flag(pt, flags) if _FINE.search(re.sub(r"(?i)\bnot\s+\w+\s+", "", pt)) and not re.search(
                r"(?i)\bnot\s+(?:at|within|controlled|normal)", pt) else None
            if f:
                evidence.setdefault("_metrics", {})["contradictions_replaced"] = \
                    evidence.get("_metrics", {}).get("contradictions_replaced", 0) + 1
                pt = _rewrite_flag(f)
            points.append(pt.strip())
        out.append(prefix + "; ".join(points))
    return out


_GENERIC_CONDITION_WORDS = {"type", "mellitus", "disease", "disorder", "condition", "chronic", "acute", "pain", "unspecified"}


def fix_new_condition_claims(lines, evidence):
    """'Newly recorded diabetes' for a known (K/C/O, long-term) condition, or one recorded before the latest visit, is
    rewritten to what the record shows."""
    out = []
    for line in lines:
        for m in reversed(list(_NEW_CLAIM.finditer(line))):
            window = line[max(0, m.start() - 60):m.end() + 60]
            words = set(re.findall(r"[a-z]{3,}", window.casefold()))
            significant = lambda l: {w for w in re.findall(r"[a-z]{4,}", l.casefold())} - _GENERIC_CONDITION_WORDS
            named = [c for c in evidence["conditions"]
                     if any(significant(l) & words for l in [c["name"], *c.get("recorded_as", [])] if l)]
            latest = evidence["patient"]["latest_visit"]
            if named and all(c["status"] in _KNOWN or c.get("long_term") or (c["first_seen"] and c["first_seen"] < latest)
                             for c in named):
                c = named[0]
                what = "known (K/C/O)" if c["status"] in _KNOWN or c.get("long_term") else f"recorded since {c['first_seen']}"
                line = line[:m.start()] + what + line[m.end():]
        out.append(line)
    return out


def drop_unsupported_status_claims(lines, evidence):
    """Remove sentences that call something a confirmed / working diagnosis when the record has no such condition."""
    known = " ".join(f"{c['name']} {' '.join(c['recorded_as'])}" for c in evidence["conditions"]).casefold()
    known_words = set(re.findall(r"[a-z]{3,}", known))

    def supported(sentence):
        for match in _STATUS_CLAIM.finditer(sentence):
            if re.search(r"(?i)\b(?:no|not|without)\b[^.]{0,12}$", sentence[:match.start()]):
                continue  # "no confirmed diagnosis" claims nothing
            words =[w for w in re.findall(r"[a-z]{3,}", match.group(1).casefold()) if w not in {"and", "the", "with"}]
            if words and not any(w in known_words or any(SequenceMatcher(None, w, k).ratio() >= 0.8 for k in known_words)
                                 for w in words[:2]):
                return False
        return True

    kept = []
    for line in lines:
        prefix = re.match(r"^(\s*-\s+)?", line).group(0)
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", line[len(prefix):]) if s and supported(s)]
        if sentences:
            kept.append(prefix + " ".join(sentences))
    return kept


_ACTIVE_CLAIM = re.compile(r"(?i)\b(?:active|current(?:ly)?|ongoing|still\s+(?:recorded|present)|at\s+(?:the\s+)?latest\s+visit)\b")
_NOT_ACTIVE = re.compile(r"(?i)\b(?:not|no\s+longer|resolved|earlier|previous(?:ly)?|past|last\s+(?:noted|recorded|seen)|since)\b")
_LABEL_STOP = {"and", "the", "with", "of", "or", "type", "unspecified"}


def _is_current(condition, latest, evidence=None):
    """The rule-based brief's 'Active': recorded at the latest visit, known history, on the long-term list, or still
    treated: a medicine for it is on the latest prescription (a latest visit that writes no diagnosis but continues the
    BP medicines has not stopped the hypertension)."""
    if condition["last_seen"] == latest or condition["status"] in _KNOWN or condition.get("long_term"):
        return True
    label = condition_label(condition)
    return bool(evidence) and any(m.get("status") == "current" and label in (m.get("for") or [])
                                  for m in evidence.get("medicines") or [])


def _label_in(label, words):
    """True when every significant word of a condition label appears (allowing spelling variants) in the point."""
    need = [w for w in re.findall(r"[a-z]{3,}", label.casefold()) if w not in _LABEL_STOP]
    return bool(need) and all(w in words or any(SequenceMatcher(None, w, x).ratio() >= 0.8 for x in words) for w in need)


def drop_unsupported_active_claims(lines, evidence):
    """Remove points that call a condition active / current / ongoing when the record last has it before the latest
    visit (and it is neither known history nor on the long-term list). E.g. 'Active: low back pain' when low back
    pain was last recorded at #22 and the latest visit, #23, was for hand pain."""
    latest = evidence["patient"]["latest_visit"]
    conditions = evidence["conditions"]

    def supported(point):
        if not _ACTIVE_CLAIM.search(point) or _NOT_ACTIVE.search(point):
            return True
        words = set(re.findall(r"[a-z]{3,}", point.casefold()))
        named = [c for c in conditions
                 if any(_label_in(label, words) for label in [c["name"], *c.get("recorded_as", [])] if label)]
        return all(_is_current(c, latest, evidence) for c in named)

    def rewrite(point):
        """Keep a recorded condition: the 'active / current' wording goes and the last visit is stated."""
        words = set(re.findall(r"[a-z]{3,}", point.casefold()))
        named = [c for c in conditions if any(_label_in(label, words) for label in [c["name"], *c.get("recorded_as", [])] if label)]
        stale = [c for c in named if not _is_current(c, latest, evidence)]
        point = _ACTIVE_CLAIM.sub("", point)
        point = re.sub(r"\s{2,}", " ", re.sub(r"^\s*[:,-]\s*", "", point)).strip()
        return point + (" (" + ", ".join(f"{_short_label(c)} last recorded {_last_enc(c)}" for c in stale) + ")" if stale else "")

    kept = []
    for line in lines:
        prefix = re.match(r"^(\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*[^*]+\*\*:?\s*)?)", line).group(0)
        points = re.split(r";\s+(?![^()]*\))", line[len(prefix):])
        points = [p if supported(p) else rewrite(p) for p in points if p.strip()]
        points = [p for p in points if p]
        if points:
            kept.append(prefix + "; ".join(points))
    return kept


_LAST_AT = re.compile(r"(?i)\blast\s+(?:recorded\s+|seen\s+|noted\s+)?(?:at\s+|on\s+)?#(\d+)")
_MISSED_CLAIM = re.compile(r"(?i)\bmissed\b[^;]*\b(?:follow[\s-]*ups?|calls?|reviews?|appointments?|visits?)\b|"
                           r"\b(?:follow[\s-]*ups?|calls?|reviews?|appointments?)\b[^;]*\bmissed\b")


def fix_record_contradictions(lines, evidence):
    """Correct or drop what the record contradicts outright:
    - '<condition> (…, last #n)' where the condition was last recorded at another encounter: n is set to the real one
      (the condition named nearest before 'last #n');
    - a missed follow-up / call / review when the record has no missed follow-up (no rule flag, no follow-up gap)."""
    conditions = evidence["conditions"]
    missed_on_record = any(re.search(r"(?i)missed follow-up", f.get("title") or "") for f in evidence["rule_flags"]) \
        or any(re.search(r"(?i)\bmissed\b", str(g)) for g in evidence.get("followup_gaps") or [])

    def last_of(text):
        """The condition named last in text (by position of its label)."""
        low, best = text.casefold(), (-1, None)
        for c in conditions:
            for label in [c["name"], *c.get("recorded_as", [])]:
                pos = low.rfind(label.casefold()) if label else -1
                if pos > best[0]:
                    best = (pos, c)
        return best[1]

    def fix(point):
        if _MISSED_CLAIM.search(point) and not missed_on_record:
            evidence.setdefault("_metrics", {})["contradicted_points_removed"] = \
                evidence.get("_metrics", {}).get("contradicted_points_removed", 0) + 1
            return ""
        out, start = [], 0
        for m in _LAST_AT.finditer(point):
            c = last_of(point[start:m.start()]) or last_of(point[:m.start()])
            real = _last_enc(c) if c else ""
            out.append(point[start:m.start()])
            out.append(m.group(0).replace(f"#{m.group(1)}", real) if real and real != f"#{m.group(1)}" else m.group(0))
            start = m.end()
        return "".join(out) + point[start:]

    kept = []
    for line in lines:
        prefix = re.match(r"^(\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*[^*]+\*\*:?\s*)?)", line).group(0)
        points = [fix(p) for p in re.split(r";\s+(?![^()]*\))", line[len(prefix):]) if p.strip()]
        points = [p for p in points if p]
        if points:
            kept.append(prefix + "; ".join(points))
    return kept


_DESPITE = re.compile(r"(?i)\bdespite\b")
_CHANGE = re.compile(r"(?i)\b(stopped|discontinued|withdrawn|switched|changed|started|added|begun|reduced|increased)\b"
                     r"[^;]*?\b(because\s+of|because|due\s+to|owing\s+to|following|after)\b\s*([^;]*)")


def _first_number(text):
    match = re.search(r"\d+(?:\.\d+)?", str(text or ""))
    return match.group(0) if match else None


def drop_unsupported_timing_claims(lines, evidence):
    """Check the ordering words a model adds against the record's dates.

    - "despite X" after a lab result: kept only when X was first prescribed before the result's date. Otherwise the
      clause is replaced by what the dates show (X started on or after the result; result undated).
    - "X stopped / started / changed after Y": dropped unless the dates show Y happened while on X (stopped) or before
      X began (started). "because / due to / owing to Y": dropped; the record does not state why a medicine changed.
    """
    meds = [m for m in evidence["medicines"] if m.get("first")]
    labs = evidence["abnormal_labs"]
    timeline = evidence["timeline"]

    def meds_in(text):
        found = []
        for m in meds:
            word = re.sub(r"[^a-z]", "", m["name"].split()[0].casefold())[:5]
            if len(word) >= 4 and re.search(rf"\b{word}", text, re.I):
                found.append(m)
        return found

    def labs_in(text):
        return [g for g in labs if (_first_number(g["value"]) and re.search(rf"(?<![\d.]){re.escape(_first_number(g['value']))}(?![\d])", text))
                or g["test"].casefold() in text.casefold()]

    def event_date(text):
        hit = labs_in(text)
        if hit:
            return min((g["date"] for g in hit if g["date"]), default=None)
        words = {w for w in re.findall(r"[a-z]{4,}", text.casefold())} - {"with", "after", "from", "this", "that"}
        for r in timeline:
            if words & set(re.findall(r"[a-z]{4,}", (r["complaint"] + " " + r["diagnosis"]).casefold())):
                return r["date"]
        return None

    def check(point):
        # Order words first, on the model's own text (the despite note added below also says "started ... after").
        change = _CHANGE.search(point)
        if change:
            verb, link, event = change.group(1).casefold(), change.group(2).casefold(), change.group(3)
            named = meds_in(point[:change.start(2)])
            if not link.startswith(("after", "following")):
                return None  # a stated cause the record does not give
            when = event_date(event)
            for m in named:
                if not when:
                    return None
                if verb in ("stopped", "discontinued", "withdrawn") and not (m["first"] <= when <= m["last"]):
                    return None
                if verb in ("started", "added", "begun") and (m["first"] < when or m.get("first_recorded")):
                    return None
        improved = re.search(r"(?i)\bimprov\w*\b", point)
        if improved:
            via = re.search(r"(?i)\s*\(?\b(?:on|with|after(?:\s+starting)?)\s+([A-Z][\w-]+(?:\s*(?:\+|,|and)\s*[A-Z][\w-]+)*)\)?",
                            point[improved.end():])
            named = meds_in(via.group(1)) if via else []
            if named:
                dates = re.findall(r"\d{4}-\d{2}-\d{2}", point)
                bp_flag = next((f for f in evidence["rule_flags"] if str(f["title"]).startswith("BP ")), None)
                dates += re.findall(r"\d{4}-\d{2}-\d{2}", str((bp_flag or {}).get("evidence") or ""))
                earlier = min(dates) if dates else evidence["patient"].get("first_visit")
                if earlier and any(m["first"] >= earlier for m in named):
                    start = improved.end() + via.start()
                    point = point[:start] + point[improved.end() + via.end():]
        despite = _DESPITE.search(point)
        if despite:
            named = meds_in(point[despite.end():])
            results = labs_in(point[:despite.start()]) or labs_in(point)
            if named and results:
                dates = [g["date"] for g in results]
                firsts = sorted(m["first"] for m in named)
                if not all(dates) or any(d <= firsts[-1] for d in dates):
                    names = " + ".join(m["name"] for m in named)
                    when = start_phrase(min(named, key=lambda m: m["first"]))
                    note = (f" ({names} {when}, on or after this result)" if all(dates)
                            else f" (result undated; {names} also recorded)")
                    point = point[:despite.start()].rstrip(" ,;") + note
        return point

    kept = []
    for line in lines:
        prefix = re.match(r"^(\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*[^*]+\*\*:?\s*)?)", line).group(0)
        points = [check(p_) for p_ in re.split(r";\s+(?![^()]*\))", line[len(prefix):]) if p_.strip()]
        points = [p_ for p_ in points if p_]
        if points:
            kept.append(prefix + "; ".join(points))
    return kept


_STOP_CLAIM = re.compile(r"(?i)\b(?:stopped|discontinued|withdrawn|ceased|not\s+continued|no\s+longer\s+(?:on|prescribed|given|taken)|"
                         r"dropped|was\s+stopped|has\s+been\s+stopped)\b")


def _stop_supported(m, encounter_dates):
    """A medicine may be called stopped only when its last course had ended by the next visit after it was written."""
    if m.get("at_latest_visit"):
        return False
    later = sorted(d for d in encounter_dates.values() if d and d > (m.get("last") or ""))
    if not later:
        return False
    return not (m.get("course_end") and m["course_end"] >= later[0])


def drop_unsupported_stop_claims(lines, evidence):
    """Remove a point saying a medicine was stopped / not continued when its last course (date + duration) was still
    running at the next visit, or it is current."""
    meds = [(m, _med_word(m["name"])) for m in evidence["medicines"] if _med_word(m["name"])]
    dates = evidence.get("encounter_dates") or {}

    def supported(point):
        if not _STOP_CLAIM.search(point):
            return True
        named = [m for m, word in meds if re.search(rf"\b{re.escape(word[:5])}", point, re.I)]
        return all(_stop_supported(m, dates) for m in named)

    out = []
    for line in lines:
        prefix = re.match(r"^(\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*[^*]+\*\*:?\s*)?)", line).group(0)
        points = [p for p in re.split(r";\s+(?![^()]*\))", line[len(prefix):]) if p.strip() and supported(p)]
        if points:
            out.append(prefix + "; ".join(points))
    return out


_CAUSE = re.compile(r"(?i)\s*[,(]?\s*\b(?:(?:possibly|probably|likely|perhaps)\s+)?(?:due\s+to|secondary\s+to|caused\s+by|"
                    r"induced\s+by|because\s+of|attributable\s+to|related\s+to|from|an?\s+effect\s+of|effect\s+of|"
                    r"(?:may|could|might)\s+(?:be|reflect|relate\s+to)|consistent\s+with|after|"
                    r"(?:may|could|might)\s+have\s+(?:worsened|raised|caused|contributed\s+to|led\s+to|triggered)|"
                    r"worsened\s+by|contributed\s+to\s+by)\b[^;.)]*\)?")
_SYMPTOM_LINK_TITLES = {link["title"] for link in REGISTRY.get("symptom_med_links") or []}
_CAUSE_ADJ = re.compile(r"(?i)\b([a-z]+)-(?:induced|related|associated)\b")
_CAUSE_CLASS = {"steroid": "steroid", "corticosteroid": "steroid", "prednisolone": "steroid", "nsaid": "nsaid",
                "painkiller": "nsaid", "azole": "azole", "antifungal": "azole", "ppi": "ppi"}


def drop_implausible_cause_claims(lines, evidence):
    """Remove a clause naming a medicine as the cause of a lab result unless that medicine was taken within the
    plausible window before the test (abnormal_labs[].medicines_in_effect_window)."""
    meds = [(m, _med_word(m["name"])) for m in evidence["medicines"] if _med_word(m["name"])]

    def named(text):
        found = [word for _, word in meds if re.search(rf"\b{re.escape(word[:5])}", text, re.I)]
        found += [f"class:{cls}" for w, cls in _CAUSE_CLASS.items() if re.search(rf"(?i)\b{w}s?\b", text)]
        return found

    def clean_subject(point):
        """'Omnacortil may have worsened glucose': the medicine is the subject, before the causal verb."""
        m = re.search(r"(?i)\b(?:may|could|might)\s+have\s+(?:worsened|raised|caused|contributed\s+to|led\s+to|triggered)\b[^;]*", point)
        if m and named(point[:m.start()]) and not allowed(point, named(point[:m.start()])):
            return point[:m.start()].rstrip(" ,") + " recorded"
        return point

    def allowed(point, who):
        results = [g for g in evidence["abnormal_labs"] if g["test"].casefold() in point.casefold()
                   or (_first_number(g["value"]) and re.search(rf"(?<![\d.]){re.escape(_first_number(g['value']))}(?!\d)", point))]
        if not results:
            # Not a lab: a medicine may be called the cause of a symptom only when a symptom-medicine rule flag names it.
            linked = " ".join(str(f.get("evidence") or "") for f in evidence.get("rule_flags") or []
                              if f.get("title") in _SYMPTOM_LINK_TITLES).casefold()
            return all(not w.startswith("class:") and w[:5] in linked for w in who)
        window = " ".join(w for g in results for w in g.get("medicines_in_effect_window") or []).casefold()
        effect_words = {"steroid": "steroid", "nsaid": "nsaid", "azole": "azole", "ppi": "ppi"}
        return all((w[6:] in window) if w.startswith("class:") and w[6:] in effect_words else w[:5] in window for w in who)

    def clean(point):
        point = clean_subject(point)
        for m in reversed(list(_CAUSE.finditer(point))):
            who = named(m.group(0))
            if who and not allowed(point, who):
                point = point[:m.start()] + point[m.end():]
        for m in reversed(list(_CAUSE_ADJ.finditer(point))):
            who = named(m.group(1))
            if who and not allowed(point, who):
                point = point[:m.start()] + point[m.end():].lstrip()
        return re.sub(r"\s+([;,.])", r"\1", point).strip()

    out = []
    for line in lines:
        prefix = re.match(r"^(\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*[^*]+\*\*:?\s*)?)", line).group(0)
        points = [clean(p) for p in re.split(r";\s+(?![^()]*\))", line[len(prefix):]) if p.strip()]
        points = [p for p in points if p]
        if points:
            out.append(prefix + "; ".join(points))
    return out


# Words that name a medicine class, from the registry plus everyday synonyms the model may use.
_CLASS_WORDS = {w for e in REGISTRY["medicine_classes"] for w in re.findall(r"[a-z][a-z\-]{2,}", e["class"].casefold())
                } - {"and", "pain", "gel", "other", "incl"} | {"steroid", "statin", "diuretic", "blocker", "painkiller", "antibacterial",
                                              "ppi", "ace", "arb", "ccb", "beta-blocker", "anticoagulant", "insulin"}
_CONTROL_CLAIM = re.compile(r"(?i)\b(?:above|below|out\s+of|outside)\s+(?:the\s+)?(?:reference\s+)?(?:range|target)\b|"
                            r"\b(?:uncontrolled|poorly\s+controlled|not\s+controlled|inadequately\s+controlled)\b")


_LEVEL_TAG = re.compile(r"(?i)(?:^|(?<=;\s))\s*(?:high|medium|info)\s*(?:risk|level|priority)?\s*:\s*|"
                        r"\s*\((?:high|medium|info|low)(?:\s+(?:risk|level|priority))?\)")
# Points that say only what is absent and would fit any patient (the lab section already says so).
_FILLER_POINT = re.compile(
    r"(?i)\s*(?:no\s+(?:abnormal\s+)?lab(?:oratory)?\s+(?:results?|tests?|values?)(?:\s+(?:available|recorded|in\s+the\s+record))?|"
    r"no\s+labs?|no\s+(?:other\s+)?abnormal\s+(?:vitals?|findings|results?|labs?)|no\s+(?:other\s+|known\s+)?long-term\s+conditions?|"
    r"no\s+(?:other\s+)?(?:conditions|findings|complications)(?:\s+recorded)?|no\s+follow-?up(?:\s+visits?)?|"
    r"no\s+follow-?up\s+gaps?(?:\s+or\s+record\s+checks?)?)\s*\.?\s*")


def drop_filler_points(lines, evidence=None):
    """Remove internal priority labels ("High:", "(medium risk)") and points that only state an absence,
    including a follow-up gap ("no follow-up for boils") the evidence does not record."""
    followup_recorded = bool(evidence and (evidence["followup_gaps"] or any(
        f.get("kind") == "followup" and re.search(r"(?i)missed|pending", f["title"]) for f in evidence["rule_flags"])))
    flagged = json.dumps((evidence or {}).get("rule_flags") or [], ensure_ascii=False)
    hypertensive = any(c.get("canonical") == "Hypertension" for c in (evidence or {}).get("conditions") or [])
    diagnosed = {w for c in (evidence or {}).get("conditions") or []
                 for w in re.findall(r"[a-z]{5,}", " ".join([c["name"], *c["recorded_as"], c.get("meaning") or ""]).casefold())
                 } - {"condition", "disease", "disorder", "chronic", "acute", "infection", "mellitus", "syndrome", "general"}

    def normal_vitals_only(point):
        """A point that only reports normal readings (never asked for), unless BP is a hypertensive's control."""
        readings = [(int(s), int(d)) for s, d in re.findall(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", point)]
        pulses = [int(p) for p in re.findall(r"\b(\d{2,3})\s*bpm\b", point)]
        if not readings and not pulses or hypertensive and readings:
            return False
        normal = all(100 <= s < 140 and 60 <= d < 90 and f"{s}/{d}" not in flagged for s, d in readings)
        return normal and all(50 <= p <= 100 for p in pulses)

    def wrongly_undiagnosed(point):
        """'may indicate anaemia, not diagnosed' when the record does diagnose anaemia."""
        return bool(re.search(r"(?i)not\s+diagnosed", point)) and bool(
            set(re.findall(r"[a-z]{5,}", point.casefold())) & diagnosed)

    out = []
    for line in lines:
        prefix = re.match(r"^(\s*-\s+(?:\*\*[^*]+\*\*:?\s*)?)?", line).group(0)
        points = [_LEVEL_TAG.sub("", p).strip() for p in line[len(prefix):].split(";")]
        points = [p for p in points if p and not _FILLER_POINT.fullmatch(p)
                  and not (evidence is not None and (normal_vitals_only(p) or wrongly_undiagnosed(p)))
                  and not (evidence is not None and not followup_recorded
                           and re.match(r"(?i)no\s+(?:follow-?up|review)\b", p))]
        if points:
            out.append(prefix + "; ".join(points))
    return out


def drop_unsupported_control_claims(lines, evidence):
    """Drop a point that calls something above range / uncontrolled without a value the evidence flags.

    "Diabetes above range: weakness despite metformin" has no result behind it; "BP above 130/80" and
    "HbA1c 8.9 % above range" quote a value that is in the abnormal results or rule flags.
    """
    rows = (evidence.get("guideline_context") or {}).get("rows") or []
    flagged = json.dumps([evidence["abnormal_labs"], evidence["rule_flags"], rows], ensure_ascii=False)
    # A marker the guideline row does not call above target (at target, status unknown, undated) is not
    # "uncontrolled" whatever range the report printed.
    settled = [r["marker"] for r in rows if r.get("marker") and r["status"] != "above target"]

    def supported(point):
        if not _CONTROL_CLAIM.search(point):
            return True
        if any(re.search(rf"(?i)\b{re.escape(m)}\b", point) for m in settled):
            return False
        values = re.findall(r"\d+(?:\.\d+)?(?:/\d+)?", point)
        return any(re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", flagged) for v in values)

    out = []
    for line in lines:
        prefix = re.match(r"^(\s*-\s+(?:\*\*[^*]+\*\*:?\s*)?)?", line).group(0)
        points = [p for p in line[len(prefix):].split("; ") if p.strip() and supported(p)]
        if points:
            out.append(prefix + "; ".join(points))
    return out


_FOR_CLAUSE = re.compile(r"\s*\(?\b(?:prescribed\s+for|given\s+for|to\s+treat|treating|for)\s+([^;,.()]+?)\s*\)?(?=$|[;,.)])",
                         re.I)
_NOT_A_CONDITION = re.compile(r"(?i)^\d|\b(?:days?|weeks?|months?|years?|visits?|encounters?|now|life|long|sos)\b")


def drop_unsupported_medicine_links(lines, evidence):
    """Remove 'for <condition>' from a point that names a medicine the record does not link to that condition.

    The evidence links a medicine to a condition only through the registry (medicines[].for); the model
    must not supply the link itself ("Ezact 90 for high cholesterol"). Points are separated by ';'.
    """
    meds = [(m, _med_word(m["name"])) for m in evidence["medicines"] if _med_word(m["name"])]

    def linked(target, med):
        links = " ".join(med["for"])
        if set(conditions_matching(target)) & set(conditions_matching(links)):
            return True
        words = set(re.findall(r"[a-z]{3,}", target.casefold())) - {"the", "and", "with"}
        return bool(words) and words <= set(re.findall(r"[a-z]{3,}", links.casefold()))

    def clean_point(point):
        named = [m for m, word in meds if re.search(rf"\b{re.escape(word)}", point, re.I)]
        if not named:
            return point
        # A class in brackets right after a medicine ("BPlex (antihypertensive)") must be the record's class for it.
        for m, word in meds:
            name = re.search(rf"\b{re.escape(word)}[^;(),]{{0,20}}?\(", point, re.I)
            if not name:
                continue
            start, depth, end = name.end() - 1, 0, None
            for i in range(start, len(point)):  # the whole bracket, nested ones included
                depth += {"(": 1, ")": -1}.get(point[i], 0)
                if depth == 0:
                    end = i
                    break
            if end is None:
                continue
            claimed = set(re.findall(r"[a-z][a-z\-]{2,}", point[start:end].casefold())) & _CLASS_WORDS
            if claimed and not all(w in m.get("class", "").casefold() for w in claimed):
                point = point[:start].rstrip() + point[end + 1:]
        # "at the latest visit" is only true when every medicine named in the point was prescribed then.
        if any(not m.get("at_latest_visit", True) for m in named):
            point = re.sub(r"(?i)\s*\b(?:at|on|from)\s+(?:the\s+)?(?:latest|last|current)\s+visit\b", "", point)
        for match in reversed(list(_FOR_CLAUSE.finditer(point))):
            target = match.group(1).strip()
            if _NOT_A_CONDITION.search(target) or all(linked(target, m) for m in named):
                continue
            point = point[:match.start()] + point[match.end():]
        return point

    out = []
    for line in lines:
        prefix = re.match(r"^(\s*-\s+(?:\*\*[^*]+\*\*:?\s*)?)?", line).group(0)
        body = "; ".join(clean_point(p) for p in line[len(prefix):].split("; "))
        out.append(prefix + re.sub(r"\s+([;,.])", r"\1", body).strip())
    return out


def _short_ref(g):
    return f" (ref {g['reference']})" if g.get("reference") else ""


def _first_clause(text):
    """'Indicates anaemia; lower values ...' -> 'indicates anaemia' (lower-cased, for use mid-sentence)."""
    clause = re.split(r"[;.]", str(text or ""), maxsplit=1)[0].strip()
    return clause[:1].lower() + clause[1:]


def _result(g):
    repeat = " (reported twice)" if len(re.findall(r"#\d+", g["encounters"])) == 2 or "–" in g["encounters"] else ""
    return f"{g['test']} {g['value']}{_short_ref(g)}{repeat}"


def _expand_ranges(text):
    """'(#1–3)' / '#2-#4' -> '#1, #2, #3': an encounter range written as its encounters."""
    def expand(m):
        a, b = int(m.group(1)), int(m.group(2))
        return ", ".join(f"#{n}" for n in range(a, b + 1)) if a < b <= a + 60 else m.group(0)
    return re.sub(r"#(\d+)\s*[–-]\s*#?(\d+)(?![\d/])", expand, str(text))


def _compress_ranges(text):
    """'#1, #2, #3' -> '#1–3' (three or more consecutive encounters in one citation list)."""
    def compress(m):
        nums = [int(n) for n in re.findall(r"\d+", m.group(0))]
        if len(nums) < 3 or nums != list(range(nums[0], nums[-1] + 1)):
            return m.group(0)
        return f"#{nums[0]}–{nums[-1]}"
    return re.sub(r"#\d+(?:,\s*#\d+){2,}", compress, str(text))


def _clip(text, words):
    """At most `words` words of record text, cut at the last clause break (';', '·', ',') inside the limit when there
    is one, never with an ellipsis: a cut point must read as complete."""
    # Encounter lists shortened upstream end in "…"; a point never carries it.
    parts = str(text or "").replace(" …", "").replace("…", "").split()
    if len(parts) <= words:
        return " ".join(parts)
    head = " ".join(parts[:words])
    # Cut only at a break outside brackets, so "(#2, #5, #7" is never left open.
    depth, breaks = 0, []
    for i, ch in enumerate(head):
        depth += ch in "([" and 1 or (-1 if ch in ")]" else 0)
        if depth == 0 and (ch in ";," or head[i:i + 2] == " ·"):
            breaks.append(i)
    cut = max((b for b in breaks if b > len(head) // 3), default=None)
    if cut is None:
        # No clause break inside the limit: the whole first clause (up to a little longer), never cut mid-phrase.
        full = " ".join(parts)
        nxt = re.search(r"[;·]|,\s", full[len(head):])
        end = len(head) + nxt.start() if nxt else len(full)
        if len(full[:end].split()) <= words * 2:
            head = full[:end]
    else:
        head = head[:cut]
    if head.count("'") % 2:  # never end inside a quote
        head = head[:head.rfind("'")].rstrip(" ,;·")
        head = re.sub(r"\s+(?:for|to|of|and|with|in|on)$", "", head)
    if head.count("(") > head.count(")"):
        head = head[:head.rfind("(")]
    return head.rstrip(" ;,·")


def _tidy_title(title):
    """A rule-flag title as readable text: no '(3)' counts, no '(s)', no 'incl.' or 'at/above'."""
    title = re.sub(r"\s*\(\d+\)$", "", str(title or ""))
    title = re.sub(r"(\w)\(s\)", r"\1s", title).replace("incl.", "including").replace("at/above", "at or above")
    return title


def _flag_point(f, words=16):
    """'<title> — <first clause of the evidence>' for a rule flag added to the brief."""
    evidence = re.split(r"\s·\s", str(f.get("evidence") or ""))[0]
    label = time_label(f) if f.get("combo") and f.get("time") == "past" else ""  # never 'current' for a stopped combination
    return label + (f"{_tidy_title(f['title'])} — {_clip(evidence, words)}" if evidence else _tidy_title(f["title"]))


def _capped(items, limit):
    """The first `limit` distinct items, then '+N more'."""
    items = [i for i in dict.fromkeys(items) if i]
    return items[:limit] + ([f"+{len(items) - limit} more"] if len(items) > limit else [])


def _bullet(theme, points):
    """One brief theme; its points are separated by '; ' so each renders as its own line."""
    points = [p.rstrip(". ") for p in points if p]
    return f"- **{theme}:** " + "; ".join(points) + "." if points else ""


_THEMES = ("Course", "Conditions", "Key findings", "Treatment", "Gaps to check")
_THEME_OF_KIND = {"medication": "Treatment", "pattern": "Course", "data": "Gaps to check", "followup": "Gaps to check"}
_GENERIC_WORDS = {"symptom", "symptoms", "reading", "readings", "record", "structured", "medicine", "medicines",
                  "visit", "visits", "above", "usual", "latest", "without", "while", "condition", "result", "results",
                  "daily"}


def _anchors(*texts):
    """Words and numbers that show a point was made: distinctive title words, values and durations."""
    text = " ".join(str(t or "") for t in texts)
    words = {w for w in re.findall(r"[a-z][a-z\-]{4,}", text.casefold()) if w not in _GENERIC_WORDS}
    numbers = set(re.findall(r"\d+(?:[-–/]\d+)+|\d{3,}(?:\.\d+)?", text))
    return words, numbers


def _mentioned(brief, words, numbers):
    low = brief.casefold()
    return (any(re.search(rf"\b{re.escape(w)}", low) for w in words)
            or any(re.search(rf"(?<![\d\-/]){re.escape(n)}(?![\d])", brief) for n in numbers))


# Importance within a level: current overdoses, dual NSAIDs and interactions first; calcium / PPI duplicates and record
# quality last. Lower number = more important.
_IMPORTANCE = [
    (0, r"^Daily dose above|^Critical lab|^Two oral NSAIDs|ARB / ACE inhibitor|azole antifungal with a calcium"),
    (1, r"^Oral steroid with|^Repeated oral NSAIDs|^Oral NSAIDs for about|: .+ out of range$|^Creatinine|"
        r"^Visual symptoms|missed follow-up|^Low BP|^Dizziness"),
    (2, r"^BP |^BMI|not settling$|^Recurrent .+ steroids|^No .+ in the record|azole antifungal without|^Possible"),
    (3, r"^Same-class|^Medicine load|^Impossible vital|^Temperature below|^Height varies|^Weight entry error|report\(s\) not linked"),
]
_LEVEL_RANK = {"high": 0, "verify": 1, "medium": 2, "info": 3, "data_quality": 4}


def flag_importance(f):
    """Sort key: level first, then the clinical weight of the rule within the level."""
    rank = next((r for r, pattern in _IMPORTANCE if re.search(pattern, str(f.get("title") or ""))), 2)
    return _LEVEL_RANK.get(f.get("level"), 2), rank


def _clauses(text):
    return [c.strip(" .") for c in re.split(r"\s·\s|;\s", str(text or "")) if c.strip(" .")]


def _rewrite_flag(f):
    """A rule flag as a point in the brief's own style: what was found, with its key values, never the raw
    'title — evidence' paste ('Recurrent eczema with repeated steroids — Eczema')."""
    title, evidence = _tidy_title(f["title"]), str(f.get("evidence") or "")
    clauses = _clauses(evidence)
    title_words = set(re.findall(r"[a-z]{4,}", title.casefold()))
    # Clauses that add something the title does not already say.
    useful = [c for c in clauses if set(re.findall(r"[a-z]{4,}", c.casefold())) - title_words or re.search(r"\d", c)]

    m = re.match(r"(?i)^Recurrent (.+) with repeated steroids$", title)
    if m:
        used = [c for c in useful if re.search(r"(?i)\b(?:oral|topical)\b", c)]
        return f"{m.group(1).capitalize()} keeps returning with repeated steroid use" + (
            f" ({'; '.join(_clip(c, 8) for c in used[:2])})" if used else "")
    m = re.match(r"(?i)^(.+) not settling$", title)
    if m:
        visits = re.search(r"At (\d+) of (\d+) visits", evidence)
        treatments = re.search(r"(\d+) different (?:treatments|medicines)", evidence)
        detail = ", ".join(filter(None, [visits and f"{visits.group(1)} of {visits.group(2)} visits",
                                         treatments and f"{treatments.group(1)} treatments tried"]))
        return f"{m.group(1)} still recorded at the latest visit" + (f" ({detail})" if detail else "")
    m = re.match(r"(?i)^(\d+) missed follow-up visits?( including the latest)?$", title)
    if m:
        return f"{m.group(1)} missed follow-up visit(s){m.group(2) or ''} ({_clip(evidence, 10)})"
    status = re.search(r"(not all on the latest prescription \([^)]*\)|not on the latest prescription \(last #\d+, [\d-]+\)|on the latest prescription \(#\d+\))", evidence)
    status_text = f" ({status.group(1)})" if status else ""
    m = re.match(r"(?i)^oral nsaids for about .+ days in total$", title)
    if m:
        days = re.search(r"over (\d+) prescription day", evidence)
        return title + (f" (over {days.group(1)} prescription days)" if days else "") + status_text
    days = re.match(r"On (\d+) day\(s\): (.+)", clauses[0] if clauses else "")
    if days:
        example = _clip(re.sub(r"\s*·.*$", "", days.group(2)), 12)
        count = int(days.group(1))
        return f"{title} on {count} day{'s' if count > 1 else ''} ({'e.g. ' if count > 1 else ''}{example}){status_text}"
    detail = _clip(useful[0], 12) if useful else ""
    first_word = detail.split()[0] if detail else ""
    # Lower-case an ordinary first word only ("Tingling" -> "tingling"), never a code ("HbA1c") or, in a medication
    # flag, a medicine name ("Diclofenac").
    if (f.get("kind") != "medication" and first_word.isalpha() and first_word[:1].isupper()
            and first_word[1:].islower()):
        detail = detail[0].lower() + detail[1:]  # "Tingling sensation ..." -> "tingling sensation ..."
    return (f"{title}: {detail}" if detail else title) + status_text


def _said(brief, f, evidence):
    """True when the brief already makes this flag's point, in any words. Strongest sign: it repeats one of the
    flag's values or dates ('7.6', '2026-07-10', '180') together with any word of the title. Otherwise most of the
    title's words (two thirds), or half of them plus a medicine or other name from the flag's evidence."""
    low = brief.casefold()
    stop = {"the", "and", "with", "for", "from", "oral", "visit", "visits", "record", "latest", "usual", "including",
            "repeated", "type", "mellitus", "disease"}
    words = {w for w in re.findall(r"[a-z][a-z0-9]{2,}", _tidy_title(f["title"]).casefold()) if w not in stop}
    pattern = lambda w: rf"\b{re.escape(w[:6])}" if len(w) >= 5 else rf"\b{re.escape(w)}\b"
    hits = sum(1 for w in words if re.search(pattern(w), low))
    ratio = hits / len(words) if words else 0
    text = str(f.get("evidence") or "") + " " + str(f["title"])
    values = set(re.findall(r"\d{4}-\d{2}-\d{2}|\d+\.\d+|(?<![\d.#])\d{3,}(?![\d.])", text))
    value_hit = any(re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", brief) for v in values)
    title_low = f["title"].casefold()
    names = {w.casefold() for w in re.findall(r"\b[A-Z][A-Za-z]{3,}", str(f.get("evidence") or ""))
             if w.casefold() not in title_low}
    name_hit = any(n in low for n in names)
    return (value_hit and hits >= 1) or ratio >= 0.65 or (ratio >= 0.5 and (not names or name_hit))


_CONDITION_WORDS = {w for entry in REGISTRY["condition_map"] for w in re.findall(r"[a-z]{4,}", entry["name"].casefold())}


def _flag_score(point, f):
    """How well a single point states this flag (0 = not at all): share of the title's words, plus a value or date."""
    low = point.casefold()
    stop = {"the", "and", "with", "for", "from", "oral", "visit", "visits", "record", "latest", "usual", "including",
            "repeated", "type", "mellitus", "disease"}
    words = {w for w in re.findall(r"[a-z][a-z0-9]{2,}", _tidy_title(f["title"]).casefold()) if w not in stop}
    if not words:
        return 0
    pattern = lambda w: rf"\b{re.escape(w[:6])}" if len(w) >= 5 else rf"\b{re.escape(w)}\b"
    hits = sum(1 for w in words if re.search(pattern(w), low))
    # The flag's own subject (NSAIDs, dose, swelling ...) must be named; condition names alone ("diabetes, hypertension")
    # do not make a point about a flag that merely mentions them.
    core = [w for w in words if w not in _CONDITION_WORDS]
    if core and not any(re.search(pattern(w), low) for w in core):
        hits = 0
    ratio = hits / len(words)
    values = set(re.findall(r"\d{4}-\d{2}-\d{2}|\d+\.\d+|(?<![\d.#])\d{3,}(?![\d.])", str(f.get("evidence") or "") + " " + f["title"]))
    value_hit = any(re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", point) for v in values)
    if ratio >= 0.65 or (value_hit and hits >= 1):
        return ratio + (0.5 if value_hit else 0)
    if hits >= 2 and ratio >= 0.5:  # half the title's own words ('impossible … reading'): a weaker statement of it
        return ratio
    observed = {w for w in re.findall(r"\b[a-z]{5,}\b", str(f.get("evidence") or "")) if w not in
                {"about", "visits", "record", "latest", "prescription", "total", "stated", "durations", "added", "since"}}
    # Distinct stems: 'reading' and 'readings' are one word of the point, not two.
    if len({w[:6] for w in observed if re.search(rf"\b{re.escape(w)}", low)}) >= 2:
        return 0.5
    return 0


def _point_flag(point, flags):
    """The rule flag a point states, if any: the best-scoring one."""
    scored = [(_flag_score(point, f), i) for i, f in enumerate(flags)]
    best = max(scored, default=(0, None))
    return flags[best[1]] if best[0] > 0 else None


def _points(lines):
    out = []
    for line in lines:
        prefix = re.match(r"^(\s*(?:[-*•]|\d+[.)])?\s*(?:\*\*[^*]+\*\*:?\s*)?)", line).group(0)
        out += [x for x in re.split(r";\s+(?![^()]*\))", line[len(prefix):]) if x.strip()]
    return out


def ensure_key_points(lines, evidence):
    """Add, from the record, any must-not-miss point the model left out.

    Must-not-miss: every high and medium rule flag (BMI, creatinine, dose above maximum, duplicate NSAIDs ...), and
    conditions held on the
    patient's long-term list. A point counts as made when the brief uses a distinctive word or value from it;
    otherwise it is appended to its theme (or a new theme bullet), so it never depends on the model.
    """
    brief = "\n".join(lines)
    # Every high, verify and medium flag must be stated in Key findings (BMI too); a point elsewhere does not count.
    def theme_of(line):
        match = re.match(r"^\s*-\s*\*\*([^*:]+):?\*\*", line)
        return match.group(1).strip().casefold() if match else None
    key_points = _points([l for l in lines if theme_of(l) == "key findings"])
    stated = {id(_point_flag(pt, evidence["rule_flags"])) for pt in key_points}
    missing = {}
    # A high / verify / medium flag already stated under another theme is moved (in the model's words), not repeated.
    must = [f for f in evidence["rule_flags"] if f.get("level") in ("high", "verify", "medium") and not is_gap_flag(f)]
    known = {}  # point text -> the flag it states, for ordering by severity
    moved, rest = [], []
    for line in lines:
        theme = theme_of(line)
        if theme in (None, "key findings", "conditions", "course"):
            rest.append(line)
            continue
        prefix = re.match(r"^(\s*-\s*\*\*[^*]+\*\*:?\s*)", line).group(0)
        keep = []
        for pt in [x.strip().rstrip(".") for x in re.split(r";\s+(?![^()]*\))", line[len(prefix):]) if x.strip()]:
            f = _point_flag(pt, must)
            if f and id(f) not in stated:
                stated.add(id(f))
                moved.append(pt)
                known[pt] = f
            else:
                keep.append(pt)
        if keep:
            rest.append(prefix + "; ".join(keep) + ".")
    if moved:
        lines = rest
        missing["Key findings"] = moved
        key_points += moved
    # Missing-test items: in Gaps to check only (moved there from Key findings, added there when absent).
    gap_points = _points([l for l in lines if theme_of(l) == "gaps to check"])
    gap_flags = [f for f in evidence["rule_flags"] if is_gap_flag(f) and f.get("level") in ("high", "verify", "medium")]
    relocated = []
    for i, line in enumerate(lines):
        if theme_of(line) != "key findings":
            continue
        prefix = re.match(r"^(\s*-\s*\*\*[^*]+\*\*:?\s*)", line).group(0)
        pts = [x.strip().rstrip(".") for x in re.split(r";\s+(?![^()]*\))", line[len(prefix):]) if x.strip()]
        gone = [x for x in pts if _point_flag(x, gap_flags)]
        if gone:
            relocated += gone
            rest_pts = [x for x in pts if x not in gone]
            lines[i] = prefix + "; ".join(rest_pts) + "." if rest_pts else ""
    lines = [l for l in lines if l]
    key_points = [x for x in key_points if x not in relocated]
    gap_add = relocated + [_rewrite_flag(f) for f in gap_flags if not any(_flag_score(x, f) >= 0.5 for x in gap_points + relocated)]
    if gap_add:
        missing["Gaps to check"] = gap_add
    for f in sorted((f for f in evidence["rule_flags"] if (f.get("level") in ("high", "verify", "medium")
                     or str(f["title"]).startswith("BMI")) and not is_gap_flag(f)), key=flag_importance):
        if id(f) not in stated and not any(_flag_score(pt, f) >= 0.5 for pt in key_points):
            point = _rewrite_flag(f)
            known[point.rstrip(". ")] = f
            missing.setdefault("Key findings", []).append(point)
    conditions_line = next((l for l in lines if re.match(r"^\s*-\s*\*\*Conditions", l)), "")
    line_words = set(re.findall(r"[a-z]{3,}", conditions_line.casefold()))
    from clinical_snapshot import THRESHOLDS
    # Named when any label, or either half of a registry group name ('GERD / acidity'), is in the line.
    named = lambda c: any(_label_in(seg.strip(), line_words) for label in [c["name"], *c.get("recorded_as", [])] if label
                          for seg in [label, *label.split("/")] if seg.strip())
    for c in evidence["conditions"]:
        # Every confirmed and known diagnosis is named in Conditions.
        if c["status"] in ("confirmed", "known diagnosis") and not c.get("long_term") and not named(c):
            missing.setdefault("Conditions", []).append(f"{_short_label(c)} ({c['status']})")
    # Recorded working / suspected diagnoses the brief left out (a stricter check may have pruned them): added back,
    # most visits first, up to brief.max_other_conditions; they are recorded diagnoses, never "unsupported".
    others = sorted((c for c in evidence["conditions"] if c["status"] in ("working diagnosis", "suspected") and not named(c)),
                    key=lambda c: (c["visits"], c["last_seen"]), reverse=True)  # most visits, then most recent
    cap = THRESHOLDS["brief"].get("max_other_conditions", 4)
    if others:
        missing.setdefault("Conditions", []).append(
            "Also recorded: " + ", ".join(f"{_short_label(c)} ({c['status']}, last {_last_enc(c)})" for c in others[:cap])
            + (f" +{len(others) - cap} more" if len(others) > cap else ""))
    for c in evidence["conditions"]:
        if c.get("long_term"):
            words, _ = _anchors(c["name"], " ".join(c["recorded_as"]))
            if not _mentioned(brief, words | {w.casefold() for w in re.findall(r"[A-Z]{2,6}", c["name"])}, set()):
                missing.setdefault("Conditions", []).append(f"Long-term: {condition_label(c)} ({c['long_term']})")
    out = list(lines)
    for theme in _THEMES:
        points = missing.get(theme)
        if not points:
            continue
        at = next((i for i, line in enumerate(out) if theme_of(line) == theme.casefold()), None)
        if at is not None:
            out[at] = out[at].rstrip().rstrip(".") + "; " + "; ".join(points) + "."
        else:
            # Keep theme order: insert before the first bullet of a later theme.
            later = {t.casefold() for t in _THEMES[_THEMES.index(theme) + 1:]}
            pos = next((i for i, line in enumerate(out) if theme_of(line) in later), len(out))
            out.insert(pos, _bullet(theme, points))
    added = sum(len(v) for v in missing.values())
    if added:
        metrics = evidence.setdefault("_metrics", {})
        metrics["flags_added"] = metrics.get("flags_added", 0) + added
    return order_key_findings(out, evidence, known)


_GAP_TITLE = re.compile(r"(?i)\b(?:no|without)\b.*\bin the record\b|monitoring due|no related lab tests|"
                        r"without liver tests")


def is_gap_flag(f):
    """A flag about a test the record lacks: belongs to Gaps to check, never Key findings."""
    return bool(_GAP_TITLE.search(str(f.get("title") or "")))


def order_key_findings(lines, evidence, known=None):
    """Key findings points by severity: high, verify, medium, then points that state no flag. Info / data-quality
    items leave (BMI kept); two points stating one flag become one; medium items are capped (clinical_thresholds brief)."""
    from clinical_snapshot import THRESHOLDS
    max_medium = THRESHOLDS["brief"]["max_medium"]

    def flag_of(point):
        return (known or {}).get(point) or _point_flag(point, evidence["rule_flags"])

    def rank(point):
        f = flag_of(point)
        return _LEVEL_RANK.get(f.get("level"), 3) if f else 2.5
    out = []
    for line in lines:
        m = re.match(r"^(\s*-\s*\*\*Key findings:?\*\*:?\s*)", line)
        if not m:
            out.append(line)
            continue
        points = [p.strip().rstrip(".") for p in re.split(r";\s+(?![^()]*\))", line[m.end():]) if p.strip()]
        kept, seen, medium = [], {}, 0
        ranked = sorted(points, key=rank)
        flags_here = [flag_of(pt) for pt in ranked]

        def absorbed(f):
            """True when another flag here, at least as severe, already carries this flag's evidence."""
            core = re.split(r"\s·\s", str(f.get("evidence") or ""))[0].strip()
            return bool(core) and any(g is not f and g and _LEVEL_RANK.get(g.get("level"), 3) <= _LEVEL_RANK.get(f.get("level"), 3)
                                      and all(c.strip() in str(g.get("evidence") or "") for c in core.split(";") if c.strip())
                                      for g in flags_here)
        for pt in ranked:
            f = flag_of(pt)
            if f and absorbed(f):
                continue
            level = (f or {}).get("level")
            if f and (level == "info" or f.get("category")) and not str(f["title"]).startswith("BMI"):
                continue
            if f and id(f) in seen:  # the same flag twice: the fuller point stays
                if len(pt) > len(kept[seen[id(f)]]):
                    kept[seen[id(f)]] = pt
                continue
            if level == "medium":
                medium += 1
                if medium > max_medium:
                    continue
            if f:
                seen[id(f)] = len(kept)
            kept.append(pt)
        cap = THRESHOLDS["brief"].get("max_key_findings", 8)
        hidden = max(0, medium - max_medium) + max(0, len(kept) - cap)
        kept = kept[:cap]
        if hidden:
            kept.append(f"+{hidden} more finding{'s' if hidden != 1 else ''} in the Clinical Snapshot")
        out.append(m.group(1) + "; ".join(kept) + ".")
    return out


def _last_enc(condition):
    found = re.findall(r"#(\d+)", _expand_ranges(condition["encounters"]))  # '#1–3' ends at #3
    return f"#{found[-1]}" if found else ""


def _short_label(condition):
    """The condition's main name: the first spelling the record uses, or the registry name."""
    return (condition["recorded_as"] or [condition["name"]])[0]


# Snapshot flags that describe the patient's presentation (symptoms, vitals) rather than record quality.
_PRESENTATION_FLAG = re.compile(r"(?i)\bsymptoms?\b|out-of-range vitals")


def _is_presentation_flag(f):
    return f.get("kind") == "risk" or (f.get("kind") == "data" and _PRESENTATION_FLAG.search(f.get("title") or ""))


def _short_encs(row):
    """A recurring complaint's encounters: listed when 3 or fewer (a run as '#3–5'), else '#first … #last, n visits'."""
    nums = row.get("encounter_numbers") or [int(n) for n in re.findall(r"\d+", str(row.get("encounters") or ""))]
    if not nums:
        return str(row.get("encounters") or "")
    if len(nums) <= 3 or nums[-1] - nums[0] == len(nums) - 1:
        return _encs(nums)
    return f"#{nums[0]} … #{nums[-1]}, {len(nums)} visits"


def _course_bullet(evidence):
    """One line: what the patient presented with, then what keeps recurring, each complaint cited with its own
    encounter(s). Age, sex, visit count and dates are left to the common patient header."""
    rows = evidence["timeline"]
    if not rows:
        return ""
    recurring = (evidence.get("recurring_complaints") or [])[:2]
    repeated = {r["complaint"].casefold() for r in recurring}
    # Presenting: the first two encounters' complaints (diagnosis when no complaint), each with the encounters it was raised at.
    cites = {}
    for r in rows[:2]:
        phrases = _complaint_phrases(r["complaint"]) or [p.strip().casefold() for p in _DX_SPLIT.split(r["diagnosis"] or "") if len(p.strip()) >= 2]
        for ph in phrases:
            if ph.casefold() not in repeated:
                cites.setdefault(_clip(ph, 5), []).append(int(r["enc"][1:]))
    groups = {}
    for ph, encs in list(cites.items())[:4]:
        groups.setdefault(_encs(encs), []).append(ph)
    parts = []
    if groups:
        parts.append("Presented with " + ", ".join(f"{', '.join(phs)} ({e})" for e, phs in groups.items()))
    for r in recurring:
        parts.append(f"recurring {_clip(r['complaint'].lower(), 5)} ({_short_encs(r)})")
    if not parts:
        return ""
    line = " · ".join(parts)
    return "- **Course:** " + line[:1].upper() + line[1:] + "."


def _conditions_bullet(evidence):
    """Conditions active at the latest visit (with known history), then those not recorded since."""
    conditions, latest = evidence["conditions"], evidence["patient"]["latest_visit"]
    if not conditions:
        return _bullet("Conditions", ["No diagnosis is recorded"])
    status_text = {"document only": "only in documents"}

    def label(c):
        visits = f", {c['visits']} visits" if c["visits"] > 1 else ""
        long_term = ", long-term" if c.get("long_term") and c["status"] not in _KNOWN else ""
        return f"{_short_label(c)} ({status_text.get(c['status'], c['status'])}{visits}{long_term})"

    current = [c for c in conditions if _is_current(c, latest, evidence)]
    earlier = [c for c in conditions if c not in current]
    points = []
    if current:
        points.append("Active: " + ", ".join(_capped([label(c) for c in current], 4)))
    if earlier:
        points.append("Not recorded since: " + ", ".join(
            _capped([f"{_short_label(c)} (last {_last_enc(c)})" for c in earlier], 3)))
    # How the course behaves across visits (not settling, visits closer together, diagnosis evolution).
    patterns = sorted((f for f in evidence["rule_flags"] if f.get("kind") == "pattern"),
                      key=lambda f: _LEVEL_RANK.get(f.get("level"), 3))
    points += [_flag_point({**f, "evidence": f["evidence"].split(";")[0]}, 16) for f in patterns[:2]]
    return _bullet("Conditions", points)


def _lab_value(g):
    return f"{g['test']} {g['value']}{_short_ref(g)}"


def _guideline_points(evidence, *kinds):
    """'<topic>: <finding>' for the guideline_context rows of these kinds, as the no-AI brief writes them."""
    rows = (evidence.get("guideline_context") or {}).get("rows") or []
    return [f"{r['topic']}: {_clip(r['finding'], 16)}" for r in rows if r["kind"] in kinds]


def _findings_bullet(evidence):
    """Guideline control and lab rows, abnormal results read against the recorded condition (or the condition they
    point to), then flagged symptoms and vitals."""
    groups = evidence["abnormal_labs"]
    points, shown = _guideline_points(evidence, "condition", "lab")[:3], set()
    # A condition whose control a guideline row already states is not repeated as "above range".
    guided = {r["condition"] for r in (evidence.get("guideline_context") or {}).get("rows") or [] if r.get("marker")}
    by_condition = {}
    for g in groups:
        if g["related_conditions"] and not guided & set(g["related_conditions"]):
            by_condition.setdefault(tuple(g["related_conditions"]), []).append(g)
    for names, items in by_condition.items():
        shown.update(id(g) for g in items[:2])
        meds = sorted({m for g in items for m in g["related_medicines"]})
        before = sorted({m for g in items for m in g.get("medicines_started_before_result") or []})
        after = sorted({m for g in items for m in g.get("medicines_started_on_or_after_result") or []})
        earlier = sorted({m for g in items for m in g.get("earlier_related_medicines") or []})
        undated = any(not g["date"] for g in items[:2])
        direction = "above" if items[0]["direction"] == "high" else "below"
        # "despite" only when every shown result is dated after its medicines started.
        if before and not undated and not after:
            tail = f" despite {' + '.join(before)}"
        elif after:
            tail = f"; {_join(after)}: on or after this result"
        elif meds and undated:
            tail = f"; result undated, so not placed against {' + '.join(meds)}"
        elif meds:
            tail = f"; {' + '.join(meds)} on the latest prescription"
        elif earlier:
            tail = f", not on the latest prescription: {_join(earlier)}"
        else:
            tail = ", no medicine recorded for it"
        points.append(f"{_join(names)} {direction} range: {_join(_lab_value(g) for g in items[:2])}" + tail)
    for pc in evidence["possible_conditions"]:
        basis = [g for g in groups if id(g) not in shown and _controls(pc["condition"], g["test_as_written"], g["direction"])]
        if basis:
            shown.update(id(g) for g in basis[:2])
            points.append(f"Possible {pc['condition'].lower()} (not diagnosed): "
                          + _join(f"{_lab_value(g)} {g['status'].lower()}" for g in basis[:2]))
    for g in [g for g in groups if id(g) not in shown and not g["related_conditions"]][:max(0, 3 - len(points))]:
        shown.add(id(g))
        meaning = _first_clause(g["significance"])
        points.append(f"{_lab_value(g)} {g['status'].lower()}" + (f": {meaning}" if meaning else ""))
    hidden = sum(1 for g in groups if id(g) not in shown)
    if hidden:
        points.append(f"+{hidden} more out-of-range result(s) in the lab details")
    flags = sorted((f for f in evidence["rule_flags"] if _is_presentation_flag(f)),
                   key=lambda f: _LEVEL_RANK.get(f.get("level"), 3))
    points += [_flag_point(f, 10) for f in flags[:2]]
    if not points:
        count = evidence["lab_counts"]["results"]
        return _bullet("Key findings", [f"All {count} evaluated result(s) are within range"]) if count else ""
    return _bullet("Key findings", points)


def _running_text(m, latest):
    """'Amlodipine added 28 Sep, 10-day course to ~8 Oct, not rewritten on 1 Oct'."""
    try:
        days = (datetime.date.fromisoformat(m["course_end"]) - datetime.date.fromisoformat(m["last"])).days
    except (TypeError, ValueError):
        days = None
    when = "added" if m.get("first") == m.get("last") else "last written"
    return (f"{m['name']} {when} {short_date(m['last'])}" + (f", {days}-day course" if days else "")
            + f" to ~{short_date(m['course_end'])}" + (f", not rewritten on {short_date(latest)}" if latest else ""))


def _treatment_bullet(evidence, risks=True):
    """Current medicines grouped by the condition they treat, medicines dropped for a condition, medicine risks
    (risks=False: medicines only, the risks being stated in Key findings)."""
    meds = evidence["medicines"]
    if not meds:
        return _bullet("Treatment", ["No prescription is recorded"])
    # Every encounter on the latest day: two departments that day both wrote the current medicines.
    rows = evidence["timeline"]
    latest_enc = _encs([int(r["enc"][1:]) for r in rows if r["date"] == rows[-1]["date"]]) if rows else "the latest visit"
    current = [m for m in meds if m.get("status", "current" if m["at_latest_visit"] else "") == "current"]
    running = [m for m in meds if m.get("status") == "course running, not renewed"]
    by_condition, other = {}, []
    for m in current:
        name = m["name"] + (" (dose not specified)" if str(m.get("regimen") or "").startswith("dose not specified") else "")
        if m["for"]:
            by_condition.setdefault(_join(m["for"]), []).append(name)
        else:
            other.append(name)
    groups = [f"{cond}: {' + '.join(names)}" for cond, names in by_condition.items()]
    # Every current medicine is named: condition groups past the third join one point rather than being dropped.
    points = groups[:3] + (["; ".join(groups[3:])] if groups[3:] else [])
    if other:
        points.append(("Also: " if points else f"At {latest_enc}: ") + ", ".join(_capped(other, 3)))
    if running:  # an earlier course still running on the latest visit date, not written again there
        latest_date = evidence["patient"].get("latest_visit") or ""
        points.append("Course running, not renewed: " + ", ".join(_capped([_running_text(m, latest_date) for m in running], 2)))
    if not current and not running:
        points.append(f"No medicine at {latest_enc}")
    stopped = [f"{m['name']} for {_join(m['for'])} (last {m['last']})"
               for m in meds if m["for"] and _stop_supported(m, evidence.get("encounter_dates") or {})]
    if stopped:
        points.append("Not on the latest prescription: " + ", ".join(_capped(stopped, 2)))
    if not risks:
        return _bullet("Treatment", points)
    risks = sorted((f for f in evidence["rule_flags"] if f.get("kind") == "medication" and f.get("level") in ("high", "verify", "medium")),
                   key=flag_importance)
    for f in risks[:1]:
        # What this record shows (medicine, dose, encounter), not the rule's generic significance.
        first = re.split(r"\s·\s|;\s", f.get("evidence") or f.get("significance") or "")[0]
        points.append(f"⚠ {_tidy_title(f['title'])} — {_clip(first, 16).rstrip(' ;,.')}")
    points += [f"⚠ {p}" for p in _guideline_points(evidence, "medication")[:max(0, 2 - len(risks[:1]))]]
    return _bullet("Treatment", points)


def _gaps_bullet(evidence):
    """What the record shows as missing, due or inconsistent."""
    flags = evidence["rule_flags"]
    items = _guideline_points(evidence, "followup") + [f["title"] for f in flags if f.get("kind") == "followup"]
    items += [_clip(item.rstrip("."), 14) for item in evidence["followup_gaps"]]
    items += [f["title"] for f in flags if f.get("kind") == "data" and not _is_presentation_flag(f)
              and f.get("level") in ("high", "medium")]
    if evidence["unverified_labs"]:
        items.append(f"{len(evidence['unverified_labs'])} result(s) could not be interpreted, check the source")
    return _bullet("Gaps to check", _capped(items, 3))


_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NESTED = re.compile(r"\(([^()]*)\(([^()]*)\)([^()]*)\)")
_WORD_SLASH = re.compile(r"\b([A-Za-z]{4,})/(?=[A-Za-z]{4,}\b)")
_STOPWORDS = {"the", "and", "with", "for", "from", "on", "in", "of", "at", "to", "a", "an", "is", "are", "was", "no",
              "not", "latest", "prescription", "visit", "visits", "record", "days", "day", "about", "since"}


def short_date(iso, ref_year=""):
    """'2026-08-14' -> '14 Aug' ('14 Aug 2025' when the year is not the latest visit's)."""
    m = _ISO.fullmatch(str(iso or ""))
    if not m:
        return str(iso or "")
    y, mo, d = m.groups()
    return f"{int(d)} {_MONTHS[int(mo) - 1]}" + (f" {y}" if ref_year and y != ref_year else "")


def _flatten_brackets(text):
    """'(a (b), c (d))' -> '(a, b, c, d)': brackets inside brackets become commas."""
    out, depth = [], 0
    for ch in str(text):
        if ch == "(":
            depth += 1
            if depth > 1:
                out.append(", ")
                continue
        elif ch == ")":
            depth = max(0, depth - 1)
            if depth >= 1:
                continue
        out.append(ch)
    return re.sub(r"\s*,\s*(?:,\s*)+", ", ", re.sub(r"\s+,", ",", "".join(out))).replace("(, ", "(")


def _shorten(text, words):
    """One short sentence: asides in brackets (not citations) go from the end until the text fits, then a clause cut."""
    while len(text.split()) > words:
        asides = [m for m in re.finditer(r"\s*\((?!#\d)[^()]*\)", text)]
        if not asides:
            break
        m = asides[-1]
        text = text[:m.start()] + text[m.end():]
    return _clip(text, words)


def plain_text(text, ref_year=""):
    """One date format, no nested brackets, no slash lists."""
    text = _ISO.sub(lambda m: short_date(m.group(0), ref_year), str(text))
    text = _flatten_brackets(text)
    text = re.sub(r"\s+/\s+", ", ", text)
    text = _WORD_SLASH.sub(r"\1, ", text)
    return re.sub(r"\s{2,}", " ", text)


def _content_words(text):
    return {w for w in re.findall(r"[a-z][a-z0-9]{2,}", str(text).casefold()) if w not in _STOPWORDS}


def _same_fact(a, b):
    wa, wb = _content_words(a), _content_words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5


def _section(lines, name):
    for i, line in enumerate(lines):
        m = re.match(rf"^(\s*-\s*\*\*{name}:?\*\*:?\s*)", line)
        if m:
            return i, m.group(1), [p.strip().rstrip(".") for p in re.split(r";\s+(?![^()]*\))", line[m.end():]) if p.strip()]
    return None, "", []


def finalise_brief(lines, evidence):
    """The doctor-facing format, for any brief (model or rule-built):
    - Key findings: current high, current medium, then past; one point per clinical problem (flags sharing a
      'problem' are one finding); at most max_key_findings; each one short sentence.
    - Treatment: the current medicines with their purpose and the 'not on the latest prescription' items, no risks.
    - Gaps: only what Key findings do not already state.
    - Everywhere: one date format ('14 Aug'), no nested brackets, no slash lists.
    Records which flags each Key finding stands for in evidence['_key_findings'] (for the dashboard links)."""
    from clinical_snapshot import THRESHOLDS
    brief = THRESHOLDS["brief"]
    flags = evidence.get("rule_flags") or []
    dates = sorted(d for d in (evidence.get("encounter_dates") or {}).values() if d)
    ref_year = dates[-1][:4] if dates else ""
    out = list(lines)

    k, head, points = _section(out, "Key findings")
    kf_flags = []
    if k is not None:
        more = sum(int(n) for p in points for n in re.findall(r"^\+(\d+) more", p))
        points = [p for p in points if not re.match(r"^\+\d+ more", p)]

        def order(item):
            _, f = item
            if not f:
                return (1, 2.5)
            return (0 if f.get("time", "current") == "current" else 1, _LEVEL_RANK.get(f.get("level"), 3))
        items = sorted(((p, _point_flag(p, flags)) for p in points), key=order)
        kept, problems = [], {}
        for p, f in items:
            prob = (f or {}).get("problem")
            if prob and prob in problems:
                kept[problems[prob]][1].append(f)
                continue
            if prob:
                problems[prob] = len(kept)
            kept.append((p, [f] if f else []))
        cap = brief.get("max_key_findings", 6)
        more += max(0, len(kept) - cap)
        kept = kept[:cap]
        # Flags of the same problem the model did not mention are part of that finding too.
        for prob, idx in problems.items():
            if idx < len(kept):
                for f in flags:
                    if f.get("problem") == prob and f not in kept[idx][1] and f.get("level") in ("high", "verify", "medium"):
                        kept[idx][1].append(f)
        words = brief.get("key_finding_words", 22)
        texts = []
        for n, (p, fs) in enumerate(kept):
            fs.sort(key=lambda f: order((None, f)))
            kept[n] = (p, fs)
            cites = re.findall(r"\((?:#\d+(?:,\s*)?)+\)", p)
            p = plain_text(p, ref_year)
            lead = fs[0] if fs else None
            if lead:  # a count the model wrote differently from the flag ('2 missed follow-ups' when the flag has 1)
                fc = re.match(r"^(\d+)\s+([a-z-]+)", _tidy_title(lead["title"]), re.I)
                pc = re.match(r"^(\d+)(\s+[a-z-]+)", p, re.I)
                if fc and pc and fc.group(1) != pc.group(1) and pc.group(2).strip()[:5].casefold() == fc.group(2)[:5].casefold():
                    p = plain_text(_tidy_title(lead["title"]), ref_year)  # the flag's own count
            if len(fs) >= 2 and lead.get("problem"):
                # One clinical problem: led by its current, most severe flag, then the others by name.
                state = time_label(lead, ref_year)
                others = [_tidy_title(f["title"]) for f in fs[1:3]]
                title = _tidy_title(lead["title"])
                prefix = "" if title.casefold().startswith(lead["problem"].casefold()) else f"{lead['problem']} — "
                p = (f"{state}{prefix}{title[:1].lower() if prefix else title[:1]}{title[1:]}"
                     + (f", also {', '.join(o[:1].lower() + o[1:] for o in others)}" if others else "")).replace(";", ",")
            body = _shorten(re.sub(r"\s*\((?:#\d+(?:,\s*)?)+\)", "", p), words).rstrip(" .;,—:")
            body = body[:1].upper() + body[1:]
            texts.append(body + (f" {cites[-1]}" if cites else ""))
        if more:
            texts.append(f"+{more} more finding{'s' if more != 1 else ''} in the Clinical Snapshot")
        out[k] = head + "; ".join(texts) + "."
        kf_flags = [[f.get("title") for f in fs] for _, fs in kept]
    evidence["_key_findings"] = kf_flags
    kf_text = [p for p in _section(out, "Key findings")[2]]

    # Course: always the one-line record-built summary (presenting complaints, then recurring), for any brief.
    c, _, _ = _section(out, "Course")
    course = _course_bullet(evidence)
    if c is not None:
        if course:
            out[c] = course
        else:
            out.pop(c)
    elif course:
        out.insert(0, course)

    t, head_t, _ = _section(out, "Treatment")
    if t is not None:
        out[t] = _treatment_bullet(evidence, risks=False)

    g, head_g, gaps = _section(out, "Gaps to check")
    if g is not None:
        kf_titles = {x for fs in kf_flags for x in fs}
        keep = [p for p in gaps if not any(_same_fact(p, q) for q in kf_text)
                and not ((_point_flag(p, flags) or {}).get("title") in kf_titles)]
        if keep:
            out[g] = head_g + "; ".join(keep) + "."
        else:
            out.pop(g)
    return [plain_text(x, ref_year) for x in out]


def brief_violations(lines, evidence):
    """What breaks the doctor-facing format (finalise_brief): more than max_key_findings Key findings, a past finding
    above a current one, 'stopped', a fact repeated across Key findings / Treatment / Gaps, nested brackets, slash
    lists, ISO dates, risk statements in Treatment. [] when the brief is fine."""
    from clinical_snapshot import THRESHOLDS
    out = []
    text = "\n".join(lines)
    if re.search(r"(?i)\bstopped\b", text):
        out.append("says 'stopped'")
    if _ISO.search(text):
        out.append("ISO date")
    if re.search(r"\([^()]*\(", re.sub(r"\((?:#\d+(?:,\s*)?)+\)", "", text)):
        out.append("nested brackets")
    if re.search(r"\s/\s", text) or _WORD_SLASH.search(text):
        out.append("slash list")
    _, _, kf = _section(lines, "Key findings")
    kf = [p for p in kf if not re.match(r"^\+\d+ more", p)]
    if len(kf) > THRESHOLDS["brief"].get("max_key_findings", 6):
        out.append(f"{len(kf)} Key findings")
    flags = evidence.get("rule_flags") or []
    by_title = {f.get("title"): f for f in flags}
    linked = evidence.get("_key_findings") or []
    times = [((by_title.get(linked[i][0]) if i < len(linked) and linked[i] else None) or _point_flag(p, flags) or {}).get("time")
             for i, p in enumerate(kf)]
    times = [t for t in times if t]
    if any(a == "past" and b == "current" for a, b in zip(times, times[1:])):
        out.append("past finding above a current one")
    _, _, tr = _section(lines, "Treatment")
    _, _, gaps = _section(lines, "Gaps to check")
    if any(re.search(r"(?i)\brisk\b|⚠", p) for p in tr):
        out.append("risk statement in Treatment")
    for name, pts in (("Treatment", tr), ("Gaps", gaps)):
        if any(_same_fact(p, q) for p in pts for q in kf):
            out.append(f"fact repeated in Key findings and {name}")
    return out


@traced(name="snapshot.deterministic_insights")
def deterministic_insights(evidence):
    """A 4-5 theme doctor's brief from the record alone: course, conditions, key findings, treatment, gaps.

    Each theme holds a few short points separated by '; '. Used when the model output is missing or unusable,
    and as the no-AI brief. Every point is built from the evidence pack, so it holds for any patient and
    states nothing the record does not.
    """
    bullets = [_course_bullet(evidence), _conditions_bullet(evidence), _findings_bullet(evidence),
               _treatment_bullet(evidence), _gaps_bullet(evidence)]
    bullets = [b for b in bullets if b]
    return bullets if len(bullets) > 1 else ["- The record holds too little clinical data for patient-specific insights."]


def apply_course_line(markdown, evidence):
    """A review's Course bullet replaced by the one-line record-built Course (also for reviews cached before the
    one-line rule), so every served brief shows the same Course."""
    course = _course_bullet(evidence)
    if not course or not markdown:
        return markdown
    lines = str(markdown).split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^\s*[-*•]\s*\*\*Course:?\*\*", line):
            lines[i] = course
            return "\n".join(lines)
    return markdown


def time_label(f, ref_year=""):
    """The time label a point about a flag starts with: 'Current at 1 Oct: ' for a current flag, 'Past (last 17 Sep): '
    for a medicine combination that stopped (snapshot tag_combo_time), '' otherwise. Never 'current' for a past flag."""
    if f.get("time") == "current" and f.get("as_of"):
        return f"Current at {short_date(f['as_of'], ref_year)}: "
    if f.get("time") == "past" and f.get("combo") and f.get("as_of"):
        return f"Past (last {short_date(f['as_of'], ref_year)}): "
    return ""


def fix_time_labels(markdown, evidence):
    """Every 'Current at <date>: <point>' checked against the flag the point leads with: dropped when that flag rests
    only on undated reports, rewritten 'Past (last <date>): ' when it is a stopped medicine combination, left as written
    otherwise. Applied to model and cached briefs alike."""
    flags = evidence.get("rule_flags") or []
    dates = sorted(d for d in (evidence.get("encounter_dates") or {}).values() if d)
    ref_year = dates[-1][:4] if dates else ""

    def fix(m):
        f = _point_flag(re.split(r",\s+also\s", m.group(2))[0], flags)  # the finding is led by its first flag
        if not f:
            return m.group(0)
        if f.get("undated"):
            return m.group(2)
        if f.get("time") == "past":
            return time_label(f, ref_year) + m.group(2)
        return m.group(0)
    return re.sub(r"(Current at [^:;]{1,20}:\s*)([^;\n]+)", fix, str(markdown or ""))


def drop_undated_dates(markdown, evidence):
    """Kept for callers: the time-label check (fix_time_labels)."""
    return fix_time_labels(markdown, evidence)

def refresh_record_lines(markdown, evidence):
    """A served brief (model or cached) brought in line with the record as it is now: the one-line Course, the
    Treatment line with each medicine's status (current / course running, not renewed / not on latest prescription),
    and every time label (fix_time_labels)."""
    text = apply_course_line(markdown, evidence)
    treatment = _treatment_bullet(evidence, risks=False)
    if treatment:
        dates = sorted(d for d in (evidence.get("encounter_dates") or {}).values() if d)
        treatment = plain_text(treatment, dates[-1][:4] if dates else "")
        text = re.sub(r"(?m)^\s*[-*•]\s*\*\*Treatment:?\*\*.*$", lambda m: treatment, text, count=1)
    return fix_time_labels(text, evidence)
