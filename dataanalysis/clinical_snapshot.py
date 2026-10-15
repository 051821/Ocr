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

from medication_journey import compute_medication_journey
from patient_chat import (REGISTRY, _CONDITION_LABS, _PAIN_MEDS, _med_display, build_condition_trend,
                          complaint_index)

_NSAID = REGISTRY["nsaid_pattern"]
_CONTROL_LABS = REGISTRY["control_labs"]
_TOPICAL = r"(?i)\b(?:gel|cream|oint(?:ment)?|spray|lotion|patch)\b"
_LEVEL_ORDER = {"high": 0, "medium": 1, "info": 2}


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
    rows = []
    for idx, v in enumerate(patient_data.get("visits") or [], 1):
        rows += [(lab, f"Encounter #{idx}") for lab in v.get("lab_results") or []]
    for doc in patient_data.get("unlinked_documents") or []:
        rows += [(lab, "undated report") for lab in doc.get("lab_results") or []]
    return rows


def _first_word(name):
    clean = re.sub(r"(?i)^(?:ta|tb|tab(?:let)?|cap(?:sule)?|syp|syr(?:up)?|inj(?:ection)?)\.?\s+", "", str(name or ""))
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
    numbers = list(numbers)
    return ", ".join(f"#{n}" for n in numbers[:6]) + (" …" if len(numbers) > 6 else "")


def build_snapshot(patient_data):
    visits = patient_data.get("visits") or []
    if not visits:
        return None
    trend = build_condition_trend(patient_data)
    conditions = {r["condition"]: r for r in trend["conditions"]}
    present = lambda names: [n for n in names if n in conditions]
    complaints = complaint_index(visits)
    meds_by_visit = [[_med_display(m.get("name")) for m in v.get("db_medications") or [] if m.get("name")] for v in visits]
    distinct_meds = list(dict.fromkeys(m for meds in meds_by_visit for m in meds))
    meds_matching = lambda pattern: [m for m in distinct_meds if re.match(pattern, m, re.I)]
    labs = _lab_rows(patient_data)
    covered = set()  # (test name, value) already reported by a condition-level flag
    flags = []

    def flag(level, title, evidence, significance=""):
        flags.append({"level": level, "title": title, "evidence": evidence, "significance": significance})

    def lab_key(lab):
        return (str(lab.get("test_name") or "").strip().casefold(), str(lab.get("value")))

    # Per-condition control markers, for every documented condition that has one.
    abnormal_control = {}
    for name, row in conditions.items():
        rule = _CONTROL_LABS.get(name)
        related_pattern = rule["pattern"] if rule else _CONDITION_LABS.get(name)
        if not related_pattern:
            continue
        meds = " + ".join(row["main_medicines"]) or "no related medicine recorded"
        related = [(l, s) for l, s in labs if re.search(related_pattern, str(l.get("test_name") or ""), re.I)]
        out = [(l, s) for l, s in related if rule and _out_of_range(l, rule["direction"]) and not _implausible(l)]
        if out:
            abnormal_control[name] = out
            covered.update(lab_key(l) for l, _ in out)
            values = "; ".join(f"{re.sub(r'(?i)glycosylated haemoglobin', '', str(l.get('test_name'))).strip()} "
                               f"{l.get('value')} ({s})" for l, s in out[:3])
            flag(rule["level"], f"{name}: {rule['label']} out of range", f"{values} · on {meds}",
                 "Values outside the reference range" + (" despite current therapy." if row["main_medicines"] else "."))
        elif not related:
            label = rule["label"] if rule else "related monitoring tests"
            flag("medium" if rule else "info", f"{name} without {label} in the record",
                 f"Followed {row['first_seen']} to {row['last_seen']} ({row['visits']} visits) · on {meds}",
                 "Control cannot be assessed from the record.")

    # Condition-linked symptoms from the chief complaints.
    for name in conditions:
        for rule in REGISTRY["condition_symptoms"].get(name, []):
            hits = {}
            for phrase, encs in complaints.items():
                if re.search(rule["pattern"], phrase, re.I):
                    hits.setdefault(_trim_to_match(phrase, rule["pattern"]), set()).update(encs)
            if hits:
                significance = rule["significance"]
                if rule.get("requires_abnormal_control") and name not in abnormal_control:
                    significance = ""
                flag(rule["level"], rule["title"], "; ".join(f"{p} ({_encs(sorted(e))})" for p, e in hits.items()),
                     significance)

    # Blood pressure trend
    bp = [(i, d, tuple(int(x) for x in re.findall(r"\d+", val)[:2])) for i, d, val in _vital_series(visits, "BP")
          if len(re.findall(r"\d+", val)) >= 2]
    htn = conditions.get("Hypertension")
    if bp:
        first, last = bp[0], bp[-1]
        raised = [i for i, _, (s, dia) in bp if s >= 140 or dia >= 90]
        lower_target = present(REGISTRY["bp_lower_target_conditions"])
        target = (130, 80) if lower_target else (140, 90)
        evidence = f"{first[2][0]}/{first[2][1]} ({first[1]}) → {last[2][0]}/{last[2][1]} ({last[1]})"
        if raised:
            evidence += f" · ≥140/90 at {_encs(raised)}"
        if htn and htn["main_medicines"]:
            evidence += " · on " + " + ".join(htn["main_medicines"])
        if last[2][0] >= target[0] or last[2][1] >= target[1]:
            improved = last[2][0] < first[2][0] or last[2][1] < first[2][1]
            flag("medium" if htn else "info", f"BP above {target[0]}/{target[1]} at latest visit", evidence,
                 ("Improved from first visit but not" if improved else "Not") + " at the usual target"
                 + (f" with {_join(lower_target)}." if lower_target else "."))
        elif raised:
            flag("info", "BP improved; earlier readings raised", evidence)

    # Weight / BMI
    weights = [(i, d, _num(v)) for i, d, v in _vital_series(visits, "Weight") if _num(v)]
    heights = [_num(v) for _, _, v in _vital_series(visits, "Height") if _num(v)]
    if weights and heights:
        height = Counter(round(h) for h in heights).most_common(1)[0][0]
        if 120 <= height <= 220:
            bmi = weights[-1][2] / (height / 100) ** 2
            cardiometabolic = present(REGISTRY["cardiometabolic_conditions"])
            if bmi >= 25:
                flag("medium" if bmi >= 30 else "info", f"BMI {bmi:.1f} ({'obese' if bmi >= 30 else 'overweight'})",
                     f"Weight {weights[0][2]:g} → {weights[-1][2]:g} kg, height ~{height} cm",
                     f"Adds cardiometabolic risk alongside {_join(cardiometabolic)}." if cardiometabolic else "")
        if max(heights) - min(heights) >= 5:
            flag("info", "Vitals data quality", f"Height recorded between {min(heights):g} and {max(heights):g} cm")

    # Pulse
    fast = [(i, _num(v)) for i, _, v in _vital_series(visits, "Pulse") if (_num(v) or 0) > 100]
    if fast:
        flag("info", "Pulse above 100", ", ".join(f"{p:g} bpm at #{i}" for i, p in fast))

    # Repeated oral NSAIDs with a condition they can worsen (topical gels counted separately)
    nsaid_risk = present(REGISTRY["nsaid_risk_conditions"])
    nsaid_visits = [i for i, meds in enumerate(meds_by_visit, 1)
                    if any(re.match(_NSAID, m, re.I) and not re.search(_TOPICAL, m) for m in meds)]
    topical = sorted({m for meds in meds_by_visit for m in meds if re.match(_NSAID, m, re.I) and re.search(_TOPICAL, m)})
    if len(nsaid_visits) >= 2 and nsaid_risk:
        names = [m for m, _ in Counter(m for meds in meds_by_visit for m in meds
                                       if re.match(_NSAID, m, re.I) and not re.search(_TOPICAL, m)).most_common()]
        flag("high" if len(nsaid_risk) >= 2 else "medium", f"Repeated oral NSAIDs with {_join(nsaid_risk)}",
             f"{', '.join(names[:3])} at {len(nsaid_visits)} visits ({_encs(nsaid_visits)})"
             + (f" · plus topical {', '.join(topical[:2])}" if topical else ""),
             "NSAIDs can raise BP, worsen fluid retention and affect kidney function.")
    elif topical and nsaid_risk:
        flag("info", "Topical NSAIDs in record", ", ".join(topical[:3]))

    # Kidney function
    renal_meds = meds_matching(REGISTRY["renal_caution_meds"])
    kidney_risk = present(REGISTRY["kidney_risk_conditions"])
    for lab, src in labs:
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
        hits = {p: e for p, e in complaints.items()
                if re.search(link["symptom"], p, re.I) and (not link["site"] or re.search(link["site"], p, re.I))}
        if not hits:
            continue
        linked = [(meds_matching(m["pattern"]), m["note"]) for m in link["meds"]]
        linked = [(names, note) for names, note in linked if names]
        if link["only_if_med"] and not linked:
            continue
        names = [n for group, _ in linked for n in group]
        flag("medium", link["title"], "; ".join(f"{p} ({_encs(e)})" for p, e in hits.items())
             + (f" · on {', '.join(names[:3])}" if names else ""),
             " ".join(note for _, note in linked))

    # ── Anomalies from extracted documents ───────────────────────────────
    doc_items = [(i, f"#{i} {d.get('document_label', 'Document')}", d)
                 for i, v in enumerate(visits, 1) for d in v.get("documents") or []]
    doc_items += [(None, f"{d.get('document_label', 'Document')} (not linked)", d)
                  for d in patient_data.get("unlinked_documents") or []]
    record_dx = {i: str(v.get("db_provisional_dx") or v.get("db_confirmed_dx") or v.get("provisional_diagnosis") or "").strip()
                 for i, v in enumerate(visits, 1)}
    # Results ahead of (or without) a matching diagnosis, for every condition with a control marker.
    for cond, rule in _CONTROL_LABS.items():
        first = conditions[cond]["encounters"][0] if cond in conditions else None
        early = []
        for enc, where, d in doc_items:
            for lab in d.get("lab_results") or []:
                name = str(lab.get("test_name") or "")
                if (not re.search(rule["pattern"], name, re.I) or not _out_of_range(lab, rule["direction"])
                        or _implausible(lab) or (rule["direction"] == "any" and re.search(r"(?i)\bhdl", name))):
                    continue
                if (enc and (first is None or enc < first)) or (enc is None and first is None):
                    covered.add(lab_key(lab))
                    then = f"; record diagnosis then: {record_dx.get(enc) or 'none'}" if enc else ""
                    early.append(f"{name} {lab.get('value')} ({lab.get('status')}) at {where}{then}")
        if early:
            flag(rule["level"],
                 f"{cond} pattern result before it was documented" if first else f"{cond} pattern result with no matching diagnosis",
                 "; ".join(list(dict.fromkeys(early))[:2])
                 + (f" · {cond} first documented #{first}" if first else f" · no {cond} diagnosis in the record"),
                 "Document finding not reflected in the structured diagnosis.")

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
             "Values far outside the range; likely an OCR misread." if all_odd else significance)
    rest = [abnormal[k] for k in abnormal if k not in used]
    if rest:
        flag("info", "Other abnormal results", "; ".join(_point(e) for e in rest))

    # Record vs document mismatches (diagnosis and vitals).
    mismatches = []
    for i, v in enumerate(visits, 1):
        for d in v.get("documents") or []:
            cs = d.get("clinical_summary") or {}
            doc_dx = str(cs.get("provisional_diagnosis") or cs.get("confirmed_diagnosis") or "").strip()
            rec = record_dx.get(i, "")
            if (doc_dx and rec and doc_dx.casefold() not in rec.casefold() and rec.casefold() not in doc_dx.casefold()
                    and not re.search(r"(?i)\b(?:bp|pulse|spo2|sign|signature|weight|height)\b", doc_dx)):
                mismatches.append(f"#{i}: document '{doc_dx[:40]}' vs record '{rec[:40]}'")
        for text in v.get("discrepancies") or []:
            if "structured record shows" in text:
                mismatches.append(f"#{i}: " + re.sub(r"^Document \d+:\s*", "", text).rstrip("."))
    if mismatches:
        flag("medium", f"Record vs document mismatch ({len(mismatches)})", "; ".join(mismatches[:4]) + (" …" if len(mismatches) > 4 else ""),
             "The scanned document and the structured record disagree.")

    # Out-of-range vitals recorded in documents.
    odd_vitals = []
    for enc, where, d in doc_items:
        for row in (d.get("clinical_summary") or {}).get("vitals") or []:
            label, value = str(row.get("label") or ""), str(row.get("value") or "")
            celsius = re.search(r"([\d.]+)\s*°C", value)
            pressure = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", value)
            number = _num(value)
            odd = ((label == "Temperature" and celsius and not 35.0 <= float(celsius.group(1)) < 38.0)
                   or (label == "BP" and pressure and (int(pressure.group(1)) >= 140 or int(pressure.group(2)) >= 90))
                   or (label.startswith("SpO") and number and number < 94)
                   or (label in ("Pulse", "Heart Rate") and number and number > 100))
            if odd:
                odd_vitals.append(f"{label} {value} ({where})")
    if odd_vitals:
        flag("medium", "Out-of-range vitals in documents", "; ".join(odd_vitals[:4]),
             "Could be a true finding or an OCR/unit error; verify.")

    # Symptoms and medicines that appear only in documents.
    record_text = " ".join(str(v.get("db_chief_complaint") or v.get("chief_complaint") or "") for v in visits).casefold()
    doc_symptoms = {}
    for enc, where, d in doc_items:
        for sym in (d.get("clinical_summary") or {}).get("symptoms") or []:
            words = [w for w in re.findall(r"[a-z]{4,}", sym.casefold()) if w not in ("pain", "since", "days")]
            if words and not any(w in record_text for w in words):
                doc_symptoms.setdefault(sym[:50], where)
    if doc_symptoms:
        flag("info", "Symptoms only in documents", "; ".join(f"{s} ({w})" for s, w in list(doc_symptoms.items())[:3]))
    rx_names = {_first_word(_med_display(m.get("name"))) for v in visits for m in v.get("db_medications") or []}
    doc_only = {}
    for enc, where, d in doc_items:
        for med in d.get("medications_found") or []:
            word = _first_word(med)
            # Fuzzy match so spelling variants (Cetirizine / citrizine, Vaseline / vasaline) count as prescribed.
            if word and not any(SequenceMatcher(None, word, rx).ratio() >= 0.75 for rx in rx_names):
                doc_only.setdefault(med, where)
    if doc_only:
        flag("info", "Medicines only in documents", "; ".join(f"{m} ({w})" for m, w in list(doc_only.items())[:4]))

    # Follow-up adherence
    missed = [(i, str(f.get("scheduled_date") or "")[:10]) for i, v in enumerate(visits, 1)
              for f in v.get("followups") or [] if "MISSED" in str(f.get("status", "")).upper()]
    if missed:
        latest_missed = missed[-1][0] == len(visits)
        flag("high" if latest_missed else "medium",
             f"{len(missed)} missed follow-up(s)" + (" incl. the latest" if latest_missed else ""),
             ", ".join(f"{d} (after #{i})" for i, d in missed),
             "Latest scheduled review did not happen." if latest_missed else "")

    # Chronic pain and record gaps
    pain = next((r for name, r in conditions.items() if name.startswith("Musculoskeletal") or "pain" in name.casefold()), None)
    if pain and pain["visits"] >= 5:
        flag("info", "Recurrent musculoskeletal pain", f"{pain['visits']} visits, {pain['status']}")
    unlinked = patient_data.get("unlinked_documents") or []
    if unlinked:
        flag("info", f"{len(unlinked)} report(s) not linked to a visit", "Lab dates unknown")

    flags.sort(key=lambda f: _LEVEL_ORDER[f["level"]])
    docs = [d for v in visits for d in v.get("documents") or []] + list(unlinked)
    age_sex = next(((d.get("clinical_summary") or {}).get("age_sex") for d in docs
                    if (d.get("clinical_summary") or {}).get("age_sex")), None)

    timeline = [{"label": r["condition"], "encounters": set(r["encounters"])} for r in trend["conditions"][:5]]
    if bp:
        timeline.append({"label": "BP ≥140/90", "encounters": {i for i, _, (s, dia) in bp if s >= 140 or dia >= 90},
                         "kind": "alert"})
    if missed:
        timeline.append({"label": "Missed follow-up", "encounters": {i for i, _ in missed}, "kind": "alert"})
    abnormal_report_encs = {enc for enc, _, d in doc_items if enc for lab in d.get("lab_results") or []
                            if re.search(r"(?i)abnormal|high|low", str(lab.get("status") or ""))}
    if abnormal_report_encs:
        timeline.append({"label": "Abnormal report results", "encounters": abnormal_report_encs, "kind": "alert"})

    return {
        "age_sex": age_sex,
        "encounters": len(visits),
        "span": trend["span"],
        "active": [r["condition"] for r in trend["conditions"] if r["status"].startswith("ongoing")],
        "active_meds": len({m.casefold() for m in meds_by_visit[-1]}) if meds_by_visit else 0,
        "flags": flags,
        "timeline": timeline,
        "dates": [_date(v) for v in visits],
    }


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
    docs = [d for v in patient_data.get("visits") or [] for d in v.get("documents") or []]
    docs += list(patient_data.get("unlinked_documents") or [])
    return next(((d.get("clinical_summary") or {}).get("age_sex") for d in docs
                 if (d.get("clinical_summary") or {}).get("age_sex")), None)


def build_clinical_course(patient_data):
    """Concise, patient-focused course: headline, what stayed consistent, diagnosis
    and medicine shifts, and a month-by-month timeline. Deterministic, no LLM."""
    visits = patient_data.get("visits") or []
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
    ongoing = [f"{r['condition']} (since {_month(r['first_seen'])})" for r in trend["conditions"]
               if r["status"].startswith("ongoing")]
    other = [r["condition"] for r in trend["conditions"] if not r["status"].startswith("ongoing")]
    headline = (f"{age_sex + ' · ' if age_sex else ''}{len(visits)} encounters, {_month(dates[0])} to {_month(latest)}. "
                + (f"Ongoing {' and '.join(ongoing)}." if ongoing else "No ongoing chronic condition documented.")
                + (f" Intermittent {', '.join(other)}." if other else ""))
    return {"headline": headline, "consistent": consistent, "diagnosis_shifts": dx_events,
            "medicine_shifts": med_events, "timeline": timeline}
