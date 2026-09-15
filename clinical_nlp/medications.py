"""
Medication extraction from free text such as:
    Tab amlodipine 5 mg od
    Cap pantoprazole 40mg 1-0-0 before food
    Tab ecopsrin av 75 20 hs       <- combo drug, two dose numbers
    Tab telmisartan 40 od          <- unit sometimes missing in OCR'd text
    ao tramadol 37 mg bd          <- OCR corruption of 'tab'
    Tab ealciumd31000mg od        <- OCR glued drug + dose
    ab b complex od               <- OCR prefix + drug with no numeric dose

Deterministic rule-based and dictionary-driven extraction without LLM.
OCR typos and common abbreviations are handled via preprocessing and RapidFuzz
normalization.
"""
from __future__ import annotations
import re
from normalization.medication import normalize_medication, normalize_frequency

_DOSAGE_FORMS = (
    r"(?:tab|tablet|tablets|cap|caps|capsule|capsules|inj|injection|"
    r"syp|syrup|susp|suspension|drops|drop|ointment|oint|cream|gel|lotion|"
    r"spray|powder|patch|inhaler|ab|ao|tb|ta|t\.|c\.|cap\.|tab\.)"
)

# Comprehensive frequency / instruction / duration patterns
_FREQ_ALTS = (
    r"od|bd|tds|tid|bid|qid|hs|sos|prn|stat|weekly|monthly|"
    r"1-0-0|1-1-1|1-0-1|0-0-1|1-1-1-1|1\s*0\s*0|1\s*0\s*1|1\s*1\s*1|0\s*0\s*1|"
    r"1/2\s*[-–—]{1,2}\s*0|0\s*[-–—]{1,2}\s*1/2|1\s*[-–—]{1,2}\s*1|"
    r"bbr|bbd|b\.d|b\.d\.|b/d|o\.d|o\.d\.|t\.d\.s|t\.d\.s\.|"
    r"once\s+daily|twice\s+daily|thrice\s+daily|"
    r"daily|once|twice|thrice|"
    r"morning|afternoon|evening|night|bedtime|"
    r"after\s+food|before\s+food|with\s+food|"
    r"empty\s+(?:stomach|syomach|stmach)|"
    r"(?:x|×|\*)\s*\d+\s*(?:days?|d\b)|"
    r"for\s+\d+\s*(?:days?|weeks?|months?)|"
    r"\d+\s*(?:days?|weeks?)\b"
)

_DOSE_UNITS = r"mg|mcg|ug|g|ml|iu|u|cap|caps|tab|tabs"

_DOSE_LINE_RE = re.compile(
    rf"^(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>{_DOSE_UNITS})?$",
    re.IGNORECASE,
)

_FREQ_LINE_RE = re.compile(
    rf"^(?P<freq>{_FREQ_ALTS})$",
    re.IGNORECASE,
)

_DURATION_LINE_RE = re.compile(r"^\d+\s*(?:days?|weeks?|months?)$", re.IGNORECASE)

_BLOCK_END_RE = re.compile(
    r"^(?:chief\s+complaint|complaints?|diagnosis|impression|"
    r"provisional\s+diagnosis|confirmed\s+diagnosis|"
    r"investigations?|biochemistry|history|physical\s+examination|"
    r"notes?\b|advice\b)",
    re.IGNORECASE,
)

_COLUMN_HEADER_RE = re.compile(
    r"^(?:dosage|frequency|duration(?:\s*/\s*timing)?|center\s*id.*|"
    r"id\s*[:#].*|id\s+[a-z0-9]{5,})$",
    re.IGNORECASE,
)

# Words that identify a line as laboratory or administrative text, not a medication
_REJECT_KEYWORDS = {
    "glucose", "creatinine", "albumin", "cholesterol", "hemoglobin",
    "haemoglobin", "urea", "sodium", "potassium", "bilirubin", "sgot",
    "sgpt", "ast", "alt", "hba1c", "tsh", "triglycerides", "sugar",
    "protein", "phosphatase", "phosphatse", "urine", "routine",
    "investigation", "biochemistry", "microscopic", "ref. interval",
    "patient", "doctor", "hospital", "department", "signature", "registration",
    "room", "ticket", "date", "address", "occupation", "opd", "uhid", "age",
    "sex", "gender", "complaints", "history", "examination", "diagnosis",
    "nagpur", "mumbai", "pulse", "weight", "height", "bp", "spo2",
    "route", "freq", "days", "physical", "provisional", "sonography",
    "radiology", "pathology", "consultation", "dispensery", "procedures",
    "food", "after", "before", "empty", "stomach", "syomach", "diet",
    "daibetic", "diabetic", "notes", "duration", "timing", "dosage", "frequency",
    "impression", "advice", "adv", "ref", "refer",
}

_MED_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9\-\.\s/]{0,35}?)\s+"
    rf"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>{_DOSE_UNITS})?(?![/\w])"
    rf"(?:\s+(?P<dose2>\d+(?:\.\d+)?)\s*(?P<unit2>{_DOSE_UNITS})?(?![/\w]))?"
    rf"(?:\s+(?P<freq>{_FREQ_ALTS}))?",
    re.IGNORECASE,
)

_MED_NO_DOSE_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9\-\.\s/]{0,35}?)\s+"
    rf"(?P<freq>{_FREQ_ALTS})(?![/\w])",
    re.IGNORECASE,
)


def _merge_prescription_lines(lines: list[str]) -> list[str]:
    """
    Merge multi-line prescription entries where the medication name and its dose/timing
    appear on adjacent lines in flattened OCR output:
    e.g.
        Line 1: TELMISARTAN
        Line 2: 40mg
        Line 3: METFORMIN
        Line 4: 500 Mg
    """
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # Look ahead: if next line starts with a dose (e.g. "40mg", "500 Mg", "1500 ug")
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r"^\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml|iu|u|cap|tab)?\b", next_line, re.IGNORECASE):
                # Only merge if current line does not already end with a numeric dose
                if not re.search(r"\d+\s*(?:mg|mcg|ug|g|ml|iu|u)\b", line, re.IGNORECASE):
                    line = f"{line} {next_line}"
                    i += 1
        merged.append(line)
        i += 1
    return merged


def _preprocess_med_line(line: str) -> str:
    """Fix common OCR word-glue issues in prescription lines."""
    # Strip leading numbering, bullets, Rx: "1.", "a.", "-", "Rx", etc.
    line = re.sub(
        r"^(?:(?:\d+|[a-zA-Z])[\.\)\:\-]|[-*•℞]|(?:rx[:\.\s\-]?))\s*",
        "", line, flags=re.IGNORECASE,
    )
    # Split attached drug+vitamin/digit: e.g. "ealciumd3" -> "ealcium d3"
    line = re.sub(r"(?i)\b(ealcium|calcium)d(\d)", r"\1 d\2", line)
    # Split d3 followed by dose: e.g. "d31000mg" -> "d3 1000 mg"
    line = re.sub(r"(?i)\bd(\d)(\d{2,})", r"d\1 \2", line)
    # Split dose and unit: e.g. "1000mg" -> "1000 mg", "40mg" -> "40 mg", "1cap" -> "1 cap"
    line = re.sub(rf"(?i)(\d+)({_DOSE_UNITS})\b", r"\1 \2", line)
    return line.strip()


def _is_rejected_name(name: str) -> bool:
    """Filter out non-medication terms like administrative labels or IDs."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", name.lower()).strip()
    if not cleaned or len(cleaned) < 2:
        return True
    # Pure numbers
    if cleaned.isdigit():
        return True
    # Alphanumeric codes / IDs: e.g. 4FOAB62C, DS-TMC-012
    if re.fullmatch(r"[a-z0-9]{6,}", cleaned) and any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned):
        return True
    # Pure dosage form
    if cleaned in {"tab", "cap", "lotion", "ointment", "cream", "syrup", "drops", "inj"}:
        return True
    # Pure context / administrative keywords
    tokens = set(cleaned.split())
    if tokens & _REJECT_KEYWORDS:
        return True
    return False


def _parse_dose_token(line: str):
    processed = _preprocess_med_line(line)
    m = _DOSE_LINE_RE.match(processed)
    if not m:
        return None
    try:
        dose = float(m.group("dose"))
    except ValueError:
        return None
    unit = (m.group("unit") or "").lower() or None
    if unit == "caps":
        unit = "cap"
    if unit == "tabs":
        unit = "tab"
    return {"dose": dose, "unit": unit, "source": line.strip()}


def _parse_freq_token(line: str):
    processed = _preprocess_med_line(line)
    m = _FREQ_LINE_RE.match(processed)
    if not m:
        return None
    return m.group("freq").lower()


def _inline_prescription(content: str):
    """Parse a single-line prescription: name + optional dose + optional frequency."""
    m = _MED_LINE_RE.search(content)
    if m:
        name_raw = m.group("name").strip()
        name_raw = re.sub(rf"\s+(?:{_DOSAGE_FORMS})\b", "", name_raw, flags=re.IGNORECASE).strip()
        unit = (m.group("unit") or "").lower() or None
        freq_raw = (m.group("freq") or "").lower() or None
        try:
            dose = float(m.group("dose"))
        except ValueError:
            dose = None
        return name_raw, dose, unit, freq_raw

    m_nodose = _MED_NO_DOSE_RE.search(content)
    if m_nodose:
        name_raw = m_nodose.group("name").strip()
        name_raw = re.sub(rf"\s+(?:{_DOSAGE_FORMS})\b", "", name_raw, flags=re.IGNORECASE).strip()
        freq_raw = (m_nodose.group("freq") or "").lower() or None
        return name_raw, None, None, freq_raw

    name_raw = re.sub(rf"\s+(?:{_FREQ_ALTS})\b.*$", "", content, flags=re.IGNORECASE).strip()
    name_raw = re.sub(rf"\s+(?:{_DOSAGE_FORMS})\b", "", name_raw, flags=re.IGNORECASE).strip()
    return name_raw, None, None, None


def _flush_orphan_doses(drugs: list[dict], orphans: list[dict]) -> None:
    """
    Flattened OCR often emits a leftover dose *after* the next drug already
    received its own dose (column lag). Assign leftover doses backwards onto
    earlier drugs that still have no dose. If several leftovers exist for one
    nameless slot, keep the last one (closest to the following drug block).
    """
    if not orphans:
        return
    doseless_idx = [i for i, d in enumerate(drugs) if d["dose"] is None]
    assigned = False
    while orphans and doseless_idx:
        orphan = orphans.pop()
        target = drugs[doseless_idx.pop()]
        target["dose"] = orphan["dose"]
        target["unit"] = orphan["unit"]
        if orphan.get("freq_raw"):
            target["frequency_raw"] = orphan["freq_raw"]
        extra = orphan.get("source") or ""
        if extra and extra not in (target.get("source_text") or ""):
            target["source_text"] = f"{target['source_text']} {extra}".strip()
        assigned = True
    # Unnamed leftover doses (e.g. a 1mg line with no drug name) must not
    # steal AFTER FOOD / BEFORE FOOD from the next real prescription.
    if assigned:
        orphans.clear()


def _extract_from_form_layout(text: str) -> list[dict]:
    """
    Reconstruct prescriptions from multi-line OPD forms where OCR stacks
    name / dose / timing on separate lines, sometimes out of column order.
    """
    drugs: list[dict] = []
    orphans: list[dict] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _BLOCK_END_RE.match(line):
            _flush_orphan_doses(drugs, orphans)
            break
        if _COLUMN_HEADER_RE.match(line) or _DURATION_LINE_RE.match(line):
            continue
        if len(line) > 80:
            continue

        dose_tok = _parse_dose_token(line)
        if dose_tok:
            current = drugs[-1] if drugs else None
            if current is not None and current["dose"] is None:
                current["dose"] = dose_tok["dose"]
                current["unit"] = dose_tok["unit"]
                current["source_text"] = f"{current['source_text']} {line}".strip()
            else:
                orphans.append(dose_tok)
            continue

        freq_raw = _parse_freq_token(line)
        if freq_raw:
            if orphans:
                orphans[-1]["freq_raw"] = freq_raw
                orphans[-1]["source"] = f"{orphans[-1].get('source', '')} {line}".strip()
            elif drugs:
                if not drugs[-1]["frequency_raw"]:
                    drugs[-1]["frequency_raw"] = freq_raw
                drugs[-1]["source_text"] = f"{drugs[-1]['source_text']} {line}".strip()
            continue

        processed = _preprocess_med_line(line)
        prefix_m = re.match(rf"^(?:{_DOSAGE_FORMS}\.?\s+)", processed, re.IGNORECASE)
        had_prefix = bool(prefix_m)
        content = processed[prefix_m.end():].strip() if prefix_m else processed
        if not content:
            continue

        name_raw, dose, unit, inline_freq = _inline_prescription(content)
        if not name_raw or _is_rejected_name(name_raw):
            continue

        canonical, matched = normalize_medication(name_raw)
        if not canonical:
            continue
        if not matched and not had_prefix:
            continue

        _flush_orphan_doses(drugs, orphans)
        drugs.append({
            "name_raw": name_raw,
            "name": canonical,
            "name_matched": matched,
            "dose": dose,
            "unit": unit,
            "frequency_raw": inline_freq,
            "had_prefix": had_prefix,
            "source_text": line,
        })

    _flush_orphan_doses(drugs, orphans)
    return [_finalize_med(d) for d in drugs]


def _finalize_med(parsed: dict) -> dict:
    freq_raw = parsed.get("frequency_raw")
    frequency = normalize_frequency(freq_raw)
    matched = parsed.get("name_matched", False)
    had_prefix = parsed.get("had_prefix", False)
    unit = parsed.get("unit")
    dose = parsed.get("dose")

    confidence = 0.65
    if matched:
        confidence += 0.25
    if freq_raw:
        confidence += 0.05
    if had_prefix:
        confidence += 0.05
    if dose is not None and unit:
        confidence += 0.05
    elif dose is not None and not unit:
        confidence -= 0.05
    confidence = max(0.2, min(confidence, 0.95))

    return {
        "name_raw": parsed["name_raw"],
        "name": parsed["name"],
        "name_matched": matched,
        "dose": dose,
        "unit": unit,
        "frequency_raw": freq_raw,
        "frequency": frequency,
        "confidence": round(confidence, 2),
        "needs_verification": not matched or confidence < 0.75,
        "source_text": parsed.get("source_text") or parsed["name_raw"],
    }


def _med_name_overlap(a: str, b: str) -> bool:
    aa, bb = a.lower(), b.lower()
    if aa == bb:
        return True
    return aa in bb or bb in aa


def _merge_med_lists(primary: list[dict], secondary: list[dict]) -> list[dict]:
    merged = list(primary)
    for med in secondary:
        existing = next((m for m in merged if _med_name_overlap(m["name"], med["name"])), None)
        if existing is None:
            merged.append(med)
            continue
        if existing.get("dose") is None and med.get("dose") is not None:
            existing["dose"] = med["dose"]
            existing["unit"] = med["unit"]
            existing["source_text"] = med.get("source_text") or existing["source_text"]
        if not existing.get("frequency") and med.get("frequency"):
            existing["frequency"] = med["frequency"]
            existing["frequency_raw"] = med["frequency_raw"]
    return merged


def extract_medications(text: str):
    """
    Returns a list of dicts:
    {name_raw, name, matched, dose, unit, frequency_raw, frequency,
     confidence, needs_verification, source_text}
    """
    results = []
    if not text:
        return results

    form_results = _extract_from_form_layout(text)

    seen_names = set()
    raw_lines = text.splitlines()
    merged_lines = _merge_prescription_lines(raw_lines)

    for original_line in merged_lines:
        line = original_line.strip()
        if not line:
            continue

        if any(kw in line.lower() for kw in _REJECT_KEYWORDS):
            # Check if it's purely an administrative or lab heading line
            tokens = set(re.findall(r"[a-z]+", line.lower()))
            if tokens <= _REJECT_KEYWORDS:
                continue

        processed = _preprocess_med_line(line)
        if not processed:
            continue

        # Strip dosage prefix from beginning if present: e.g. "Tab Telma" -> "Telma"
        prefix_m = re.match(rf"^(?:{_DOSAGE_FORMS}\.?\s+)", processed, re.IGNORECASE)
        had_prefix = bool(prefix_m)
        if prefix_m:
            content = processed[prefix_m.end():].strip()
        else:
            content = processed

        # Also detect dosage form suffix: e.g. "Luliconazole lotion daily"
        suffix_m = re.search(rf"\s+(?:{_DOSAGE_FORMS})\b", content, re.IGNORECASE)
        had_suffix = bool(suffix_m)

        # Pass A: Prescription line with explicit dose: "Telma 40mg od", "Terbinafine 250mg - Daily"
        m = _MED_LINE_RE.search(content)
        if m:
            name_raw = m.group("name").strip()
            # Clean dosage form from name if it was captured as suffix
            name_raw = re.sub(rf"\s+(?:{_DOSAGE_FORMS})\b", "", name_raw, flags=re.IGNORECASE).strip()
            unit = (m.group("unit") or "").lower() or None
            freq_raw = (m.group("freq") or "").lower() or None
            dose2 = m.group("dose2")

            if _is_rejected_name(name_raw):
                continue

            # Without a unit, require a frequency code, a known dosage prefix,
            # or a second combo dose to prevent matching random text
            if not unit and not freq_raw and not dose2 and not had_prefix:
                continue

            try:
                dose = float(m.group("dose"))
            except ValueError:
                continue

            canonical, matched = normalize_medication(name_raw)
            if not canonical or len(name_raw) < 2:
                continue

            frequency = normalize_frequency(freq_raw)

            confidence = 0.65
            if matched:
                confidence += 0.25
            if freq_raw:
                confidence += 0.05
            if had_prefix or had_suffix:
                confidence += 0.05
            if not unit:
                confidence -= 0.1
            confidence = max(0.2, min(confidence, 0.95))

            needs_verification = not matched or confidence < 0.75

            source_text = original_line
            if dose2:
                source_text += f"  [combo dose detected: {dose}/{dose2}{unit or ''}]"

            med_key = (canonical.lower(), dose, unit, frequency)
            if med_key not in seen_names:
                seen_names.add(med_key)
                results.append({
                    "name_raw": name_raw,
                    "name": canonical,
                    "name_matched": matched,
                    "dose": dose,
                    "unit": unit,
                    "frequency_raw": freq_raw,
                    "frequency": frequency,
                    "confidence": round(confidence, 2),
                    "needs_verification": needs_verification,
                    "source_text": source_text,
                })
            continue

        # Pass B: Prescription line without numeric dose: e.g. "Luliconazole lotion daily", "Cap B complex OD"
        # Allowed if dosage prefix or suffix is present OR normalized name matches known formulary.
        m_nodose = _MED_NO_DOSE_RE.search(content)
        if m_nodose:
            name_raw = m_nodose.group("name").strip()
            name_raw = re.sub(rf"\s+(?:{_DOSAGE_FORMS})\b", "", name_raw, flags=re.IGNORECASE).strip()
            freq_raw = (m_nodose.group("freq") or "").lower() or None

            if _is_rejected_name(name_raw):
                continue

            canonical, matched = normalize_medication(name_raw)
            has_form = had_prefix or had_suffix
            if len(name_raw) >= 2 and (matched or has_form):
                frequency = normalize_frequency(freq_raw)
                if matched:
                    confidence = 0.85 if has_form else 0.75
                    needs_verification = False
                else:
                    # Dynamic drug name structurally identified by dosage form
                    confidence = 0.65
                    needs_verification = True

                med_key = (canonical.lower(), None, None, frequency)
                if med_key not in seen_names:
                    seen_names.add(med_key)
                    results.append({
                        "name_raw": name_raw,
                        "name": canonical,
                        "name_matched": matched,
                        "dose": None,
                        "unit": None,
                        "frequency_raw": freq_raw,
                        "frequency": frequency,
                        "confidence": round(confidence, 2),
                        "needs_verification": needs_verification,
                        "source_text": original_line,
                    })
                continue

        # Pass C: Prescription line with explicit dosage-form prefix (Tab, Cap, etc.)
        # but no inline dose or frequency code (e.g. "TAB ECOPSRIN AV", "TAB NEXITO PLUS").
        # These are explicit prescription lines where timing was written separately.
        if had_prefix:
            name_raw = content.strip()
            name_raw = re.sub(rf"\s+(?:{_DOSAGE_FORMS})\b", "", name_raw, flags=re.IGNORECASE).strip()
            name_raw = re.sub(rf"\s+(?:{_FREQ_ALTS})\b.*$", "", name_raw, flags=re.IGNORECASE).strip()
            if not _is_rejected_name(name_raw) and len(name_raw) >= 3:
                canonical, matched = normalize_medication(name_raw)
                if canonical and (matched or len(name_raw) >= 4):
                    confidence = 0.80 if matched else 0.55
                    med_key = (canonical.lower(), None, None, None)
                    if med_key not in seen_names:
                        seen_names.add(med_key)
                        results.append({
                            "name_raw": name_raw,
                            "name": canonical,
                            "name_matched": matched,
                            "dose": None,
                            "unit": None,
                            "frequency_raw": None,
                            "frequency": None,
                            "confidence": round(confidence, 2),
                            "needs_verification": not matched,
                            "source_text": original_line,
                        })

    if form_results:
        return _merge_med_lists(form_results, results)
    return results