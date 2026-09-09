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
    r"(?:tab|cap|inj|syp|syrup|susp|drops|ointment|oint|cream|gel|lotion|"
    r"ab|ao|tb|ta|t\.|c\.|cap\.|tab\.)"
)

_FREQ_ALTS = (
    r"od|bd|tds|tid|bid|qid|hs|sos|prn|stat|weekly|"
    r"1-0-0|1-1-1|1-0-1|0-0-1|1-1-1-1|1\s*0\s*0|1\s*0\s*1|1\s*1\s*1|0\s*0\s*1|"
    r"bbr|bbd|b\.d|b\.d\.|b/d|o\.d|o\.d\.|t\.d\.s|t\.d\.s\."
)

_MED_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9\-\.\s/]{0,35}?)\s+"
    r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml|iu|u)?(?![/\w])"
    rf"(?:\s+(?P<dose2>\d+(?:\.\d+)?)\s*(?P<unit2>mg|mcg|g|ml|iu|u)?(?![/\w]))?"
    rf"(?:\s+(?P<freq>{_FREQ_ALTS}))?",
    re.IGNORECASE,
)

_MED_NO_DOSE_RE = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9\-\.\s/]{0,35}?)\s+"
    rf"(?P<freq>{_FREQ_ALTS})(?![/\w])",
    re.IGNORECASE,
)

# Words that identify a line as a laboratory result, not a medication
_LAB_KEYWORDS = {
    "glucose", "creatinine", "albumin", "cholesterol", "hemoglobin",
    "haemoglobin", "urea", "sodium", "potassium", "bilirubin", "sgot",
    "sgpt", "ast", "alt", "hba1c", "tsh", "triglycerides", "sugar",
    "protein", "phosphatase", "phosphatse", "urine", "routine",
    "investigation", "biochemistry", "microscopic", "ref. interval",
}


def _preprocess_med_line(line: str) -> str:
    """Fix common OCR word-glue issues in prescription lines."""
    # Split attached drug+vitamin/digit: e.g. "ealciumd3" -> "ealcium d3"
    line = re.sub(r"(?i)\b(ealcium|calcium)d(\d)", r"\1 d\2", line)
    # Split d3 followed by dose: e.g. "d31000mg" -> "d3 1000 mg"
    line = re.sub(r"(?i)\bd(\d)(\d{2,})", r"d\1 \2", line)
    # Split dose and unit: e.g. "1000mg" -> "1000 mg", "40mg" -> "40 mg"
    line = re.sub(r"(?i)(\d+)(mg|mcg|g|ml|iu)\b", r"\1 \2", line)
    return line.strip()


def extract_medications(text: str):
    """
    Returns a list of dicts:
    {name_raw, name, matched, dose, unit, frequency_raw, frequency,
     confidence, needs_verification, source_text}
    """
    results = []
    if not text:
        return results

    seen_names = set()

    for original_line in text.splitlines():
        line = original_line.strip()
        if not line:
            continue

        if any(kw in line.lower() for kw in _LAB_KEYWORDS):
            continue

        processed = _preprocess_med_line(line)

        # Strip dosage prefix from beginning if present
        prefix_m = re.match(rf"^(?:{_DOSAGE_FORMS}\.?\s+)", processed, re.IGNORECASE)
        had_prefix = bool(prefix_m)
        if prefix_m:
            content = processed[prefix_m.end():].strip()
        else:
            content = processed

        # Pass A: Prescription line with explicit dose
        m = _MED_LINE_RE.search(content)
        if m:
            name_raw = m.group("name").strip()
            unit = (m.group("unit") or "").lower() or None
            freq_raw = (m.group("freq") or "").lower() or None
            dose2 = m.group("dose2")

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
            if had_prefix:
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

        # Pass B: Prescription line without numeric dose (e.g. "ab b complex od", "cap vitamin d3 weekly")
        # Allowed if dosage prefix was present OR normalized name matches known formulary
        m_nodose = _MED_NO_DOSE_RE.search(content)
        if m_nodose:
            name_raw = m_nodose.group("name").strip()
            freq_raw = (m_nodose.group("freq") or "").lower() or None

            canonical, matched = normalize_medication(name_raw)
            if matched and len(name_raw) >= 1:
                frequency = normalize_frequency(freq_raw)
                confidence = 0.85 if had_prefix else 0.75

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
                        "needs_verification": False,
                        "source_text": original_line,
                    })

    return results