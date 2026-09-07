"""
Medication extraction from free text such as:
    Tab amlodipine 5 mg od
    Cap pantoprazole 40mg 1-0-0 before food
    ecopsrin av 75 20 hs

OCR is unreliable, so anything that doesn't cleanly match a known
medication name is still returned, but with matched=False and
needs_verification=True rather than being silently dropped or guessed at.
"""
from __future__ import annotations
import re
from normalization.medication import normalize_medication, normalize_frequency

_DOSAGE_FORMS = r"(?:tab|cap|inj|syp|syrup|susp|drops)"

_MED_LINE_RE = re.compile(
    rf"(?:{_DOSAGE_FORMS}\.?\s+)?"
    r"(?P<name>[A-Za-z][A-Za-z\-]{2,30}(?:\s+[A-Za-z\-]{2,20}){0,2}?)\s+"
    r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml|iu)(?![/\w])"
    r"(?:\s+(?P<freq>od|bd|tds|tid|bid|qid|hs|sos|prn|stat|1-0-0|1-1-1|1-0-1|weekly))?",
    re.IGNORECASE,
)

# Words that identify a line as a laboratory result, not a medication —
# prevents e.g. "Random glucose 310.87 mg/dL" from being misread as a drug.
_LAB_KEYWORDS = {
    "glucose", "creatinine", "albumin", "cholesterol", "hemoglobin",
    "haemoglobin", "urea", "sodium", "potassium", "bilirubin", "sgot",
    "sgpt", "ast", "alt", "hba1c", "tsh", "triglycerides", "sugar",
}


def extract_medications(text: str):
    """
    Returns a list of dicts:
    {name_raw, name, matched, dose, unit, frequency_raw, frequency,
     confidence, needs_verification, source_text}
    """
    results = []
    if not text:
        return results

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if any(kw in line.lower() for kw in _LAB_KEYWORDS):
            continue

        for m in _MED_LINE_RE.finditer(line):
            name_raw = m.group("name").strip()
            dose = float(m.group("dose"))
            unit = m.group("unit").lower()
            freq_raw = (m.group("freq") or "").lower() or None

            canonical, matched = normalize_medication(name_raw)
            frequency = normalize_frequency(freq_raw)

            confidence = 0.5
            if matched:
                confidence += 0.3
            if freq_raw:
                confidence += 0.1
            confidence = min(confidence, 0.95)

            needs_verification = not matched or confidence < 0.7

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
                "source_text": line,
            })

    return results
