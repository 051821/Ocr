"""
Laboratory result extraction.

Deliberately regex/rule-based rather than relying on generic NER, per the
task spec ("do not assume NER alone can reliably extract laboratory
values"). Handles the common Indian lab-report free-text patterns:

    Serum Creatinine : 1.25 mg/dL
    Reference: 0.6-1.2

    SGOT = 13.5 U/L
    Random glucose  310.87  mg/dL   Ref 70-140
"""
from __future__ import annotations
import re
from normalization.laboratory import normalize_lab_name

_LAB_LINE_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 /()\-]{1,40}?)"
    r"(?:\s*[:=]\s*|\s+)"                      # mandatory separator: prevents
                                                 # digits inside the name (e.g.
                                                 # "HbA1c") being mistaken for
                                                 # the start of the value
    r"(?P<value>-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mg/dl|g/dl|u/l|iu/l|mmol/l|meq/l|%|mmhg|ng/ml|miu/l|mcg/dl|/ul)?",
    re.IGNORECASE,
)

_REF_RANGE_RE = re.compile(
    r"ref(?:erence)?\.?:?\s*(?P<low>-?\d+(?:\.\d+)?)\s*[-to]{1,3}\s*(?P<high>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# words that indicate the "name" match is not actually a lab test
_STOPWORDS = {"date", "age", "sex", "id", "visit", "patient", "doctor", "dr"}


def extract_labs(text: str):
    """
    Returns a list of dicts:
    {name_raw, name, matched, value, unit, reference_low, reference_high,
     abnormal, confidence, source_text}
    """
    results = []
    if not text:
        return results

    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 200:
            continue

        m = _LAB_LINE_RE.search(line)
        if not m:
            continue

        name_raw = m.group("name").strip()
        if name_raw.lower() in _STOPWORDS:
            continue
        # skip if the "name" is itself mostly numeric (false match)
        if re.fullmatch(r"[\d\s./-]+", name_raw):
            continue

        try:
            value = float(m.group("value"))
        except (TypeError, ValueError):
            continue

        unit = (m.group("unit") or "").strip() or None

        ref_match = _REF_RANGE_RE.search(line)
        ref_low = ref_high = None
        abnormal = None
        if ref_match:
            ref_low = float(ref_match.group("low"))
            ref_high = float(ref_match.group("high"))
            abnormal = not (ref_low <= value <= ref_high)

        canonical, matched = normalize_lab_name(name_raw)

        # confidence heuristic: has unit + has reference range = high
        confidence = 0.6
        if unit:
            confidence += 0.15
        if ref_match:
            confidence += 0.15
        if matched:
            confidence += 0.1
        confidence = min(confidence, 0.99)

        results.append({
            "name_raw": name_raw,
            "name": canonical,
            "name_matched": matched,
            "value": value,
            "unit": unit,
            "reference_low": ref_low,
            "reference_high": ref_high,
            "abnormal": abnormal,
            "confidence": round(confidence, 2),
            "source_text": line,
        })

    return results
