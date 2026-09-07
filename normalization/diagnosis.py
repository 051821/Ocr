"""
Diagnosis normalization.

Controlled dictionary today; designed so a SNOMED-CT / ICD-10 / UMLS lookup
can be dropped in later without changing any calling code — just swap out
DIAGNOSIS_MAP / add a lookup() call to an external service inside
normalize_diagnosis().
"""
from __future__ import annotations
import re

# key: normalized form (canonical) -> list of raw surface forms / abbreviations
_DIAGNOSIS_SYNONYMS = {
    "Hypertension": ["htn", "hypertension", "high blood pressure"],
    "Type 2 diabetes mellitus": [
        "t2dm", "type 2 dm", "type ii diabetes", "type 2 diabetes",
        "diabetes mellitus type 2", "niddm",
    ],
    "Type 1 diabetes mellitus": ["t1dm", "type 1 dm", "type i diabetes", "iddm"],
    "Chronic kidney disease": ["ckd", "chronic kidney disease", "chronic renal failure", "crf"],
    "Coronary artery disease": ["cad", "coronary artery disease", "ihd", "ischemic heart disease"],
    "Hypothyroidism": ["hypothyroidism", "hypothyroid"],
    "Dyslipidemia": ["dyslipidemia", "hyperlipidemia"],
    "Osteoarthritis": ["oa", "osteoarthritis"],
    "Chronic obstructive pulmonary disease": ["copd", "chronic obstructive pulmonary disease"],
    "Anemia": ["anemia", "anaemia"],
    "Obesity": ["obesity"],
    "Asthma": ["asthma"],
}

# Build a fast reverse lookup: normalized surface form -> canonical name
_REVERSE = {}
for canonical, synonyms in _DIAGNOSIS_SYNONYMS.items():
    for s in synonyms:
        _REVERSE[s.strip().lower()] = canonical


def _clean(raw: str) -> str:
    raw = raw.strip().lower()
    raw = re.sub(r"[^a-z0-9\s]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def normalize_diagnosis(raw_text: str):
    """
    Returns (canonical_name, matched: bool).
    If no match is found in the controlled dictionary, the cleaned-up
    original text is returned with matched=False so callers can flag it
    for review rather than silently dropping / inventing a diagnosis.
    """
    if not raw_text:
        return None, False

    cleaned = _clean(raw_text)
    if cleaned in _REVERSE:
        return _REVERSE[cleaned], True

    # try a loose "contains" match as a fallback (still controlled — only
    # matches known synonyms, never invents a new diagnosis)
    for surface, canonical in _REVERSE.items():
        if surface and surface in cleaned:
            return canonical, True

    return raw_text.strip(), False


def split_diagnosis_string(raw_text: str):
    """
    Splits a free-text diagnosis field ("Htn t2dm", "HTN, T2DM, CKD") into
    individual diagnosis candidates. Handles common separators used in
    Indian clinical notes: commas, '+', '/', 'and', multiple spaces.
    """
    if not raw_text:
        return []
    parts = re.split(r"[,+/;\n]| and |&", raw_text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    # further split space-separated abbreviation runs like "Htn t2dm ckd"
    expanded = []
    for p in parts:
        tokens = p.split()
        if len(tokens) > 1 and all(len(t) <= 6 for t in tokens):
            expanded.extend(tokens)
        else:
            expanded.append(p)
    return expanded
