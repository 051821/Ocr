"""
Diagnosis normalization.

Controlled dictionary today; designed so a SNOMED-CT / ICD-10 / UMLS lookup
can be dropped in later without changing any calling code — just swap out
DIAGNOSIS_MAP / add a lookup() call to an external service inside
normalize_diagnosis().
"""
from __future__ import annotations
import re
from rapidfuzz import process, fuzz

# key: normalized form (canonical) -> list of raw surface forms / abbreviations
_DIAGNOSIS_SYNONYMS = {
    "Hypertension": ["htn", "hypertension", "high blood pressure", "essential hypertension"],
    "Type 2 diabetes mellitus": [
        "t2dm", "12dm", "type 2 dm", "type-2 dm", "type ii diabetes", "type 2 diabetes",
        "diabetes mellitus type 2", "niddm", "dm2", "dm 2", "dm-2", "t2 d",
    ],
    "Type 1 diabetes mellitus": ["t1dm", "type 1 dm", "type i diabetes", "iddm", "dm1", "dm 1"],
    "Chronic kidney disease": ["ckd", "chronic kidney disease", "chronic renal failure", "crf"],
    "Coronary artery disease": ["cad", "coronary artery disease", "ihd", "ischemic heart disease"],
    "Hypothyroidism": ["hypothyroidism", "hypothyroid"],
    "Dyslipidemia": ["dyslipidemia", "hyperlipidemia", "high cholesterol"],
    "Osteoarthritis": ["oa", "osteoarthritis", "degenerative joint disease"],
    "Chronic obstructive pulmonary disease": ["copd", "chronic obstructive pulmonary disease"],
    "Anemia": ["anemia", "anaemia", "iron deficiency anemia"],
    "Obesity": ["obesity", "overweight"],
    "Asthma": ["asthma", "bronchial asthma"],
    "Gastritis": ["gastritis", "acute gastritis", "chronic gastritis"],
    "Gastroesophageal reflux disease": ["gerd", "gastroesophageal reflux disease", "acid reflux", "reflux"],
    "Acid peptic disease": ["apd", "acid peptic disease", "peptic ulcer", "peptic ulcer disease", "pud"],
    "Fatty liver disease": ["fatty liver", "nafld", "hepatic steatosis", "fatty liver disease"],
    "Urinary tract infection": ["uti", "urinary tract infection"],
    "Spondylosis": ["spondylosis", "cervical spondylosis", "lumbar spondylosis"],
    "Neuropathy": ["neuropathy", "diabetic neuropathy", "peripheral neuropathy"],
    "Retinopathy": ["retinopathy", "diabetic retinopathy"],
    "Nephropathy": ["nephropathy", "diabetic nephropathy"],
    "Pneumonia": ["pneumonia", "bronchopneumonia"],
    "Bronchitis": ["bronchitis", "acute bronchitis"],
    "Headache": ["headache", "migraine"],
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


def normalize_diagnosis(raw_text: str) -> tuple[str | None, bool]:
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

    # Check whole word boundary match (never substring match without boundary!)
    for surface, canonical in _REVERSE.items():
        if surface and re.search(rf"\b{re.escape(surface)}\b", cleaned):
            return canonical, True

    # Fuzzy match with RapidFuzz for longer condition names (len >= 5)
    surfaces = [s for s in _REVERSE.keys() if len(s) >= 5]
    match = process.extractOne(cleaned, surfaces, scorer=fuzz.ratio, score_cutoff=85)
    if match:
        best_surface, score, _ = match
        if abs(len(cleaned) - len(best_surface)) <= 3:
            return _REVERSE[best_surface], True

    return raw_text.strip().title(), False


def split_diagnosis_string(raw_text: str) -> list[str]:
    """
    Splits a free-text diagnosis field ("Htn t2dm", "HTN, T2DM, CKD") into
    individual diagnosis candidates. Handles common separators used in
    Indian clinical notes: commas, '+', '/', 'and', multiple spaces.
    """
    if not raw_text:
        return []
    parts = re.split(r"[,+/;\n]| and |&", raw_text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    # further split space-separated abbreviation runs like "Htn t2dm ckd" or "Htn 12dm"
    expanded = []
    for p in parts:
        tokens = p.split()
        if len(tokens) > 1 and all(len(t) <= 6 for t in tokens):
            expanded.extend(tokens)
        else:
            expanded.append(p)
    return expanded

