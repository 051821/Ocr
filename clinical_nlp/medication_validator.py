"""
Context-based validation for NER medication candidates.

No hardcoded drug names.  A candidate is accepted when the
surrounding prescription text provides enough structural
evidence (dosage form, dose unit, frequency word, etc.).
"""
from __future__ import annotations
import re
from typing import Optional

# -- Structural / context vocabularies ----------------------------------------

_DOSAGE_FORMS = {
    "tab", "tablet", "tablets", "cap", "caps", "capsule", "capsules",
    "syrup", "suspension", "solution", "soln", "injection", "inj",
    "cream", "ointment", "gel", "lotion", "drops", "drop",
    "spray", "powder", "patch", "inhaler", "suppository", "sachet",
}

_FREQ_WORDS = {
    "od", "bd", "tds", "qid", "hs", "stat", "weekly", "monthly",
    "daily", "once", "twice", "thrice",
    "morning", "evening", "night", "after", "before", "food",
}

_DOSE_UNITS = {
    "mg", "mcg", "g", "ml", "iu", "units", "%", "mg/ml", "mcg/ml",
}

_CONTEXT_WORDS = _FREQ_WORDS | {
    "medicine", "medication", "drug", "dose", "dosage",
    "frequency", "duration", "prescribed", "prescription",
    "take", "taking",
}


# -- Helpers ------------------------------------------------------------------

_ADMIN_IDENTIFIERS = {"TMC", "NMC", "UPHC", "OPD", "UHID", "REG", "IPD", "DS-TMC", "FOBDA9CH8", "6E55C6C3"}
_STRUCTURAL_NON_MEDS = {
    "stomach", "syomach", "empty", "empty stomach", "empty syomach",
    "diet", "food", "after food", "before food", "daibetic", "diabetic",
    "notes", "duration", "timing", "complaint", "chief", "impression",
    "center", "advice", "adv", "ref", "refer",
}


def _is_identifier(text: str) -> bool:
    """Reject patient IDs / accession numbers (e.g. 4FOAB62C, 1700012940, DS-TMC-006)."""
    t = text.strip()
    if t.upper() in _ADMIN_IDENTIFIERS:
        return True
    if re.fullmatch(r"\d+", t):
        return True
    if re.fullmatch(r"[A-Z0-9]{6,}", t.upper()):
        letters = sum(c.isalpha() for c in t)
        digits  = sum(c.isdigit() for c in t)
        if letters >= 1 and digits >= 1:
            return True
    if re.match(r"^(?:ds[-_]?tmc|tmc[-_]?\d+|fobda)", t.lower()):
        return True
    return False


def _is_structural(text: str) -> bool:
    """Reject bare dosage-form words (LOTION, TAB) and generic context words."""
    lower = text.lower().strip()
    return lower in _DOSAGE_FORMS or lower in _CONTEXT_WORDS or lower in _STRUCTURAL_NON_MEDS


def _too_short(text: str) -> bool:
    return len(re.sub(r"[^A-Za-z]", "", text)) < 3


def _context_score(source: str, candidate: str, window: int = 120) -> float:
    """
    Score how prescription-like the text surrounding *candidate* looks.
    Returns [0, 1].  Does NOT look up drug names.
    """
    lower_src = source.lower()
    pos = lower_src.find(candidate.lower())
    if pos == -1:
        return 0.0

    ctx = lower_src[max(0, pos - window): pos + len(candidate) + window]
    score = 0.0

    for form in _DOSAGE_FORMS:
        if re.search(rf"\b{re.escape(form)}\b", ctx):
            score += 0.15

    for word in _CONTEXT_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", ctx):
            score += 0.08

    for unit in _DOSE_UNITS:
        if re.search(rf"\b\d+(?:\.\d+)?\s*{re.escape(unit)}\b", ctx):
            score += 0.20

    return min(score, 1.0)


# -- Public API ---------------------------------------------------------------

def validate_ner_medication(
    candidate: str,
    ner_score: float,
    source_text: str,
    min_final_score: float = 0.50,
) -> Optional[float]:
    """
    Validate one NER medication candidate against its surrounding text.

    Returns the combined confidence score (float) if the candidate passes,
    or None if it should be rejected.
    """
    candidate = re.sub(r"^[^\w]+|[^\w]+$", "", candidate.strip())
    candidate = re.sub(r"\s+", " ", candidate).strip()

    if not candidate:                  return None
    if _is_identifier(candidate):     return None
    if _is_structural(candidate):     return None
    if _too_short(candidate):         return None

    ctx_score = _context_score(source_text, candidate)
    final     = (ner_score * 0.60) + (ctx_score * 0.40)

    return round(final, 3) if final >= min_final_score else None
