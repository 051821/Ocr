"""
Assertion / negation / temporality detection.

Rule-based NegEx-style approach: look at a window of text around the
matched entity for trigger phrases. This is intentionally conservative —
it only flags what it has direct textual evidence for.
"""
from __future__ import annotations
import re

NEGATION_TRIGGERS = [
    "denies", "denied", "no evidence of", "no signs of", "not present",
    "negative for", "ruled out", "r/o negative", "absent", "without",
    "no h/o", "no history of", r"\bno\b",
]

HISTORICAL_TRIGGERS = [
    "h/o", "history of", "known case of", "k/c/o", "previously diagnosed",
    "old", "prior", "past history",
]

SUSPECTED_TRIGGERS = [
    "possible", "probable", "suspected", "likely", "r/o", "rule out",
    "?", "susp",
]

# Generic explanatory / reference language found in lab reports and
# guideline text — e.g. "High Alkaline Phosphatase may indicate liver
# disease or certain bone disorders". This describes what a finding
# *can* mean in general, not a statement about this specific patient, so
# any diagnosis/entity caught in this kind of sentence should never be
# scored as a confirmed present-tense finding — regardless of which
# disease or drug name it happens to be.
REFERENCE_TEXT_TRIGGERS = [
    "may indicate", "can indicate", "could indicate", "suggestive of",
    "differential includes", "differential diagnosis", "is associated with",
    "consistent with", "may suggest", "can be caused by", "may occur in",
    "seen in", "commonly caused by", "reference:", "guideline", "guidelines",
]

_WINDOW_CHARS = 40

# Matches a sentence-ending "." only when NOT flanked by digits, so
# decimal values like "1.25 mg/dL" or "6.91 g/dL" — extremely common in
# lab reports — are never mistaken for a sentence boundary. Also treats
# newlines as boundaries, since OCR'd reports frequently have no
# terminal punctuation at all between logical lines.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<!\d)\.(?!\d)|\n")


def _window(text: str, start: int, end: int) -> str:
    lo = max(0, start - _WINDOW_CHARS)
    return text[lo:end].lower()


def _containing_sentence(text: str, start: int, end: int) -> str:
    """
    Returns the sentence/line containing the entity, bounded by the
    nearest real sentence-ending punctuation or newline on either side
    (decimal points are never treated as boundaries — see
    _SENTENCE_BOUNDARY_RE). Used for checks, like reference-text
    triggers, where the relevant cue can sit further from the entity
    than the standard NegEx window covers.
    """
    lo = 0
    for m in _SENTENCE_BOUNDARY_RE.finditer(text, 0, start):
        lo = m.end()

    hi_match = _SENTENCE_BOUNDARY_RE.search(text, end)
    hi = hi_match.start() if hi_match else len(text)

    return text[lo:hi].lower()


def detect_assertion(text: str, entity_start: int, entity_end: int) -> str:
    """
    Returns one of: 'present', 'negated', 'historical', 'suspected'.
    Checks the text immediately preceding the entity (the standard NegEx
    window) for trigger phrases. Historical/suspected checked before
    negation so "no history of X" doesn't get mis-tagged past historical.

    Reference/explanatory text (e.g. "Alkaline Phosphatase may indicate
    liver disease or certain bone disorders") is checked against the
    whole containing sentence and, if matched, is treated as 'suspected'
    — this is a statement about what a finding *can* mean in general,
    not a confirmed observation about this patient, and the check is
    generic (any disease/drug name), not tied to specific terms.
    """
    ctx = _window(text, entity_start, entity_end)

    for trig in HISTORICAL_TRIGGERS:
        if re.search(re.escape(trig), ctx):
            return "historical"

    sentence_ctx = _containing_sentence(text, entity_start, entity_end)
    for trig in REFERENCE_TEXT_TRIGGERS:
        if re.search(re.escape(trig), sentence_ctx):
            return "suspected"

    for trig in SUSPECTED_TRIGGERS:
        if re.search(re.escape(trig), ctx):
            return "suspected"

    for trig in NEGATION_TRIGGERS:
        if re.search(trig, ctx):
            return "negated"

    return "present"


def is_in_reference_text(text: str, entity_start: int, entity_end: int) -> bool:
    """
    Returns True if the entity is situated inside an explanatory or guideline
    sentence describing what a test can indicate in general, rather than a
    patient diagnosis.
    """
    sentence_ctx = _containing_sentence(text, entity_start, entity_end)
    for trig in REFERENCE_TEXT_TRIGGERS:
        if re.search(re.escape(trig), sentence_ctx):
            return True
    return False