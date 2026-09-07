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

_WINDOW_CHARS = 40


def _window(text: str, start: int, end: int) -> str:
    lo = max(0, start - _WINDOW_CHARS)
    return text[lo:end].lower()


def detect_assertion(text: str, entity_start: int, entity_end: int) -> str:
    """
    Returns one of: 'present', 'negated', 'historical', 'suspected'.
    Checks the text immediately preceding the entity (the standard NegEx
    window) for trigger phrases. Historical/suspected checked before
    negation so "no history of X" doesn't get mis-tagged past historical.
    """
    ctx = _window(text, entity_start, entity_end)

    for trig in HISTORICAL_TRIGGERS:
        if re.search(re.escape(trig), ctx):
            return "historical"

    for trig in SUSPECTED_TRIGGERS:
        if re.search(re.escape(trig), ctx):
            return "suspected"

    for trig in NEGATION_TRIGGERS:
        if re.search(trig, ctx):
            return "negated"

    return "present"
