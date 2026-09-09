"""
"""
from __future__ import annotations
import re
from datetime import datetime

_DATE_PATTERNS = [
    (r"\b(\d{4})-(\d{2})-(\d{2})\b", "%Y-%m-%d"),
    (r"\b(\d{2})/(\d{2})/(\d{4})\b", "%d/%m/%Y"),
    (r"\b(\d{2})-(\d{2})-(\d{4})\b", "%d-%m-%Y"),
]


def extract_explicit_dates(text: str):
    """Returns a list of ISO date strings found explicitly in the text."""
    found = []
    if not text:
        return found
    for pattern, fmt in _DATE_PATTERNS:
        for m in re.finditer(pattern, text):
            try:
                dt = datetime.strptime(m.group(0), fmt)
                found.append(dt.date().isoformat())
            except ValueError:
                continue
    return found


def resolve_event_date(text: str, visit_date: str | None) -> str | None:
    """
    Primary rule from the spec: visit_date is used unless a more specific
    event_date is available. We only override visit_date with an explicit
    in-text date when exactly one unambiguous date is found near the
    relevant text; otherwise we conservatively fall back to visit_date.
    """
    explicit = extract_explicit_dates(text)
    if len(explicit) == 1:
        return explicit[0]
    return visit_date
