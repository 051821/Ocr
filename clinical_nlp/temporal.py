"""
"""
from __future__ import annotations
import re
from datetime import datetime

_DATE_FORMATS = [
    # ISO: 2026-08-06, 2026/08/06
    (re.compile(r"\b(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\b"), "%Y-%m-%d"),
    # Standard DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    (re.compile(r"\b(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{4})\b"), "%d/%m/%Y"),
    # 2-digit year: 24/8/20, 21/07/26, 24-08-20
    (re.compile(r"\b(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{2})\b"), "%d/%m/%y"),
]

_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_TEXT_MONTH_RE = re.compile(
    r"\b(?P<day>\d{1,2})[-/\s](?P<mon>[A-Za-z]{3,9})[-/\s](?P<year>\d{2,4})\b",
    re.IGNORECASE,
)


def extract_explicit_dates(text: str):
    """Returns a list of ISO date strings found explicitly in the text."""
    found = []
    if not text:
        return found

    # 1. Textual month dates: 21-Jul-2026, 21 July 2026
    for m in _TEXT_MONTH_RE.finditer(text):
        mon_str = m.group("mon")[:3].lower()
        if mon_str in _MONTH_NAMES:
            try:
                day = int(m.group("day"))
                mon = _MONTH_NAMES[mon_str]
                yr = int(m.group("year"))
                if yr < 100:
                    yr += 2000
                dt = datetime(yr, mon, day)
                found.append(dt.date().isoformat())
            except ValueError:
                pass

    # 2. Numeric dates
    for regex, fmt in _DATE_FORMATS:
        for m in regex.finditer(text):
            # Normalize separators to match format string
            raw = m.group(0)
            if fmt == "%Y-%m-%d":
                parts = re.split(r"[/\-]", raw)
                try:
                    dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                    found.append(dt.date().isoformat())
                except ValueError:
                    pass
            elif fmt == "%d/%m/%Y":
                parts = re.split(r"[/\.-]", raw)
                try:
                    dt = datetime(int(parts[2]), int(parts[1]), int(parts[0]))
                    found.append(dt.date().isoformat())
                except ValueError:
                    pass
            elif fmt == "%d/%m/%y":
                parts = re.split(r"[/\.-]", raw)
                try:
                    yr = int(parts[2]) + 2000
                    dt = datetime(yr, int(parts[1]), int(parts[0]))
                    found.append(dt.date().isoformat())
                except ValueError:
                    pass

    return list(dict.fromkeys(found))


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
