"""
Medication name normalization + frequency code parsing.
Designed so RxNorm can be plugged into normalize_medication() later.
"""
from __future__ import annotations
import re

_MED_SYNONYMS = {
    "Amlodipine": ["amlodipine", "amlong", "amlopres"],
    "Telmisartan": ["telmisartan", "telma"],
    "Metformin": ["metformin", "glycomet", "glucophage"],
    "Atorvastatin": ["atorvastatin", "atorva"],
    "Aspirin": ["aspirin", "ecosprin", "ecopsrin", "asa"],
    "Clopidogrel": ["clopidogrel", "clopilet"],
    "Losartan": ["losartan"],
    "Glimepiride": ["glimepiride", "amaryl"],
    "Pantoprazole": ["pantoprazole", "pan", "pantocid"],
    "Levothyroxine": ["levothyroxine", "thyronorm", "eltroxin"],
}

_REVERSE = {}
for canonical, synonyms in _MED_SYNONYMS.items():
    for s in synonyms:
        _REVERSE[s.strip().lower()] = canonical

_FREQUENCY_MAP = {
    "od": "once_daily", "1-0-0": "once_daily", "hs": "once_daily_night",
    "bd": "twice_daily", "1-0-1": "twice_daily", "bid": "twice_daily",
    "tds": "thrice_daily", "tid": "thrice_daily", "1-1-1": "thrice_daily",
    "qid": "four_times_daily",
    "sos": "as_needed", "prn": "as_needed",
    "stat": "immediately",
    "weekly": "weekly", "od weekly": "weekly",
}


def normalize_medication(raw_name: str):
    """Returns (canonical_name, matched: bool)."""
    if not raw_name:
        return None, False
    cleaned = re.sub(r"[^a-z0-9\s]", "", raw_name.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned in _REVERSE:
        return _REVERSE[cleaned], True
    for surface, canonical in _REVERSE.items():
        if surface and re.search(rf"\b{re.escape(surface)}\b", cleaned):
            return canonical, True
    # not in dictionary -> return title-cased raw token, flagged unmatched
    return raw_name.strip().title(), False


def normalize_frequency(raw_freq: str):
    if not raw_freq:
        return None
    key = raw_freq.strip().lower()
    return _FREQUENCY_MAP.get(key, key or None)
