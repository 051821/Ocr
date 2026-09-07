"""
Laboratory test name normalization. Designed so LOINC can be plugged
into normalize_lab_name() later.
"""
from __future__ import annotations
import re

_LAB_SYNONYMS = {
    "AST": ["sgot", "ast", "aspartate aminotransferase"],
    "ALT": ["sgpt", "alt", "alanine aminotransferase"],
    "Creatinine": ["creatinine", "s creatinine", "serum creatinine"],
    "Random blood glucose": ["blood sugar random", "rbs", "random glucose",
                              "random blood glucose", "random blood sugar"],
    "Fasting blood glucose": ["fbs", "fasting blood sugar", "fasting glucose"],
    "HbA1c": ["hba1c", "glycated hemoglobin", "glycosylated hemoglobin"],
    "Albumin": ["albumin", "s albumin", "serum albumin"],
    "Hemoglobin": ["hb", "hemoglobin", "haemoglobin"],
    "Urea": ["urea", "blood urea"],
    "Total cholesterol": ["total cholesterol", "cholesterol total"],
    "LDL": ["ldl", "ldl cholesterol"],
    "HDL": ["hdl", "hdl cholesterol"],
    "Triglycerides": ["tg", "triglycerides"],
    "TSH": ["tsh", "thyroid stimulating hormone"],
    "Sodium": ["na", "sodium"],
    "Potassium": ["k", "potassium"],
}

_REVERSE = {}
for canonical, synonyms in _LAB_SYNONYMS.items():
    for s in synonyms:
        _REVERSE[s.strip().lower()] = canonical


def normalize_lab_name(raw_name: str):
    """Returns (canonical_name, matched: bool)."""
    if not raw_name:
        return None, False
    cleaned = re.sub(r"[^a-z0-9\s]", " ", raw_name.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned in _REVERSE:
        return _REVERSE[cleaned], True
    for surface, canonical in _REVERSE.items():
        if surface and surface == cleaned:
            return canonical, True
    return raw_name.strip(), False
