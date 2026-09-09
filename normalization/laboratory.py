"""
Laboratory test name normalization. Designed so LOINC can be plugged
into normalize_lab_name() later.
"""
from __future__ import annotations
import re
from rapidfuzz import process, fuzz

_LAB_SYNONYMS = {
    # --- Liver & Enzymes ---
    "AST": ["sgot", "ast", "aspartate aminotransferase", "serum sgot"],
    "ALT": ["sgpt", "alt", "alanine aminotransferase", "serum sgpt"],
    "Alkaline phosphatase": [
        "alkaline phosphatase", "alp", "serum alkaline phosphatase",
        "alkaline phosphatse", "alk phosphatase",
    ],
    "Bilirubin total": [
        "bilirubin total", "bilirubin, total", "serum bilirubin, total",
        "serum bilirubin total", "total bilirubin", "s bilirubin total",
    ],
    "Bilirubin direct": [
        "bilirubin direct", "bilirubin, direct", "serum bilirubin, direct",
        "serum bilirubin direct", "direct bilirubin", "conjugated bilirubin",
    ],
    "Bilirubin indirect": [
        "bilirubin indirect", "bilirubin, indirect", "serum bilirubin, indirect",
        "serum bilirubin indirect", "indirect bilirubin", "unconjugated bilirubin",
    ],
    "Total protein": ["total protein", "serum total protein", "s total protein"],
    "Albumin": ["albumin", "s albumin", "serum albumin"],
    "Globulin": ["globulin", "serum globulin", "s globulin"],
    "Albumin globulin ratio": ["a g ratio", "ag ratio", "a/g ratio", "albumin globulin ratio"],

    # --- Renal Function & Electrolytes ---
    "Creatinine": ["creatinine", "s creatinine", "serum creatinine"],
    "Urea": ["urea", "blood urea", "serum blood urea", "serum urea"],
    "Blood urea nitrogen": ["bun", "blood urea nitrogen", "bun blood urea nitrogen"],
    "Uric acid": ["uric acid", "serum uric acid", "s uric acid"],
    "Sodium": ["sodium", "serum sodium", "s sodium", "na+"],
    "Potassium": ["potassium", "serum potassium", "s potassium", "k+"],
    "Chloride": ["chloride", "serum chloride", "cl-"],
    "Calcium": ["serum calcium", "s calcium", "calcium total"],
    "Phosphorus": ["phosphorus", "serum phosphorus", "phosphate"],

    # --- Glucose & Diabetes ---
    "Random blood glucose": [
        "blood sugar random", "rbs", "random glucose",
        "random blood glucose", "random blood sugar", "blood glucose random",
    ],
    "Fasting blood glucose": [
        "fbs", "fasting blood sugar", "fasting glucose", "blood glucose fasting",
    ],
    "Postprandial blood glucose": [
        "ppbs", "post prandial blood sugar", "postprandial blood glucose", "pp glucose",
    ],
    "HbA1c": [
        "hba1c", "glycated hemoglobin", "glycosylated hemoglobin", "glycosylated hemoglobin hba1c",
    ],

    # --- Lipid Profile ---
    "Total cholesterol": ["total cholesterol", "cholesterol total", "serum cholesterol"],
    "LDL": ["ldl", "ldl cholesterol", "ldl direct"],
    "HDL": ["hdl", "hdl cholesterol"],
    "Triglycerides": ["tg", "triglycerides", "serum triglycerides"],
    "VLDL": ["vldl", "vldl cholesterol"],

    # --- Thyroid Profile ---
    "TSH": ["tsh", "thyroid stimulating hormone", "ultra sensitive tsh"],
    "Free T3": ["free t3", "ft3"],
    "Free T4": ["free t4", "ft4"],

    # --- CBC / Complete Blood Count ---
    "Hemoglobin": ["hb", "hemoglobin", "haemoglobin", "hgb"],
    "RBC": ["rbc", "rbc count", "red blood cells", "total rbc count", "total rbc"],
    "WBC": [
        "wbc", "wbc count", "total leucocyte count", "total leukocyte count",
        "tlc", "white blood cells", "leucocyte count",
    ],
    "Platelets": ["platelets", "platelet count", "total platelet count", "plt"],
    "Packed cell volume": ["pcv", "packed cell volume", "hematocrit", "haematocrit", "hct"],
    "MCV": ["mcv", "mean corpuscular volume"],
    "MCH": ["mch", "mean corpuscular hemoglobin"],
    "MCHC": ["mchc", "mean corpuscular hemoglobin concentration"],
    "RDW": ["rdw", "red cell distribution width", "rdw cv", "rdw sd"],
    "Neutrophils": ["neutrophils", "polymorphs", "neutrophil count", "poly"],
    "Lymphocytes": ["lymphocytes", "lymphocyte count", "lympho"],
    "Monocytes": ["monocytes", "monocyte count", "mono"],
    "Eosinophils": ["eosinophils", "eosinophil count", "eosino"],
    "Basophils": ["basophils", "basophil count", "baso"],
    "ESR": ["esr", "erythrocyte sedimentation rate"],

    # --- Urine Routine Examination ---
    "Urine quantity": ["quantity", "urine quantity", "volume", "urine volume"],
    "Urine colour": ["colour", "color", "urine colour", "urine color"],
    "Urine appearance": ["appearance", "urine appearance", "transparency"],
    "Urine pH": ["ph", "urine ph"],
    "Urine specific gravity": [
        "specific gravity", "sp gravity", "urine specific gravity", "sp gr", "sp. gravity",
    ],
    "Urine albumin": ["urine albumin", "albumin urine", "urine protein", "protein urine"],
    "Urine glucose": ["urine glucose", "glucose urine", "urine sugar", "sugar urine"],
    "Urine ketone": ["ketone", "ketones", "urine ketones", "urine ketone", "acetone"],
    "Urine blood": ["urine blood", "blood urine", "occult blood"],
    "Urine bile salts": ["bile salts", "urine bile salts"],
    "Urine bile pigments": ["bile pigments", "urine bile pigments"],
    "Urine urobilinogen": ["urobilinogen", "urine urobilinogen"],
    "Urine nitrite": ["nitrite", "urine nitrite"],
    "Pus cells": ["pus cells", "pus cell", "urine pus cells", "pus cells /hpf"],
    "Epithelial cells": ["epithelial cells", "epithelial cell", "urine epithelial cells"],
    "Urine RBC": ["urine rbc", "rbc urine", "red blood cells urine"],
    "Crystals": ["crystal", "crystals", "urine crystals", "calcium oxalate"],
    "Casts": ["casts", "urine casts", "hyaline casts", "granular casts"],
    "Bacteria": ["bacteria", "urine bacteria"],
    "Amorphous deposit": [
        "amorphous deposit", "amorphous deposits", "amorphous urates", "amorphous phosphates",
    ],
}

_REVERSE = {}
for canonical, synonyms in _LAB_SYNONYMS.items():
    for s in synonyms:
        _REVERSE[s.strip().lower()] = canonical


def normalize_lab_name(raw_name: str, in_urine_context: bool = False) -> tuple[str | None, bool]:
    """Returns (canonical_name, matched: bool)."""
    if not raw_name:
        return None, False

    cleaned = re.sub(r"[^a-z0-9\s]", " ", raw_name.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Contextual check for urine reports:
    # If in urine context, bare terms like "Albumin", "Glucose", "Blood", "Protein", "Quantity", "Colour"
    # map to Urine-specific tests.
    if in_urine_context:
        urine_overrides = {
            "albumin": "Urine albumin",
            "protein": "Urine albumin",
            "glucose": "Urine glucose",
            "sugar": "Urine glucose",
            "blood": "Urine blood",
            "rbc": "Urine RBC",
            "quantity": "Urine quantity",
            "colour": "Urine colour",
            "color": "Urine colour",
            "appearance": "Urine appearance",
            "ph": "Urine pH",
            "specific gravity": "Urine specific gravity",
            "ketone": "Urine ketone",
            "ketones": "Urine ketone",
            "pus cells": "Pus cells",
            "epithelial cells": "Epithelial cells",
            "crystal": "Crystals",
            "crystals": "Crystals",
            "bacteria": "Bacteria",
            "amorphous deposit": "Amorphous deposit",
        }
        if cleaned in urine_overrides:
            return urine_overrides[cleaned], True

    if cleaned in _REVERSE:
        return _REVERSE[cleaned], True

    # Exact equality across reversed surfaces
    for surface, canonical in _REVERSE.items():
        if surface and surface == cleaned:
            return canonical, True

    # Fuzzy match with RapidFuzz for slight OCR corruptions (e.g. 'alkaline phosphatse')
    surfaces = list(_REVERSE.keys())
    match = process.extractOne(cleaned, surfaces, scorer=fuzz.ratio, score_cutoff=85)
    if match:
        best_surface, score, _ = match
        if abs(len(cleaned) - len(best_surface)) <= 3 and len(best_surface) >= 4:
            return _REVERSE[best_surface], True

    return raw_name.strip(), False