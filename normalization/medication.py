"""
Medication name normalization + frequency code parsing.
Designed so RxNorm can be plugged into normalize_medication() later.
"""
from __future__ import annotations
import re
from rapidfuzz import process, fuzz

_MED_SYNONYMS = {
    "Amlodipine": ["amlodipine", "amlong", "amlopres", "amlo"],
    "Telmisartan": ["telmisartan", "telma", "telmikem", "telpres", "telvas"],
    "Metformin": ["metformin", "glycomet", "glucophage", "gluconorm"],
    "Atorvastatin": ["atorvastatin", "atorva", "storvas", "atocor"],
    "Aspirin": ["aspirin", "ecosprin", "ecopsrin", "asa", "aspirin ec"],
    "Clopidogrel": ["clopidogrel", "clopilet", "deplatt"],
    "Losartan": ["losartan", "losar", "repace"],
    "Glimepiride": ["glimepiride", "amaryl", "zoryl"],
    "Pantoprazole": ["pantoprazole", "pan", "pantocid", "pantodac"],
    "Levothyroxine": ["levothyroxine", "thyronorm", "eltroxin"],
    # Analgesics & Pain management
    "Tramadol": ["tramadol", "ultram", "tramazac", "tramasure"],
    "Paracetamol": ["paracetamol", "pcm", "calpol", "dolo", "crocin", "acetaminophen"],
    "Ibuprofen": ["ibuprofen", "brufen", "combiflam"],
    "Diclofenac": ["diclofenac", "voveran"],
    "Aceclofenac": ["aceclofenac", "hifenac", "zerodol"],
    # GI medications
    "Rabeprazole": ["rabeprazole", "razo", "happi", "rabicip"],
    "Omeprazole": ["omeprazole", "omez", "ocid"],
    "Esomeprazole": ["esomeprazole", "nexpro"],
    "Ranitidine": ["ranitidine", "aciloc", "rantac"],
    "Domperidone": ["domperidone", "vomistop"],
    "Ondansetron": ["ondansetron", "emset", "zofran"],
    # Vitamins & Supplements
    "Calcium + Vitamin D3": [
        "calcium d3", "calcium and vitamin d3", "calcium", "ealciumd3", "ealcium",
        "shelcal", "gemcal", "cipcal", "calcitriol", "cholecalciferol",
    ],
    "Vitamin B Complex": [
        "b complex", "b-complex", "vitamin b complex", "vitamin b", "becosules",
        "neurobion", "neurobion forte", "mecobalamin", "methylcobalamin",
        "folvite", "folic acid", "supradyn", "multivitamin", "multivitamins",
    ],
    # Cardiovascular
    "Atenolol": ["atenolol", "betacard", "aten"],
    "Metoprolol": ["metoprolol", "betaloc", "metolar"],
    "Cilnidipine": ["cilnidipine", "cilacar"],
    "Enalapril": ["enalapril", "envas"],
    "Ramipril": ["ramipril", "cardace"],
    "Rosuvastatin": ["rosuvastatin", "rosuvas", "rosave"],
    # Antidiabetics
    "Glibenclamide": ["glibenclamide", "daonil"],
    "Glipizide": ["glipizide", "minidiab"],
    "Teneligliptin": ["teneligliptin", "tenepure", "zita"],
    "Vildagliptin": ["vildagliptin", "galvus"],
    "Sitagliptin": ["sitagliptin", "januvia"],
    "Dapagliflozin": ["dapagliflozin", "forxiga"],
    "Empagliflozin": ["empagliflozin", "jardiance"],
    "Voglibose": ["voglibose", "volibo"],
    "Insulin": ["insulin", "human mixtard", "lantus", "novorapid", "glargine"],
    # Antibiotics
    "Amoxicillin": ["amoxicillin", "mox", "amoxyclav", "augmentin", "moxikind"],
    "Azithromycin": ["azithromycin", "azithral", "azee"],
    "Ciprofloxacin": ["ciprofloxacin", "ciplox", "cifran"],
    "Cefixime": ["cefixime", "taxim o", "taxim-o", "zifi"],
    "Cefuroxime": ["cefuroxime", "ceftum"],
    "Metronidazole": ["metronidazole", "flagyl"],
    # Respiratory & Allergy
    "Montelukast": ["montelukast", "montair"],
    "Levocetirizine": ["levocetirizine", "levocet", "teczine", "cetirizine"],
    "Salbutamol": ["salbutamol", "asthalin"],
}

# Build a fast reverse lookup: normalized surface form -> canonical name
_REVERSE = {}
for canonical, synonyms in _MED_SYNONYMS.items():
    for s in synonyms:
        _REVERSE[s.strip().lower()] = canonical

_FREQUENCY_MAP = {
    "od": "once_daily", "o.d": "once_daily", "o.d.": "once_daily", "od.": "once_daily",
    "1-0-0": "once_daily", "1 0 0": "once_daily", "od daily": "once_daily",
    "hs": "once_daily_night", "h.s": "once_daily_night", "h.s.": "once_daily_night",
    "hs.": "once_daily_night", "0-0-1": "once_daily_night", "0 0 1": "once_daily_night",
    "bd": "twice_daily", "b.d": "twice_daily", "b.d.": "twice_daily", "bd.": "twice_daily",
    "b/d": "twice_daily", "1-0-1": "twice_daily", "1 0 1": "twice_daily", "bid": "twice_daily",
    "bbr": "twice_daily", "bbd": "twice_daily",  # common OCR typos for bd
    "tds": "thrice_daily", "tid": "thrice_daily", "1-1-1": "thrice_daily", "1 1 1": "thrice_daily",
    "t.d.s": "thrice_daily", "t.d.s.": "thrice_daily",
    "qid": "four_times_daily", "1-1-1-1": "four_times_daily",
    "sos": "as_needed", "s.o.s": "as_needed", "s.o.s.": "as_needed", "prn": "as_needed",
    "stat": "immediately",
    "weekly": "weekly", "od weekly": "weekly", "once a week": "weekly",
}


def normalize_medication(raw_name: str) -> tuple[str | None, bool]:
    """Returns (canonical_name, matched: bool)."""
    if not raw_name:
        return None, False

    cleaned = re.sub(r"[^a-z0-9\s]", " ", raw_name.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned in _REVERSE:
        return _REVERSE[cleaned], True

    # Exact word-boundary search
    for surface, canonical in _REVERSE.items():
        if surface and re.search(rf"\b{re.escape(surface)}\b", cleaned):
            return canonical, True

    # Fuzzy matching using RapidFuzz for OCR typo resilience (e.g. 'ealcium' -> 'calcium')
    surfaces = list(_REVERSE.keys())
    match = process.extractOne(cleaned, surfaces, scorer=fuzz.ratio, score_cutoff=80)
    if match:
        best_surface, score, _ = match
        # Ensure length isn't wildly different for short abbreviations
        if abs(len(cleaned) - len(best_surface)) <= 3 and len(best_surface) >= 3:
            return _REVERSE[best_surface], True

    # Not in dictionary -> return title-cased raw token, flagged unmatched
    return raw_name.strip().title(), False


def normalize_frequency(raw_freq: str) -> str | None:
    if not raw_freq:
        return None
    key = raw_freq.strip().lower()
    return _FREQUENCY_MAP.get(key, key or None)

