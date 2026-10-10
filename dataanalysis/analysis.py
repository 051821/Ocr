import os
import re
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from core.db import db_pool
from core.llm_client import complete as llm_complete
from storage.ai_analysis_csv import get_llm_cache, set_llm_cache

logger = logging.getLogger(__name__)

# ============================================================
# Medication name normalization map (OCR raw → clean display)
# ============================================================
_REFERENCE_DIR = Path(__file__).resolve().parent / "data" / "reference"

def _load_reference(name):
    with (_REFERENCE_DIR / name).open(encoding="utf-8") as stream:
        return json.load(stream)

MED_NAME_NORMALIZE = _load_reference("medication_normalize.json")
MEDICATION_SIDE_EFFECTS_DB = _load_reference("medication_side_effects.json")

# ============================================================
# Text cleaning utilities
# ============================================================

def normalize_medical_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        s = text.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                parsed = json.loads(s)
                return normalize_medical_text(parsed)
            except Exception as exc:
                logger.warning("Unable to normalize serialized medical text: %s", exc)
        return text
    if isinstance(text, list):
        lines = [normalize_medical_text(item) for item in text if item is not None]
        return "\n".join([l for l in lines if l])
    if isinstance(text, dict):
        return json.dumps(text, ensure_ascii=False)
    return str(text)


def _normalize_text_input(text_lines):
    if not text_lines:
        return []
    text_str = normalize_medical_text(text_lines)
    return [line.strip() for line in text_str.splitlines() if line and line.strip()]


def normalize_med_name(raw_name: str) -> str:
    """Clean OCR medicine names while preserving unfamiliar names."""
    if not raw_name:
        return "Unknown / unreadable medicine"
    raw = re.sub(r"\s+", " ", str(raw_name)).strip(" .,:;|*-")
    key = raw.lower()
    if key in MED_NAME_NORMALIZE:
        return MED_NAME_NORMALIZE[key]
    raw = re.sub(
        r"(?i)^(?:tab(?:let)?|cap(?:sule)?|syr(?:up)?|inj(?:ection)?|cream|"
        r"lotion|oint(?:ment)?|gel|drops?|susp(?:ension)?)\.?\s+", "", raw
    ).strip()
    if re.fullmatch(r"(?i)(?:\d+(?:\.\d+)?\s*(?:mg|ml|mcg|gm|%|iu)|"
                    r"od|bd|tds|qid|sos|m-0-n|m-0-0|0-0-n|1-0-1|1-1-1)", raw):
        return "Unknown / unreadable medicine"
    return raw if raw.isupper() else raw[:1].upper() + raw[1:]


def _is_garbled(line: str) -> bool:
    """Return True if line is OCR garbage: purely illegible, repeated-word loops, non-latin noise."""
    if not line.strip():
        return True
    # All-ILLEGIBLE line
    if re.match(r'^[\s\[\]A-Z/:.]*\[(?:ILLEGIBLE|UNCLEAR)\][\s\[\]A-Z/:.]*$', line, re.I):
        return True
    # Line is mostly non-ASCII (Hindi noise, OCR artefacts)
    non_ascii = sum(1 for c in line if ord(c) > 127)
    if non_ascii > len(line) * 0.45:
        return True
    # Repeated-word loop detection (e.g. "गुलामी " × 30)
    words = line.split()
    if len(words) > 4:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.25:
            return True
    return False


def _clean_line(line: str) -> str:
    """Strip markdown artifacts, leading/trailing punctuation, [ILLEGIBLE] tags."""
    line = re.sub(r'\[(?:ILLEGIBLE|UNCLEAR)\]', '', line, flags=re.I)
    line = re.sub(r'\*\*|\*|#{1,3}', '', line)
    line = re.sub(r'\|', ' ', line)
    line = re.sub(r'\s{2,}', ' ', line)
    return line.strip(' .,;:-')


def _parse_frequency(raw: str) -> str:
    """Convert coded frequency strings to human-readable form."""
    raw = raw.strip().upper()
    mapping = {
        "M-0-N":  "Morning & Night (twice daily)",
        "M-0-0":  "Morning only (once daily)",
        "0-0-N":  "Night only (once daily)",
        "0-0-0":  "As directed",
        "1-0-1":  "Morning & Night",
        "1-1-1":  "Three times daily",
        "OD":     "Once daily",
        "BD":     "Twice daily",
        "TDS":    "Three times daily",
        "QID":    "Four times daily",
        "SOS":    "As needed",
        "AFTER FOOD": "After food",
        "BEFORE FOOD": "Before food",
    }
    for k, v in mapping.items():
        if k in raw:
            return v
    return raw.title() if raw else "Not documented"


# ============================================================
# Structured clinical extraction from OCR lines
# ============================================================

def extract_clinical_summary_from_ocr(text_lines: list, visit_date: str = None) -> dict:
    """
    Parses OCR clean_text lines into a fully structured clinical summary.
    Returns dict with keys:
        age_sex, chief_complaint, symptoms, provisional_diagnosis, confirmed_diagnosis,
        clinical_impression, vitals, medications, lab_results, clinical_notes,
        follow_up, extraction_stats
    Never fabricates data — uses 'Not clearly documented' for uncertain/missing fields.
    """
    empty_result = {
        "age_sex": None,
        "chief_complaint": None,
        "symptoms": [],
        "provisional_diagnosis": None,
        "confirmed_diagnosis": None,
        "clinical_impression": None,
        "vitals": [],
        "medications": [],
        "lab_results": [],
        "clinical_notes": [],
        "follow_up": None,
        "extraction_stats": {
            "total_lines": 0,
            "garbled_lines": 0,
            "extracted_fields": 0,
            "uncertain_fields": 0,
        }
    }

    text_lines = _normalize_text_input(text_lines)
    if not text_lines:
        return empty_result

    stats = {"total_lines": len(text_lines), "garbled_lines": 0,
              "extracted_fields": 0, "uncertain_fields": 0}

    age_sex = None
    chief_complaint = None
    symptoms = []
    provisional_dx = None
    confirmed_dx = None
    impression = None
    vitals = []
    medications = []
    lab_results = []
    notes = []
    follow_up = None

    # State machine for medication table parsing
    in_med_table = False
    current_med = {}

    def flush_med():
        nonlocal current_med
        if current_med.get("name"):
            medications.append(current_med)
            current_med = {}

    for line in text_lines:
        raw = str(line).strip()
        if not raw:
            continue

        stats["total_lines"] += 1

        # Drop garbled lines
        if _is_garbled(raw):
            stats["garbled_lines"] += 1
            continue

        cl = _clean_line(raw)
        cl_lower = cl.lower()
        if len(cl) < 2:
            continue

        # ------ Age / Sex ------
        if age_sex is None:
            m = re.search(r'(?i)(?:age[:/\s]+)?(\d{1,3})\s*/\s*(m(?:ale)?|f(?:emale)?)\b', cl)
            if m:
                age_v = m.group(1)
                sex_v = "Female" if m.group(2).lower().startswith("f") else "Male"
                age_sex = f"{age_v} years / {sex_v}"
                stats["extracted_fields"] += 1
                continue
            m2 = re.search(r'(?i)(?:age[:/\s-]+)?(\d{1,3})\s*(?:y|yrs?|years?)\s*[-/]\s*(male|female|m|f)\b', cl)
            if m2:
                age_v = m2.group(1)
                sex_v = "Female" if m2.group(2).lower().startswith("f") else "Male"
                age_sex = f"{age_v} years / {sex_v}"
                stats["extracted_fields"] += 1
                continue

        # ------ Chief Complaint ------
        if re.search(r'(?i)\bchief\s+complaint\b', cl):
            rest = re.sub(r'(?i)^\s*(?:\*\*)?chief\s+complaint\s*(?:\*\*)?[:\s-]*', '', cl).strip()
            if rest and len(rest) > 3:
                chief_complaint = rest.strip('*')
                stats["extracted_fields"] += 1
            continue  # skip the header line itself

        # ------ Diagnosis / Impression headers ------
        if re.search(r'(?i)\bdiagnosis\s*/\s*impression\b|\bdiagnosis\s*[:]\s*$|\bimpression\s*[:]\s*$', cl):
            in_med_table = False
            continue

        # ------ Provisional Diagnosis from document ------
        prov_match = re.search(r'(?i)(?:provostion\s*diagnosis|provisional\s*(?:diagnosis|dx)|prov\.?\s*dx)\s*[:\s/-]*([A-Za-z0-9\s(),.-]+)', cl)
        if prov_match:
            cand = prov_match.group(1).strip('*').strip()
            if cand and len(cand) > 2 and not cand.lower().startswith(('dr', 'doctor')):
                provisional_dx = cand
                stats["extracted_fields"] += 1
                continue
        elif re.search(r'(?i)\bprovisional\s+(?:diagnosis|dx)\b', cl):
            rest = re.sub(r'(?i)^\s*provisional\s+(?:diagnosis|dx)\s*[:\s-]*', '', cl).strip('*').strip()
            if rest and len(rest) > 2:
                provisional_dx = rest
                stats["extracted_fields"] += 1
            continue

        # ------ Confirmed Diagnosis from document ------
        if re.search(r'(?i)\bconfirmed\s+(?:diagnosis|dx)\b', cl):
            rest = re.sub(r'(?i)^\s*confirmed\s+(?:diagnosis|dx)\s*[:\s-]*', '', cl).strip('*').strip()
            if rest and len(rest) > 2:
                confirmed_dx = rest
                stats["extracted_fields"] += 1
            continue

        # ------ Clinical Impression ------
        # Tier 1: structural — ANY line explicitly labeled as an impression/
        # assessment/dx, regardless of which disease it names. This is what
        # lets the app recognise a diagnosis it has never seen before,
        # instead of only ones in the hardcoded keyword list below.
        if impression is None:
            imp_label_match = re.search(
                r'(?i)^\s*(?:\*\*)?(?:clinical\s+)?(?:impression|assessment|diagnosis|dx)\s*[:\-]\s*(?:\*\*)?(.+)',
                cl
            )
            if imp_label_match:
                candidate = imp_label_match.group(1).strip('*').strip()
                if candidate and len(candidate) >= 2:
                    impression = candidate.title() if candidate.isupper() else candidate
                    stats["extracted_fields"] += 1
                    continue

        # Tier 2: known-condition keyword fallback — still useful as a
        # secondary signal when a diagnosis is mentioned inline without an
        # explicit label, but no longer the ONLY way to detect a diagnosis.
        if impression is None and re.search(
            r'(?i)\b(?:eczema|lsc|tinea[\w\s]*|dermatitis|psoriasis|urticaria|folliculitis|scabies|dysmenorrhea|herpes\s+zoster|gastroenteritis|t2dm|htn|diabetes)\b', cl
        ):
            candidate = cl.strip('*').strip()
            if len(candidate) >= 3:
                impression = candidate.title()
                stats["extracted_fields"] += 1

        # ------ Medication table header detection ------
        if re.search(r'(?i)\bmedications?\b.*\bstrength\b|\br\s+medications\b', cl):
            flush_med()
            in_med_table = True
            continue

        # ------ Medication table rows ------
        # Detect a row that starts with a known medication or "TAB/CAP/CREAM/LOTION/OINT" prefix
        med_header_match = re.match(
            r'(?i)^(?:tab(?:let)?|cap(?:sule)?|syr(?:up)?|inj(?:ection)?|cream|lotion|oint(?:ment)?|gel|drops?|susp(?:ension)?)\.?\s+(.+)',
            cl
        )
        known_med_match = re.search(
            r'(?i)\b(fluconazole|citrizine|cetirizine|miconazole|lobet|vasaline|topisal|augmentin|fucibet|omnacortil|atarax|pantoprazole|omeprazole|amlodipine|telmisartan|metformin|ecosprin|paracetamol|azithromycin|amoxicillin|cefixime|ciprofloxacin|levocetirizine|montelukast|ibuprofen|diclofenac)\b',
            cl
        )
        # Bare "<Name> <strength unit>" with no dosage-form word at all
        # (e.g. "ZoltrimForte 500mg TDS") — still structural, still works
        # for a medicine never seen before.
        bare_med_match = re.search(
            r'\b([A-Z][A-Za-z\-]{2,24})\s+(\d+(?:\.\d+)?)\s*(mg|ml|mcg|gm|%|iu)\b', cl
        )

        # A line shaped like a lab-report row (numeric reference range or a
        # lab unit such as mg/dL, U/L, cells/cumm...) is never a medication
        # row, even mid-table — this is what previously let lab components
        # like Albumin/Urea/Creatinine/Direct/Indirect/Total/MCHC show up as
        # "medications identified".
        row_is_lab = _looks_like_lab_row(cl) or (
            bare_med_match is not None and _is_lab_term(bare_med_match.group(1))
        )

        if (in_med_table or med_header_match or known_med_match or bare_med_match) and not row_is_lab:
            # Try to parse pipe-delimited medication table row
            parts = [p.strip().strip('*').strip() for p in re.split(r'\|', cl)]
            if len(parts) >= 2 and (med_header_match or known_med_match or parts[0]):
                raw_name_candidate = parts[0] if parts[0] else (
                    med_header_match.group(0) if med_header_match else
                    (known_med_match.group(1) if known_med_match else "")
                )
                if _is_lab_term(raw_name_candidate):
                    continue
                flush_med()
                raw_name = raw_name_candidate
                strength = parts[1] if len(parts) > 1 else ""
                freq_raw = parts[2] if len(parts) > 2 else ""
                duration = parts[3] if len(parts) > 3 else ""
                extra_note = parts[4] if len(parts) > 4 else ""

                def _nd(v):
                    v = v.strip() if v else ""
                    return None if v in ("", "—", "-", "—", "N/A") else v

                med_entry = {
                    "name": normalize_med_name(raw_name),
                    "strength": _nd(strength) or "Not documented",
                    "frequency": _parse_frequency(freq_raw) if _nd(freq_raw) else "Not documented",
                    "duration": _nd(duration) or "Not documented",
                    "notes": _nd(extra_note),
                    "source": "Document",
                }
                medications.append(med_entry)
                stats["extracted_fields"] += 1
                continue

            # Single-line med (not pipe delimited). Fires for ANY dosage-form
            # prefixed line (med_header_match) — not just names on the
            # hardcoded known-medicine list — so an unfamiliar medicine
            # still gets a structured entry, not just a bare name pill.
            if known_med_match or med_header_match or bare_med_match:
                flush_med()
                if known_med_match:
                    raw_name = known_med_match.group(1)
                elif med_header_match:
                    # First word(s) after the dosage-form prefix, stopping
                    # before any strength/number token.
                    rest = med_header_match.group(1)
                    name_m = re.match(r'([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,1})', rest)
                    raw_name = name_m.group(1) if name_m else rest.split()[0]
                else:
                    raw_name = bare_med_match.group(1)
                # Try to pick up strength in same line
                strength_m = re.search(r'(\d+(?:\.\d+)?)\s*(mg|ml|gm|mcg|%)', cl, re.I)
                freq_m = re.search(r'(m-0-n|m-0-0|0-0-n|0-0-0|1-0-1|1-1-1|od|bd|tds|qid|sos|after\s+food|before\s+food)', cl, re.I)
                dur_m = re.search(r'(\d+)\s*(days?|weeks?|months?)', cl, re.I)
                medications.append({
                    "name": normalize_med_name(raw_name),
                    "strength": f"{strength_m.group(1)} {strength_m.group(2).lower()}" if strength_m else "Not documented",
                    "frequency": _parse_frequency(freq_m.group(0)) if freq_m else "Not documented",
                    "duration": f"{dur_m.group(1)} {dur_m.group(2).lower()}" if dur_m else "Not documented",
                    "notes": None,
                    "source": "Document",
                })
                stats["extracted_fields"] += 1
                continue

        in_med_table = False  # stop med table state if no match

        # ------ Vitals ------
        vitals_patterns = [
            (r'(?i)\bspo2\b\s*[:\-]?\s*([\d.]+)\s*%?',           "SpO₂",    lambda m: f"{m.group(1)}%"),
            (r'(?i)\bpulse\b\s*[:\-]?\s*([\d.]+)\s*(?:bpm|/min)?', "Pulse",  lambda m: f"{m.group(1)} bpm"),
            (r'(?i)\bhr\b\s*[:\-]?\s*([\d.]+)',                    "Heart Rate", lambda m: f"{m.group(1)} bpm"),
            (r'(?i)\bb\.?p\.?\b\s*[:\-]?\s*([\d]+/[\d]+)\s*(?:mmhg)?', "BP", lambda m: f"{m.group(1)} mmHg"),
            (r'(?i)\bweight\s*[-:–]?\s*([\d.]+)\s*(kg|lbs?)?',    "Weight",  lambda m: f"{m.group(1)} {(m.group(2) or 'kg').lower()}"),
            (r'(?i)\bwt\s*[-:–]?\s*([\d.]+)\s*(kg)?',             "Weight",  lambda m: f"{m.group(1)} kg"),
            (r'(?i)\bheight\s*[-:–]?\s*([\d.]+)\s*(cm|m)?',       "Height",  lambda m: f"{m.group(1)} {(m.group(2) or 'cm').lower()}"),
            (r'(?i)\bht\s*[-:–]?\s*([\d.]+)\s*(cm)?',             "Height",  lambda m: f"{m.group(1)} cm"),
            (r'(?i)\btemp(?:erature)?\s*[:\-]?\s*([\d.]+)\s*(°?[cf]?)',
             "Temperature", lambda m: _format_temp(m.group(1), m.group(2))),
        ]

        vital_found = False
        for pattern, label, fmt in vitals_patterns:
            m = re.search(pattern, cl)
            if m:
                value_str = fmt(m)
                # Dedup: same label already recorded?
                existing = next((v for v in vitals if v["label"] == label), None)
                if existing:
                    if existing["value"] != value_str:
                        existing["discrepancy"] = value_str  # flag conflict
                else:
                    vitals.append({
                        "label": label,
                        "value": value_str,
                        "date": visit_date[:10] if visit_date else None,
                        "source": "Document",
                        "discrepancy": None,
                    })
                vital_found = True
                stats["extracted_fields"] += 1
        if vital_found:
            continue

        # ------ Symptoms ------
        if re.search(r'(?i)\b(itching|itch|dark\s+patch|rash|lesion|scaling|pain|burning|discharge|redness|swelling|dryness|cracking|blister|wound)\b', cl):
            # Clean symptom line
            sym = re.sub(r'(?i)^\s*(?:complaints?|symptoms?|c/o)[:\s-]*', '', cl).strip('*').strip()
            if sym and len(sym) > 3 and sym not in symptoms:
                symptoms.append(sym)
            stats["extracted_fields"] += 1
            continue

        # ------ Follow-up ------
        if re.search(r'(?i)\bnext\s+follow[-\s]?up\b|\bfollow\s+up\s+at\b|\bfollow[-\s]up\s*:', cl):
            rest = re.sub(r'(?i)^\s*(?:\*\*)?next\s+follow[-\s]?up\s*:?\s*(?:\*\*)?', '', cl).strip()
            if rest and len(rest) > 1:
                follow_up = rest.strip('*')
                stats["extracted_fields"] += 1
            continue

        # ------ Clinical notes (refer, review, etc.) ------
        if re.search(r'(?i)\b(refer\s+to|review|counsel|advice|instructions?)\b', cl):
            note = cl.strip('*').strip()
            if note and len(note) > 4 and note not in notes:
                notes.append(note)

    # Populate lab results using comprehensive lab extractor
    lab_results = extract_lab_results_from_text(text_lines)
    if lab_results:
        stats["extracted_fields"] += len(lab_results)

    # Deduplicate medications by name
    seen_med_names = {}
    deduped_meds = []
    for med in medications:
        key = med["name"].lower()
        if key not in seen_med_names:
            seen_med_names[key] = True
            deduped_meds.append(med)

    stats["total_lines"] = len(text_lines)
    stats["extracted_fields"] = min(stats["extracted_fields"], len(text_lines))

    return {
        "age_sex":              age_sex,
        "chief_complaint":      chief_complaint,
        "symptoms":             symptoms,
        "provisional_diagnosis": provisional_dx,
        "confirmed_diagnosis":  confirmed_dx,
        "clinical_impression":  impression,
        "vitals":               vitals,
        "medications":          deduped_meds,
        "lab_results":          lab_results,
        "clinical_notes":       notes,
        "follow_up":            follow_up,
        "extraction_stats":     stats,
    }


def _format_temp(value_str: str, unit_str: str) -> str:
    """Format temperature, flagging unit conflicts."""
    try:
        val = float(value_str)
        unit = unit_str.strip().upper().replace("°", "") if unit_str else ""
        if unit == "F" or (not unit and val > 45):  # likely Fahrenheit
            celsius = round((val - 32) * 5 / 9, 1)
            return f"{val}°F ({celsius}°C calculated)"
        elif unit == "C" or val < 45:
            fahrenheit = round(val * 9 / 5 + 32, 1)
            return f"{val}°C ({fahrenheit}°F calculated)"
        return f"{value_str}{unit_str}"
    except Exception:
        return value_str


# ============================================================
# Standard medication extractor (for aggregation)
# ============================================================

_MED_STOPWORDS = re.compile(
    r'^(?:dose|dosage|frequency|duration|timing|notes|after|before|food|'
    r'days?|weeks?|months?|morning|night|evening|once|twice|daily|patient|'
    r'advice|follow|review|refer)$', re.I
)

# Lab test / lab-report vocabulary. These are NOT medicines, but a lab report
# rendered as a pipe-delimited or "name <value> <unit>" table looks
# structurally identical to a medication table, which previously caused
# things like "Albumin", "Urea", "Creatinine", "Direct", "Indirect", "Total"
# and "MCHC" to be reported to the user as medications. Any candidate name
# that is (or contains only) one of these terms is never treated as a
# medicine, regardless of which pattern matched it.
_LAB_TERMS = {
    "albumin", "globulin", "protein", "total protein", "bilirubin", "direct",
    "indirect", "total", "total bilirubin", "urea", "blood urea", "creatinine",
    "uric acid", "calcium", "sodium", "potassium", "chloride", "phosphorus",
    "magnesium", "glucose", "blood sugar", "random glucose", "fasting glucose",
    "post prandial", "hba1c", "cholesterol", "triglycerides", "hdl", "ldl",
    "vldl", "hdl-cholesterol", "ldl-cholesterol", "vldl-cholesterol",
    "sgot", "sgpt", "ast", "alt", "alkaline phosphatase", "ggt",
    "hemoglobin", "haemoglobin", "hb", "pcv", "mcv", "mch", "mchc", "rdw",
    "tlc", "dlc", "wbc", "rbc", "platelets", "platelet count", "esr",
    "neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils",
    "polymorphs", "tsh", "t3", "t4", "ft3", "ft4", "vitamin d", "vitamin b12",
    "iron", "ferritin", "sample", "method", "test", "result", "units",
    "reference", "interval", "biological", "value", "range", "investigation",
    "specimen", "report", "age", "sex", "date", "lab", "laboratory",
}


def _looks_like_lab_row(line: str) -> bool:
    """True if a line has the shape of a lab-report row: a numeric value
    alongside a reference RANGE (e.g. '3.5 - 5.0') or a lab unit — patterns
    a medication row (strength/frequency/duration) doesn't normally have."""
    if re.search(r'[\d.]+\s*[-–—]\s*[\d.]+', line):
        return True
    if re.search(r'(?i)\b(?:mg/dl|g/dl|gm/dl|u/l|iu/l|mmol/l|ng/ml|pg/ml|'
                 r'miu/ml|cells/cumm|lakhs/cumm|millions/cumm|fl)\b', line):
        return True
    if re.search(r'(?i)\bbio\.?\s*ref\.?\s*interval\b|\breference\s*range\b|'
                 r'\bnormal\s*range\b', line):
        return True
    return False


def _is_lab_term(name: str) -> bool:
    n = re.sub(r'[^a-z0-9 ]', '', name.lower()).strip()
    if not n:
        return True
    if n in _LAB_TERMS:
        return True
    # e.g. "serum albumin", "total leucocyte count" — a lab-term as one of
    # the words, with no other substantive word alongside it.
    words = n.split()
    if all(w in _LAB_TERMS or w in ("serum", "count", "level", "levels",
                                     "in", "of", "the", "and", "for", "with", "to", "at", "on")
           for w in words):
        return True
    return False


def extract_medications_from_text(text_lines):
    """Dynamically extract medicine names, including unfamiliar OCR names."""
    meds = set()
    lines = _normalize_text_input(text_lines)
    dosage_forms = (
        r"tab(?:let)?|cap(?:sule)?|syr(?:up)?|inj(?:ection)?|oint(?:ment)?|"
        r"cream|lotion|gel|drops?|susp(?:ension)?|powder|spray|patch|solution|soap"
    )
    structural_patterns = [
        rf"(?i)\b(?:{dosage_forms})\.?\s+([A-Za-z][A-Za-z0-9&+\-]*(?:\s+[A-Za-z][A-Za-z0-9&+\-]*){{0,2}})",
        r"\b([A-Z][A-Za-z0-9&+\-]{2,35})\s+\d+(?:\.\d+)?\s*(?:mg|ml|mcg|gm|g|%|iu)\b(?!\s*/)",
        r"\b([A-Za-z][A-Za-z0-9&+\-]{2,35})\s+\d+(?:\.\d+)?\s*(?:mg|ml|mcg|gm|g|%|iu)\b(?!\s*/)",
    ]
    for line in lines:
        clean = _clean_line(line)
        if not clean:
            continue
        is_lab_row = _looks_like_lab_row(clean)
        parts = [p.strip(" *") for p in re.split(r"\|", clean)]
        if len(parts) >= 2 and not is_lab_row:
            first = re.sub(rf"(?i)^(?:{dosage_forms})\.?\s+", "", parts[0]).strip()
            if (len(first) >= 3 and not _MED_STOPWORDS.match(first)
                    and not _is_lab_term(first)
                    and not re.search(r"(?i)^(?:medicine|medication|drug|name)$", first)
                    and not re.search(r"(?i)\b(?:strength|frequency|duration|dose|timing)\b", first)):
                meds.add(normalize_med_name(first))
        if is_lab_row:
            # A line with a reference range or lab unit is a lab result,
            # never a medication row — skip the structural-pattern pass too.
            continue
        for pat in structural_patterns:
            for match in re.findall(pat, clean):
                name = match if isinstance(match, str) else match[0]
                name = re.sub(rf"(?i)^(?:{dosage_forms})\.?\s+", "", name).strip()
                if len(name) >= 3 and not _MED_STOPWORDS.match(name) and not _is_lab_term(name):
                    meds.add(normalize_med_name(name))

    known_pat = (
        r"(?i)\b(fluconazole|citrizine|cetirizine|miconazole|lobet|vasaline|"
        r"topisal|augmentin|fucibet|omnacortil|atarax|pantoprazole|omeprazole|"
        r"amlodipine|telmisartan|metformin|ecosprin|paracetamol|azithromycin|"
        r"amoxicillin|cefixime|ciprofloxacin|levocetirizine|montelukast|"
        r"ibuprofen|diclofenac)\b"
    )
    for line in lines:
        for m in re.findall(known_pat, line):
            meds.add(normalize_med_name(m.strip()))

    return sorted(m for m in meds if m and m.lower() not in {
        "medicine", "medication", "drug", "name", "unknown"
    } and not _is_lab_term(m))



_UNKNOWN_MED_FALLBACK = (
    "This medicine isn't in our local reference list, so we can't show its "
    "specific side effects here — please check the package insert or ask "
    "your pharmacist/doctor."
)


def _ai_lookup_side_effects(unmatched_meds):
    """Look up unknown names in one request and cache each name permanently."""
    if not unmatched_meds:
        return {}
    cache_keys = {med: f"side-effects:v1:{med.strip().casefold()}" for med in unmatched_meds}
    results, missing = {}, []
    for med, key in cache_keys.items():
        try:
            cached = get_llm_cache(key)
            if cached is None:
                missing.append(med)
                continue
            value = json.loads(cached)
            if isinstance(value, str) and value.strip():
                results[med] = f"{value.strip()} (AI-assisted general reference; verify with a pharmacist or clinician)."
        except Exception as exc:
            logger.warning("Side-effect cache read failed for %s: %s", med, exc)
            missing.append(med)
    if not missing:
        return results
    med_list_str = "\n".join(f"- {m}" for m in missing)
    prompt = (
        "You are a medication-reference assistant. Resolve common brand, generic, and minor OCR spelling variants only when confident. "
        "For each exact input name, return 3 to 6 established common side effects as one short string. "
        "Use null only when the medicine cannot be identified. Do not give treatment, dose, or diagnosis advice. "
        "Return one valid JSON object and no prose: exact input name -> string or null.\n\n"
        f"MEDICINE NAMES:\n{med_list_str}"
    )
    try:
        response = llm_complete([{"role": "user", "content": prompt}], "side_effects", max_tokens=300)
        content = response["content"].strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I).strip()
        json_match = re.search(r"\{.*\}", content, flags=re.S)
        data = json.loads(json_match.group(0) if json_match else content)
        if not isinstance(data, dict):
            return results
        returned = {str(key).strip().casefold(): value for key, value in data.items()}
        for med in missing:
            text = returned.get(med.casefold())
            value = text.strip().rstrip(".") if isinstance(text, str) and text.strip() else None
            try:
                set_llm_cache(cache_keys[med], json.dumps(value, ensure_ascii=False))
            except Exception as exc:
                logger.warning("Side-effect cache write failed for %s: %s", med, exc)
            if value:
                results[med] = f"{value} (AI-assisted general reference; verify with a pharmacist or clinician)."
        return results
    except Exception as exc:
        logger.warning("AI side-effect lookup failed: %s", exc)
        return results
def get_possible_side_effects(medication_list, use_ai=True):
    side_effects = {}
    unmatched = []

    for med in medication_list:
        if med.lower() in {"unknown / unreadable medicine", "unknown"}:
            side_effects[med] = _UNKNOWN_MED_FALLBACK
            continue
        med_lower = med.lower()
        matched = [effects for key, effects in MEDICATION_SIDE_EFFECTS_DB.items()
                   if key in med_lower]
        if matched:
            side_effects[med] = " ".join(matched)
        else:
            unmatched.append(med)

    # For locally unknown names, ask the model dynamically. A null or failed
    # lookup remains explicit; it must never be presented as a drug-specific
    # side-effect profile.
    if unmatched and use_ai:
        ai_hits = _ai_lookup_side_effects(unmatched)
        for med in unmatched:
            side_effects[med] = ai_hits.get(
                med,
                "Unable to identify this medicine confidently. Check the spelling and active ingredient on the package, then ask a pharmacist or clinician.",
            )
    elif unmatched:
        for med in unmatched:
            side_effects[med] = _UNKNOWN_MED_FALLBACK

    return side_effects


def _evaluate_lab_status(val_s, ref_s):
    if not val_s or not ref_s:
        return "Not Evaluated"
    try:
        value_match = re.search(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", str(val_s))
        if not value_match:
            return "Not Evaluated"
        val_clean = float(value_match.group(0).replace(",", ""))
        ref_str = str(ref_s).strip()

        range_m = re.search(r'([\d.]+)\s*[-–—]\s*([\d.]+)', ref_str)
        if range_m:
            rmin, rmax = float(range_m.group(1)), float(range_m.group(2))
            if val_clean < rmin:
                return "Abnormal (Low)"
            elif val_clean > rmax:
                return "Abnormal (High)"
            else:
                return "Normal"

        lt_m = re.search(r'[<≤]\s*=?\s*([\d.]+)', ref_str)
        if lt_m:
            limit = float(lt_m.group(1))
            return "Normal" if val_clean <= limit else "Abnormal (High)"

        gt_m = re.search(r'[>≥]\s*=?\s*([\d.]+)', ref_str)
        if gt_m:
            limit = float(gt_m.group(1))
            return "Normal" if val_clean >= limit else "Abnormal (Low)"

        status_word = re.search(r"\b(normal|high|elevated|low|decreased|abnormal)\b", ref_str, re.I)
        if status_word:
            status = status_word.group(1).casefold()
            if status == "normal":
                return "Normal"
            if status in {"high", "elevated"}:
                return "Abnormal (High)"
            if status in {"low", "decreased"}:
                return "Abnormal (Low)"
            return "Abnormal"

    except Exception as exc:
        logger.warning("Unable to evaluate lab status for value %r: %s", val_s, exc)
    return "Not Evaluated"


def extract_lab_results_from_text(text_lines):
    raw_lines = _normalize_text_input(text_lines)
    if not raw_lines:
        return []

    results = []
    seen_tests = set()

    def add_result(tname, val, ref):
        tname = re.sub(r'[*`|_]+', '', tname).strip(' :|*,')
        val = str(val).strip(' :|*')
        ref = str(ref).strip(' :|*')
        if not tname or len(tname) < 2 or re.match(r'^(?:age|date|page|wt|ht|sample|method|result|units|bio|investigation|interpretation)', tname, re.I):
            return
        key = (tname.lower(), val)
        if key in seen_tests:
            return
        seen_tests.add(key)
        status = _evaluate_lab_status(val, ref)
        results.append({
            "test_name": tname,
            "value": val,
            "reference": ref if ref else "Not Specified",
            "status": status
        })

    # Pass 1: Line-by-line parsing
    for line in raw_lines:
        cl = line.strip()
        if not cl:
            continue

        if '|' in cl:
            parts = [p.strip() for p in cl.split('|') if p.strip()]
            if len(parts) >= 2:
                t_candidate = parts[0]
                val_found = None
                ref_found = ""
                for p in parts[1:]:
                    val_m = re.search(r'^[<>≤≥]?\s*(\d+(?:\.\d+)?)', p)
                    if val_m and not val_found:
                        val_found = val_m.group(1)
                    elif re.search(r'[\d.]+\s*[-–—]\s*[\d.]+|[<>≤≥]\s*[\d.]+', p) and not ref_found:
                        ref_found = p
                if t_candidate and val_found:
                    add_result(t_candidate, val_found, ref_found)
                    continue

        hba1c_m = re.search(r'(?i)\b(HbA1C|HbA1c)\s+([\d.]+)\s*%\s*(?:Non-Diabetic\s*:?\s*)?([<>≤≥]?\s*[\d.]+)', cl)
        if hba1c_m:
            add_result(hba1c_m.group(1), hba1c_m.group(2), f"< {hba1c_m.group(3)}".replace('< <', '<'))
            continue

        inline_m = re.search(
            r'(?i)^\s*(Serum\s+[A-Za-z0-9\s(),\/\-\"\']+?)\s*:\s*([\d.]+)\s*(?:[A-Za-z/%.]+)?\s*(?:Bio\.?\s*Ref\.?\s*Interval|Desirable|Normal|Major\s+Risk|Negative\s+Risk)?\s*([<>≤≥]?\s*\d+(?:\.\d+)?(?:\s*[-–—]\s*\d+(?:\.\d+)?)?)',
            cl
        )
        if inline_m:
            add_result(inline_m.group(1), inline_m.group(2), inline_m.group(3))
            continue

        direct_m = re.search(
            r'(?i)^\s*([A-Za-z0-9\s(),\/\-\"\']+?)\s+(?:Sample\s+Type:[^:]*)?(?:Method:[^:]*)?\s*([\d.]+)\s*(?:mg/dL|g/dL|U/L|mmol/L|gm/dl|Millions/Cumm|fL|Pg|Cells/Cumm|Lakhs/Cumm|%|g/dl)?\s+([<>≤≥]?\s*\d+(?:\.\d+)?(?:\s*[-–—]\s*\d+(?:\.\d+)?)?)',
            cl
        )
        if direct_m:
            t_name = direct_m.group(1).strip()
            v_val = direct_m.group(2)
            r_ref = direct_m.group(3)
            if re.match(r'(?i)^(?:Serum|Blood|Hb|HbA1c|Hemoglobin|Total|PCV|MCV|MCH|MCHC|RDW|Platelet|Polymorphs|Lymphocytes|Monocytes|Eosinophils|Basophils)', t_name):
                add_result(t_name, v_val, r_ref)
                continue

    # Pass 2: Multi-line window scanning for split OCR lines
    n = len(raw_lines)
    test_keywords = (
        "serum calcium", "serum uric acid", "serum sodium", "serum potassium",
        "blood sugar random", "serum sgot", "serum sgpt", "serum alkaline phosphatase",
        "serum total protein", "serum albumin", "serum bilirubin", "serum blood urea",
        "serum creatinine", "hemoglobin", "total rbc count", "platelet count"
    )

    for i in range(n):
        l_lower = raw_lines[i].lower().strip(' :*')
        matched_kw = next((kw for kw in test_keywords if kw in l_lower), None)
        if matched_kw:
            test_name = raw_lines[i].strip(' :*')
            val = None
            ref = ""
            for j in range(i + 1, min(i + 7, n)):
                line_j = raw_lines[j].strip()
                if not val:
                    val_m = re.search(r'(?:Result\s*:?\s*)?^:?\s*([<>≤≥]?\s*\d+(?:\.\d+)?)', line_j, re.I)
                    if val_m and not re.search(r'^(?:19|20)\d\d', line_j):
                        val = val_m.group(1).strip()

                if not ref:
                    ref_m = re.search(r'(?:Bio\.?\s*Ref\.?\s*Interval\s*:?\s*)?([<>≤≥]?\s*\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?|[<>≤≥]\s*[\d.]+)', line_j, re.I)
                    if ref_m:
                        ref = ref_m.group(1).strip()

            if test_name and val:
                add_result(test_name, val, ref)

    return results


# ============================================================
# AI-assisted extraction fallback
# ============================================================
# Regex is structural (dosage-form words, "Diagnosis:" labels, table rows)
# so it already catches medicines/diagnoses it has never seen the NAME of
# before. But some documents phrase things in ways no structural rule
# anticipates (odd layout, no labels at all). For exactly those cases —
# real document text exists, yet regex found neither a diagnosis nor a
# medication — we ask the model to have a look, instead of quietly
# reporting "not documented".

def _ai_extract_full_clinical_summary(text_lines):
    """Extract a weak or unresolved OCR document in one cached request."""
    if not text_lines:
        return None
    document = "\n".join(text_lines)[:12000]
    prompt = (
        "Extract facts explicitly present in this clinical document. Do not diagnose or infer. "
        "Preserve medicine, diagnosis, lab, and value strings as written; use null or [] when absent. "
        "Return only JSON with keys: age_sex, chief_complaint, symptoms, clinical_impression, "
        "provisional_diagnosis, confirmed_diagnosis, vitals[{label,value}], medications[{name,strength,frequency,duration}], "
        "lab_results[{test_name,value,reference}], follow_up, clinical_notes.\n\n"
        f"DOCUMENT:\n{document}"
    )
    try:
        response = llm_complete([{"role": "user", "content": prompt}], "extraction", max_tokens=300)
        content = re.sub(r"^```(?:json)?|```$", "", response["content"].strip(), flags=re.M).strip()
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("AI document extraction failed: %s", exc)
        return None


def _extraction_looks_weak(clinical_summary) -> bool:
    """True when the deterministic parser extracted no meaningful content."""
    fields = ("chief_complaint", "clinical_impression", "provisional_diagnosis",
              "confirmed_diagnosis", "symptoms", "medications", "vitals",
              "lab_results", "age_sex")
    return not any(clinical_summary.get(field) for field in fields)


def _has_unresolved_medicine_or_diagnosis(text_lines, clinical_summary):
    meds = {str(item.get("name", "")).casefold() for item in clinical_summary.get("medications", []) if isinstance(item, dict)}
    meds.update(str(name).casefold() for name in extract_medications_from_text(text_lines))
    for line in text_lines:
        candidate = re.search(r"(?i)\b([A-Za-z][A-Za-z0-9+&-]{2,})\s+\d+(?:\.\d+)?\s*(?:mg|ml|mcg|gm|g|iu)\b", line)
        if candidate and candidate.group(1).casefold() not in meds:
            return True
        diagnosis = re.search(r"(?i)\b(?:diagnosis|impression)\s*[:\-]\s*(.{3,100})", line)
        known_dx = " ".join(str(clinical_summary.get(key) or "") for key in
                             ("clinical_impression", "provisional_diagnosis", "confirmed_diagnosis")).casefold()
        if diagnosis and diagnosis.group(1).strip().casefold() not in known_dx:
            return True
    return False


def _augment_with_ai_if_gaps(doc_lines, clinical_summary):
    """One Tier 2 extraction only when deterministic parsing leaves a gap."""
    if not doc_lines or not (
        _extraction_looks_weak(clinical_summary)
        or _has_unresolved_medicine_or_diagnosis(doc_lines, clinical_summary)
    ):
        return clinical_summary
    extracted = _ai_extract_full_clinical_summary(doc_lines)
    if not extracted:
        return clinical_summary

    scalar_fields = ("age_sex", "chief_complaint", "clinical_impression",
                     "provisional_diagnosis", "confirmed_diagnosis", "follow_up")
    ai_fields = []
    for field in scalar_fields:
        if not clinical_summary.get(field) and extracted.get(field):
            clinical_summary[field] = str(extracted[field])[:300]
            ai_fields.append(field)
    for field, limit in (("symptoms", 10), ("clinical_notes", 5), ("vitals", 10), ("lab_results", 30)):
        if not clinical_summary.get(field) and extracted.get(field):
            clinical_summary[field] = extracted[field][:limit]
            ai_fields.append(field)
    current_meds = {str(item.get("name", "")).casefold() for item in clinical_summary.get("medications", []) if isinstance(item, dict)}
    for item in extracted.get("medications", [])[:20]:
        name = str(item.get("name") or "").strip()
        clean = normalize_med_name(name)
        if clean and clean.casefold() not in current_meds and not _is_lab_term(clean):
            clinical_summary.setdefault("medications", []).append({
                "name": clean,
                "strength": item.get("strength"),
                "frequency": item.get("frequency"),
                "duration": item.get("duration"),
                "ai_assisted": True,
            })
            current_meds.add(clean.casefold())
            ai_fields.append("medications")
    clinical_summary["ai_assisted_fields"] = sorted(set(ai_fields))
    clinical_summary.setdefault("extraction_stats", {})["ai_assisted"] = True
    return clinical_summary
# ============================================================
# Main data fetch
# ============================================================

def fetch_patient_history(patient_id):
    """
    """
    patient_id = str(patient_id).strip()
    conn = db_pool.getconn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    v.patient_id,
                    v.id AS vid,
                    v.created_at,
                    v.chief_complaint,
                    v.vitals_json,

                    COALESCE(d.documents, '[]'::jsonb) AS documents,
                    COALESCE(p.prescriptions, '[]'::jsonb) AS prescriptions

                FROM visit v

                LEFT JOIN LATERAL (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'doc_id', de.id,
                            'clean_text', de.clean_text
                        )
                        ORDER BY de.id
                    ) AS documents
                    FROM document_extraction de
                    WHERE de.visit_id = v.id
                ) d ON true

                LEFT JOIN LATERAL (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'prescription_id', pe.id,
                            'provisional_diagnosis', pe.provisional_diagnosis,
                            'confirmed_diagnosis', pe.confirmed_diagnosis,
                            'instructions', pe.instructions,
                            'patient_instructions', pe.patient_instructions,
                            'medications', COALESCE(m.medications, '[]'::jsonb),
                            'followups', COALESCE(f.followups, '[]'::jsonb)
                        )
                        ORDER BY pe.id
                    ) AS prescriptions

                    FROM prescription pe

                    LEFT JOIN LATERAL (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'medication_name', pi.medication_name,
                                'dosage', pi.dosage,
                                'frequency', pi.frequency,
                                'duration', pi.duration
                            )
                            ORDER BY pi.medication_name
                        ) AS medications
                        FROM prescriptionitem pi
                        WHERE pi.prescription_id = pe.id
                    ) m ON true

                    LEFT JOIN LATERAL (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'scheduled_date', fs.scheduled_date,
                                'status', fs.status,
                                'sequence_no', fs.sequence_no,
                                'resolution', fs.resolution,
                                'resolution_notes', fs.resolution_notes,
                                'is_emergency', fs.is_emergency
                            )
                            ORDER BY fs.scheduled_date ASC
                        ) AS followups
                        FROM followup_schedule fs
                        WHERE fs.source_prescription_id = pe.id
                    ) f ON true

                    WHERE pe.visit_id = v.id
                ) p ON true

                WHERE v.patient_id = %s::uuid

                ORDER BY v.created_at ASC NULLS LAST;
                """,
                (patient_id,),
            )
            rows = cur.fetchall()

        visits = {}

        for (pid, vid, created_at, chief_complaint, vitals_json,
             documents_jsonb, prescriptions_jsonb) in rows:

            visit_id = str(vid)

            # ---- build rx_items and medications from aggregated prescriptions ----
            rx_items = []
            seen_rx_items = set()
            medications = []
            seen_prescriptions = set()
            prov_dx = conf_dx = None
            followups = []
            seen_followup_keys = set()

            for pe in (prescriptions_jsonb or []):
                prescription_id = pe.get("prescription_id")
                if prescription_id not in seen_prescriptions:
                    seen_prescriptions.add(prescription_id)
                    prov_dx = prov_dx or pe.get("provisional_diagnosis")
                    conf_dx = conf_dx or pe.get("confirmed_diagnosis")
                    for rx in (pe.get("instructions"), pe.get("patient_instructions")):
                        if rx:
                            s = normalize_medical_text(rx)
                            if s and s not in medications:
                                medications.append(s)
                    # collect follow-ups for this prescription
                    for fu in (pe.get("followups") or []):
                        fkey = (str(fu.get("scheduled_date", "")), str(fu.get("sequence_no", "")))
                        if fkey not in seen_followup_keys:
                            seen_followup_keys.add(fkey)
                            followups.append({
                                "scheduled_date":  str(fu.get("scheduled_date", "")),
                                "status":          fu.get("status", ""),
                                "sequence_no":     fu.get("sequence_no"),
                                "resolution":      fu.get("resolution"),
                                "resolution_notes":fu.get("resolution_notes"),
                                "is_emergency":    fu.get("is_emergency", False),
                            })
                for med in (pe.get("medications") or []):
                    med_name = med.get("medication_name")
                    if med_name and str(med_name).strip():
                        item_key = (prescription_id,
                                    str(med_name).strip().lower(),
                                    str(med.get("dosage") or "").strip().lower(),
                                    str(med.get("frequency") or "").strip().lower(),
                                    str(med.get("duration") or "").strip().lower())
                        if item_key not in seen_rx_items:
                            seen_rx_items.add(item_key)
                            rx_items.append({
                                "name":      str(med_name).strip(),
                                "dosage":    med.get("dosage"),
                                "frequency": med.get("frequency"),
                                "duration":  med.get("duration"),
                                "source":    "Prescription (database)",
                            })

            # ---- build documents dict from aggregated documents ----
            documents = {}
            for doc in (documents_jsonb or []):
                doc_id = doc.get("doc_id")
                clean_text = doc.get("clean_text")
                if doc_id is not None and clean_text:
                    if doc_id not in documents:
                        documents[doc_id] = {"doc_id": doc_id, "lines": []}
                    doc_lines = documents[doc_id]["lines"]
                    for line in normalize_medical_text(clean_text).splitlines():
                        ls = line.strip()
                        if ls and ls not in doc_lines:
                            doc_lines.append(ls)

            visits[visit_id] = {
                "visit_id":              visit_id,
                "visit_date":            created_at.isoformat() if created_at else None,
                "chief_complaint":       chief_complaint,
                "provisional_diagnosis": prov_dx,
                "confirmed_diagnosis":   conf_dx,
                "medications":           medications,
                "rx_items":              rx_items,
                "vitals":                vitals_json,
                "documents":             documents,
                "followups":             followups,
            }

        formatted = []
        all_meds = set()
        all_db_meds = set()
        all_doc_meds = set()

        for v in visits.values():
            # Stable order: by doc_id so "Document 1/2/3" is consistent.
            documents_list = [v["documents"][k] for k in sorted(v["documents"].keys())]
            has_document = len(documents_list) > 0

            db_vitals_json = v["vitals"] or {}
            if isinstance(db_vitals_json, str):
                try:
                    db_vitals_json = json.loads(db_vitals_json)
                except Exception:
                    db_vitals_json = {}
            db_vital_rows = _parse_vitals_json(db_vitals_json, v["visit_date"])

            discrepancies = []
            documents_formatted = []
            visit_doc_meds = set()     # medicines extracted from documents ONLY
            visit_labs = []

            if has_document:
                for doc_num, doc in enumerate(documents_list, start=1):
                    doc_lines = doc["lines"]

                    clinical_summary = extract_clinical_summary_from_ocr(
                        doc_lines, visit_date=v["visit_date"]
                    )
                    # If structural regex parsing came up empty-handed for
                    # diagnosis/medications despite real text being present,
                    # fall back to an AI-assisted pass instead of just
                    # giving up on anything not in the hardcoded lists.
                    # Keep record fetch deterministic; AI enrichment is
                    # triggered explicitly from the relevant UI action.

                    # clinical_summary["lab_results"] already IS the regex
                    # extraction, plus anything Tier 2 above backfilled for
                    # an unfamiliar document layout — reuse it rather than
                    # recomputing a fresh regex-only pass that would throw
                    # the AI-assisted labs away.
                    doc_labs = clinical_summary.get("lab_results") or extract_lab_results_from_text(doc_lines)
                    doc_meds = extract_medications_from_text(doc_lines)
                    for med in clinical_summary.get("medications", []):
                        if med.get("name"):
                            doc_meds.append(med["name"])
                    doc_meds = sorted(set(doc_meds))

                    visit_doc_meds.update(doc_meds)
                    visit_labs.extend(doc_labs)

                    documents_formatted.append({
                        "document_label":    f"Document {doc_num}",
                        "doc_id":            doc["doc_id"],
                        "extracted_text":    doc_lines,
                        "clinical_summary":  clinical_summary,
                        "medications_found": doc_meds,
                        "lab_results":       doc_labs,
                    })

                    # ---- Discrepancy detection (DB structured vs this document) ----
                    if v["chief_complaint"] and clinical_summary["chief_complaint"]:
                        db_cc  = v["chief_complaint"].strip().lower()
                        ocr_cc = clinical_summary["chief_complaint"].strip().lower()
                        if db_cc not in ocr_cc and ocr_cc not in db_cc:
                            discrepancies.append(
                                f"{documents_formatted[-1]['document_label']}: chief complaint differs between structured record and document.")

                    if v["provisional_diagnosis"] and clinical_summary["provisional_diagnosis"]:
                        db_pd  = v["provisional_diagnosis"].strip().lower()
                        ocr_pd = clinical_summary["provisional_diagnosis"].strip().lower()
                        if db_pd not in ocr_pd and ocr_pd not in db_pd:
                            discrepancies.append(
                                f"{documents_formatted[-1]['document_label']}: provisional diagnosis differs between structured record and document.")

                    for ocr_vital in clinical_summary["vitals"]:
                        label_lower = ocr_vital["label"].lower().replace("₂", "2")
                        for db_key, db_val in db_vitals_json.items():
                            if any(kw in db_key.lower() for kw in label_lower.split()):
                                try:
                                    db_numeric = float(re.sub(r'[^\d.]', '', str(db_val)))
                                    ocr_numeric = float(re.sub(r'[^\d.]', '', ocr_vital["value"].split()[0]))
                                    if abs(db_numeric - ocr_numeric) > 2:
                                        ocr_vital["source"] = "Document"
                                        discrepancies.append(
                                            f"{documents_formatted[-1]['document_label']}: {ocr_vital['label']} — "
                                            f"structured record shows {db_val}, document shows {ocr_vital['value'].split()[0]}."
                                        )
                                except Exception as exc:
                                    logger.warning("Unable to compare structured and OCR vital values: %s", exc)

            # Database medicines come ONLY from prescriptionitem; prescription
            # instruction text is no longer mined for medicine names.
            visit_db_meds = sorted({m["name"] for m in v["rx_items"]})

            all_db_meds.update(visit_db_meds)
            all_doc_meds.update(visit_doc_meds)
            all_meds.update(visit_db_meds)
            all_meds.update(visit_doc_meds)

            formatted.append({
                "visit_id":             v["visit_id"],
                "visit_date":           v["visit_date"],
                "chief_complaint":      v["chief_complaint"] or (documents_formatted[0]["clinical_summary"].get("chief_complaint") if documents_formatted else None),
                "provisional_diagnosis": v["provisional_diagnosis"] or (documents_formatted[0]["clinical_summary"].get("provisional_diagnosis") or documents_formatted[0]["clinical_summary"].get("clinical_impression") if documents_formatted else None),
                "confirmed_diagnosis":  v["confirmed_diagnosis"] or (documents_formatted[0]["clinical_summary"].get("confirmed_diagnosis") if documents_formatted else None),
                "db_chief_complaint":   v["chief_complaint"],
                "db_provisional_dx":    v["provisional_diagnosis"],
                "db_confirmed_dx":      v["confirmed_diagnosis"],
                "db_instructions":      v["medications"],
                "db_vitals_rows":       db_vital_rows,
                "db_vitals_json":       db_vitals_json,
                "has_document":         has_document,
                "documents":            documents_formatted,
                "db_medications":       v["rx_items"],
                "document_medications": sorted(visit_doc_meds),
                "medications_found":    sorted(set(visit_db_meds) | visit_doc_meds),
                "lab_results":          visit_labs,
                "discrepancies":        discrepancies,
                "followups":            v.get("followups", []),
            })

        return {
            "patient_id":       str(patient_id),
            "total_visits":     len(formatted),
            "visits":           formatted,
            "all_medications":  sorted(all_meds),
            "all_db_medications":       sorted(all_db_meds),
            "all_document_medications": sorted(all_doc_meds),
            "possible_side_effects": get_possible_side_effects(sorted(all_meds), use_ai=False),
        }

    finally:
        db_pool.putconn(conn)


def _parse_vitals_json(vitals_json: dict, visit_date: str) -> list:
    """Convert structured DB vitals_json into a list of vital rows."""
    if not vitals_json:
        return []
    date_str = visit_date[:10] if (visit_date and len(visit_date) >= 10) else "Not recorded"
    rows = []
    label_map = {
        "spo2":        ("SpO₂",       lambda v: f"{v}%"),
        "pulse":       ("Pulse",      lambda v: f"{v} bpm"),
        "heart_rate":  ("Heart Rate", lambda v: f"{v} bpm"),
        "bp":          ("BP",         lambda v: f"{v} mmHg"),
        "blood_pressure": ("BP",      lambda v: f"{v} mmHg"),
        "weight":      ("Weight",     lambda v: f"{v} kg"),
        "weight_kg":   ("Weight",     lambda v: f"{v} kg"),
        "height":      ("Height",     lambda v: f"{v} cm"),
        "height_cm":   ("Height",     lambda v: f"{v} cm"),
        "temp":        ("Temperature", lambda v: f"{v}"),
        "temperature": ("Temperature", lambda v: f"{v}"),
        "rr":          ("Respiratory Rate", lambda v: f"{v} /min"),
    }
    seen = set()
    for raw_key, raw_val in vitals_json.items():
        key_lower = raw_key.lower().replace(" ", "_")
        for k, (label, fmt) in label_map.items():
            if k in key_lower and label not in seen:
                try:
                    display_val = fmt(raw_val)
                except Exception:
                    display_val = str(raw_val)
                rows.append({
                    "label":  label,
                    "value":  display_val,
                    "date":   date_str,
                    "source": "Structured record",
                })
                seen.add(label)
                break
    return rows


def fetch_all_patient_ids():
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT patient_id::text FROM visit WHERE patient_id IS NOT NULL
                UNION
                SELECT DISTINCT patient_id::text FROM document_extraction WHERE patient_id IS NOT NULL
                ORDER BY 1
                """
            )
            return [str(r[0]) for r in cur.fetchall() if r[0]]
    finally:
        db_pool.putconn(conn)
