"""
Laboratory result extraction.

Supports Complete Blood Count (CBC), Serum Biochemistry, and Urine Routine
Examinations without relying on an LLM. Two coordinated extraction passes:

PASS 1 — same-line format:
    Serum Creatinine : 1.25 mg/dL 0.6-1.2
    SGOT = 13.5 U/L <46
    Blood Sugar Random : 310.87 mg/dL 70-140

PASS 2 — columnar / multi-line / qualitative format (common in scanned lab reports):
    Bacteria : ABSENT /HPF
    Pus Cells : 0-1 /HPF 0-2
    pH : 6.5 5.0 - 7.0
    Specific Gravity : 1.005 1.015-1.025
    Serum Albumin with value on preceding line (:3.96 3.5-5.2)
"""
from __future__ import annotations
import re
from normalization.laboratory import normalize_lab_name, _LAB_SYNONYMS

_UNIT_ALTS = (
    r"mg/dl|gm/dl|g/dl|u/l|iu/l|mmol/l|meq/l|%|mmhg|ng/ml|miu/l|mcg/dl|/ul|"
    r"/hpf|/lpf|millions/cumm|million/cumm|lakhs/cumm|lakh/cumm|cells/cumm|cumm|fl|pg|sec|seconds|mm/hr|ml"
)

_QUAL_RESULTS = (
    r"absent|present|nil|clear|hazy|cloudy|turbid|yellow|pale yellow|straw|"
    r"positive|negative|trace|\+{1,4}"
)

_LAB_LINE_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 ,/()\-]{1,50}?)"
    r"(?:\s*[:=]\s*|\s+)"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_UNIT_ALTS})"
    r"(?![/\w])",
    re.IGNORECASE,
)

_REF_RANGE_RE = re.compile(
    r"(?:ref(?:erence)?[,\.]?:?\s*|bio[,\.]?\s*ref[,\.]?\s*interval\s*)?"
    r"(?P<low>-?\d+(?:\.\d+)?)\s*[-to]{1,3}\s*(?P<high>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_LT_REF_RE = re.compile(r"(?:ref(?:erence)?[,\.]?:?\s*|bio[,\.]?\s*ref[,\.]?\s*interval\s*)?<=?\s*(?P<high>-?\d+(?:\.\d+)?)", re.IGNORECASE)
_GT_REF_RE = re.compile(r"(?:ref(?:erence)?[,\.]?:?\s*|bio[,\.]?\s*ref[,\.]?\s*interval\s*)?>=?\s*(?P<low>-?\d+(?:\.\d+)?)", re.IGNORECASE)

_MED_CONTEXT_WORDS = {
    "tab", "cap", "inj", "syp", "syrup", "susp", "drops",
    "od", "bd", "tds", "tid", "bid", "qid", "hs", "sos", "prn", "stat",
}

_NON_LAB_KEYWORDS = {
    "age", "sex", "age/sex", "htn", "t2dm", "dm", "diagnosis", "diagnoses",
    "guideline", "guidelines", "ada", "wallach", "wallachs", "wallach's",
    "reference", "ref", "icd", "note", "notes", "visit", "date", "doctor",
    "dr", "id", "patient", "years", "yrs", "year", "since", "chief",
    "complaint", "history", "medications", "medication", "labs", "lab",
    "pin", "plot", "code", "reg", "phone", "mob", "registration",
}

# Physiological sanity limits for blood tests to reject row-shifted or non-clinical numbers
_PHYSIOLOGICAL_LIMITS = {
    # LFT
    "Albumin": (0.5, 8.0),
    "Total protein": (1.0, 15.0),
    "Globulin": (0.5, 10.0),
    "Bilirubin total": (0.01, 40.0),
    "Bilirubin direct": (0.01, 30.0),
    "Bilirubin indirect": (0.01, 30.0),
    "AST": (1.0, 5000.0),
    "ALT": (1.0, 5000.0),
    "Alkaline phosphatase": (5.0, 3000.0),
    # KFT / Electrolytes
    "Creatinine": (0.1, 30.0),
    "Urea": (2.0, 350.0),
    "Blood urea nitrogen": (1.0, 150.0),
    "Uric acid": (0.5, 25.0),
    "Sodium": (80.0, 200.0),
    "Potassium": (1.0, 12.0),
    "Calcium": (3.0, 20.0),
    # CBC — crucial limits to prevent row-shifted values from mapping to wrong test
    "Hemoglobin": (1.0, 30.0),
    "RBC": (1.0, 9.0),                  # millions/cumm — rejects 12.7 (Hb) being mapped to RBC
    "Packed cell volume": (10.0, 70.0), # % — rejects 4.59 (RBC) being mapped to PCV
    "MCV": (40.0, 140.0),               # fL
    "MCH": (10.0, 50.0),                # pg
    "MCHC": (15.0, 45.0),               # g/dL
    "RDW": (8.0, 35.0),                 # %
    "WBC": (500.0, 150000.0),           # cells/cumm
    "Platelets": (0.2, 20.0),           # in lakhs/cumm (or 20k to 2M)
    "ESR": (0.0, 150.0),                # mm/hr
    # Glucose / Lipids
    "Random blood glucose": (10.0, 1200.0),
    "Fasting blood glucose": (10.0, 1000.0),
    "HbA1c": (2.0, 25.0),
    "Total cholesterol": (30.0, 1000.0),
    "Triglycerides": (20.0, 2500.0),
    "HDL": (5.0, 150.0),
    "LDL": (10.0, 500.0),
    "TSH": (0.001, 150.0),
    "PVC":(0.0, 100.0),
}

# Standard reference intervals to determine abnormality when unprinted in OCR report
_DEFAULT_REFERENCE_INTERVALS = {
    "RDW": (11.0, 16.0),
    "Hemoglobin": (12.0, 16.0),
    "RBC": (3.8, 5.5),
    "Packed cell volume": (36.0, 50.0),
    "MCV": (80.0, 100.0),
    "MCH": (27.0, 32.0),
    "MCHC": (31.5, 35.5),
    "WBC": (4000.0, 11000.0),
    "Platelets": (1.5, 4.5),
    "Neutrophils": (40.0, 80.0),
    "Lymphocytes": (20.0, 40.0),
    "Monocytes": (2.0, 10.0),
    "Eosinophils": (1.0, 6.0),
    "Basophils": (0.0, 2.0),
    "Epithelial cells": (0.0, 5.0),
    "Creatinine": (0.6, 1.2),
    "Urea": (10.0, 50.0),
    "Blood urea nitrogen": (6.0, 20.0),
    "Bilirubin total": (0.2, 1.2),
    "Bilirubin direct": (0.0, 0.4),
    "Bilirubin indirect": (0.2, 0.8),
    "Total protein": (6.0, 8.3),
    "Albumin": (3.5, 5.2),
    "Globulin": (2.0, 3.5),
    "AST": (0.0, 45.0),
    "ALT": (0.0, 45.0),
    "Alkaline phosphatase": (40.0, 140.0),
    "Random blood glucose": (70.0, 140.0),
    "Fasting blood glucose": (70.0, 100.0),
    "HbA1c": (4.0, 5.7),
    "Sodium": (135.0, 145.0),
    "Potassium": (3.5, 5.1),
    "HDL": (40.0, 88.0),
    "LDL": (0.0, 100.0),
    "Total cholesterol": (125.0, 200.0),
    "Triglycerides": (50.0, 150.0),
    "Calcium": (8.5, 10.5),
    "Uric acid": (3.5, 7.2),
}

# Permitted clinical units per test to reject units mistakenly copied from neighboring rows/columns
_EXPECTED_UNITS = {
    "Albumin": {"g/dl", "gm/dl", "g/l"},
    "Total protein": {"g/dl", "gm/dl", "g/l"},
    "Globulin": {"g/dl", "gm/dl", "g/l"},
    "Hemoglobin": {"g/dl", "gm/dl", "g/l"},
    "RBC": {"millions/cumm", "million/cumm", "/cumm", "cumm"},
    "WBC": {"cells/cumm", "/cumm", "cumm", "/ul"},
    "Platelets": {"lakhs/cumm", "lakh/cumm", "cells/cumm", "/cumm", "cumm"},
    "Packed cell volume": {"%"},
    "MCV": {"fl"},
    "MCH": {"pg"},
    "MCHC": {"g/dl", "gm/dl", "%"},
    "RDW": {"%"},
    "Neutrophils": {"%"},
    "Lymphocytes": {"%"},
    "Monocytes": {"%"},
    "Eosinophils": {"%"},
    "Basophils": {"%"},
    "Epithelial cells": {"/hpf", "/lpf"},
    "Creatinine": {"mg/dl", "mg/l"},
    "Urea": {"mg/dl", "mg/l"},
    "Blood urea nitrogen": {"mg/dl", "mg/l"},
    "Bilirubin total": {"mg/dl", "mg/l"},
    "Bilirubin direct": {"mg/dl", "mg/l"},
    "Bilirubin indirect": {"mg/dl", "mg/l"},
    "AST": {"u/l", "iu/l", ".u/l"},
    "ALT": {"u/l", "iu/l", ".u/l"},
    "Alkaline phosphatase": {"u/l", "iu/l", ".u/l"},
    "Random blood glucose": {"mg/dl", "mg/l"},
    "Fasting blood glucose": {"mg/dl", "mg/l"},
    "HbA1c": {"%"},
    "Sodium": {"mmol/l", "meq/l"},
    "Potassium": {"mmol/l", "meq/l", ".mmol/l"},
    "HDL": {"mg/dl", "mg/l"},
    "LDL": {"mg/dl", "mg/l"},
    "Total cholesterol": {"mg/dl", "mg/l"},
    "Triglycerides": {"mg/dl", "mg/l"},
    "Calcium": {"mg/dl", "mg/l"},
    "Uric acid": {"mg/dl", "mg/l"},
}


def _looks_like_medication_line(line_lower: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", line_lower))
    return bool(tokens & _MED_CONTEXT_WORDS)


def _looks_like_non_lab_name(name_raw: str) -> bool:
    cleaned = re.sub(r"[^a-z0-9\s/']", "", name_raw.lower()).strip()
    if cleaned in _NON_LAB_KEYWORDS:
        return True
    tokens = set(re.findall(r"[a-z0-9']+", cleaned))
    return bool(tokens & _NON_LAB_KEYWORDS)


def _extract_labs_line_based(text: str, in_urine_context: bool = False):
    """Pass 1: name, value, and unit all on the same line."""
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 200:
            continue

        line_lower = line.lower()
        if _looks_like_medication_line(line_lower):
            continue

        m = _LAB_LINE_RE.search(line)
        if not m:
            continue

        name_raw = m.group("name").strip()
        if _looks_like_non_lab_name(name_raw) or re.fullmatch(r"[\d\s./-]+", name_raw):
            continue

        try:
            value = float(m.group("value"))
        except (TypeError, ValueError):
            continue

        unit = m.group("unit").strip()

        canonical, matched = normalize_lab_name(name_raw, in_urine_context=in_urine_context)
        if not canonical:
            continue

        # Check physiological plausibility
        if canonical in _PHYSIOLOGICAL_LIMITS:
            lo_lim, hi_lim = _PHYSIOLOGICAL_LIMITS[canonical]
            if not (lo_lim <= value <= hi_lim):
                continue

        ref_low = ref_high = None
        abnormal = None
        ref_match = _REF_RANGE_RE.search(line[m.end():])
        lt_match = _LT_REF_RE.search(line[m.end():])
        gt_match = _GT_REF_RE.search(line[m.end():])
        if ref_match:
            ref_low = float(ref_match.group("low"))
            ref_high = float(ref_match.group("high"))
            abnormal = not (ref_low <= value <= ref_high)
        elif lt_match:
            ref_high = float(lt_match.group("high"))
            abnormal = value >= ref_high
        elif gt_match:
            ref_low = float(gt_match.group("low"))
            abnormal = value <= ref_low

        # Fallback to clinically standard reference intervals if unprinted in report
        if ref_low is None and ref_high is None and canonical in _DEFAULT_REFERENCE_INTERVALS:
            def_low, def_high = _DEFAULT_REFERENCE_INTERVALS[canonical]
            ref_low, ref_high = def_low, def_high
            if value is not None:
                abnormal = not (ref_low <= value <= ref_high)

        confidence = 0.85 if matched else 0.65
        if ref_match or lt_match or gt_match:
            confidence += 0.1
        confidence = min(confidence, 0.99)

        results.append({
            "name_raw": name_raw,
            "name": canonical,
            "name_matched": matched,
            "value": value,
            "qualitative_value": None,
            "unit": unit,
            "reference_low": ref_low,
            "reference_high": ref_high,
            "abnormal": abnormal,
            "confidence": round(confidence, 2),
            "source_text": line,
        })

    return results


_ALLOWED_TEST_PREFIXES = {"", "serum", "s", "s.", "blood", "total", "urine", "plasma"}
_NON_TEST_LINE_KEYWORDS = {"method", "impedence", "impedance", "calculated", "colorimetric", "analyzer", "end of report", "processed at", "sample type"}


def _is_valid_test_line(text: str, match_start: int) -> bool:
    """Ensure matched test name is on its own test line, not inside an explanatory sentence or footnote."""
    line_text = text[:match_start].split("\n")[-1].strip().lower()
    if any(kw in line_text for kw in _NON_TEST_LINE_KEYWORDS):
        return False
    line_prefix = re.sub(r"[^a-z\s\.]", "", line_text).strip()
    words = line_prefix.split()
    return all(w in _ALLOWED_TEST_PREFIXES for w in words)


def _extract_labs_proximity(text: str, in_urine_context: bool = False):
    """
    Pass 2: Columnar / multi-line search for both numeric and qualitative tests.
    Handles lines starting with ':' immediately below or near the test name.
    """
    results = []
    if not text:
        return results

    claimed_positions = []

    # Sort surfaces by length descending so "blood urea nitrogen" matches before "blood urea"
    surfaces = sorted(
        ((s, c) for c, syns in _LAB_SYNONYMS.items() for s in syns),
        key=lambda sc: -len(sc[0]),
    )

    for surface, canonical_candidate in surfaces:
        # Require word boundary
        pattern = re.compile(rf"\b{re.escape(surface)}\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            # If position already claimed by ANY test, skip to avoid double counting
            if any(abs(m.start() - pos) < 35 for pos in claimed_positions):
                continue

            # Ensure the match is an actual test line (e.g. "Serum Sodium", "HDL-Cholesterol"),
            # and NOT an entity mention inside an educational sentence (e.g. "elevate potassium level",
            # "High levels of HDL cholesterol are associated with...")
            if not _is_valid_test_line(text, m.start()):
                continue

            canonical, matched = normalize_lab_name(surface, in_urine_context=in_urine_context)

            # Window right after the matched name
            window = text[m.end(): m.end() + 120]
            backward_window = text[max(0, m.start() - 60): m.start()]
            source_snippet = text[max(0, m.start() - 20): m.end() + 60].replace("\n", " ").strip()

            value = None
            qual_val = None
            unit = None
            ref_low = ref_high = None
            abnormal = None
            back_num_match = re.search(r"[:=]\s*(-?\d+(?:\.\d+)?)\s*(?:[A-Za-z/]+)?\s*$", backward_window.strip())

            # 1. Look for value anchored by colon or equals, or immediately following newline
            colon_m = re.search(r"[:=]\s*(?P<val>[^\n\r]+)", window[:45])
            if not colon_m:
                colon_m = re.match(r"^\s*\n\s*(?P<val>-?\d+(?:\.\d+)?)\b", window[:30])

            if colon_m:
                raw_val = colon_m.group("val").strip()

                # Reject date strings (e.g. "21/07/26", "2026-08-06", "21-07-26") — these are dates, NOT lab values
                # Note: single decimal point like "25.64" is a valid float, NOT a date
                if re.search(r"^\d{1,4}[/\-]\d{1,2}|^\d{1,2}\.\d{1,2}\.\d{2,4}", raw_val):
                    raw_val = ""

                # Is it numeric? (e.g. "96", "14.43", "310.87", "30.9", "139.8", "6.5", "1.005")
                num_m = re.match(r"^(-?\d+(?:\.\d+)?)\s*(?:ml)?(?![A-Za-z0-9])", raw_val, re.IGNORECASE)
                # Is it qualitative? (e.g. "ABSENT", "CLEAR", "Yellow", "POSITIVE", "NIL", "REACTIVE", "TRACE")
                qual_m = re.match(rf"^({_QUAL_RESULTS})\b", raw_val, re.IGNORECASE)
                # Is it microscopic count range? (e.g. "0-1", "0-2 /HPF")
                range_m = re.match(r"^(\d+\s*-\s*\d+)\s*(/hpf|/lpf)?", raw_val, re.IGNORECASE)

                if num_m:
                    try:
                        val_float = float(num_m.group(1))
                        # Plausibility check against address lines like "LSnop No 30"
                        if not any(kw in window[:colon_m.end()].lower() for kw in ("no ", "no.", "reg", "phone", "pin", "code")):
                            if canonical in _PHYSIOLOGICAL_LIMITS:
                                lo_lim, hi_lim = _PHYSIOLOGICAL_LIMITS[canonical]
                                if lo_lim <= val_float <= hi_lim:
                                    value = val_float
                            else:
                                value = val_float
                    except ValueError:
                        pass
                elif qual_m:
                    qual_val = qual_m.group(1).upper()
                    unit = "/HPF" if "/hpf" in window[:60].lower() else None
                    abnormal = qual_val in ("PRESENT", "POSITIVE", "REACTIVE", "DETECTED", "TRACE") or ("+" in qual_val)
                elif range_m:
                    qual_val = range_m.group(1).strip()
                    unit = (range_m.group(2) or "").upper() or "/HPF"
                    abnormal = False

            value_from_backward = False

            # Check backward window if value not found forward (for layouts like ":3.96 \n 3.5-5.2 \n Serum Albumin")
            if value is None and qual_val is None and back_num_match:
                try:
                    val_candidate = float(back_num_match.group(1))
                    if canonical in _PHYSIOLOGICAL_LIMITS:
                        lo_lim, hi_lim = _PHYSIOLOGICAL_LIMITS[canonical]
                        if lo_lim <= val_candidate <= hi_lim:
                            value = val_candidate
                            value_from_backward = True
                    else:
                        value = val_candidate
                        value_from_backward = True
                except ValueError:
                    pass

            if value is None and qual_val is None:
                continue

            # Parse unit and reference interval — restrict unit search to same line
            same_line_window = window.split("\n")[0]
            unit_match = re.search(_UNIT_ALTS, same_line_window[:60], re.IGNORECASE)
            if unit_match and not unit:
                unit = unit_match.group(0).lower()
            if not unit and backward_window:
                same_line_back = backward_window.split("\n")[-1]
                unit_match_back = re.search(_UNIT_ALTS, same_line_back, re.IGNORECASE)
                if unit_match_back:
                    unit = unit_match_back.group(0).lower()

            # Validate unit against _EXPECTED_UNITS to reject misplaced units (e.g. 'fl' on Albumin)
            if canonical in _EXPECTED_UNITS:
                expected_set = _EXPECTED_UNITS[canonical]
                if unit and unit.lower() not in expected_set:
                    # Incompatible unit from neighboring column/row — normalize to primary expected unit
                    unit = next(iter(expected_set))
                elif not unit:
                    unit = next(iter(expected_set))

            # Ref search window: from after the colon, stopping at the NEXT test's value line
            ref_search_start = colon_m.end() if colon_m else 0
            raw_ref_window = window[ref_search_start: ref_search_start + 90]
            next_val_m = re.search(r"\n\s*[:=]\s*-?\d", raw_ref_window)
            ref_search_window = raw_ref_window[:next_val_m.start()] if next_val_m else raw_ref_window

            ref_match = _REF_RANGE_RE.search(ref_search_window)
            if not ref_match and value_from_backward and backward_window:
                ref_match = _REF_RANGE_RE.search(backward_window)
            lt_match = _LT_REF_RE.search(ref_search_window)
            gt_match = _GT_REF_RE.search(ref_search_window)

            if ref_match:
                try:
                    ref_low = float(ref_match.group("low"))
                    ref_high = float(ref_match.group("high"))
                    if value is not None:
                        abnormal = not (ref_low <= value <= ref_high)
                except ValueError:
                    pass
            elif lt_match:
                try:
                    ref_high = float(lt_match.group("high"))
                    if value is not None:
                        abnormal = value >= ref_high
                except ValueError:
                    pass
            elif gt_match:
                try:
                    ref_low = float(gt_match.group("low"))
                    if value is not None:
                        abnormal = value <= ref_low
                except ValueError:
                    pass

            # Fallback to clinically standard reference intervals if unprinted in report
            if ref_low is None and ref_high is None and canonical in _DEFAULT_REFERENCE_INTERVALS:
                def_low, def_high = _DEFAULT_REFERENCE_INTERVALS[canonical]
                ref_low, ref_high = def_low, def_high
                if value is not None:
                    abnormal = not (ref_low <= value <= ref_high)


            confidence = 0.85 if matched else 0.65

            results.append({
                "name_raw": surface,
                "name": canonical,
                "name_matched": matched,
                "value": value,
                "qualitative_value": qual_val,
                "unit": unit,
                "reference_low": ref_low,
                "reference_high": ref_high,
                "abnormal": abnormal,
                "confidence": round(confidence, 2),
                "source_text": source_snippet,
            })
            claimed_positions.append(m.start())

    return results


def extract_labs(text: str):
    """
    Returns a list of dicts:
    {name_raw, name, matched, value, qualitative_value, unit, reference_low,
     reference_high, abnormal, confidence, source_text}
    """
    results = []
    if not text:
        return results

    in_urine = bool(re.search(r"\burine\b", text, re.IGNORECASE))

    # Pass 1: same line
    pass1 = _extract_labs_line_based(text, in_urine_context=in_urine)
    results.extend(pass1)

    # Pass 2: multi-line / qualitative / columnar
    pass2 = _extract_labs_proximity(text, in_urine_context=in_urine)

    # Avoid duplicate additions between pass 1 and pass 2
    seen_keys = set()
    for item in pass1:
        seen_keys.add((item["name"].lower(), item["value"]))

    for item in pass2:
        key = (item["name"].lower(), item["value"])
        if item["value"] is not None and key in seen_keys:
            continue
        results.append(item)

    return results