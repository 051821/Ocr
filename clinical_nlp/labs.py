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
    r"mg/dl|g/dl|u/l|iu/l|mmol/l|meq/l|%|mmhg|ng/ml|miu/l|mcg/dl|/ul|"
    r"/hpf|/lpf|cells/cumm|cumm|lakhs/cumm|fl|pg|sec|seconds|mm/hr|ml"
)

_QUAL_RESULTS = (
    r"absent|present|nil|clear|hazy|cloudy|turbid|yellow|pale yellow|straw|"
    r"positive|negative|trace|\+{1,4}"
)

_LAB_LINE_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 /()\-]{1,40}?)"
    r"(?:\s*[:=]\s*|\s+)"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_UNIT_ALTS})"
    r"(?![/\w])",
    re.IGNORECASE,
)

_REF_RANGE_RE = re.compile(
    r"(?:ref(?:erence)?\.?:?\s*|bio\.?\s*ref\.?\s*interval\s*)?"
    r"(?P<low>-?\d+(?:\.\d+)?)\s*[-to]{1,3}\s*(?P<high>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_LT_REF_RE = re.compile(r"<\s*(?P<high>-?\d+(?:\.\d+)?)", re.IGNORECASE)

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

# Physiological sanity limits for blood tests to reject non-clinical serial numbers
_PHYSIOLOGICAL_LIMITS = {
    "Albumin": (0.5, 8.0),
    "Total protein": (1.0, 15.0),
    "Bilirubin total": (0.01, 40.0),
    "Bilirubin direct": (0.01, 30.0),
    "Bilirubin indirect": (0.01, 30.0),
    "Creatinine": (0.1, 30.0),
    "Urea": (2.0, 350.0),
    "Blood urea nitrogen": (1.0, 150.0),
    "Sodium": (80.0, 200.0),
    "Potassium": (1.0, 12.0),
    "AST": (1.0, 5000.0),
    "ALT": (1.0, 5000.0),
    "Alkaline phosphatase": (5.0, 3000.0),
    "Hemoglobin": (1.0, 30.0),
    "Random blood glucose": (10.0, 1200.0),
    "Fasting blood glucose": (10.0, 1000.0),
    "HbA1c": (2.0, 25.0),
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
        if ref_match:
            ref_low = float(ref_match.group("low"))
            ref_high = float(ref_match.group("high"))
            abnormal = not (ref_low <= value <= ref_high)
        elif lt_match:
            ref_high = float(lt_match.group("high"))
            abnormal = value >= ref_high

        confidence = 0.85 if matched else 0.65
        if ref_match or lt_match:
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

            # 1. The test result is ALWAYS anchored by a colon ':' or '=' directly following the test name
            colon_m = re.search(r"[:=]\s*(?P<val>[^\n\r]+)", window[:45])
            if colon_m:

                raw_val = colon_m.group("val").strip()

                # Is it numeric? (e.g. "96", "14.43", "310.87", "30.9", "139.8", "6.5", "1.005", "9ML")
                num_m = re.match(r"^(-?\d+(?:\.\d+)?)\s*(?:ml)?(?![A-Za-z0-9])", raw_val, re.IGNORECASE)
                # Is it qualitative? (e.g. "ABSENT", "CLEAR", "Yellow", "POSITIVE", "NIL")
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
                    abnormal = qual_val in ("PRESENT", "POSITIVE") or ("+" in qual_val)
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

            # Parse unit and reference interval
            unit_match = re.search(_UNIT_ALTS, window[:60], re.IGNORECASE)
            if unit_match and not unit:
                unit = unit_match.group(0).lower()
            # Also check backward window for unit (inverted layout: value before name)
            if not unit and backward_window:
                unit_match_back = re.search(_UNIT_ALTS, backward_window, re.IGNORECASE)
                if unit_match_back:
                    unit = unit_match_back.group(0).lower()

            # Ref search window: from after the colon, stopping at the NEXT test's value line.
            # We stop at the next "\n: digit" or "\n= digit" pattern — that signals another test's result,
            # preventing ref ranges from one test leaking into an adjacent test (e.g. Sodium 135-145 → Potassium).
            ref_search_start = colon_m.end() if colon_m else 0
            raw_ref_window = window[ref_search_start: ref_search_start + 90]
            next_val_m = re.search(r"\n\s*[:=]\s*-?\d", raw_ref_window)
            ref_search_window = raw_ref_window[:next_val_m.start()] if next_val_m else raw_ref_window

            ref_match = _REF_RANGE_RE.search(ref_search_window)
            # Only search backward for ref when the value itself came from the backward window (inverted layout).
            # This prevents cross-test leakage where the PREVIOUS test's ref appears in the backward window.
            if not ref_match and value_from_backward and backward_window:
                ref_match = _REF_RANGE_RE.search(backward_window)
            lt_match = _LT_REF_RE.search(ref_search_window)

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