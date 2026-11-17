"""Identify, sanity-check and evaluate lab results from any source (regex, shorthand notes, AI extraction).

Every lab result passes through validate_lab(), which uses data/reference/lab_catalog.json to:
- recognise the test under any name or shorthand ("HIAC-C", "S.Creat", "Hb");
- reject values a human body cannot have (a row number read as Sodium "9") as Unverified;
- repair a dropped decimal point ("HbA1c 51%" -> 5.1 %) but keep it Unverified until checked;
- evaluate against the standard adult range when the report gives none (marked as such);
- mark values beyond critical limits as Critical.

It also finds "Investigation advised" lists (tests ordered, not results) and parses shorthand results
written in notes ("TSH -0.4", "Hb:6.5 gm/dl", "HIV-NR"). Panel names without values ("LFT-@") are not results.
"""
import json
import re
from pathlib import Path

with (Path(__file__).resolve().parent / "data" / "reference" / "lab_catalog.json").open(encoding="utf-8") as _f:
    CATALOG = json.load(_f)["analytes"]

NORMAL, HIGH, LOW = "Normal", "Abnormal (High)", "Abnormal (Low)"
CRITICAL_HIGH, CRITICAL_LOW = "Critical (High)", "Critical (Low)"
UNVERIFIED = "Unverified (check source)"
NOT_EVALUATED = "Not Evaluated"
REACTIVE = "Abnormal (Reactive)"
DERIVED = "Derived (not interpreted)"
# Values a report calculates from another result (eAG and mean plasma glucose come from HbA1c): shown, never
# interpreted, counted or listed as an earlier result of a measured test.
_DERIVED_NAME = re.compile(r"(?i)estimated\s+average\s+glucose|\be\.?a\.?g\b|mean\s+(?:plasma|blood)\s+glucose")


def is_derived(test_name):
    return bool(_DERIVED_NAME.search(str(test_name or "")))

_NOT_SPECIFIED = {"", "not specified", "none", "n/a", "-", "null"}

# ── Test names ────────────────────────────────────────────────────────────────

_OCR_SWAPS = (("1", "l"), ("1", "i"), ("0", "o"))


def _alias_pattern(alias):
    """'s.creat' -> s[\\s.\\-_]*creat; separators inside an alias match any run of separators."""
    parts = [re.escape(p) for p in re.split(r"[\s.\-_]+", alias.strip()) if p]
    return r"[\s.,()\-_]*".join(parts)  # "Bilirubin, Direct" / "Bilirubin (direct)" -> bilirubin direct


def _compile(spec):
    aliases = set(spec["aliases"])
    for alias in list(aliases):  # hba1c -> hbalc, hbaic (OCR reads 1 as l / i)
        for digit, letter in _OCR_SWAPS:
            if digit in alias and re.search(r"[a-z]", alias):
                aliases.add(alias.replace(digit, letter))
    alternation = "|".join(sorted((_alias_pattern(a) for a in aliases), key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alternation})(?![a-z])")


_PATTERNS = {key: _compile(spec) for key, spec in CATALOG.items()}


def find_analytes(text):
    """Non-overlapping [(start, end, key)] for every test named in text, longest name winning."""
    low = str(text or "").casefold()
    found = [(m.start(), m.end(), key) for key, pattern in _PATTERNS.items() for m in pattern.finditer(low)]
    found.sort(key=lambda f: (-(f[1] - f[0]), f[0]))
    taken = []
    for start, end, key in found:
        if all(end <= s or start >= e for s, e, _ in taken):
            taken.append((start, end, key))
    return sorted(taken)


def identify(test_name):
    """Catalog key for a test name, or None. The longest matching name wins (Indirect Bilirubin, not Bilirubin)."""
    found = find_analytes(test_name)
    return max(found, key=lambda f: (f[1] - f[0], -f[0]))[2] if found else None


def clean_test_name(name):
    """'7. Serum Creatinine (Serum) (Serum Biochemistry-…) (Blood)' -> 'Serum Creatinine'."""
    name = re.sub(r"^\s*\d{1,3}\s*[.)]\s*", "", str(name or ""))
    name = re.sub(r"\s*\([^)]*\)", "", name)
    return re.sub(r"\s+", " ", name).strip(" :-|*,") or str(name).strip()


# ── Values and units ─────────────────────────────────────────────────────────

_UNITS = (
    (r"mmol\s*/\s*mol", "mmol/mol"),
    (r"(?:mmol|meq)\s*/\s*l", "mmol/l"),
    (r"[uµμ]mol\s*/\s*l", "umol/l"),
    (r"mgs?\s*/\s*dl|mg\s*%", "mg/dl"),
    (r"mg\s*/\s*mmol", "mg/mmol"),
    (r"mg\s*/\s*g(?:m|ram)?\b|(?:[uµμ]g|mcg)\s*/\s*mg", "mg/g"),
    (r"mg\s*/\s*l", "mg/l"),
    (r"g(?:ms?)?\s*/\s*dl|g(?:ms?)?\s*%", "g/dl"),
    (r"[uµμ]\s*iu\s*/\s*ml|miu\s*/\s*l|m[uµμ]\s*/\s*l", "uiu/ml"),
    (r"i?u\s*/\s*l", "u/l"),
    (r"ng\s*/\s*ml", "ng/ml"),
    (r"ng\s*/\s*dl", "ng/dl"),
    (r"pg\s*/\s*ml", "pg/ml"),
    (r"(?:[uµμ]g|mcg)\s*/\s*dl", "ug/dl"),
    (r"ml\s*/\s*min(?:\s*/\s*1\.73\s*m(?:2|²|\^2)?)?", "ml/min/1.73m2"),
    (r"(?:lakhs?|lacs?)(?:\s*/\s*cu\.?\s*mm)?", "lakh/cumm"),
    (r"(?:x\s*)?10\s*\^?\s*3\s*/\s*[uµμ]l|thou\s*/\s*[uµμ]l|k\s*/\s*[uµμ]l|(?:x\s*)?10\s*\^?\s*9\s*/\s*l", "10^3/ul"),
    (r"(?:x\s*)?10\s*\^?\s*6\s*/\s*[uµμ]l|mill?ions?\s*/\s*cu\.?\s*mm|(?:x\s*)?10\s*\^?\s*12\s*/\s*l", "10^6/ul"),
    (r"(?:cells\s*)?/\s*cu\.?\s*mm|(?:cells\s*)?/\s*[uµμ]l|/\s*mm3", "/cumm"),
    (r"f\.?l\b", "fl"),
    (r"pg\b", "pg"),
    (r"mm\s*/\s*(?:1st\s*)?h(?:ou)?r", "mm/hr"),
    (r"sec(?:onds?|s)?\b", "sec"),
    (r"%", "%"),
)
_UNIT_PATTERNS = [(re.compile(p, re.I), canon) for p, canon in _UNITS]
_UNIT_DISPLAY = {"g/dl": "g/dL", "mg/dl": "mg/dL", "mg/l": "mg/L", "mmol/l": "mmol/L", "umol/l": "µmol/L",
                 "uiu/ml": "µIU/mL", "u/l": "U/L", "ng/ml": "ng/mL", "ng/dl": "ng/dL", "pg/ml": "pg/mL",
                 "ug/dl": "µg/dL", "mg/g": "mg/g", "mg/mmol": "mg/mmol", "ml/min/1.73m2": "mL/min/1.73m²", "lakh/cumm": "lakh/cumm",
                 "10^3/ul": "×10³/µL", "10^6/ul": "×10⁶/µL", "/cumm": "/cumm", "fl": "fL", "mm/hr": "mm/hr"}

_NUMBER = re.compile(r"(?P<cmp>[<>≤≥]=?)?\s*(?P<num>\d+(?:[.,]\d+)*)")


def parse_unit(text):
    """Canonical unit at the start of text; None when there is none, '?' for an unrecognised word."""
    rest = str(text or "").strip().lstrip(":=-").strip()
    if not rest:
        return None
    for pattern, canon in _UNIT_PATTERNS:
        if pattern.match(rest):
            return canon
    return "?" if re.match(r"[a-zµμ]", rest, re.I) else None


def _to_float(text):
    """'6.5' -> 6.5, '6,5' -> 6.5, '2,50,000' -> 250000 (Indian / Western digit grouping)."""
    if "," in text:
        groups = text.split(",")
        if all(len(g) in (2, 3) for g in groups[1:]) and "." not in text:
            return float(text.replace(",", ""))
        text = text.replace(",", ".", 1).replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_value(value):
    """(comparator, number, number as written, unit) from a value string; number is None if absent."""
    match = _NUMBER.search(str(value or ""))
    if not match:
        return None, None, None, None
    return (match.group("cmp") or ""), _to_float(match.group("num")), match.group("num"), \
        parse_unit(str(value)[match.end():])


def _fmt(number):
    return f"{number:g}"


def _unit_label(unit):
    return _UNIT_DISPLAY.get(unit, unit or "")


# ── Reference ranges ─────────────────────────────────────────────────────────

def evaluate_against_reference(val_s, ref_s):
    """Status of a value against a report's own reference text (range, '< x', '> x' or a status word)."""
    if not val_s or not ref_s:
        return NOT_EVALUATED
    _, value, _, _ = parse_value(val_s)
    if value is None:
        return NOT_EVALUATED
    ref = str(ref_s).strip()
    range_m = re.search(r"([\d.]+)\s*[-–—]\s*([\d.]+)", ref)
    if range_m:
        low, high = float(range_m.group(1)), float(range_m.group(2))
        return LOW if value < low else HIGH if value > high else NORMAL
    lt_m = re.search(r"[<≤]\s*=?\s*([\d.]+)", ref)
    if lt_m:
        return NORMAL if value <= float(lt_m.group(1)) else HIGH
    gt_m = re.search(r"[>≥]\s*=?\s*([\d.]+)", ref)
    if gt_m:
        return NORMAL if value >= float(gt_m.group(1)) else LOW
    word = re.search(r"\b(normal|high|elevated|low|decreased|abnormal)\b", ref, re.I)
    if word:
        return {"normal": NORMAL, "high": HIGH, "elevated": HIGH, "low": LOW, "decreased": LOW}.get(
            word.group(1).casefold(), "Abnormal")
    return NOT_EVALUATED


def _has_reference(ref):
    return str(ref or "").strip().casefold() not in _NOT_SPECIFIED and bool(re.search(r"\d|normal|high|low", str(ref), re.I))


def _standard_range(unit_spec, sex):
    return unit_spec.get(f"ref_{sex}") if sex in ("f", "m") and unit_spec.get(f"ref_{sex}") else unit_spec.get("ref")


def _range_text(low, high, unit):
    unit = f" {_unit_label(unit)}" if unit else ""
    if low is not None and high is not None:
        return f"{_fmt(low)} - {_fmt(high)}{unit}"
    return f"≤ {_fmt(high)}{unit}" if high is not None else f"≥ {_fmt(low)}{unit}"


def sex_from(age_sex):
    """'f' / 'm' / None from an age/sex string such as '29 Yrs. F' or 'Male, 54'."""
    text = str(age_sex or "").casefold()
    if re.search(r"\b(?:f|female|fem)\b", text):
        return "f"
    if re.search(r"\b(?:m|male)\b", text):
        return "m"
    return None


# ── Qualitative and unreadable results ───────────────────────────────────────

_QUALITATIVE = (
    (re.compile(r"(?i)(?:^|[^a-z])(?:nr|non[\s-]?reactive|negative|neg|-ve|not\s+detected|absent|nil)(?![a-z])"), "Non-reactive", NORMAL),
    (re.compile(r"(?i)(?:^|[^a-z])(?:reactive|positive|pos|\+ve|detected|present)(?![a-z])"), "Reactive", REACTIVE),
)
_ILLEGIBLE = re.compile(r"(?i)^\W*$|^[\s\-:=>]*[@?#~*]+[\s\-.]*$|illegible|unreadable|not\s+legible")


def _qualitative(text):
    for pattern, label, status in _QUALITATIVE:  # negatives first: "non-reactive" also contains "reactive"
        if pattern.search(str(text or "")):
            return label, status
    return None


# ── Validation ───────────────────────────────────────────────────────────────

def _pick_unit(units, number, unit):
    """(unit, None) to evaluate with, or (None, reason) when the unit can't be settled."""
    if unit == "?":
        return None, "unit not recognised"
    if unit:
        return (unit, None) if unit in units else (None, f"unit {unit} not in catalog")
    default = next(iter(units))
    fits = [u for u, spec in units.items() if spec["plausible"][0] <= number <= spec["plausible"][1]]
    if not fits or default in fits:
        return default, None
    if len(fits) == 1:
        return fits[0], None
    return None, "unit unclear"


def _repair_decimal(number, written, low, high):
    """51 -> 5.1 when only a dropped decimal point makes the value possible."""
    if "." in written or "," in written or number < 10:
        return None
    for divisor in (10, 100):
        if low <= number / divisor <= high:
            return number / divisor
    return None


def validate_lab(lab, sex=None):
    """Return a copy of one lab result with a checked value, status, reference and an explanatory note."""
    out = dict(lab)
    out["test_name"] = clean_test_name(lab.get("test_name"))
    value = str(lab.get("value") if lab.get("value") is not None else "").strip()
    reference = lab.get("reference")
    key = identify(lab.get("test_name"))
    if key and identify(out["test_name"]) != key:  # the stripped "(direct)" was what named the test
        out["test_name"] = str(lab.get("test_name")).strip()
    spec = CATALOG.get(key) if key else None
    notes = [lab["note"]] if lab.get("note") else []

    def done(status, note=None):
        out["status"] = status
        if note:
            notes.append(note)
        if notes:
            out["note"] = " ".join(notes)
        return out

    if is_derived(lab.get("test_name")):
        return done(DERIVED, "Calculated from another result (HbA1c); not a separate measurement, so not interpreted.")
    if spec:
        out["analyte"] = spec["name"]
    if not spec:
        if not lab.get("status") or lab.get("status") == NOT_EVALUATED:
            return done(evaluate_against_reference(value, reference))
        return out

    if spec.get("qualitative"):
        result = _qualitative(value)
        if result:
            out["value"] = result[0]
            return done(result[1])
        return done(UNVERIFIED, f"Result '{value}' is not a recognised reactive / non-reactive reading.")

    if spec.get("panel"):
        return done(UNVERIFIED, f"{spec['name']} result could not be read as values ('{value}'); see source document.")

    comparator, number, written, unit = parse_value(value)
    if number is None:
        if _ILLEGIBLE.search(value):
            return done(UNVERIFIED, f"Result unreadable ('{value}'); see source document.")
        return done(lab.get("status") or NOT_EVALUATED)
    if unit is None and reference:
        unit = parse_unit(re.sub(r"^[\s\d.,<>≤≥=\-–—]+", "", str(reference))) if re.search(r"[a-z%]", str(reference), re.I) else None
        unit = None if unit == "?" else unit

    unit, problem = _pick_unit(spec["units"], number, unit)
    if problem:
        # Can't check plausibility in an unknown unit: keep the report's own evaluation.
        return done(evaluate_against_reference(value, reference) if _has_reference(reference) else NOT_EVALUATED,
                    f"Not checked against standard limits ({problem}).")
    unit_spec = spec["units"][unit]
    low, high = unit_spec["plausible"]

    repaired = None
    if not low <= number <= high:
        repaired = None if comparator else _repair_decimal(number, written, low, high)
        if repaired is None:
            return done(UNVERIFIED, f"{written} {_unit_label(unit)} is outside the possible range for {spec['name']} "
                                    f"({_fmt(low)}–{_fmt(high)} {_unit_label(unit)}); likely a misread value or row number.")
    number = repaired if repaired is not None else number
    shown = f"{comparator}{_fmt(number)}" + (f" {_unit_label(unit)}" if unit else "")

    # A percentage range that cannot belong to this test (Lymphocytes "1 - 10") was read from the wrong row
    # of the scan; percentages have no unit scale to explain the difference, so the standard range is used.
    printed = re.search(r"([\d.]+)\s*[-–—]\s*([\d.]+)", str(reference or "")) if _has_reference(reference) else None
    standard = _standard_range(unit_spec, sex)
    printed_low, printed_high = (_to_float(printed.group(1)), _to_float(printed.group(2))) if printed else (None, None)
    if (unit == "%" and printed_low is not None and printed_high is not None and standard
            and standard[0] is not None and standard[1] is not None
            and (printed_high < standard[0] or printed_low > standard[1])):
        notes.append(f"Printed range '{reference}' does not fit {spec['name']} (likely misaligned in the scan); "
                     "standard range used.")
        reference = None

    # Sex-specific range: a printed range that sits closer to the other sex's standard range (a report printing the
    # male PCV 40-50 for a woman) is replaced by the patient's own standard range.
    own_ref = unit_spec.get(f"ref_{sex}") if sex in ("f", "m") else None
    other_ref = unit_spec.get("ref_m" if sex == "f" else "ref_f") if sex in ("f", "m") else None
    if (printed_low is not None and printed_high is not None and own_ref and other_ref and None not in own_ref
            and None not in other_ref):
        to_own = abs(printed_low - own_ref[0]) + abs(printed_high - own_ref[1])
        to_other = abs(printed_low - other_ref[0]) + abs(printed_high - other_ref[1])
        if to_other < to_own:
            notes.append(f"Printed range '{reference}' matches the {'male' if sex == 'f' else 'female'} range; "
                         f"{'female' if sex == 'f' else 'male'} standard range used.")
            reference = None
            out["reference_source"] = "standard"

    # A reversed reference: '<40' printed for a test where only a LOW value matters (HDL) is the lower limit.
    lt_only = re.fullmatch(r"\s*[<≤]\s*=?\s*([\d.]+)\s*[a-zA-Z/%]*\s*", str(reference or ""))
    standard_ref = _standard_range(unit_spec, sex)
    if lt_only and standard_ref and standard_ref[0] is not None and standard_ref[1] is None:
        limit = float(lt_only.group(1))
        notes.append(f"Printed '{reference}' read as the lower limit (for {spec['name']} only a low value is adverse).")
        status = NORMAL if number >= limit else LOW
    elif _has_reference(reference):
        status = evaluate_against_reference(shown, reference)
    else:
        status = NOT_EVALUATED
    if status == NOT_EVALUATED:
        ref = _standard_range(unit_spec, sex)
        if ref and (ref[0] is not None or ref[1] is not None):
            status = LOW if ref[0] is not None and number < ref[0] else HIGH if ref[1] is not None and number > ref[1] else NORMAL
            out["reference"] = _range_text(ref[0], ref[1], unit) + " (standard adult range; report gave none)"
            out["reference_source"] = "standard"
    critical = unit_spec.get("critical") or [None, None]
    if critical[0] is not None and number < critical[0]:
        status = CRITICAL_LOW
    elif critical[1] is not None and number > critical[1]:
        status = CRITICAL_HIGH

    out["value"] = shown
    if repaired is not None:
        return done(UNVERIFIED, f"Value {value} is not possible for {spec['name']}; decimal point assumed ({shown}). "
                                f"If correct: {status}.")
    return done(status)


def _check_rbc_indices(labs):
    """Mark an RBC count Unverified when it disagrees with Hb, HCT, MCV and MCH from the same report.

    MCH = Hb x 10 / RBC and HCT = RBC x MCV / 10; when both relations fail by more than 30% while Hb and HCT
    agree with each other, the RBC count is the misread value (a row shift in the scan).
    """
    def find(key):
        for lab in labs:
            if identify(lab.get("test_name")) == key and not str(lab.get("status") or "").startswith("Unverified"):
                number = parse_value(lab.get("value"))[1]
                if number:
                    return lab, number
        return None, None

    rbc, rbc_n = find("rbc")
    hb, hb_n = find("haemoglobin")
    if not rbc or not hb:
        return
    _, mch_n = find("mch")
    _, mcv_n = find("mcv")
    _, hct_n = find("pcv")
    off = lambda expected, printed: abs(expected - printed) / printed > 0.3
    checks = []
    if mch_n:
        checks.append(off(hb_n * 10 / rbc_n, mch_n))
    if mcv_n and hct_n:
        checks.append(off(rbc_n * mcv_n / 10, hct_n))
    hb_hct_agree = not hct_n or not off(hb_n * 3, hct_n)
    if checks and all(checks) and hb_hct_agree:
        rbc["status"] = UNVERIFIED
        rbc["note"] = " ".join(filter(None, [rbc.get("note"), "RBC count does not agree with Hb, HCT and red-cell "
                                             "indices on the same report; likely misread."]))


def validate_labs(labs, sex=None):
    checked = [validate_lab(lab, sex) for lab in labs or [] if isinstance(lab, dict)]
    _check_rbc_indices(checked)
    return checked


# ── Ordered tests ("Investigation advised") ──────────────────────────────────

_ADVISED_HEADER = re.compile(
    r"(?i)^\W*(?:(?:investigations?|invx?|inv\.|tests?|labs?|lab\s+tests?|blood\s+tests?)\s*"
    r"(?:advised|adv\.?|ordered|requested|suggested|recommended|planned|to\s+be\s+done|to\s+do)\b"
    r"|(?:advised|adv\.?)\s+(?:investigations?|tests?|labs?)\b)")
_LIST_ITEM = re.compile(r"^\s*(?:\d{1,3}\s*[.)]|[-•*▪●])\s*[A-Za-z]")


def find_advised_tests(lines):
    """(line indexes that belong to 'Investigation advised' lists, ordered test names)."""
    lines = [str(line or "") for line in lines]
    excluded, names = set(), []
    i = 0
    while i < len(lines):
        header = _ADVISED_HEADER.match(lines[i])
        if not header:
            i += 1
            continue
        excluded.add(i)
        inline = lines[i][header.end():].strip(" :-")
        names += [clean_test_name(t) for t in re.split(r"[,;]", inline) if clean_test_name(t)]
        j = i + 1
        # The list runs while lines are numbered/bulleted items, or bare test names with no values.
        while j < len(lines) and (_LIST_ITEM.match(lines[j])
                                  or (find_analytes(lines[j]) and not re.search(r"\d", lines[j]))):
            excluded.add(j)
            if clean_test_name(lines[j]):
                names.append(clean_test_name(lines[j]))
            j += 1
        i = j
    return excluded, list(dict.fromkeys(names))


def is_new_test_row(line):
    """True for a line that starts another numbered/bulleted test entry."""
    return bool(_LIST_ITEM.match(str(line or "")))


# ── Shorthand results in notes ───────────────────────────────────────────────

_MED_LINE = re.compile(r"(?i)\b(?:tab|tabs|tablet|cap|caps|capsule|syp|syrup|inj|injection|oint|ointment|drops?|"
                       r"sachet|od|bd|tds|bid|tid|qid|sos|stat)\b\.?")
_SEPARATOR_WORDS = {"serum", "plasma", "blood", "level", "levels", "value", "result", "results", "is", "was", "of",
                    "fasting", "random", "report", "reported", "came", "at", "total", "count"}
_SEGMENT_VALUE = re.compile(r"(?P<sep>[^\d\n]{0,25}?)(?P<cmp>[<>≤≥]=?)?\s*(?P<num>\d+(?:[.,]\d+)*)(?P<rest>.*)", re.S)


def _separator_ok(sep):
    words = re.findall(r"[a-z]+", re.sub(r"\([^)]*\)", "", sep.casefold()))
    return all(len(w) <= 2 or w in _SEPARATOR_WORDS for w in words)


def _plausible_somewhere(spec, number, written):
    return any(u["plausible"][0] <= number <= u["plausible"][1]
               or _repair_decimal(number, written, *u["plausible"]) is not None for u in spec["units"].values())


def extract_shorthand_labs(lines, skip=()):
    """Results written as shorthand in notes: 'TSH -0.4', 'Hb:6.5 gm/dl', 'HIAC-C -51%', 'HIV/HBsAg-NR'."""
    results = []
    for i, line in enumerate(str(line or "") for line in lines):
        if i in skip or _MED_LINE.search(line):
            continue
        matches = find_analytes(line)
        for n, (start, end, key) in enumerate(matches):
            spec = CATALOG[key]
            segment = line[end:matches[n + 1][0] if n + 1 < len(matches) else len(line)]
            written = line[start:end + len(segment)].strip(" ,;")
            note = f"Written as '{written}'."
            if spec.get("qualitative"):
                result = _qualitative(segment)
                if result:
                    results.append({"test_name": spec["name"], "value": result[0], "reference": "", "note": note})
                continue
            if spec.get("panel"):
                continue  # a panel name ("LFT-@", "CBC-Ave") carries no value of its own: not a result row
            match = _SEGMENT_VALUE.match(segment)
            if not match or not _separator_ok(match.group("sep")):
                continue
            rest = match.group("rest")
            if re.match(r"\s*[/\-.]\s*\d", rest):  # a date (12/03) or a BP-style pair, not a value
                continue
            if re.match(r"\s*(?:%|[a-zµμ/]+)?\s*[-–—]\s*\d", rest, re.I):  # "is 4% - 6 %": a range, not a value
                continue
            number = _to_float(match.group("num"))
            unit = parse_unit(rest)
            if number is None or unit == "?" or (unit and unit not in spec["units"]):
                continue
            # A bare number is only taken as a result when it is possible for this test (or one decimal slip away).
            if not unit and not _plausible_somewhere(spec, number, match.group("num")):
                continue
            value = f"{match.group('cmp') or ''}{match.group('num')}" + (f" {_unit_label(unit)}" if unit else "")
            results.append({"test_name": spec["name"], "value": value, "reference": "", "note": note})
    return results


def _value_key(value):
    """'7.6 %', '7.60' and '7.6%' are one value; text values compare case- and space-insensitively."""
    text = str(value if value is not None else "").strip()
    m = re.match(r"^\s*[<>≤≥]?\s*(-?\d+(?:\.\d+)?)", text)
    return f"{float(m.group(1)):g}" if m else re.sub(r"\s+", " ", text.casefold())


def same_day_duplicate_key(date, test_name, value):
    """Key under which two lab rows are one result: same date, same test (catalog key, else cleaned name), same value.
    An undated row only matches another undated row."""
    test = identify(test_name) or clean_test_name(test_name).casefold()
    return (str(date or "undated")[:10], test, _value_key(value))


def drop_same_day_duplicates(rows, date, test, value, source=None):
    """Lab rows with a repeated result removed: the same test with the same value on the same date (a report entered
    twice, or a report image and its typed copy). Kept: a different test that day, the same test with a different
    value that day, the same test and value on another date. The kept row's 'duplicate_sources' names the copies.
    date / test / value / source are functions reading a row."""
    kept, first = [], {}
    for r in rows:
        key = same_day_duplicate_key(date(r), test(r), value(r))
        if key in first:
            if source:
                first[key].setdefault("duplicate_sources", []).append(source(r))
            continue
        first[key] = r
        kept.append(r)
    return kept
