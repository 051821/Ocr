"""Post-generation check that an LLM chat answer only states record facts it was given.

Each answer line is checked on its own. A line is dropped when it cites an encounter the patient
doesn't have, a date that is not in the evidence, or a dose / lab / BP value of this patient that the
evidence doesn't contain. General medical knowledge (side effects, what a high value means) is left
alone: numbers in a line that talks about thresholds or typical ranges are not treated as record facts.

Names are checked too: a line recording a diagnosis or a prescription is dropped when that condition or
medicine is not in the evidence, and a line saying a test, medicine or condition was not recorded is
dropped when the patient's record does have it (at the encounters the line or question is about).
"""
import json
import re
from datetime import datetime

from core.tracing import traced

_NUMBER = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])")
# Encounter numbers are 1-3 digits not followed by a date separator, so "Encounter #1, 2025-01-10" is one reference.
_ENC_NUM = r"(\d{1,3})(?!\d|[/.-]\d)"
_ENCOUNTER_REF = re.compile(rf"(?i)\b(?:encounters?|enc)\s*#?\s*{_ENC_NUM}(?:\s*(?:-|–|to|and|,)\s*#?\s*{_ENC_NUM})?|#\s*{_ENC_NUM}")
_VISIT_REF = re.compile(rf"(?i)\bvisits?\s+#?\s*{_ENC_NUM}")
_DOSE = re.compile(r"(?i)(?<![\d.])(\d+(?:\.\d+)?)\s*(mg|mcg|µg|g|units?|iu|ml)\b")
_BP = re.compile(r"(?<![\d/])(\d{2,3})\s*/\s*(\d{2,3})(?![\d/])")
_LAB_VALUE = re.compile(r"(?i)(?<![\d.])(\d+(?:\.\d+)?)\s*(%|mg/dl|mmol/l|g/dl|uiu/ml|miu/l|ng/ml|pg/ml|"
                        r"ml/min(?:/1\.73\s*m2)?|meq/l|u/l|iu/l|cells/\S+|x10\S*)")
# A line using any of these is about general knowledge or reference ranges, not the patient's own values.
_THRESHOLD = re.compile(r"(?i)\b(?:above|below|over|under|target|goal|threshold|cut-?off|range|less than|"
                        r"greater than|more than|at least|up to|typically|usually|generally|normal is|"
                        r"exceed\w*|maximum|max)\b|[<>≤≥]")

# Lines stating that the record holds a diagnosis / a prescription, or that it lacks something.
_DX_CLAIM = re.compile(r"(?i)\b(?:diagnos\w*|assessment|assessed|impression|confirmed|provisional|known\s+case|k/c/o)\b")
_RX_CLAIM = re.compile(r"(?i)\b(?:prescribed|started|given|taking|takes|continued|switched|added|stopped|"
                       r"discontinued|restarted|(?:is|was|were)\s+on)\b")
_ABSENT_CLAIM = re.compile(r"(?i)\b(?:not|never|no)\s+(?:\w+\s+){0,3}?(?:recorded|documented|mentioned|available|"
                           r"done|measured|checked|tested|prescribed|found|reported)\b|\bno\s+record\b")

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DAY_MONTH_YEAR = re.compile(rf"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH},?\s+(\d{{4}})\b")
_MONTH_DAY_YEAR = re.compile(rf"(?i)\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")
_MONTH_YEAR = re.compile(rf"(?i)\b{_MONTH},?\s+(\d{{4}})\b")


def _num(text):
    return f"{float(text):g}"


def _unit(text):
    unit = text.casefold()
    return {"units": "unit", "iu": "unit", "µg": "mcg"}.get(unit, unit)


def _iso(year, month, day):
    try:
        return datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _month(name):
    return _MONTHS.index(name[:3].casefold()) + 1


def _dates(text):
    """(dates, 'YYYY-MM' months) written in any common format.

    Each date is a set of ISO readings: 04/05/2025 can be day-first or month-first, and either may be meant.
    """
    dates = [{_iso(y, m, d)} for y, m, d in _ISO_DATE.findall(text)]
    dates += [{_iso(y, b, a), _iso(y, a, b)} for a, b, y in _NUMERIC_DATE.findall(text)]
    dates += [{_iso(y, _month(mon), d)} for d, mon, y in _DAY_MONTH_YEAR.findall(text)]
    dates += [{_iso(y, _month(mon), d)} for mon, d, y in _MONTH_DAY_YEAR.findall(text)]
    dates = [readings - {None} for readings in dates if readings - {None}]
    months = {f"{y}-{_month(mon):02d}" for mon, y in _MONTH_YEAR.findall(text)}
    return dates, months


def _facts(text):
    """The evidence as separate facts, casefolded: one string per JSON object (all its plain values together, so a
    lab's test name, value and date are one fact) or per line when the evidence is not JSON."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return [line.casefold() for line in str(text or "").splitlines() if line.strip()]
    facts = []

    def walk(node):
        if isinstance(node, dict):
            plain = [str(v) for v in node.values() if not isinstance(v, (dict, list))]
            plain += [str(i) for v in node.values() if isinstance(v, list) for i in v if not isinstance(i, (dict, list))]
            if plain:
                facts.append(" ".join(plain).casefold())
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    walk(item)
                else:
                    facts.append(str(item).casefold())
        else:
            facts.append(str(node).casefold())

    walk(data)
    return facts


# A duration stated as a span ("over 4 months", "for 27 days"): it must be one the evidence gives, never the model's sum.
_SPAN = re.compile(r"(?i)\b(?:over|for|across|within|spanning|in)\s+(?:about\s+|~\s*)?(\d+(?:\.\d+)?)\s*"
                   r"(day|week|month|year)s?\b")


def _number_in(number, fact):
    return re.search(rf"(?<![\d.]){re.escape(number)}(?![\d])", fact) is not None


class _Evidence:
    """Facts the LLM was given, normalised for lookup."""

    def __init__(self, text):
        self.facts = _facts(text)
        self.numbers = {_num(n) for n in _NUMBER.findall(text)}
        self.doses = {(_num(n), _unit(u)) for n, u in _DOSE.findall(text)}
        self.bp = {(a, b) for a, b in _BP.findall(text)}
        dates, months = _dates(text)
        self.dates = set().union(*dates)
        self.months = {d[:7] for d in self.dates} | months
        self.visit_ids = {m.casefold() for m in re.findall(r"(?i)\bvisit\s+(\S+?)\b", text)}


def _line_problems(line, evidence, visit_count, medicines, lab_words):
    problems = []
    for match in _ENCOUNTER_REF.finditer(line):
        for number in filter(None, match.groups()):
            if not 1 <= int(number) <= visit_count:
                problems.append(("encounter", f"#{number}"))
    for number in _VISIT_REF.findall(line):
        if not (1 <= int(number) <= visit_count or number in evidence.visit_ids):
            problems.append(("encounter", f"visit {number}"))

    dates, months = _dates(line)
    problems += [("date", min(readings)) for readings in dates if not readings & evidence.dates]
    # "Apr 12, 2025" also matches as the month "Apr ... 2025"; that month was checked with the full date.
    covered = {d[:7] for readings in dates for d in readings}
    problems += [("date", m) for m in months - covered if m not in evidence.months]

    # Spans are computed in code and given in the evidence; one the model worked out itself is not trusted.
    for n, unit in _SPAN.findall(line):
        if not any(re.search(rf"(?<![\d.]){re.escape(n)}\s*~?\s*{unit}", fact) for fact in evidence.facts):
            problems.append(("span", f"{n} {unit}s"))
    if _THRESHOLD.search(line):
        return problems
    low = line.casefold()
    if any(name in low for name in medicines):
        problems += [("dose", f"{n} {u}") for n, u in _DOSE.findall(line) if (_num(n), _unit(u)) not in evidence.doses]
    if any(re.search(rf"\b{re.escape(word)}\b", low) for word in lab_words):
        for match in _LAB_VALUE.finditer(line):
            n, u = match.group(1), match.group(2)
            if _num(n) not in evidence.numbers:
                problems.append(("lab value", f"{n} {u}"))
                continue
            # The value must belong to the same test: the nearest lab word before it has to share a fact with it
            # ("creatinine 7.6" fails when 7.6 is the HbA1c).
            before = low[max(0, match.start() - 60):match.start()]
            named = [(before.rfind(w), w) for w in lab_words if re.search(rf"\b{re.escape(w)}\b", before)]
            if named:
                label = max(named)[1]
                if not any(label in fact and _number_in(n, fact) for fact in evidence.facts):
                    problems.append(("lab value", f"{label} {n} {u}"))
    if re.search(r"\b(?:bp|blood pressure)\b", low):
        problems += [("bp", f"{a}/{b}") for a, b in _BP.findall(line) if (a, b) not in evidence.bp]
    return problems


def _word_in(word, text):
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _line_encounters(line):
    numbers = [int(n) for m in _ENCOUNTER_REF.finditer(line) for n in m.groups() if n]
    return set(numbers + [int(n) for n in _VISIT_REF.findall(line)])


def _name_problems(line, evidence_low, drugs, conditions, lab_words, record, scope):
    """Diagnoses / medicines the line records but the evidence lacks; record facts the line calls missing."""
    low = line.casefold()
    if _ABSENT_CLAIM.search(line):
        # "Aspirin was not prescribed" is true when aspirin is nowhere in the evidence: only the record can refute it.
        if not record:
            return []
        encounters = _line_encounters(line) or scope
        text = " ".join(t for i, t in record.items() if not encounters or i in encounters)
        named = [w for w in lab_words | drugs if _word_in(w, low) and _word_in(w, text)]
        named += [name for pattern, name in conditions if pattern.search(line) and pattern.search(text)]
        return [("absent claim", name) for name in named]
    problems = []
    if _DX_CLAIM.search(line):
        problems += [("diagnosis", name) for pattern, name in conditions
                     if pattern.search(line) and not pattern.search(evidence_low)]
    if _RX_CLAIM.search(line):
        problems += [("medicine", name) for name in drugs if _word_in(name, low) and name not in evidence_low]
    return problems


@traced(name="rag.verify_answer", process_inputs=lambda i: {"answer": i.get("answer"), "visit_count": i.get("visit_count")})
def verify_answer(answer, evidence_text, visit_count, medicines=(), lab_names=(),
                  drug_names=(), conditions=(), record=None, scope=None):
    """Drop answer lines stating record facts that are not in evidence_text.

    drug_names: known medicine names (the patient's own plus a reference list) for prescription claims.
    conditions: (compiled pattern, name) pairs for diagnosis claims.
    record: {encounter number: casefolded text of the patient's whole record at it} (0 = not tied to one),
    used to reject claims that something was not recorded. scope: encounter numbers the question is about.
    Returns (answer, problems) where problems is a list of (kind, value) for every dropped claim.
    """
    evidence = _Evidence(evidence_text)
    evidence_low = evidence_text.casefold()
    medicines = {str(m).casefold() for m in medicines if m and len(str(m)) >= 4}
    drugs = medicines | {str(d).casefold() for d in drug_names if d and len(str(d)) >= 4}
    lab_words = {w for name in lab_names if name for w in re.findall(r"[a-z0-9]+", str(name).casefold()) if len(w) >= 3}
    lab_words -= {"serum", "plasma", "blood", "total", "count", "fasting", "random", "test", "level"}
    scope = set(scope or ())
    kept, problems = [], []
    for line in answer.splitlines():
        found = [] if line.lstrip().startswith("#") else (
            _line_problems(line, evidence, visit_count, medicines, lab_words)
            + _name_problems(line, evidence_low, drugs, conditions, lab_words, record, scope))
        if found:
            problems += found
        else:
            kept.append(line)
    return "\n".join(kept).strip(), problems
