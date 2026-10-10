"""Shared clinical review sanitization, validation, and deterministic fallbacks."""
import json
import re
from pathlib import Path

from review_facts import build_review_facts


REVIEW_SECTIONS = ("Overall Pattern", "Key Insights", "Review Points", "Evidence")
OUTPUT_SECTIONS = REVIEW_SECTIONS[:3] + ("General Patient Considerations", "Evidence")
REQUIRED_ENDING = "Retrospective record-based observations only; clinician review is required for interpretation and decision-making."
_REFERENCE = Path(__file__).resolve().parent / "data" / "reference" / "considerations.json"
with _REFERENCE.open(encoding="utf-8") as _stream:
    _CONSIDERATION_RULES = json.load(_stream)

_IMPERATIVE = re.compile(
    r"\b(?:prescribe|start|stop|increase|decrease|switch|adjust|administer|take|hold)\b"
    r".{0,80}\b(?:medication|medicine|drug|dose|tablet|capsule|\d+\s*(?:mg|mcg|ml|iu))\b",
    re.I,
)


def build_considerations(patient_data):
    facts = patient_data.get("_review_facts") or build_review_facts(patient_data)
    diagnoses = " ".join(
        str(value or "") for fact in facts
        for value in (fact.get("provisional_diagnosis"), fact.get("confirmed_diagnosis"))
    ).casefold()
    rules = []
    for keyword, messages in _CONSIDERATION_RULES.items():
        if keyword == "followup":
            has_gap = any(
                status.casefold() in {"missed", "pending", "overdue"}
                for fact in facts for status in fact.get("followup_status", [])
            )
            if has_gap:
                rules.extend(messages)
        elif keyword == "general" or keyword in diagnoses:
            rules.extend(messages)
    return list(dict.fromkeys(rules))[:5]


def _trim_incomplete_lines(lines, max_words=400):
    kept, count = [], 0
    for line in lines:
        words = len(line.split())
        if count + words > max_words:
            break
        if line.startswith("- ") and line.rstrip().endswith((",", ":")):
            break
        if line and not line.startswith(("#", "- ")) and not re.search(r"[.!?)]$", line):
            break
        kept.append(line)
        count += words
    return kept


def sanitize_review(markdown, patient_data):
    sections = {name: [] for name in REVIEW_SECTIONS}
    current = None
    for raw in str(markdown or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("```") or line == REQUIRED_ENDING:
            continue
        if line.startswith("#"):
            heading = re.sub(r"^#+\s*", "", line).strip()
            current = next((name for name in REVIEW_SECTIONS if heading.casefold() == name.casefold()), None)
            continue
        if current is None or line.startswith("|") or re.fullmatch(r"[-|: ]+", line):
            continue
        if _IMPERATIVE.search(line):
            safe = [part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip() and not _IMPERATIVE.search(part)]
            line = " ".join(safe)
            if not line:
                continue
        sections[current].append(line)

    content = {name: _trim_incomplete_lines(lines, 380) for name, lines in sections.items()}
    if not content["Overall Pattern"]:
        content["Overall Pattern"] = ["The supplied record contains limited information for a longitudinal pattern."]
    if not content["Key Insights"]:
        content["Key Insights"] = ["- No additional longitudinal insight could be verified from the supplied facts."]
    if not content["Review Points"]:
        content["Review Points"] = ["- Confirm that extracted record details match the source documents."]
    if not content["Evidence"]:
        content["Evidence"] = ["- No specific source reference was returned by the model."]

    output = ["# AI Clinical Review"]
    for section in REVIEW_SECTIONS[:3]:
        output.append(f"## {section}")
        output.extend(content[section])
    output.append("## General Patient Considerations")
    output.extend(f"- {line}" for line in build_considerations(patient_data))
    output.append("## Evidence")
    output.extend(content["Evidence"][:5])
    output.extend(("", REQUIRED_ENDING))
    return "\n".join(output)


def validate_review(markdown):
    """Validate only the final review's required Markdown structure."""
    if not markdown or not markdown.startswith("# AI Clinical Review"):
        return False
    if not markdown.rstrip().endswith(REQUIRED_ENDING):
        return False
    headings = re.findall(r"^## (.+)$", markdown, flags=re.M)
    if headings != list(OUTPUT_SECTIONS):
        return False
    overall = re.search(r"## Overall Pattern\s+(.+?)(?=\n## )", markdown, flags=re.S)
    return bool(overall and overall.group(1).strip())


def validate_model_review(markdown):
    """Check the model's four required sections before deterministic additions."""
    if not markdown or not markdown.startswith("# AI Clinical Review"):
        return False
    if not markdown.rstrip().endswith(REQUIRED_ENDING):
        return False
    headings = re.findall(r"^## (.+)$", markdown, flags=re.M)
    if headings != list(REVIEW_SECTIONS):
        return False
    overall = re.search(r"## Overall Pattern\s+(.+?)(?=\n## )", markdown, flags=re.S)
    return bool(overall and overall.group(1).strip())


def deterministic_review(patient_data):
    facts = patient_data.get("_review_facts") or build_review_facts(patient_data)
    diagnoses = sorted({
        str(value).strip() for fact in facts
        for value in (fact.get("provisional_diagnosis"), fact.get("confirmed_diagnosis"))
        if value
    })
    visits = patient_data.get("visits", []) or []
    if diagnoses:
        overall = f"The record documents {', '.join(diagnoses[:3])} across {len(visits)} encounter(s). The available facts support record review but do not establish additional conclusions."
    else:
        overall = f"The record contains {len(visits)} encounter(s), with no diagnosis text available in the normalized facts. The available facts support record review but do not establish additional conclusions."
    insights = []
    abnormal = [lab for fact in facts for lab in fact.get("labs", []) if "abnormal" in str(lab.get("status", "")).casefold() or str(lab.get("status", "")).casefold() in ("high", "low")]
    if abnormal:
        insights.append(f"- **Recorded laboratory findings:** {len(abnormal)} lab result(s) are marked abnormal in the source record.")
    discrepancies = sum(fact.get("discrepancy_count", 0) for fact in facts)
    insights.append(f"- **Record reconciliation:** {discrepancies} discrepancy item(s) are documented across the supplied encounters.")
    review_points = ["- Verify extracted details against the source documents before clinical interpretation."]
    evidence = [f"- Visit {fact['visit_id']} ({str(fact.get('date') or 'date not recorded')[:10]})." for fact in facts[-5:]]
    if not evidence:
        evidence = ["- No encounter source references are available."]
    parts = [
        "# AI Clinical Review", "## Overall Pattern", overall,
        "## Key Insights", *insights[:5], "## Review Points", *review_points,
        "## General Patient Considerations", *[f"- {item}" for item in build_considerations(patient_data)],
        "## Evidence", *evidence, "", REQUIRED_ENDING,
    ]
    return "\n".join(parts)
