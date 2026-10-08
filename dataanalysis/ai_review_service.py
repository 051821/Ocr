"""Cache-first orchestration shared by the Streamlit clinical review UI."""
import json
import logging
import re

from Zai_analysis import (
    PROMPT_VERSION,
    REQUIRED_ENDING,
    REVIEW_SECTIONS,
    analyze_patient_incremental,
    analyze_patient_with_ai,
)
from storage.ai_analysis_csv import (
    FULL_RERUN_THRESHOLD,
    compute_source_hash,
    get_patient_row,
    upsert_ai_analysis,
)

logger = logging.getLogger(__name__)


def _read_review(row):
    try:
        value = json.loads(row.get("analysis", ""))
        return value.get("raw_markdown", "") if isinstance(value, dict) else str(value)
    except (TypeError, ValueError):
        return ""


def _valid_review(markdown):
    """Validate review against clinical safety, length, and structure requirements."""
    if not markdown or len(markdown.split()) > 550 or not markdown.strip().endswith(REQUIRED_ENDING):
        return False
    if not markdown.startswith("# AI Clinical Review") or "|" in markdown:
        return False
    if any(f"## {section}" not in markdown for section in REVIEW_SECTIONS):
        return False
    if any(old in markdown.casefold() for old in ("vital signs stability", "single visit limitation")):
        return False

    contents = {name: [] for name in REVIEW_SECTIONS}
    current = None
    for line in markdown.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
        elif current in contents and line.strip() != REQUIRED_ENDING:
            contents[current].append(line.strip())

    # Ensure Overall Pattern is non-empty
    if not " ".join(contents["Overall Pattern"]).strip():
        return False

    # Check bullets
    insights = [line for line in contents["Key Insights"] if line.startswith(("- ", "* ", "• "))]
    evidence = [line for line in contents["Evidence"] if line.startswith(("- ", "* ", "• "))]
    considerations = [line for line in contents["General Patient Considerations"] if line.startswith(("- ", "* ", "• "))]

    if not (1 <= len(insights) <= 5):
        return False
    if len(evidence) > 5:
        return False
    if not (1 <= len(considerations) <= 5) and not any(contents["General Patient Considerations"]):
        return False

    # Check unsafe prescription or dosing instructions
    unsafe = re.compile(
        r"\bprescrib\w*\b|\b(?:start|stop|increase|decrease|switch|adjust)\s+(?:metformin|medication|dose|drug)\b|\b\d+\s*(?:mg|mcg|ml)\b",
        re.I,
    )
    if unsafe.search(markdown):
        return False

    # Check unsupported diagnosis claims
    unsupported_dx = re.compile(
        r"\b(?:strong likelihood of T2DM|metabolic syndrome|liver function abnormalities|may indicate anemia or dehydration)\b",
        re.I,
    )
    if unsupported_dx.search(markdown):
        return False

    return True


def is_review_current(patient_data):
    """Check if the cache contains a current, valid review for the exact patient and source hash."""
    row = get_patient_row(str(patient_data.get("patient_id", "")))
    return bool(
        row
        and row.get("source_data_hash") == compute_source_hash(patient_data)
        and row.get("prompt_version") == PROMPT_VERSION
        and _valid_review(_read_review(row))
    )


def get_or_generate_review(patient_data, force_refresh=False):
    """Return a current CSV review or generate and persist exactly one update."""
    patient_id = str(patient_data.get("patient_id", ""))
    current_hash = compute_source_hash(patient_data)
    row = get_patient_row(patient_id)
    same_source = bool(row and row.get("source_data_hash") == current_hash)
    current_prompt = bool(row and row.get("prompt_version") == PROMPT_VERSION)

    if same_source and current_prompt and not force_refresh:
        markdown = _read_review(row)
        if _valid_review(markdown):
            return {
                "raw_markdown": markdown,
                "model_name": row.get("model_name", ""),
                "model_version": row.get("model_version", ""),
                "cache_hit": True,
                "status": "Cached review; no LLM call.",
            }

    previous_count = int((row or {}).get("source_visit_count") or 0)
    mismatch_count = int((row or {}).get("hash_mismatch_count") or 0)
    try:
        previous_context = json.loads((row or {}).get("source_context") or "null")
    except (TypeError, ValueError):
        previous_context = None

    should_increment = bool(
        row
        and not force_refresh
        and current_prompt
        and not same_source
        and (int(patient_data.get("total_visits", 0)) > previous_count or previous_context)
        and mismatch_count + 1 < FULL_RERUN_THRESHOLD
    )

    if should_increment:
        report = analyze_patient_incremental(_read_review(row), patient_data, previous_count, previous_context)
        next_mismatch = mismatch_count + 1
        status = "Incremental review generated and saved."
    else:
        report = analyze_patient_with_ai(patient_data)
        next_mismatch = 0
        status = "Review refreshed and saved."

    markdown = report.get("raw_markdown", "")
    if not _valid_review(markdown):
        raise RuntimeError("AI review generation returned malformed or incomplete content; previous cache was preserved")

    upsert_ai_analysis(
        patient_data,
        {"raw_markdown": markdown},
        report.get("model_name", ""),
        report.get("model_version", ""),
        mismatch_count=next_mismatch,
    )
    return {**report, "cache_hit": False, "status": status}
