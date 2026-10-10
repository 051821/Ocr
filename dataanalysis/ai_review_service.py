"""Cache-first orchestration shared by the Streamlit clinical review UI."""
import json
import logging

from Zai_analysis import analyze_patient_incremental, analyze_patient_with_ai
from review_facts import PROMPT_VERSION, build_review_facts, compute_source_hash, visit_hashes
from review_rules import deterministic_review, sanitize_review, validate_model_review, validate_review
from storage.ai_analysis_csv import (
    FULL_RERUN_THRESHOLD,
    get_patient_row,
    upsert_ai_analysis,
)


logger = logging.getLogger(__name__)


def _read_review(row):
    return str((row or {}).get("review_md") or "")


def _valid_review(markdown):
    """Compatibility wrapper for structure-only review validation."""
    return validate_review(markdown)


def _ensure_facts(patient_data):
    facts = patient_data.get("_review_facts")
    hashes = patient_data.get("_review_visit_hashes")
    source_hash = patient_data.get("_review_source_hash")
    if facts is None or hashes is None or source_hash is None:
        facts = build_review_facts(patient_data)
        hashes = visit_hashes(patient_data, facts)
        source_hash = compute_source_hash(patient_data.get("patient_id", ""), hashes)
        patient_data["_review_facts"] = facts
        patient_data["_review_visit_hashes"] = hashes
        patient_data["_review_source_hash"] = source_hash
    return facts, hashes, source_hash


def is_review_current(patient_data):
    """Check the cached review using source facts prepared by the caller."""
    source_hash = patient_data.get("_review_source_hash")
    if not source_hash:
        return False
    row = get_patient_row(str(patient_data.get("patient_id", "")))
    return bool(
        row
        and row.get("source_hash") == source_hash
        and row.get("prompt_version") == PROMPT_VERSION
        and _valid_review(_read_review(row))
    )


def get_or_generate_review(patient_data, force_refresh=False):
    """Return a cached review or generate and persist one review update."""
    _, current_hashes, current_hash = _ensure_facts(patient_data)
    patient_id = str(patient_data.get("patient_id", ""))
    row = get_patient_row(patient_id)
    old_hashes = json.loads((row or {}).get("visit_hashes") or "{}")
    same_source = bool(row and row.get("source_hash") == current_hash)
    current_prompt = bool(row and row.get("prompt_version") == PROMPT_VERSION)

    if same_source and current_prompt and not force_refresh:
        markdown = _read_review(row)
        if _valid_review(markdown):
            return {
                "raw_markdown": markdown,
                "model_name": row.get("model", ""),
                "model_version": "",
                "generated_at": row.get("generated_at", ""),
                "cache_hit": True,
                "status": "Cached review; no LLM call.",
            }

    mismatch_count = int((row or {}).get("mismatch_count") or 0)
    changed_ids = {
        visit_id for visit_id, value in current_hashes.items()
        if old_hashes.get(str(visit_id)) != value
    }
    should_increment = bool(
        row and row.get("source_hash") and not force_refresh and current_prompt
        and changed_ids and set(old_hashes).issubset(current_hashes)
        and mismatch_count + 1 < FULL_RERUN_THRESHOLD
    )

    if should_increment:
        changed_visits = [
            visit for visit in patient_data.get("visits", [])
            if str(visit.get("visit_id", "")) in changed_ids
        ]
        incremental_data = {**patient_data, "visits": changed_visits, "total_visits": len(patient_data.get("visits", []))}
        generate = lambda feedback=None: analyze_patient_incremental(
            _read_review(row), incremental_data, 0, None, retry_feedback=feedback
        )
        next_mismatch = mismatch_count + 1
        status = "Incremental review generated and saved."
    else:
        generate = lambda feedback=None: analyze_patient_with_ai(patient_data, retry_feedback=feedback)
        next_mismatch = 0
        status = "Review refreshed and saved."

    try:
        report = generate()
        markdown = report.get("raw_markdown", "")
        if not validate_model_review(markdown):
            report = generate("Use exactly the four required headings, a non-empty Overall Pattern, and the required exact ending.")
            markdown = report.get("raw_markdown", "")
    except Exception as exc:
        logger.warning("AI review generation failed; using deterministic record summary: %s", exc)
        report = {"raw_markdown": "", "model_name": "Deterministic fallback", "model_version": ""}
        markdown = ""
    if validate_model_review(markdown):
        markdown = sanitize_review(markdown, patient_data)
    if not _valid_review(markdown):
        markdown = deterministic_review(patient_data)
    report["raw_markdown"] = markdown
    saved_row = upsert_ai_analysis(
        patient_data,
        markdown,
        report.get("model_name", ""),
        report.get("model_version", ""),
        mismatch_count=next_mismatch,
        source_hash=current_hash,
        hashes=current_hashes,
    )
    return {**report, "generated_at": saved_row.get("generated_at", ""), "cache_hit": False, "status": status}
