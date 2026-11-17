"""Cache-first orchestration shared by the Streamlit clinical review UI."""
import json
import logging

import quick_review
from core.tracing import annotate_run, traced
from Zai_analysis import analyze_patient_incremental, analyze_patient_with_ai
from review_facts import PROMPT_VERSION, build_review_facts, compute_source_hash, visit_hashes
from review_rules import deterministic_review, sanitize_review, validate_model_review, validate_review
from storage.ai_analysis_csv import (
    FULL_RERUN_THRESHOLD,
    get_patient_row,
    upsert_ai_analysis,
)


logger = logging.getLogger(__name__)

# Model label stored with a template review. Such rows are never treated as
# current, so the next load retries the LLM instead of locking in the fallback.
FALLBACK_MODEL = "Deterministic fallback"


def _read_review(row):
    return str((row or {}).get("review_md") or "")


def _read_quick(row):
    """The stored AI quick tier ({'headline', 'lines', 'model'}) or None."""
    try:
        return json.loads(row.get("quick_json") or "null")
    except (TypeError, ValueError):
        return None


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


def _one_line_course(markdown, patient_data):
    """Course as one line (presenting · recurring), and no visit date on undated-only findings, for generated and
    cached reviews alike."""
    from clinical_insights import build_insight_evidence, refresh_record_lines
    try:
        return refresh_record_lines(markdown, build_insight_evidence(patient_data))
    except Exception as exc:  # never lose a review over the Course line
        logger.warning("Course line not applied: %s", exc)
        return markdown


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
        and row.get("model") != FALLBACK_MODEL
        and _valid_review(_read_review(row))
    )


@traced(name="ai_review.get_or_generate_review")
def get_or_generate_review(patient_data, force_refresh=False):
    """Return a cached review or generate and persist one review update."""
    _, current_hashes, current_hash = _ensure_facts(patient_data)
    patient_id = str(patient_data.get("patient_id", ""))
    row = get_patient_row(patient_id)
    old_hashes = json.loads((row or {}).get("visit_hashes") or "{}")
    same_source = bool(row and row.get("source_hash") == current_hash)
    current_prompt = bool(row and row.get("prompt_version") == PROMPT_VERSION)

    is_fallback = bool(row and row.get("model") == FALLBACK_MODEL)
    if same_source and current_prompt and not force_refresh and not is_fallback:
        markdown = _read_review(row)
        if _valid_review(markdown):
            markdown = _one_line_course(markdown, patient_data)
            annotate_run({"patient_id": patient_id, "review_mode": "cached", "model": row.get("model")},
                         tags=["review:cached"])
            return {
                "raw_markdown": markdown,
                "quick_ai": _read_quick(row),
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
        row and row.get("source_hash") and not force_refresh and current_prompt and not is_fallback
        and changed_ids and set(old_hashes).issubset(current_hashes)
        and mismatch_count + 1 < FULL_RERUN_THRESHOLD
    )

    if should_increment:
        # The update sees the whole evidence pack (it is compact) plus which encounters changed.
        generate = lambda feedback=None: analyze_patient_incremental(
            _read_review(row), patient_data, changed_ids, retry_feedback=feedback
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
            report = generate("Use exactly one heading, '## Clinical Insights', followed by 4-5 bullets (Course, "
                              "Conditions, Key findings, Treatment, Gaps to check), and the required exact ending.")
            markdown = report.get("raw_markdown", "")
    except Exception as exc:
        logger.warning("AI review generation failed; using deterministic record summary: %s", exc)
        report = {"raw_markdown": "", "model_name": FALLBACK_MODEL, "model_version": ""}
        markdown = ""
    if validate_model_review(markdown):
        markdown = sanitize_review(markdown, patient_data)
    if not _valid_review(markdown):
        markdown = deterministic_review(patient_data)
        report["model_name"], report["model_version"] = FALLBACK_MODEL, ""
    markdown = _one_line_course(markdown, patient_data)
    report["raw_markdown"] = markdown
    # Quick tier: a second, small call; None (rule-built quick tier) when the model is unavailable or fails.
    report["quick_ai"] = (quick_review.generate(patient_data)
                          if report.get("model_name") != FALLBACK_MODEL else None)
    saved_row = upsert_ai_analysis(
        patient_data,
        markdown,
        report.get("model_name", ""),
        report.get("model_version", ""),
        mismatch_count=next_mismatch,
        source_hash=current_hash,
        hashes=current_hashes,
        quick_json=report["quick_ai"],
    )
    mode = "fallback" if report.get("model_name") == FALLBACK_MODEL else ("incremental" if should_increment else "full")
    annotate_run({"patient_id": patient_id, "review_mode": mode, "model": report.get("model_name"),
                  "changed_visits": len(changed_ids), "force_refresh": force_refresh}, tags=[f"review:{mode}"])
    return {**report, "generated_at": saved_row.get("generated_at", ""), "cache_hit": False, "status": status}
