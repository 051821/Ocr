"""Compact longitudinal clinical review generation through OpenRouter."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
MODULE_DIR = str(Path(__file__).resolve().parent)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from review_facts import PROMPT_VERSION
from medication_journey import compute_medication_journey
from core.llm_client import complete as llm_complete
from review_rules import REQUIRED_ENDING, REVIEW_SECTIONS, sanitize_review as _sanitize

SYSTEM_PROMPT = """You are a clinical-record review assistant. Analyze the patient’s complete medical journey using only supplied records. Summarize the timeline, documented conditions, investigations, treatments, clinical progress, response, complications, and unresolved issues.

**Prioritize red flags:** critical or worsening results, clinical deterioration, emergency indicators, persistent symptoms, abnormal trends, potential medication-related concerns, and gaps in follow-up. Explain their significance only when supported by evidence. Distinguish documented facts, uncertainties, and missing information. Highlight key turning points, conflicting records, and issues requiring clinician review. Never invent facts, diagnose, prescribe, recommend medication changes, or provide lifestyle advice.

Use exactly these Markdown headings:

## Overall Pattern
## Key Insights
## Review Points
## Evidence

Be concise, chronological where useful, and evidence-based. Include dates, values, units, and source references when available. Keep the review between 250 and 400 words. End with:

**Retrospective review only: based on supplied records, not a real-time assessment or a substitute for clinician judgment.**"""


def build_ai_review_context(patient_data):
    """Build a concise, duplicate-free summary from deterministic visit facts."""
    journey = patient_data.get("_medication_journey_cache")
    if journey is None:
        journey = compute_medication_journey(patient_data)
    shifts = {str(item.get("visit_id")): item for item in journey}
    visits = []
    for number, visit in enumerate(patient_data.get("visits", []), 1):
        visit_id = str(visit.get("visit_id", number))
        labs = visit.get("lab_results", []) or []
        abnormal = [
            [str(lab.get("test_name") or lab.get("name") or "")[:36], str(lab.get("value") or "")[:24], str(lab.get("status") or "")[:18]]
            for lab in labs
            if any(flag in str(lab.get("status") or "").casefold() for flag in ("abnormal", "high", "low"))
        ]
        changes = [str(change.get("text") or "")[:100] for change in shifts.get(visit_id, {}).get("changes", [])[:8]]
        followups = [str(item.get("status") or "")[:20] for item in visit.get("followups", [])[:5]]
        diagnosis = visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis") or visit.get("db_provisional_dx") or visit.get("provisional_diagnosis")
        visits.append({
            "id": visit_id,
            "date": str(visit.get("visit_date") or "")[:10],
            "cc": str(visit.get("db_chief_complaint") or visit.get("chief_complaint") or "")[:100],
            "dx": str(diagnosis or "")[:100],
            "med_changes": changes,
            "abnormal_labs": abnormal[:8],
            "normal_lab_count": sum(1 for lab in labs if str(lab.get("status") or "").casefold() == "normal"),
            "followup": followups,
            "discrepancy_count": len(visit.get("discrepancies", []) or []),
            "doc_ids": [str(doc.get("doc_id")) for doc in (visit.get("documents") or [])[:8] if doc.get("doc_id") is not None],
        })
    payload = {"patient_id": patient_data.get("patient_id"), "total_visits": len(visits), "visits": visits}
    while len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) > 4000 and len(payload["visits"]) > 1:
        payload["visits"].pop(0)
    return payload


def build_ai_review_prompt(patient_data, retry_feedback=None):
    context = json.dumps(build_ai_review_context(patient_data), ensure_ascii=False, separators=(",", ":"))
    feedback = f"\n\nCorrection required: {retry_feedback}" if retry_feedback else ""
    return f"Return only the required four-section Markdown review. End with this exact line: {REQUIRED_ENDING}\nLONGITUDINAL PATIENT FACTS (JSON):\n{context}{feedback}"
def build_incremental_prompt(old_analysis_markdown, patient_data, prev_visit_count=0, previous_context=None):
    """Build an update prompt containing only changed visits and prior review."""
    current = json.dumps(build_ai_review_context(patient_data), ensure_ascii=False, separators=(",", ":"))
    kept, words = [], 0
    for line in str(old_analysis_markdown or "").splitlines():
        count = len(line.split())
        if words + count > 260:
            break
        kept.append(line)
        words += count
    previous = "\n".join(kept)
    return (
        f"Update the review only when the supplied changed evidence warrants it. Preserve the required headings and exact ending: {REQUIRED_ENDING}\n\n"
        f"PREVIOUS REVIEW:\n{previous}\n\nCHANGED VISIT FACTS (JSON):\n{current}"
    )
def _call_openrouter(prompt, retry_feedback=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if retry_feedback:
        messages.append({"role": "user", "content": f"Fix the structure issue: {retry_feedback}"})
    result = llm_complete(messages, "review", temperature=0.1, max_tokens=600)
    return {"raw_markdown": result["content"], "model_name": result["model"], "model_version": ""}


def _sanitize_review(markdown, patient_data=None):
    return _sanitize(markdown, patient_data or {})
def analyze_patient_with_ai(patient_data, retry_feedback=None):
    return _call_openrouter(build_ai_review_prompt(patient_data, retry_feedback))


def analyze_patient_incremental(old_markdown, patient_data, prev_visit_count=0, previous_context=None, retry_feedback=None):
    prompt = build_incremental_prompt(old_markdown, patient_data, prev_visit_count, previous_context)
    if retry_feedback:
        prompt += f"\n\nCorrection required: {retry_feedback}"
    return _call_openrouter(prompt)
