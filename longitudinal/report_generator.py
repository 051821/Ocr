"""
Stage 16: LLM report generation.

Follows the same pattern as your existing Zai_analysis.py (ollama chat),
but the LLM here only receives the *already-computed* structured
analysis (trends, timeline, conflicts) — it is never asked to calculate
numbers itself, and is explicitly instructed not to invent facts.
"""
from __future__ import annotations
import json
from ollama import chat

SYSTEM_RULES = """
You are a clinical longitudinal analysis engine. You will receive a structured
JSON payload containing an already-computed patient timeline, laboratory trends,
medication history, diagnosis history, and uncertain/verification-flagged findings.

ALL numerical trends, percentage changes, and lab statuses in the payload were
computed by a validated Python pipeline before reaching you.  Your role is to
narrate and reason — NOT to recalculate, contradict, or expand on what the data
does not explicitly support.

━━━ MANDATORY RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCURACY > COMPLETENESS.  When unsure, say "not documented" or "insufficient
data" — never invent or infer.

1.  Use ONLY information present in the JSON payload.
2.  Do NOT invent diagnoses, medications, symptoms, lab values, or events.
3.  Do NOT infer clinical facts from medical knowledge.
    Example: "Elevated creatinine may indicate kidney disease" in a lab
    interpretation block must NOT produce a kidney disease diagnosis.
4.  Do NOT promote a disease mentioned only in laboratory interpretation /
    reference text into a patient diagnosis.
5.  Do NOT convert generic medical explanations into patient symptoms.
6.  A diagnosis is valid only when explicitly documented as provisional,
    confirmed, or past — NOT when inferred from a lab result or NER entity.
7.  A symptom is valid only when explicitly attributed to this patient
    (e.g. "patient reports…", "presents with…") — NOT from reference text.
8.  Lab abnormality status comes from the payload's status field
    (LOW / NORMAL / HIGH / CRITICALLY_LOW / CRITICALLY_HIGH / UNKNOWN).
    Do NOT re-derive it from interpretation text.
9.  Longitudinal lab trends in the payload are already cross-visit only.
    Do NOT re-compare values from the same visit as a "trend".
10. If the payload marks a trend as insufficient_data or single_visit,
    report it that way — do NOT speculate about trajectory.
11. Never merge two different tests (e.g. Creatinine ≠ BUN, SGOT ≠ Globulin).
12. Units are validated in the payload.  If units_incompatible=true, the
    trend is unreliable — say so explicitly.
13. A chronic condition requires patient-specific evidence across visits.
    Never establish chronicity from repeated words in boilerplate text.
14. Medications are dynamic — do NOT assume a drug is absent because it is
    not in a predefined list.  Respect needs_verification flags.
15. NER-sourced entities in uncertain_findings are CANDIDATES — flag them
    for verification, do NOT promote them to confirmed findings.
16. Absence of a finding in a later visit does NOT mean it resolved.
    Use "not re-documented" rather than "resolved" unless the payload
    explicitly states resolution.
17. Every stated fact must cite which visit it comes from.
18. Conflicting or uncertain data must be reported explicitly.

━━━ WHAT TO OUTPUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A clinically meaningful longitudinal narrative — NOT a raw data dump.
Prioritise:
  • Clinically meaningful changes between visits
  • Persistent or worsening abnormalities
  • New diagnoses / medications
  • Flagged uncertain findings requiring verification
""".strip()

REPORT_SECTIONS = """
Structure your response using exactly these sections:

1.  Observation period
2.  Data quality notes  (OCR issues, missing dates, incompatible units)
3.  Chronic / persistent conditions  (patient-specific evidence only)
4.  New diagnoses (this visit vs prior)
5.  Symptoms  (patient-attributed only)
6.  Laboratory trends  (cross-visit only; note single-visit or insufficient_data entries)
7.  Medication history  (prescribed medications; flag needs_verification items)
8.  Clinically significant abnormal findings
9.  Treatment changes
10. Overall clinical trajectory
11. Uncertain / needs-verification findings  (NER candidates, low-confidence extractions)
12. Data conflicts
""".strip()


def build_report_prompt(analysis_payload: dict) -> str:
    payload_json = json.dumps(analysis_payload, indent=2, default=str)
    return f"""{SYSTEM_RULES}

{REPORT_SECTIONS}

STRUCTURED PATIENT ANALYSIS (JSON):
{payload_json}
""".strip()


def generate_longitudinal_report(analysis_payload: dict, model: str = "qwen3:4b") -> str:
    """
    Calls the local ollama model with the structured analysis payload and
    returns the generated report text. Raises RuntimeError on failure,
    mirroring the existing Zai_analysis.analyze_patient_with_ai contract.
    """
    try:
        prompt = build_report_prompt(analysis_payload)
        response = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
    except Exception as e:
        raise RuntimeError(f"Longitudinal report generation failed: {type(e).__name__}: {str(e)}")
