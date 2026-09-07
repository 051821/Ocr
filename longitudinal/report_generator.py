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
You are a medical record longitudinal-analysis assistant. You will be given
a structured JSON payload containing an already-computed patient timeline,
laboratory trends, medication history, diagnosis history, and any detected
data conflicts or uncertain extractions.

Rules you must follow:
- Use only the information present in the JSON payload provided.
- Do not invent diagnoses, medications, laboratory results, or events.
- Do not invent or assume future events.
- All numerical trends (increase/decrease/percentage change) are already
  calculated for you in the payload — report them, do not recompute or
  contradict them.
- Do not treat entries already merged via deduplication as separate
  measurements; the payload has already deduplicated repeated values.
- Preserve all dates exactly as given.
- Clearly distinguish stated fact from your own interpretation.
- Explicitly mention any conflicting information or uncertain
  (needs_verification) extractions found in the payload.
- Do not draw unsupported causal conclusions (e.g. do not say a
  medication caused a lab value to change) unless the payload explicitly
  states that causal link.
- Do not make a definitive medical diagnosis from trends alone — describe
  observed patterns only.
""".strip()

REPORT_SECTIONS = """
Structure your response using exactly these sections:

1. Observation period
2. Demographics / data-quality issues
3. Chronic conditions
4. New diagnoses
5. Symptoms
6. Laboratory trends
7. Medication history
8. Important abnormal findings
9. Treatment changes
10. Overall clinical trajectory
11. Data conflicts / uncertainty
12. Evidence-based observations
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
