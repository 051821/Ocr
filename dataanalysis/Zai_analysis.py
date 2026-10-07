import requests
from config import OPENROUTER_API_KEY
from patient_timeline import create_patient_timeline

try:
    from longitudinal.adapters import adapt_from_simple_history
    from longitudinal.longitudinal import run_longitudinal_analysis
    from longitudinal.report_generator import generate_longitudinal_report
    HAS_LONGITUDINAL = True
except ImportError:
    HAS_LONGITUDINAL = False


# ============================================================
# OpenRouter Configuration
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# Supported OpenRouter models with fallback
PRIMARY_MODEL = "inclusionai/ling-3.0-flash-sante:free"
FALLBACK_MODELS = [
    "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-001"
]


# ============================================================
# Prompt for Dashboard Analysis
# ============================================================

def build_longitudinal_analysis_prompt(patient_data):

    timeline = create_patient_timeline(patient_data)

    prompt = f"""
You are a clinical documentation analysis AI assistant. Provide a structured, concise retrospective analysis of this patient's medical timeline.

CRITICAL RULES:
1. STRICTLY NO MEDICAL ADVICE / NO PRESCRIBING / NO TREATMENT ALTERATIONS. Retrospective data summary only.
2. FORMATTING REQUIREMENTS:
   - Use clean Markdown tables for Medications, Vitals, and Lab Tests.
   - Keep summaries concise and well-organized.
   - Label each visit explicitly as "Initial Visit" (Visit #1) or "Follow-Up #N" (subsequent visits).

SECTION GUIDELINES:

### 📋 Patient Past History Summary (Date-Wise)
- For EVERY visit, show a row with: Visit # | Type (Initial / Follow-Up #N) | Date | Chief Complaint | Provisional/Confirmed Dx | Medicines class given (e.g. antihistamine, topical steroid, antibiotic).
- After the table, write a 3–5 sentence narrative summarising the overall treatment trend and how complaints evolved across visits.
- Explicitly call out any notable discrepancies (diagnosis shifts, age/sex mismatches across documents, vital value conflicts).

### 💊 Identified Medications, Purpose & Complaint Match
Present a Markdown Table with these exact columns:
| Medication | Source (Rx DB / Document) | Drug Class | Primary Purpose / Indication | Matches Chief Complaint? | General Potential Side Effects (Reference Only) |

### 📈 Date-Wise Vitals & Clinical Trend Analysis
- Markdown Table for vitals over visit dates (Date | Visit Type | BP | SpO₂ | Pulse | Temp | Weight).
- After the table, bullet points for:
  • Each vital's trend (stable / worsening / improving / concerning single reading).
  • Flag any single reading outside normal range with the specific value.

### 🧪 Laboratory Test Evaluation (Normal vs Abnormal)
Present a Markdown Table with columns:
| Test Name | Result | Reference Range | Status (NORMAL / ABNORMAL) | Clinical Context |
If no labs: state clearly "No laboratory investigations documented."

### 🔮 Clinical Condition Trajectory & Outlook (Retrospective Pattern Only)
Based ONLY on the documented visit data above, describe:
- Whether the primary condition appears to be improving, stable, or recurring based on visit frequency and complaint pattern.
- Whether medication escalation or de-escalation is observable over time (e.g. systemic steroid added then removed).
- Any patterns suggesting the condition is chronic vs acute resolution.
- What follow-up gaps (long intervals between visits) suggest about the patient's engagement or condition stability.
- **Follow-up Adherence:** Using the Follow-up Schedule data, state the overall adherence rate (X of Y completed), call out any MISSED follow-ups by date, note if any were emergencies, and interpret what missed follow-ups may indicate about the condition.
- End with: "⚠️ This is a retrospective pattern observation only — not a clinical prediction or medical advice."

### 🥗 General Lifestyle & Everyday Diet Suggestions (Not Medical Advice)
- Concise bullet points grouped by condition (e.g. HTN, T2DM, Eczema, GI).
- End with a 1-line reminder to consult a doctor or registered dietitian.

PATIENT TIMELINE DATA:
{timeline}
""".strip()

    return prompt


def build_incremental_prompt(old_analysis_markdown: str, patient_data: dict, prev_visit_count: int) -> str:
    """
    Lightweight prompt used when hash changed but mismatch_count < FULL_RERUN_THRESHOLD.
    Feeds the old analysis summary + only the new visits' timeline to the model.
    Much shorter than a full re-analysis prompt.
    """
    from patient_timeline import create_patient_timeline

    all_visits  = patient_data.get("visits", [])
    new_visits  = all_visits[prev_visit_count:]          # only visits added since last run
    new_count   = len(new_visits)
    total_count = len(all_visits)

    # Build a mini patient_data containing only the new visits for timeline
    mini_data = dict(patient_data)
    mini_data["visits"] = new_visits
    mini_data["total_visits"] = new_count
    new_timeline = create_patient_timeline(mini_data)

    return f"""
You are a clinical documentation AI. An existing retrospective analysis exists for this patient
({total_count - new_count} visits previously analysed). {new_count} new visit(s) have been added.

TASK: Update the existing analysis below to reflect the new visits. Keep all unchanged sections
as-is. Only revise sections where the new visit data changes the picture
(trajectory, medications, vitals trend, labs, discrepancies, outlook).

EXISTING ANALYSIS:
{old_analysis_markdown}

NEW VISIT(S) TIMELINE ({new_count} visit(s), visit #{total_count - new_count + 1}–{total_count}):
{new_timeline}

Return the complete updated analysis in the same Markdown format.
⚠️ Retrospective data summary only — no medical advice.
""".strip()


# ============================================================
# Main Patient Analysis
# ============================================================

def analyze_patient_with_ai(patient_data):
    """
    Run retrospective clinical analysis via AI.

    Returns a dict:
        {
            "raw_markdown": str,          # Markdown text for dashboard display
            "model_name":   str,          # model that produced the output
            "model_version": str,         # empty string – OpenRouter doesn't expose this
        }

    Callers that previously expected a plain string should use result["raw_markdown"].
    """

    # --------------------------------------------------------
    # 1. Try structured longitudinal analysis first
    # --------------------------------------------------------

    if HAS_LONGITUDINAL:
        try:
            visits = adapt_from_simple_history(patient_data)
            analysis_payload = run_longitudinal_analysis(visits)
            report_text = generate_longitudinal_report(analysis_payload)
            return {
                "raw_markdown":  report_text,
                "model_name":    "longitudinal_engine",
                "model_version": "",
            }
        except Exception as e:
            print(f"Structured longitudinal analysis failed: {e}")
            print("Falling back to OpenRouter AI models...")

    # --------------------------------------------------------
    # 2. OpenRouter API Call with Fallback Models
    # --------------------------------------------------------

    prompt = build_longitudinal_analysis_prompt(patient_data)
    models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS
    last_exception = None

    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2
            }
            if "ling" in model:
                payload["reasoning"] = {"enabled": True}

            response = requests.post(
                OPENROUTER_URL,
                headers=HEADERS,
                json=payload,
                timeout=60
            )

            response.raise_for_status()
            result = response.json()

            choices = result.get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                markdown_text = choices[0]["message"]["content"]
                return {
                    "raw_markdown":  markdown_text,
                    "model_name":    model,
                    "model_version": "",   # OpenRouter does not expose version strings
                }

        except Exception as e:
            print(f"[Zai_analysis] Model '{model}' failed: {e}")
            last_exception = e
            continue

    raise RuntimeError(f"AI analysis failed across all models. Last error: {last_exception}")


# ============================================================
# Shared OpenRouter caller
# ============================================================

def _call_openrouter(prompt: str) -> dict:
    """Call OpenRouter with fallback models. Returns same dict as analyze_patient_with_ai."""
    models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS
    last_exception = None
    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            if "ling" in model:
                payload["reasoning"] = {"enabled": True}
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=60)
            resp.raise_for_status()
            choices = resp.json().get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                return {
                    "raw_markdown":  choices[0]["message"]["content"],
                    "model_name":    model,
                    "model_version": "",
                }
        except Exception as e:
            print(f"[Zai_analysis] Model '{model}' failed: {e}")
            last_exception = e
    raise RuntimeError(f"All models failed. Last: {last_exception}")


def analyze_patient_incremental(old_markdown: str, patient_data: dict, prev_visit_count: int) -> dict:
    """
    Lightweight update: send old analysis summary + new visits only.
    Used when hash changed but mismatch_count < FULL_RERUN_THRESHOLD.
    """
    prompt = build_incremental_prompt(old_markdown, patient_data, prev_visit_count)
    return _call_openrouter(prompt)