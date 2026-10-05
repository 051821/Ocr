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
- End with: "⚠️ This is a retrospective pattern observation only — not a clinical prediction or medical advice."

### 🥗 General Lifestyle & Everyday Diet Suggestions (Not Medical Advice)
- Concise bullet points grouped by condition (e.g. HTN, T2DM, Eczema, GI).
- End with a 1-line reminder to consult a doctor or registered dietitian.

PATIENT TIMELINE DATA:
{timeline}
""".strip()

    return prompt


# ============================================================
# Main Patient Analysis
# ============================================================

def analyze_patient_with_ai(patient_data):

    # --------------------------------------------------------
    # 1. Try structured longitudinal analysis first
    # --------------------------------------------------------

    if HAS_LONGITUDINAL:
        try:
            visits = adapt_from_simple_history(patient_data)
            analysis_payload = run_longitudinal_analysis(visits)
            return generate_longitudinal_report(analysis_payload)
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
                return choices[0]["message"]["content"]

        except Exception as e:


            
            print(f"[Zai_analysis] Model '{model}' failed: {e}")
            last_exception = e
            continue

    raise RuntimeError(f"AI analysis failed across all models. Last error: {last_exception}")