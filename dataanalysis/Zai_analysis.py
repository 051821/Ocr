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

SECTION GUIDELINES:

### 📋 Patient Past History Summary (Date-Wise)
- Present chronological visit progression (Date, Visit # , Chief Complaint, Provisional/Confirmed Dx, kind of medicines given example antibiotic, antigastric etc on that particular visit date).
- Give symmary of complete visit all together how trend changesover time of complaints and medicines.
- Explicitly highlight any notable discrepancies (e.g. age/sex shifts across docs, diagnosis mismatches).

### 💊 Identified Medications, Purpose & Complaint Match
Present a Markdown Table with these exact columns:
| Medication | Source (Rx DB / Document) | Primary Purpose / Indication | Matches Chief Complaint? | General Potential Side Effects (Reference Only) |

### 📈 Date-Wise Vitals & Clinical Trend Analysis
- Markdown Table for vitals over visit dates (Date | BP | SpO2 | Pulse | Temp | Weight).
- Concise bullet points highlighting overall clinical trends (e.g. BP trend, weight changes).

### 🧪 Laboratory Test Evaluation (Normal vs Abnormal)
Present a Markdown Table with columns:
| Test Name | Result | Reference Range | Status (NORMAL / ABNORMAL) | Clinical Context |

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