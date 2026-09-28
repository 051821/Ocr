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
You are a clinical documentation analysis AI assistant. Analyze this patient's longitudinal medical history timeline.

CRITICAL CONSTRAINTS AND MANDATORY RULES:
1. STRICTLY NO MEDICAL ADVICE / NO TREATMENT SUGGESTIONS / NO PRESCRIBING:
   - Do NOT suggest starting, stopping, increasing, decreasing, or altering any medication or dosage.
   - Do NOT provide medical treatment recommendations or diagnostic advice.

2. PATIENT PAST HISTORY ANALYSIS ONLY:
   - Base all commentary purely on documented past patient visit data.
   - Provide a retrospective summary of documented symptoms, diagnoses, and visit progression across dates.

3. DOCUMENTED MEDICATIONS & GENERAL SIDE EFFECTS:
   - Identify medications listed in the record.
   - For identified medications, list known general potential side effects strictly as reference information for patient awareness.
   - Explicitly note that side effect listing is educational reference context only.

4. DATE-WISE TREND ANALYSIS:
   - Highlight trends over time across visit dates for vitals, symptoms, or lab values.
   - State trends only when backed by documented dates.

5. LAB TEST REPORT EVALUATION (NORMAL VS ABNORMAL):
   - Evaluate documented laboratory test results against provided reference ranges.
   - Explicitly flag test values as NORMAL or ABNORMAL (High/Low).

6. GENERAL LIFESTYLE & EVERYDAY DIET SUGGESTIONS (NOT MEDICAL ADVICE):
   - This section is wellness/lifestyle information only, not a treatment or diet
     prescription, and must not mention doses, medications, or clinical management.
   - Based ONLY on the documented conditions/diagnoses for THIS patient (e.g. if
     diabetes/hypertension/eczema/gastroenteritis-type conditions appear in the
     timeline), give a short list of well-known, general everyday lifestyle and
     dietary habits people with those conditions commonly follow (e.g. reducing
     added salt/sugar, hydration, gentle skin care, food hygiene, regular sleep,
     light activity as tolerated).
   - Keep it generic and widely-known — no personalized meal plans, calorie
     targets, or specific quantities.
   - End the section with a one-line reminder to confirm any dietary or
     lifestyle change with their doctor or a registered dietitian.

Structure your dashboard analysis clearly using the following markdown headers:
### 📋 Patient Past History Summary (Date-Wise)
### 💊 Identified Medications & Potential Side Effects
### 📈 Date-Wise Vitals & Clinical Trend Analysis
### 🧪 Laboratory Test Evaluation (Normal vs Abnormal)
### 🥗 General Lifestyle & Everyday Diet Suggestions (Not Medical Advice)

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