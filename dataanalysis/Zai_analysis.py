"""Compact longitudinal clinical review generation through OpenRouter."""

import json
import logging
import os
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from config import OPENROUTER_API_KEY
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

from medication_journey import compute_medication_journey

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
PRIMARY_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-001",
    "inclusionai/ling-3.0-flash-sante:free",
]
PROMPT_VERSION = "clinical_ai_review_v5"
REQUIRED_ENDING = "Retrospective record-based observations only; clinician review is required for interpretation and decision-making."
REVIEW_SECTIONS = ("Overall Pattern", "Key Insights", "Review Points", "General Patient Considerations", "Evidence")

SYSTEM_PROMPT = f"""You are a senior clinical documentation review AI assisting a practicing physician. Provide rigorous, higher-order longitudinal synthesis rather than superficial restatements.

Use ONLY supplied normalized patient evidence. Do not duplicate the chronological visit list, full lab tables, or medication lists.
Do NOT independently diagnose conditions. Do NOT infer a diagnosis solely from an isolated laboratory value.
Do NOT claim causation without supporting evidence.
Do NOT make unsupported statements such as:
"strong likelihood of T2DM"
"metabolic syndrome"
"liver function abnormalities"
"may indicate anemia or dehydration"
Instead use objective clinical framing such as:
"The record documents..."
"The available records show..."
"A repeated finding is..."
"The available record does not establish the cause."
"This may warrant review."

Do NOT prescribe medication, suggest starting/stopping/changing medication, specify drug doses, or construct medical treatment plans.

SYNTHESIS GUIDELINES FOR HIGH CLINICAL VALUE:
1. Multi-System Evolution & Drug-Condition Dynamics: Synthesize how distinct disease domains interact over time (e.g. dermatological/infectious conditions transitioning into chronic cardiometabolic conditions, corticosteroid exposure in the setting of emerging glycemic dysregulation, or how lab findings correlate with documented complaints like polyuria/thirst).
2. Quantified Evidence: Cite specific documented parameters (e.g. HbA1c 7.6%, Triglycerides 162.99 mg/dL, PCV 37.9%) to substantiate observations.
3. Care Continuity Vulnerabilities: Specifically analyze how missed follow-ups or medication transitions leave clinical responses unverified.
4. Bold Topic Leads: Prefix every insight bullet with a concise bold topic lead (e.g. **Cardiometabolic & Steroid Dynamics:**, **Care Continuity & Pharmacotherapy Transition:**, **Hematological Baseline vs Inflammatory Context:**, **Cross-Source Medication Reconciliation:**).

GENERAL PATIENT CONSIDERATIONS (PRACTICAL DIET & LIFESTYLE GUIDANCE):
Include 3–5 condition-specific, practical self-care and dietary dialogue points tailored to the patient's actual documented conditions (e.g. T2DM, elevated lipids, hypertension, eczema, gastroenteritis):
- Concrete Dietary & Nutritional Guidance:
  * For glycemic/metabolic findings: practical carbohydrate distribution, portion awareness, prioritizing complex fiber-rich foods, and minimizing refined sugars/sweetened beverages.
  * For elevated triglycerides/lipids: reducing trans-fats, deep-fried snacks, and saturated fats in favor of heart-healthy unsaturated fats and whole grains.
  * For hypertension: moderate sodium restriction and avoiding heavily processed salty foods.
  * For post-gastrointestinal or dehydration complaints: steady daily hydration and easily digestible fluids.
- Condition-Specific Self-Care & Monitoring:
  * For chronic eczema/skin conditions: gentle emollient/moisturizer application within 3 minutes of lukewarm bathing to protect the epidermal barrier.
  * For glycemic therapy: patient awareness of early warning signs of hypoglycemia (sweating, tremors, dizziness) and hyperglycemia (extreme thirst, polyuria), with clear thresholds for seeking medical attention.
  * For care continuity: prompt re-engagement on missed follow-up appointments.
Prefix each consideration with a bold label (e.g. **Dietary Carbohydrate & Glycemic Management:**, **Lipid & Heart-Healthy Nutrition:**, **Dietary Sodium & Blood Pressure:**, **Skin Barrier & Emollient Care:**, **Hydration & Symptom Awareness:**, **Care Continuity & Follow-Up:**).

Return exactly this Markdown structure and no other headings:
# AI Clinical Review

## Overall Pattern
2–3 concise sentences describing the multi-system longitudinal trajectory.

## Key Insights
3–5 concise bullets with bold topic leads highlighting high-order patterns, cross-condition interactions, or reconciliation gaps.

## Review Points
0–3 bullets highlighting clinical audit items warranting physician verification.

## General Patient Considerations
3–5 practical condition-tailored dietary, lifestyle, and monitoring bullets with bold topic leads.

## Evidence
Maximum 5 concise references citing dates, visits, or documents.

Target: 300–480 words. Hard maximum: 550 words.
End exactly with:
{REQUIRED_ENDING}"""


def _compact(value):
    """Drop empty values and normalize values for compact JSON context."""
    if isinstance(value, dict):
        return {k: v for k, raw in value.items() if (v := _compact(raw)) not in (None, "", [], {})}
    if isinstance(value, list):
        return [v for raw in value if (v := _compact(raw)) not in (None, "", [], {})]
    if isinstance(value, str):
        return value.strip() or None
    return value


def build_ai_review_context(patient_data):
    """Build a structured, OCR-text-free compact evidence set for longitudinal review."""
    visits = []
    # Compute deterministic medication shifts to avoid huge repeated med lists
    try:
        med_shifts = compute_medication_journey(patient_data)
        shift_by_id = {str(item["visit_id"]): item for item in med_shifts}
    except Exception:
        shift_by_id = {}

    for number, visit in enumerate(patient_data.get("visits", []), 1):
        v_id = str(visit.get("visit_id", number))
        documents = []
        for doc in visit.get("documents", []) or []:
            summary = doc.get("clinical_summary") or {}
            documents.append({
                "document_id": doc.get("doc_id"),
                "clinical_summary": {
                    key: summary.get(key)
                    for key in (
                        "clinical_impression", "clinical_notes", "chief_complaint",
                        "provisional_diagnosis", "confirmed_diagnosis", "symptoms",
                    )
                    if summary.get(key)
                },
                "extracted_meds": summary.get("medications") or doc.get("medications_found"),
                "labs": summary.get("lab_results") or doc.get("lab_results"),
            })

        v_shifts = [c.get("text") for c in shift_by_id.get(v_id, {}).get("changes", [])]

        visits.append(_compact({
            "visit_number": number,
            "visit_id": visit.get("visit_id"),
            "date": (visit.get("visit_date") or "")[:10],
            "chief_complaint": visit.get("db_chief_complaint") or visit.get("chief_complaint"),
            "provisional_diagnosis": visit.get("db_provisional_dx") or visit.get("provisional_diagnosis"),
            "confirmed_diagnosis": visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis"),
            "medication_shifts": v_shifts if v_shifts else visit.get("db_medications"),
            "labs": visit.get("lab_results"),
            "followups": visit.get("followups"),
            "discrepancies": visit.get("discrepancies"),
            "documents": documents,
        }))

    return _compact({
        "patient_id": patient_data.get("patient_id"),
        "total_visits": patient_data.get("total_visits", len(visits)),
        "visits": visits
    })


def build_ai_review_prompt(patient_data):
    context = json.dumps(build_ai_review_context(patient_data), ensure_ascii=False, separators=(",", ":"))
    return f"{SYSTEM_PROMPT}\n\nLONGITUDINAL PATIENT EVIDENCE (JSON):\n{context}"


def build_incremental_prompt(old_analysis_markdown, patient_data, prev_visit_count, previous_context=None):
    full_context = build_ai_review_context(patient_data)
    if previous_context:
        prior_visits = {str(v.get("visit_id", v.get("visit_number"))): v for v in previous_context.get("visits", [])}
        changed = [v for v in full_context.get("visits", []) if prior_visits.get(str(v.get("visit_id", v.get("visit_number")))) != v]
    else:
        changed = full_context.get("visits", [])[max(0, prev_visit_count):]

    if not changed:
        changed = full_context.get("visits", [])[-1:]

    context = {"patient_id": full_context.get("patient_id"), "total_visits": len(changed), "visits": changed}
    previous = str(old_analysis_markdown or "")
    words = previous.split()
    previous = " ".join(words[:450])
    delta = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Update the existing review only if the new evidence materially changes it. "
        "Otherwise preserve the previous review unchanged. "
        "Return the complete final review, retaining the required headings and exact ending.\n\n"
        f"PREVIOUS REVIEW:\n{previous}\n\n"
        f"NEW OR CHANGED INFORMATION ONLY (JSON):\n{delta}"
    )


def _call_openrouter(prompt):
    last_error = None
    for model in [PRIMARY_MODEL] + FALLBACK_MODELS:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt.replace(SYSTEM_PROMPT + "\n\n", "", 1)}
                ],
                "temperature": 0.15,
                "max_tokens": 750
            }
            if "ling" in model:
                payload["reasoning"] = {"enabled": True}
            response = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=60)
            response.raise_for_status()
            choices = response.json().get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                return {
                    "raw_markdown": _sanitize_review(choices[0]["message"]["content"]),
                    "model_name": model,
                    "model_version": ""
                }
        except Exception as exc:
            logger.warning("[Zai_analysis] Model '%s' failed: %s", model, exc)
            last_error = exc
    raise RuntimeError(f"AI analysis failed across all models. Last error: {last_error}")


def _sanitize_review(markdown):
    """Normalize required structure, remove tables/unsupported diagnoses/prescriptions, and cap at 550 words."""
    sections = {name: [] for name in REVIEW_SECTIONS}
    current = None
    skip = False

    # Regex filters for unsafe medication changes or prescribing
    unsafe_prescription = re.compile(
        r"\b(?:prescrib\w*|antibiotic\w*)\b|\b(?:start|stop|increase|decrease|switch|adjust|administer|take|cease)\b.{0,60}\b(?:medicat\w*|treatment|drug|dose|antibiotic\w*|pill|therapy|regimen|metformin)\b|\b\d+\s*(?:mg|mcg|ml|units?)\b",
        re.I
    )

    # Unsupported diagnostic phrasing replacements
    unsupported_replacements = [
        (re.compile(r"\bstrong likelihood of T2DM\b", re.I), "The available record documents glycemic elevation; clinician review is required to evaluate diagnosis."),
        (re.compile(r"\bmetabolic syndrome\b", re.I), "documented metabolic parameters"),
        (re.compile(r"\bliver function abnormalities\b", re.I), "recorded liver enzyme variations"),
        (re.compile(r"\bmay indicate anemia or dehydration\b", re.I), "The available record does not establish the cause."),
    ]

    for raw_line in str(markdown).splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            heading = re.sub(r"^#+\s*", "", line).strip()
            current = next((name for name in REVIEW_SECTIONS if heading.lower() == name.lower() or heading.lower().startswith(name.lower())), None)
            skip = current is None
            continue
        if skip or not line or line == REQUIRED_ENDING or line.startswith("|") or re.fullmatch(r"[-|: ]+", line):
            continue

        # Neutralize unsupported phrases
        for pattern, replacement in unsupported_replacements:
            line = pattern.sub(replacement, line)

        # Filter unsafe prescription / dose advice (line or sentence level)
        if unsafe_prescription.search(line):
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
            safe_sentences = [s for s in sentences if not unsafe_prescription.search(s)]
            line = " ".join(safe_sentences).strip()
            if not line:
                continue

        if current:
            cap = {"Key Insights": 5, "Review Points": 3, "General Patient Considerations": 5, "Evidence": 5}.get(current)
            if cap is not None and line.startswith(("- ", "* ", "• ")) and sum(x.startswith(("- ", "* ", "• ")) for x in sections[current]) >= cap:
                continue
            sections[current].append(line)
        elif not any(sections["Overall Pattern"]):
            sections["Overall Pattern"].append(line)

    # Build sanitized output
    out = ["# AI Clinical Review"]
    for name in REVIEW_SECTIONS:
        out.append(f"## {name}")
        content = sections[name]
        if content:
            out.extend(content)
        elif name == "Review Points":
            out.append("- No additional review points identified.")
        elif name == "General Patient Considerations":
            out.append("- Discuss practical meal choices and portion awareness in the context of documented findings.")
            out.append("- Maintain appropriate daily activity as tolerated and according to clinician advice.")
            out.append("- Keep scheduled follow-up and monitoring appointments.")
        elif name == "Key Insights":
            out.append("- Documented longitudinal patterns across encounters warrant clinician review.")

    body = "\n".join(out)

    # Ensure hard limit of 550 words
    words = body.split()
    ending_words = REQUIRED_ENDING.split()
    if len(words) > 550 - len(ending_words):
        body = " ".join(words[:550 - len(ending_words)])

    return body.rstrip() + "\n\n" + REQUIRED_ENDING


def analyze_patient_with_ai(patient_data):
    return _call_openrouter(build_ai_review_prompt(patient_data))


def analyze_patient_incremental(old_markdown, patient_data, prev_visit_count, previous_context=None):
    return _call_openrouter(build_incremental_prompt(old_markdown, patient_data, prev_visit_count, previous_context))
