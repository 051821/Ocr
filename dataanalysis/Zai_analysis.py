"""Patient-specific clinical review generation through OpenRouter.

The model writes the narrative sections from the patient's evidence pack (clinical_insights), never from
raw visits; review_rules then adds the record-based lab and evidence sections and checks every model line.
"""

import sys
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
MODULE_DIR = str(Path(__file__).resolve().parent)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from review_facts import PROMPT_VERSION
from clinical_insights import build_insight_evidence, evidence_for_prompt
from core.llm_client import complete as llm_complete
from core.tracing import traced
from review_rules import REQUIRED_ENDING, REVIEW_SECTIONS, sanitize_review as _sanitize, validate_model_review
from core.config import flag as _flag

SYSTEM_PROMPT = """You are a clinical-record review assistant writing a short, patient-specific clinical brief for the treating clinician. Use only the PATIENT EVIDENCE JSON supplied; it was extracted from this patient's record.

How to read the evidence:
- conditions[].status: "confirmed", "working diagnosis" (provisional, recorded by the clinician), "known history" (stated in a complaint, e.g. K/C/O, or held on the patient's long-term condition list or health history), "suspected" (recorded with "?" or as under evaluation) or "document only" (in a scanned document, not the structured record). conditions[].long_term names the patient-level source (long-term condition list, health history): such a condition is current even when no visit records it, and must be mentioned.
- possible_conditions: conditions that out-of-range results point to but the record never diagnoses.
- abnormal_labs: out-of-range results with value, unit, reference range and its source, date, encounters (#n), earlier and later results of the same test, plain-language significance, related_conditions (the result is that recorded condition's control marker), related_medicines, also_relevant_to and in_range_context (same-day related tests that were within range).
- medicines: name, class (what the medicine is, e.g. "NSAID (pain / anti-inflammatory)"), regimen, dates, whether prescribed at the latest visit, and "for" (the recorded condition the record links it to).
- rule_flags: rule-based findings; kind is condition, pattern (how a problem behaves across visits: long-standing symptoms, not settling, visits closer together, diagnosis evolution), risk, medication, followup or data.
- health_history: conditions and habits the patient reported in the intake health-history questionnaire, with any notes or quantity. This is patient-reported history, not a recorded diagnosis: write it as "reports a history of ..." unless conditions also records it.
- clinician_notes: free text the clinician wrote at a visit (advice or impressions). A condition named only here is "noted by the clinician", not a recorded diagnosis.
- recurring_complaints: symptoms raised at two or more visits, with the visit count.
- latest_vitals: the latest structured readings. A value marked "impossible value" is an entry error: never interpret it.
- followup_gaps, record_checks, timeline (#n is the encounter number).

Rules:
- Every statement must come from the evidence. Quote values exactly with units; add the reference range in brackets, e.g. "(ref <range>)", for a value outside it. When a result appears at more than one encounter, say "reported twice" (or the count).
- Diagnoses: use the recorded status (confirmed, working diagnosis, known history, suspected, documented only in a report). A possible condition is written as "may indicate ..." or "possible ..., not diagnosed in the record", with the results or symptoms that raise it. Never name a condition that is in neither conditions nor possible_conditions.
- Medicines and conditions: say a medicine is for a condition only when that condition is in the medicine's "for". When "for" is empty, name the medicine with its class instead (e.g. "<medicine> (<class>)"); never guess what it treats.
- For every out-of-range result, say in a few plain words what it can mean (use its significance), and add its in_range_context when present (e.g. "while <related test> is within range").
- Medication considerations: say what a recorded medicine can do to this patient's findings or conditions (from rule_flags of kind medication, or the significance of a related result). Never recommend starting, stopping or changing a medicine or a dose.
- Follow-up: say what the record shows as due or missing (out-of-range results not repeated, advised tests without results, missed follow-ups, conditions without their monitoring result).
- Describe control exactly as the evidence shows it. Do not call a condition uncontrolled when its latest reading is within range. Hypertension control is shown by the BP readings; never call it unassessable when BP is recorded.
- Mention a normal value only as context for a related abnormal one; never report normal vitals or normal results on their own. Do not repeat a point or a medicine list, and do not add generic statements that would fit any patient. No questions, no lifestyle advice, no generic education. Do not state that something is absent (no labs, no other abnormal findings, no long-term conditions) unless the evidence lists it as a gap. Never write the rule-flag levels (high, medium, info).

Write a brief the clinician can read in under a minute to understand this patient's history and current condition. Use these five themes, in this order; leave a theme out only when the evidence has nothing for it:
1. **Course:** age/sex, number of visits and the period, and what the patient presented with over time (recurring_complaints, and each visit's complaint from timeline, or the first and latest when there are many); a long-standing symptom's stated duration (rule_flags of kind pattern).
2. **Conditions:** what is recorded at the latest visit, with its status and how many visits it spans; long-term and known history conditions; patient-reported health_history (including smoking, alcohol, drug allergies, surgeries and hospitalisations) where it bears on the conditions or findings; conditions not recorded since an earlier visit; course patterns from rule_flags of kind pattern.
3. **Key findings:** the most important out-of-range results with what each can mean, whether the record diagnoses what they point to (possible_conditions), and symptoms or vitals flagged in rule_flags (high and medium level first).
4. **Treatment:** medicine risks for this patient from rule_flags of kind medication first (dose above the usual maximum, a medicine that can worsen a recorded condition or reading); then medicines at the latest visit by name, with the recorded condition each is for or its class in brackets (never a list of classes alone); a chronic condition whose medicines stopped appearing; a recorded condition whose control marker is out of range on its medicine.
5. **Gaps to check:** what the record shows as missing, due or inconsistent (followup_gaps, missed follow-ups, record_checks, impossible readings).

Write analysis, not a summary: connect facts where the evidence links them (a result to its recorded condition and medicines, a symptom to a condition, a finding the record never diagnoses), and lead with what that link means, e.g. "<condition> above range: <result> despite <medicines>". Put the clinically most important point first in each theme.

Format: start with the heading line "## Clinical Insights", then 4-5 bullets, one per theme. Inside a bullet, write 2-4 short points separated by "; ". A point is a phrase of at most 15 words, optionally starting with a short label and a colon ("Active: ...", "Recurring complaints: ..."). No full paragraphs, no filler words.

## Clinical Insights
- **Course:** <point>; <point>
- **Conditions:** <point>; <point>
- **Key findings:** <point>; <point>; <point>
- **Treatment:** <point>; <point>
- **Gaps to check:** <point>; <point>

Each bullet at most 45 words; the whole list under 180 words. The full lab table and the source encounters are added from the record, so do not write them. End with this exact line:

""" + REQUIRED_ENDING

# When ENABLE_EXTERNAL_PROMPTS is on, load the system prompt from the prompt
# file and substitute the canonical REQUIRED_ENDING so both the prompt and the
# validator agree on the exact ending text.
if _flag("ENABLE_EXTERNAL_PROMPTS"):
    try:
        from core.prompts import load as _load_prompt
        SYSTEM_PROMPT = _load_prompt("review_system.md").format(required_ending=REQUIRED_ENDING)
    except Exception:
        pass  # keep the hardcoded fallback


def build_ai_review_context(patient_data):
    """The patient's evidence pack: conditions, medicines, abnormal labs, flags and follow-up gaps."""
    return build_insight_evidence(patient_data)


@traced(name="ai_review.build_prompt")
def build_ai_review_prompt(patient_data, retry_feedback=None):
    context = evidence_for_prompt(build_ai_review_context(patient_data))
    feedback = f"\n\nCorrection required: {retry_feedback}" if retry_feedback else ""
    if _flag("ENABLE_EXTERNAL_PROMPTS"):
        try:
            return _load_prompt("review_user.md").format(
                required_ending=REQUIRED_ENDING, context=context, feedback=feedback
            )
        except Exception:
            pass
    return ("Write the Clinical Insights for this patient. End with this exact line: "
            f"{REQUIRED_ENDING}\nPATIENT EVIDENCE (JSON):\n{context}{feedback}")


@traced(name="ai_review.build_incremental_prompt")
def build_incremental_prompt(old_analysis_markdown, patient_data, changed_visit_ids=()):
    """Update prompt: the previous review, the full current evidence, and which encounters changed."""
    current = evidence_for_prompt(build_ai_review_context(patient_data))
    changed_ids = {str(v) for v in changed_visit_ids}
    changed = ", ".join(f"#{i}" for i, v in enumerate(patient_data.get("visits") or [], 1)
                        if str(v.get("visit_id")) in changed_ids) or "none identified"
    kept, words = [], 0
    for line in str(old_analysis_markdown or "").splitlines():
        count = len(line.split())
        if words + count > 260:
            break
        kept.append(line)
        words += count
    previous = "\n".join(kept)
    if _flag("ENABLE_EXTERNAL_PROMPTS"):
        try:
            return _load_prompt("review_incremental.md").format(
                required_ending=REQUIRED_ENDING, previous=previous, current=current, changed=changed
            )
        except Exception:
            pass
    return (
        "Update the review for new or changed encounters. Keep previous points that the current evidence still "
        "supports, revise or drop those it no longer supports, and add what the changed encounters show. Write the "
        f"same Clinical Insights section and end with this exact line: {REQUIRED_ENDING}\n\n"
        f"PREVIOUS REVIEW:\n{previous}\n\nCHANGED ENCOUNTERS: {changed}\n\nPATIENT EVIDENCE (JSON):\n{current}"
    )


def _call_openrouter(prompt, retry_feedback=None, patient_id=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if retry_feedback:
        messages.append({"role": "user", "content": f"Fix the structure issue: {retry_feedback}"})
    result = llm_complete(messages, "review", temperature=0.1, max_tokens=600,
                          patient_id=patient_id, validate=validate_model_review)
    return {"raw_markdown": result["content"], "model_name": result["model"], "model_version": ""}


def _sanitize_review(markdown, patient_data=None):
    return _sanitize(markdown, patient_data or {})


@traced(name="ai_review.analyze_full")
def analyze_patient_with_ai(patient_data, retry_feedback=None):
    return _call_openrouter(build_ai_review_prompt(patient_data, retry_feedback),
                            patient_id=patient_data.get("patient_id"))


@traced(name="ai_review.analyze_incremental")
def analyze_patient_incremental(old_markdown, patient_data, changed_visit_ids=(), retry_feedback=None):
    prompt = build_incremental_prompt(old_markdown, patient_data, changed_visit_ids)
    if retry_feedback:
        prompt += f"\n\nCorrection required: {retry_feedback}"
    return _call_openrouter(prompt, patient_id=patient_data.get("patient_id"))
