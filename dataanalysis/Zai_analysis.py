from ollama import chat
from patient_timeline import create_patient_timeline

def build_longitudinal_analysis_prompt(patient_data):
    timeline = create_patient_timeline(patient_data)

    prompt = f"""
You are a medical record analysis assistant.

Analyze the following patient's longitudinal medical history chronologically.

Your task is to summarize documented information and identify patterns
across visits.

Rules:
- Use only information present in the records.
- Do not invent diagnoses, symptoms, treatments, or outcomes.
- Do not make unsupported medical conclusions.
- Clearly indicate uncertainty or missing information.
- Compare visits based on their chronological order.
- Focus on changes, recurring complaints, and documented progression.

Return your response using these sections:

## Overall Clinical Summary

## Chronological Timeline

## Changes Over Time

## Recurring Issues

## Latest Documented Status

## Data Gaps and Limitations

PATIENT HISTORY:
{timeline}
""".strip()

    return prompt


def analyze_patient_with_ai(patient_data):
    try:
        prompt = build_longitudinal_analysis_prompt(patient_data)

        response = chat(
            model="qwen3:4b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.message.content

    except Exception as e:
        raise RuntimeError(f"AI analysis failed: {type(e).__name__}: {str(e)}")