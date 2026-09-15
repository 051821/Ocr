import requests
import json

from config import OPENROUTER_API_KEY


URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}


# ============================================================
# MEDICAL INPUT
# ============================================================

medical_input = """
PatientAge/Sex: 67/F
Visit Date: 2026-05-05

Confirmed Diagnoses:
- Hypertension
- Type 2 Diabetes Mellitus

Symptoms:
- Bilateral knee pain
- Neck pain

Medications:
- Amlodipine 5 mg OD
- Telmisartan 40 mg OD
- Metformin 500 mg
- Aspirin 75 mg
- Tramadol 37 mg BD
- Calcium + Vitamin D3
- Vitamin B Complex

Laboratory Results:

1. Random Blood Glucose
   Value: 310.87
   Unit: mg/dL
   Reference: 70-140

2. Creatinine
   Value: 1.25
   Unit: mg/dL
   Reference: 0.6-1.2

3. Sodium
   Value: 139.8
   Unit: mmol/L
   Reference: 135-145

4. Potassium
   Value: 4.39
   Unit: mmol/L

5. Urea
   Value: 30.9
   Unit: mg/dL
   Reference: 10-50

6. BUN
   Value: 14.43
   Unit: mg/dL
   Reference: 6-20

7. AST
   Value: 13.5
   Unit: U/L
   Reference: <46

8. ALT
   Value: 14.1
   Unit: U/L
   Reference: <49

9. Alkaline Phosphatase
   Value: 96
   Unit: U/L
   Reference: 64-306

10. Total Bilirubin
    Value: 0.34
    Unit: mg/dL
    Reference: 0.3-1.2

11. Direct Bilirubin
    Value: 0.07
    Unit: mg/dL
    Reference: <0.4

12. Indirect Bilirubin
    Value: 0.27
    Unit: mg/dL
    Reference: 0.2-0.8

13. Total Protein
    Value: 6.91
    Unit: g/dL
    Reference: 6.0-8.0

14. Albumin
    Value: 3.96
    Unit: g/dL
    Reference: 3.5-5.2
"""


# ============================================================
# FIRST API CALL
# ============================================================

messages = [
    {
        "role": "user",
        "content": medical_input
    }
]

payload = {
    "model": "tencent/hy4-preview",
    "messages": messages,
    "reasoning": {
        "enabled": True
    }
}

response = requests.post(
    URL,
    headers=HEADERS,
    json=payload,
    timeout=120
)

print("First API status:", response.status_code)

if response.status_code != 200:
    print(response.text)
    response.raise_for_status()

data = response.json()

assistant_message = data["choices"][0]["message"]

print("\n================ FIRST RESPONSE ================\n")
print(assistant_message.get("content"))


# ============================================================
# PRESERVE REASONING DETAILS
# ============================================================

reasoning_details = assistant_message.get("reasoning_details")

print("\nReasoning details present:",
      reasoning_details is not None)


# ============================================================
# SECOND API CALL
# ============================================================

messages = [
    {
        "role": "user",
        "content": medical_input
    },
    {
        "role": "assistant",
        "content": assistant_message.get("content"),
        "reasoning_details": reasoning_details
    },
    {
        "role": "user",
        "content": """
Review the previous clinical analysis against the original patient data. Correct any unsupported assumptions, incorrect interpretations, or invented information.

Analyze ONLY the information provided.

Your purpose is to explain the patient's **current condition, severity, risks, precautions, and appropriate follow-up** — NOT to prescribe treatment.

### Strict Rules

1. Do not invent or modify diagnoses, symptoms, laboratory values, units, medications, doses, vitals, or medical history.
2. Preserve all laboratory values exactly as provided.
3. Do not introduce a new diagnosis from a single abnormal result.
4. Clearly distinguish **documented facts** from **clinical interpretation**.
5. If evidence is insufficient, state: **"Insufficient information to determine."**
6. Do not recommend starting, stopping, increasing, decreasing, or changing any medication.
7. Do not provide drug names or medication doses as treatment recommendations.
8. Do not tell the patient to stop prescribed medication.
9. Do not assume missing information.
10. Do not claim that a complication will definitely occur.
11. If only one visit/result is available, do not describe a longitudinal trend.
12. Do not use information from outside the provided patient data.

### Generate this report

## 1. Patient Summary

* Age/sex
* Visit date
* Documented diagnoses
* Documented symptoms

## 2. Important Findings

Identify the most important abnormal and normal findings.

For each abnormal finding, explain:

* What is abnormal
* How significant/concerning it appears
* What it may indicate
* Whether further evaluation may be appropriate

Do not convert a possible condition into a confirmed diagnosis.

## 3. Overall Severity

Give ONE overall classification:

**Low / Moderate / High / Potentially Urgent / Insufficient Information**

Then briefly explain why.

Base severity only on the documented findings.

## 4. Possible Risks

Describe complications or health risks that may be associated with the documented diagnoses and abnormalities.

Use:

* "may increase the risk of..."
* "can be associated with..."
* "could potentially lead to..."

Do not state that the patient will develop them.

Also distinguish between:

* risks suggested by the findings
* complications actually documented

## 5. General Precautions

Provide only general, low-risk precautions relevant to this patient, such as:

* following the existing treatment plan
* appropriate blood glucose/blood pressure monitoring
* appropriate diet and hydration
* physical activity as medically tolerated
* avoiding smoking
* attending follow-up appointments
* reporting worsening or new symptoms

Do not provide medication-specific instructions.

## 6. Monitoring and Follow-up

Suggest what should reasonably be discussed with the treating clinician based on the documented abnormalities, symptoms, and diagnoses.

Do not create an unnecessary list of tests.

## 7. When to Seek Medical Attention

Identify relevant warning signs that may require prompt or urgent medical evaluation.

Do not exaggerate risk.

If urgency cannot be determined from the available information, state:
**"Insufficient information to determine urgency."**

## 8. Data Quality Issues

Identify any:

* OCR errors
* conflicting information
* missing units
* uncertain values
* duplicate results
* unclear medication information
* uncertain diagnosis/entity extraction

If none are present, say:
**"No obvious data-quality issues identified."**

## 9. Final Assessment

Give a concise summary covering:

* Current documented condition
* Most important abnormality
* Severity
* Main risks
* Precautions
* Monitoring/follow-up
* Potentially urgent concerns

Do not prescribe or modify treatment.

End with:

"This is a clinical decision-support summary based only on the provided information and is not a definitive diagnosis or individualized treatment plan. Final clinical decisions should be made by the treating healthcare professional."
"""
    }
]

payload2 = {
    "model": "tencent/hy4-preview",
    "messages": messages,
    "reasoning": {
        "enabled": True
    }
}

response2 = requests.post(
    URL,
    headers=HEADERS,
    json=payload2,
    timeout=120
)

print("\nSecond API status:", response2.status_code)

if response2.status_code != 200:
    print(response2.text)
    response2.raise_for_status()

data2 = response2.json()

final_message = data2["choices"][0]["message"]

print("\n================ FINAL RESPONSE ================\n")
print(final_message.get("content"))