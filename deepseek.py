import requests
import json

from config import OPENROUTER_API_KEY


URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}


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



prompt = f"""
Analyze the following patient data.

Provide:
- Current clinical status
- Important abnormal findings
- Relevant clinical concerns
- Supported relationships between diagnoses, symptoms, medications, and laboratory results
- Appropriate clinical follow-up considerations
- Limitations of the available information
- Overall severity ONLY if adequately supported

### Strict evidence rules

1. Use ONLY information explicitly provided in the patient data.
2. For every important conclusion, identify the specific finding(s) that support it.
3. Separate:
   - FACT: directly stated or measured in the input.
   - INTERPRETATION: directly supported by the provided facts.
   - POSSIBLE CONCERN: plausible but not established; clearly label it as such.
4. Never present an inference, possibility, or risk as an established diagnosis or fact.
5. Do not infer diagnoses, causes, complications, symptoms, history, trends, disease duration, treatment response, or severity unless supported by the input.
6. Do not infer causality from associations. For example, do not assume that a laboratory abnormality was caused by a diagnosis or medication unless the data establish this.
7. A single abnormal laboratory result does not establish chronic disease, acute disease, disease progression, or poor long-term control.
8. Do not classify overall severity as mild, moderate, or severe when the available information is insufficient. State: "Overall severity cannot be determined from the available data."
9. Do not infer missing values. If eGFR, HbA1c, BP, weight, baseline laboratory values, or other information is not provided, state that it is unavailable.
10. If a laboratory reference range is not provided, do not classify the result as normal or abnormal based on an assumed reference range.
11. Do not assume medication toxicity, contraindication, interaction, or need for dose adjustment from an abnormal laboratory result alone.
12. Do not prescribe treatment or recommend starting, stopping, or changing medication.
13. Do not invent trends when only one time point is available.
14. When evidence is insufficient, explicitly state that it cannot be determined from the available information rather than filling the gap with assumptions.

### Laboratory interpretation

For each important abnormal result:
- Give the exact value and unit.
- Compare it with the supplied reference range, if available.
- Describe the abnormality conservatively.
- Explain its clinical significance only to the extent supported by the data.
- State what additional information would be needed for a stronger conclusion.


### Medication interpretation

- Do not assume medication toxicity, contraindication, interaction, or need for dose adjustment from an abnormal laboratory result alone.
- Do not prescribe treatment or recommend starting, stopping, or changing medication.
- Write comman side effect for medicines prescribed in the patient data.

### Follow-up

Follow-up considerations must be framed as information that may be useful for further clinical evaluation, not as prescriptions or treatment instructions.

Patient data:

{medical_input}

"""


response = requests.post(
    URL,
    headers=HEADERS,
    json={
        "model": "deepseek/deepseek-v4-flash-0731",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "reasoning": {
            "enabled": True
        }
    }
)

response.raise_for_status()

result = response.json()

message = result["choices"][0]["message"]

print(message.get("content"))