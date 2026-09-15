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

Visit Date: 2026-06-30

Confirmed Diagnoses:

* None documented

Provisional Diagnoses:

* Type 2 Diabetes Mellitus

Symptoms:

* Not documented

Medications:

* Ecosprin (ECOPSRIN AV)
* Metformin 500 mg
* Calcium + Vitamin D3 1000 mg
* Vitamin B Complex (METHOCOBALMIN) 1500 ug

Laboratory Results:

* None documented

Visit Date: 2026-07-07

Confirmed Diagnoses:

* None documented

Provisional Diagnoses:

* Type 2 Diabetes Mellitus

Symptoms:

* Not documented

Medications:

* Not documented

Laboratory Results:

1. Serum HDL Cholesterol
   Value: 25.64
   Unit: mg/dL
   Reference: 42-88
   Status: ABNORMAL

2. Serum Potassium
   Value: 4.16
   Unit: mmol/L
   Reference: 3.5-5.1
   Status: NORMAL

3. Serum Sodium
   Value: 138.9
   Unit: mmol/L
   Reference: 135-145
   Status: NORMAL

Visit Date: 2026-07-11

Confirmed Diagnoses:

* None documented

Provisional Diagnoses:

* Adhesive Capsulitis

Symptoms:

* Not documented

Medications:

* Ecosprin (ECOPSRIN AV)
* Metformin 500 mg
* Calcium + Vitamin D3 1000 mg
* Vitamin B Complex (METHOCOBALMIN) 1500 ug

Laboratory Results:

* None documented

Visit Date: 2026-08-14

Confirmed Diagnoses:

* None documented

Provisional Diagnoses:

* Hypertension
* Type 2 Diabetes Mellitus

Symptoms:

* Not documented

Medications:

* Telmisartan 40 mg
* Metformin 500 mg
* Omeprazole 20 mg
* Ecosprin 
* Nexito Plus 

Laboratory Results:

* None documented

Visit Date: 2026-09-08

Confirmed Diagnoses:

* None documented

Provisional Diagnoses:

* Type 2 Diabetes Mellitus

Symptoms:

* Not documented

Medications:

* Not documented

Laboratory Results:

* None documented

Longitudinal Information:

Persistent Conditions:

* Type 2 Diabetes Mellitus: documented across 5/5 visits
* Headache: documented across 3/5 visits

Conditions Documented Once:

* Adhesive Capsulitis: documented on 2026-07-11
* Hypertension: documented on 2026-08-14

Medication History:

* Ecosprin: documented on 3 visits
* Metformin: documented on 3 visits
* Calcium + Vitamin D3: documented on 2 visits
* Vitamin B Complex: documented on 2 visits
* Telmisartan: documented on 1 visit
* Omeprazole: documented on 1 visit
* Nexito Plus: possible medication; requires verification
* Nexito: possible medication; requires verification

AI/NLP Findings Requiring Verification:

Possible Medication:

* Nexito Plus
  Confidence: 0.55
  Evidence: "TAB NEXITO PLUS"

Possible Medication:

* Nexito
  Confidence: 0.65
  Evidence: "MG 20 Mg TA. ECOPSRIN AV TAB NEXITO PLUS CHIEF COMPLAIN"

Possible Diagnosis/Symptom:

* Headache
  Confidence: 0.49
  Evidence: "CHIEF COMPLAINT Headache, upper abdomen pain"

Possible Diagnosis:

* Adhesive Capsulitis
  Confidence: 0.70
  Evidence: "Adhesive capsulitis"

Possible Diagnosis/Symptom:

* Headache
  Confidence: 0.49
  Evidence: "pain, dizziness and weakness, headache"

Possible Diagnosis:

* Type 2 Diabetes Mellitus
  Confidence: 0.49
  Evidence: "DIAGNOSIS/IMPRESSION T2dm"

Possible Symptom:

* Headache
  Confidence: 0.49
  Evidence: "Insomnia, headache"

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
- Trend with dates if person is improving comparing current conditions with previous conditions

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
        "model": "inclusionai/ling-3.0-flash-sante:free",
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