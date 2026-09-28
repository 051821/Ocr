import requests
import time
import json


# ============================================================
# DYNAMO CONFIGURATION
# ============================================================

URL = "http://localhost:8000/v1/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
}


# ============================================================
# MODEL
# ============================================================

MODEL = "Qwen/Qwen3-0.6B"


# ============================================================
# MEDICAL INPUT
# ============================================================

medical_input = """
PATIENT CLINICAL SUMMARY

Patient Name: Rajesh Kumar
Age: 54 years
Gender: Male
Date of Examination: 28 September 2026

CHIEF COMPLAINT:
The patient presented with a history of intermittent chest discomfort for approximately three weeks. The discomfort is described as a dull, pressure-like sensation in the central chest, occasionally occurring after walking for 10 to 15 minutes. The patient reports that the symptoms generally improve after resting. He also complains of mild shortness of breath on exertion and occasional fatigue.

HISTORY OF PRESENT ILLNESS:
The patient has experienced progressively increasing episodes of chest discomfort during physical activity. There is no history of syncope, hemoptysis, or recent respiratory infection. He reports occasional palpitations but denies sustained episodes of rapid heartbeat. There is no history of recent trauma.

PAST MEDICAL HISTORY:
The patient was diagnosed with type 2 diabetes mellitus approximately 8 years ago and has been taking oral medication intermittently. He was diagnosed with hypertension 6 years ago. His blood pressure control has been variable because of inconsistent medication adherence. The patient also has a history of dyslipidemia.

FAMILY HISTORY:
The patient's father had coronary artery disease and underwent coronary angioplasty at the age of 62 years. His mother has a history of hypertension and type 2 diabetes mellitus.

PERSONAL HISTORY:
The patient does not currently smoke cigarettes but has a previous smoking history of approximately 10 pack-years. He reports occasional consumption of alcohol. His diet is relatively high in carbohydrates and saturated fats. Physical activity is limited because of a sedentary occupation.

VITAL SIGNS:
Blood Pressure: 148/92 mmHg
Heart Rate: 84 beats/min
Respiratory Rate: 18 breaths/min
Temperature: 98.4 °F
Oxygen Saturation: 97% on room air
Weight: 82 kg
Height: 171 cm
BMI: 28.0 kg/m²

LABORATORY INVESTIGATIONS:
Hemoglobin: 13.8 g/dL
Total Leukocyte Count: 7,600 cells/µL
Platelet Count: 2.46 lakh/µL
Fasting Blood Glucose: 132 mg/dL
HbA1c: 7.4%
Serum Creatinine: 1.02 mg/dL
Blood Urea Nitrogen: 18 mg/dL
Total Cholesterol: 214 mg/dL
LDL Cholesterol: 142 mg/dL
HDL Cholesterol: 41 mg/dL
Triglycerides: 156 mg/dL
Sodium: 139 mmol/L
Potassium: 4.3 mmol/L

PHYSICAL EXAMINATION:
The patient is conscious, oriented, and cooperative. Cardiovascular examination reveals normal S1 and S2 heart sounds without an obvious murmur. Respiratory examination reveals bilateral equal air entry with no significant wheezing or crepitations. There is no peripheral edema. Peripheral pulses are palpable bilaterally.

ELECTROCARDIOGRAM:
The ECG demonstrates normal sinus rhythm at a rate of 82 beats/min. There are nonspecific ST-T wave changes in the lateral leads. No acute ST-segment elevation is identified.

ASSESSMENT:
1. Exertional chest discomfort requiring further evaluation.
2. Type 2 diabetes mellitus with suboptimal glycemic control.
3. Essential hypertension with currently elevated blood pressure.
4. Dyslipidemia.
5. Overweight.

PLAN:
The patient was advised to continue regular monitoring of blood pressure and blood glucose levels. Medication adherence and dietary modification were discussed. Further cardiovascular evaluation, including appropriate cardiac investigations, was recommended based on clinical assessment. The patient was advised to seek urgent medical attention if chest pain becomes severe, persistent, occurs at rest, or is associated with sweating, fainting, severe shortness of breath, or other acute symptoms.

DISCLAIMER:
This is synthetic sample text created for software/OCR testing and is not medical advice or a real patient's medical record."""


# ============================================================
# SAME PROMPT AS OPENROUTER VERSION
# ============================================================

prompt = f"""
Analyze the following patient data.

Provide:

- Current clinical status
- Important abnormal findings
- Relevant clinical concerns
- Supported relationships between diagnoses, symptoms,
  medications, and laboratory results
- Appropriate clinical follow-up considerations
- Limitations of the available information
- Overall severity ONLY if adequately supported
- Trend with dates if person is improving comparing
  current conditions with previous conditions

### Strict evidence rules

1. Use ONLY information explicitly provided in the patient data.

2. For every important conclusion, identify the specific
   finding(s) that support it.

3. Separate:

   - FACT: directly stated or measured in the input.
   - INTERPRETATION: directly supported by the provided facts.
   - POSSIBLE CONCERN: plausible but not established.

4. Never present an inference, possibility, or risk as an
   established diagnosis or fact.

5. Do not infer diagnoses, causes, complications, symptoms,
   history, trends, disease duration, treatment response,
   or severity unless supported by the input.

6. Do not infer causality from associations.

7. A single abnormal laboratory result does not establish
   chronic disease, acute disease, disease progression,
   or poor long-term control.

8. Do not classify overall severity when insufficient.

9. Do not infer missing values.

10. If a laboratory reference range is not provided,
    do not classify the result as normal or abnormal.

11. Do not assume medication toxicity, contraindication,
    interaction, or dose adjustment.

12. Do not prescribe treatment or recommend changing medication.

13. Do not invent trends when only one time point is available.

14. When evidence is insufficient, explicitly state that
    it cannot be determined.

### Laboratory interpretation

For each important abnormal result:

- Give exact value and unit.
- Compare with supplied reference range.
- Describe abnormality conservatively.
- Explain significance only as supported.
- State what additional information is needed.

### Medication interpretation

- Do not assume medication toxicity or contraindication.
- Do not prescribe treatment.
- Mention common side effects of medicines explicitly
  present in the patient data.

### Follow-up

Follow-up considerations must be information that may be
useful for further clinical evaluation, not prescriptions.

Patient data:

{medical_input}
"""


# ============================================================
# BENCHMARK
# ============================================================

print("=" * 70)
print("DYNAMO + vLLM BENCHMARK")
print("=" * 70)


start_time = time.perf_counter()

response = requests.post(
    URL,
    headers=HEADERS,
    json={
        "model": MODEL,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": 0.0,

        "max_tokens": 1000
    },

    timeout=300
)

end_time = time.perf_counter()

response.raise_for_status()

result = response.json()

message = result["choices"][0]["message"]

answer = message.get("content", "")


# ============================================================
# USAGE
# ============================================================

usage = result.get("usage", {})

prompt_tokens = usage.get("prompt_tokens")
completion_tokens = usage.get("completion_tokens")
total_tokens = usage.get("total_tokens")


# ============================================================
# PERFORMANCE
# ============================================================

total_latency = end_time - start_time

if completion_tokens:
    output_tokens_per_second = (
        completion_tokens / total_latency
    )
else:
    output_tokens_per_second = None


# ============================================================
# RESULTS
# ============================================================

print("\nRESULT")
print("-" * 70)

print(answer)

print("\n" + "=" * 70)
print("PERFORMANCE")
print("=" * 70)

print(f"Total latency       : {total_latency:.3f} seconds")

if prompt_tokens is not None:
    print(f"Input tokens        : {prompt_tokens}")
else:
    print("Input tokens        : unavailable")

if completion_tokens is not None:
    print(f"Output tokens       : {completion_tokens}")
else:
    print("Output tokens       : unavailable")

if total_tokens is not None:
    print(f"Total tokens        : {total_tokens}")
else:
    print("Total tokens        : unavailable")

if output_tokens_per_second is not None:
    print(
        f"Generation speed    : "
        f"{output_tokens_per_second:.2f} tokens/sec"
    )
else:
    print("Generation speed    : unavailable")

print("=" * 70)