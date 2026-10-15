"""Retrieval evaluation for patient chat: 20 questions with the evidence each one needs.

Runs the real answer_patient_question pipeline with the LLM, answer cache and Neo4j stubbed out,
then checks the evidence it returns (what the LLM would see) against each case's expectations.

    python -m evals.retrieval_eval            # from dataanalysis/: per-case table + overall score

Expectations were written from the clinical intent of each question, not from current output,
so a failing case is a real retrieval gap rather than a broken test.
"""
import logging
from contextlib import ExitStack
from dataclasses import dataclass, field
from unittest import mock

import patient_chat

# Patient A: type 2 diabetes + hypertension; amlodipine swapped for telmisartan after ankle edema.
DIABETIC = {
    "patient_id": "eval-diabetic",
    "possible_side_effects": {
        "Metformin": "nausea, diarrhoea, metallic taste, rarely lactic acidosis",
        "Amlodipine": "ankle edema, flushing, headache",
        "Telmisartan": "dizziness, hyperkalaemia, back pain",
        "Atorvastatin": "muscle pain, raised liver enzymes",
    },
    "visits": [
        {
            "visit_id": "a1", "visit_date": "2025-01-10", "chief_complaint": "Fatigue and increased thirst",
            "provisional_diagnosis": "Type 2 diabetes mellitus",
            "db_medications": [{"name": "Metformin", "dosage": "500mg", "frequency": "BD"}],
            "lab_results": [
                {"test_name": "HbA1c", "value": "8.4 %", "status": "High"},
                {"test_name": "Fasting Glucose", "value": "162 mg/dL", "status": "High"},
            ],
            "db_vitals_rows": [{"label": "BP", "value": "150/95 mmHg"}, {"label": "Pulse", "value": "82"}],
        },
        {
            "visit_id": "a2", "visit_date": "2025-04-12", "chief_complaint": "Diabetes follow-up, headaches",
            "confirmed_diagnosis": "Type 2 diabetes mellitus with hypertension",
            "db_medications": [
                {"name": "Metformin", "dosage": "1000mg", "frequency": "BD"},
                {"name": "Amlodipine", "dosage": "5mg", "frequency": "OD"},
            ],
            "lab_results": [
                {"test_name": "HbA1c", "value": "7.6 %", "status": "High"},
                {"test_name": "LDL Cholesterol", "value": "160 mg/dL", "status": "High"},
            ],
            "db_vitals_rows": [{"label": "BP", "value": "146/92 mmHg"}],
        },
        {
            "visit_id": "a3", "visit_date": "2025-08-20", "chief_complaint": "Ankle swelling",
            "confirmed_diagnosis": "Type 2 diabetes mellitus with hypertension",
            "db_medications": [
                {"name": "Metformin", "dosage": "1000mg", "frequency": "BD"},
                {"name": "Telmisartan", "dosage": "40mg", "frequency": "OD"},
                {"name": "Atorvastatin", "dosage": "20mg", "frequency": "HS"},
            ],
            "documents": [{
                "doc_id": "a3-note", "document_label": "Consultation note",
                "clinical_summary": {"clinical_impression": "Pedal edema likely due to amlodipine",
                                     "symptoms": ["ankle swelling"]},
                "extracted_text": ["Bilateral pedal edema noted, likely due to amlodipine.",
                                   "Amlodipine stopped and switched to telmisartan 40mg.",
                                   "Atorvastatin started for raised LDL."],
            }],
        },
        {
            "visit_id": "a4", "visit_date": "2026-01-15", "chief_complaint": "Routine review",
            "confirmed_diagnosis": "Type 2 diabetes mellitus with hypertension",
            "db_medications": [
                {"name": "Metformin", "dosage": "1000mg", "frequency": "BD"},
                {"name": "Telmisartan", "dosage": "40mg", "frequency": "OD"},
                {"name": "Atorvastatin", "dosage": "20mg", "frequency": "HS"},
            ],
            "lab_results": [
                {"test_name": "HbA1c", "value": "6.9 %", "status": "High"},
                {"test_name": "Serum Creatinine", "value": "1.1 mg/dL", "status": "Normal"},
            ],
            "db_vitals_rows": [{"label": "BP", "value": "128/82 mmHg"}],
        },
    ],
}

# Patient B: hypothyroidism + CKD, no diabetes. Catches retrieval that favours diabetes-related terms.
THYROID_CKD = {
    "patient_id": "eval-thyroid-ckd",
    "possible_side_effects": {"Levothyroxine": "palpitations, insomnia, weight loss if overdosed"},
    "visits": [
        {
            "visit_id": "b1", "visit_date": "2025-02-05", "chief_complaint": "Weight gain and cold intolerance",
            "provisional_diagnosis": "Hypothyroidism",
            "db_medications": [{"name": "Levothyroxine", "dosage": "50mcg", "frequency": "OD"}],
            "lab_results": [{"test_name": "TSH", "value": "9.8 uIU/mL", "status": "High"}],
        },
        {
            "visit_id": "b2", "visit_date": "2025-06-18", "chief_complaint": "Tiredness, reduced urine output",
            "confirmed_diagnosis": "Hypothyroidism; Chronic kidney disease stage 3",
            "db_medications": [{"name": "Levothyroxine", "dosage": "50mcg", "frequency": "OD"}],
            "lab_results": [
                {"test_name": "TSH", "value": "5.2 uIU/mL", "status": "High"},
                {"test_name": "Serum Creatinine", "value": "1.8 mg/dL", "status": "High"},
            ],
            "documents": [{
                "doc_id": "b2-neph", "document_label": "Nephrology consult",
                "clinical_summary": {"clinical_impression": "CKD stage 3, creatinine rising; avoid NSAIDs"},
                "extracted_text": ["Nephrology impression: creatinine rising, consistent with CKD stage 3.",
                                   "Avoid NSAIDs such as ibuprofen and diclofenac. Repeat renal function in 3 months."],
            }],
        },
        {
            "visit_id": "b3", "visit_date": "2025-11-02", "chief_complaint": "Follow-up",
            "confirmed_diagnosis": "Hypothyroidism; Chronic kidney disease stage 3",
            "db_medications": [{"name": "Levothyroxine", "dosage": "75mcg", "frequency": "OD"}],
            "lab_results": [
                {"test_name": "TSH", "value": "3.1 uIU/mL", "status": "Normal"},
                {"test_name": "Serum Creatinine", "value": "2.0 mg/dL", "status": "High"},
            ],
        },
    ],
    "unlinked_documents": [{
        "doc_id": "b-us", "document_label": "Renal ultrasound",
        "clinical_summary": {"clinical_impression": "Bilateral small echogenic kidneys"},
        "extracted_text": ["Ultrasound abdomen: both kidneys small and echogenic, suggestive of chronic renal parenchymal disease."],
    }],
}


@dataclass
class Case:
    id: str
    patient: dict
    question: str
    expect: list                                  # each must appear (case-insensitive) in some evidence
    forbid: list = field(default_factory=list)    # none may appear in any evidence
    history: list = None


CASES = [
    # --- Patient A: medicines ---
    Case("se-metformin", DIABETIC, "What are the side effects of metformin?",
         ["lactic acidosis"], forbid=["muscle pain", "back pain"]),
    Case("se-telmisartan", DIABETIC, "Can telmisartan cause dizziness?", ["dizziness"]),
    Case("se-followup-pronoun", DIABETIC, "What are its side effects?", ["muscle pain"],
         forbid=["lactic acidosis"],
         history=[{"role": "user", "content": "Tell me about atorvastatin"},
                  {"role": "assistant", "content": "Atorvastatin 20mg was started in Encounter #3."}]),
    Case("why-amlodipine-stopped", DIABETIC, "Why was amlodipine stopped?", ["likely due to amlodipine"]),
    Case("added-after-visit-2", DIABETIC, "What medicines were added after visit 2?", ["telmisartan", "atorvastatin"]),
    Case("metformin-dose-change", DIABETIC, "Was the metformin dose changed?", ["1000mg"]),
    # --- Patient A: labs, vitals, visits ---
    Case("hba1c-values", DIABETIC, "What were the HbA1c values?", ["8.4", "7.6", "6.9"]),
    Case("ldl-meaning", DIABETIC, "What does the LDL result indicate?", ["160 mg/dl"]),
    Case("creatinine-normal", DIABETIC, "Is the creatinine normal?", ["1.1 mg/dl"]),
    Case("bp-visit-1", DIABETIC, "What was the BP in visit 1?", ["150/95"]),
    Case("visit-2-summary", DIABETIC, "What happened in visit 2?", ["amlodipine"]),
    Case("ankle-swelling-cause", DIABETIC, "Why did the patient have ankle swelling?", ["edema"]),
    Case("diet-advice", DIABETIC, "What diet should this patient follow?", ["type 2 diabetes"]),
    # --- Patient B: non-diabetes labs and conditions ---
    Case("creatinine-elevated", THYROID_CKD, "Why is creatinine elevated?", ["2.0 mg/dl", "nephrology"],
         forbid=["hba1c"]),
    Case("tsh-meaning", THYROID_CKD, "What does my TSH indicate?", ["9.8", "hypothyroidism"]),
    Case("labs-generic", THYROID_CKD, "What do my labs indicate?", ["tsh", "creatinine"]),
    Case("levothyroxine-dose", THYROID_CKD, "Was the levothyroxine dose changed?", ["75mcg"]),
    Case("ultrasound-unlinked", THYROID_CKD, "What did the ultrasound show?", ["small echogenic kidneys"]),
    Case("avoid-painkillers", THYROID_CKD, "Which painkillers should be avoided?", ["nsaids"]),
    Case("ckd-lifestyle", THYROID_CKD, "What lifestyle changes help with kidney disease?", ["chronic kidney disease"]),
]


# Patient C: the hard case. 14 visits, every one with labs, vitals and a routine progress note full of the
# same words (potassium, creatinine, renal, medicines), so the one entry that answers a question has to
# win against many near-duplicates. Key facts sit only in the middle of long scanned documents.
_ROUTINE = ("Patient reviewed in clinic with spouse. Blood pressure, potassium and creatinine reviewed with the patient. "
            "Diet and exercise counselling given; salt restriction reinforced. Medicines reconciled against the pharmacy "
            "list and adherence discussed. Foot examination and fundus check due as per diabetes protocol. Renal function "
            "and electrolytes compared with the previous visit. Home glucose log reviewed; occasional fasting readings above "
            "target. No chest pain, breathlessness or ankle swelling reported. Follow-up booked in clinic.")


def _note(doc_id, label, *key_sentences, padding=2):
    """A long scanned note: routine text, the key sentences in the middle, routine text again."""
    text = [_ROUTINE] * padding + list(key_sentences) + [_ROUTINE] * padding
    return {"doc_id": doc_id, "document_label": label, "extracted_text": text}


def _chronic_patient():
    dates = ["2024-01-08", "2024-02-26", "2024-04-15", "2024-06-03", "2024-07-22", "2024-09-09", "2024-10-28",
             "2024-12-16", "2025-02-03", "2025-03-24", "2025-05-12", "2025-06-30", "2025-08-18", "2025-10-06"]
    hba1c = [8.2, 8.0, 7.9, 8.1, 8.3, 8.6, 8.8, 9.0, 9.1, 8.4, 7.8, 7.5, 7.3, 7.2]
    creat = [1.3, 1.3, 1.4, 1.4, 1.5, 1.6, 2.1, 1.9, 1.9, 1.9, 2.0, 2.0, 1.9, 1.9]
    potassium = [4.6, 4.7, 4.6, 4.8, 5.1, 5.4, 6.2, 4.9, 4.8, 4.7, 4.8, 4.9, 4.7, 4.8]
    visits = []
    for i, date in enumerate(dates, start=1):
        meds = [{"name": "Metformin", "dosage": "500mg" if i >= 11 else "1000mg", "frequency": "BD"},
                {"name": "Ramipril", "dosage": "5mg" if i >= 7 else "10mg", "frequency": "OD"},
                {"name": "Atorvastatin", "dosage": "20mg", "frequency": "HS"},
                {"name": "Amlodipine", "dosage": "5mg", "frequency": "OD"}]
        if 5 <= i <= 6:
            meds.append({"name": "Spironolactone", "dosage": "25mg", "frequency": "OD"})
        if i >= 9:
            meds.append({"name": "Insulin Glargine", "dosage": "10 units", "frequency": "HS"})
        labs = [{"test_name": "HbA1c", "value": f"{hba1c[i - 1]} %", "status": "High"},
                {"test_name": "Serum Creatinine", "value": f"{creat[i - 1]} mg/dL", "status": "High"},
                {"test_name": "Potassium", "value": f"{potassium[i - 1]} mmol/L",
                 "status": "High" if potassium[i - 1] > 5.0 else "Normal"}]
        if i >= 10:
            labs.append({"test_name": "eGFR", "value": "35 mL/min/1.73m2" if i >= 11 else "41 mL/min/1.73m2", "status": "Low"})
        visit = {"visit_id": f"c{i}", "visit_date": date, "chief_complaint": "Diabetes and blood pressure review",
                 "confirmed_diagnosis": "Type 2 diabetes mellitus; Hypertension; Chronic kidney disease stage 3",
                 "db_medications": meds, "lab_results": labs,
                 "db_vitals_rows": [{"label": "BP", "value": f"{150 - i}/{92 - i // 2} mmHg"}],
                 "documents": [_note(f"c{i}-progress", "Progress note", f"Visit {i} routine review.")]}
        visits.append(visit)

    visits[4]["documents"][0] = _note("c5-progress", "Progress note",
                                      "Resistant hypertension despite three agents; spironolactone 25 mg started.")
    visits[6]["confirmed_diagnosis"] = "Hyperkalemia; Type 2 diabetes mellitus; Hypertension; Chronic kidney disease stage 3"
    visits[6]["documents"].append(_note(
        "c7-discharge", "Discharge summary",
        "Admitted with muscle weakness and palpitations. On admission serum potassium was 6.2 mmol/L with peaked T waves on ECG.",
        "Treated with calcium gluconate, insulin-dextrose and potassium binder. Spironolactone was discontinued and ramipril "
        "reduced to 5 mg once daily. Discharged on day 3 with potassium 4.9 mmol/L.", padding=3))
    visits[8]["documents"][0] = _note("c9-progress", "Progress note",
                                      "HbA1c 9.1% despite oral agents; basal insulin glargine 10 units at bedtime started.")
    visits[9]["documents"].append(_note("c10-eye", "Ophthalmology report",
                                        "Dilated fundus examination: mild non-proliferative diabetic retinopathy in both eyes, "
                                        "no macular edema."))
    visits[10]["documents"][0] = _note("c11-progress", "Progress note",
                                       "Metformin reduced to 500 mg twice daily because eGFR has fallen to 35.")
    visits[11]["documents"].append(_note("c12-foot", "Podiatry note",
                                         "Small neuropathic ulcer on the left great toe, 1 cm, clean base, no signs of infection."))
    return {
        "patient_id": "eval-chronic",
        "possible_side_effects": {
            "Spironolactone": "hyperkalaemia, gynaecomastia, dizziness",
            "Ramipril": "dry cough, hyperkalaemia, angioedema",
            "Atorvastatin": "muscle pain, raised liver enzymes",
            "Metformin": "nausea, diarrhoea, rarely lactic acidosis",
            "Amlodipine": "ankle edema, flushing",
            "Insulin Glargine": "hypoglycaemia, injection-site reactions",
        },
        "visits": visits,
    }


CHRONIC = _chronic_patient()

HARD_CASES = [
    Case("hard-spiro-stopped", CHRONIC, "Why was spironolactone stopped?", ["spironolactone was discontinued"]),
    Case("hard-admission", CHRONIC, "What happened during the hospital admission?", ["peaked t waves"]),
    Case("hard-eye-exam", CHRONIC, "What did the eye examination show?", ["non-proliferative"]),
    Case("hard-metformin-reduced", CHRONIC, "Why was the metformin dose reduced?", ["egfr has fallen to 35"]),
    Case("hard-changes-after-8", CHRONIC, "What medicines were changed after visit 8?", ["glargine", "500mg"]),
    Case("hard-potassium-v7", CHRONIC, "What was the potassium in visit 7?", ["6.2"]),
    Case("hard-insulin-start", CHRONIC, "When was insulin started?", ["glargine", "encounter #9"]),
    Case("hard-foot-ulcer", CHRONIC, "What does the foot ulcer note say?", ["great toe"]),
    Case("hard-kidney-labs", CHRONIC, "What do the kidney function tests indicate?", ["egfr", "creatinine"]),
    Case("hard-ramipril-dose", CHRONIC, "Was the ramipril dose changed?", ["5mg"]),
    Case("hard-spiro-side-effects", CHRONIC, "What are the side effects of spironolactone?",
         ["gynaecomastia", "spironolactone was discontinued"],
         forbid=["muscle pain", "dry cough"]),
    Case("hard-potassium-after-7", CHRONIC, "What was the potassium after visit 7?", ["4.9"]),
]

CASES = CASES + HARD_CASES


@dataclass
class Result:
    case: Case
    path: str
    found: list
    missing: list
    forbidden_hits: list
    n_evidence: int

    @property
    def recall(self):
        return len(self.found) / len(self.case.expect)

    @property
    def passed(self):
        return not self.missing and not self.forbidden_hits


def _stubs():
    """Patch out everything non-deterministic: LLM, answer cache, Neo4j, audit log."""
    stack = ExitStack()
    for target, value in (
        ("patient_chat.llm_complete", lambda *a, **k: {"content": "### Answer\nstub", "usage": {}}),
        ("patient_chat.get_llm_cache", lambda *a, **k: None),
        ("patient_chat.set_llm_cache", lambda *a, **k: None),
        ("patient_chat.is_neo4j_configured", lambda: False),
        ("patient_chat.audit_logger", logging.getLogger("retrieval_eval_null")),
    ):
        stack.enter_context(mock.patch(target, value))
    return stack


def run_case(case):
    meta = {}
    _, evidence = patient_chat.answer_patient_question(
        case.patient, case.patient["patient_id"], case.question, history=case.history, meta=meta)
    text = "\n".join(str(item.get("content", "")) for item in evidence or []).casefold()
    found = [e for e in case.expect if e.casefold() in text]
    return Result(case, meta.get("path") or meta.get("rule") or "?", found,
                  [e for e in case.expect if e not in found],
                  [f for f in case.forbid if f.casefold() in text], len(evidence or []))


def run_eval(cases=CASES):
    with _stubs():
        return [run_case(case) for case in cases]


def summary(results):
    return {
        "passed": sum(r.passed for r in results),
        "total": len(results),
        "mean_recall": sum(r.recall for r in results) / len(results),
        "forbidden_hits": sum(len(r.forbidden_hits) for r in results),
    }


def main():
    results = run_eval()
    print(f"{'case':<24} {'path':<14} {'recall':>6}  {'ev':>2}  problems")
    for r in results:
        problems = "; ".join(filter(None, (
            f"missing {r.missing}" if r.missing else "",
            f"forbidden {r.forbidden_hits}" if r.forbidden_hits else "")))
        print(f"{r.case.id:<24} {r.path:<14} {r.recall:>6.0%}  {r.n_evidence:>2}  {problems or 'ok'}")
    s = summary(results)
    print(f"\npassed {s['passed']}/{s['total']}  mean recall {s['mean_recall']:.0%}  forbidden hits {s['forbidden_hits']}")


if __name__ == "__main__":
    main()
