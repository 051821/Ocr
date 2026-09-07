"""
Self-contained smoke test using the example patient from the spec.
Run with:  python -m tests.test_pipeline_demo
No database, no ollama, no network required.
"""
import json
import logging
from longitudinal.adapters import adapt_from_simple_history
from longitudinal.longitudinal import run_longitudinal_analysis

logging.basicConfig(level=logging.INFO)

PATIENT_ID = "574c0874-4677-48e2-b60f-7a33e5e659a3"

RAW_PATIENT_DATA = {
    "patient_id": PATIENT_ID,
    "total_visits": 4,
    "visits": [
        {
            "visit_id": "v1",
            "visit_date": "2026-05-05",
            "provisional_diagnosis": "Htn t2dm",
            "confirmed_diagnosis": None,
            "medical_text": (
                "Clinical notes: Both knee pain, neck pain\n"
                "Medications:\n"
                "Tab amlodipine 5 mg od\n"
                "Tab telmisartan 40 mg od\n"
                "Tab metformin 500 mg bd\n"
                "Labs:\n"
                "Serum Creatinine : 1.25 mg/dL\n"
                "Reference: 0.6-1.2\n"
                "Random glucose 310.87 mg/dL\n"
                "SGOT = 13.5 U/L\n"
                "SGPT = 14.1 U/L\n"
                "Albumin = 3.96 g/dL\n"
            ),
        },
        {
            "visit_id": "v2",
            "visit_date": "2026-06-20",
            "provisional_diagnosis": "HTN T2DM",
            "confirmed_diagnosis": None,
            "medical_text": (
                "Medications:\nTab metformin 500 mg bd\n"
                "Labs:\nRandom glucose 250 mg/dL\n"
            ),
        },
        {
            "visit_id": "v3",
            "visit_date": "2026-07-30",
            "provisional_diagnosis": "T2DM",
            "confirmed_diagnosis": None,
            "medical_text": "Labs:\nHbA1c 8.4 %\n",
        },
        {
            "visit_id": "v4",
            "visit_date": "2026-09-01",
            "provisional_diagnosis": None,
            "confirmed_diagnosis": None,
            "medical_text": "Labs:\nHbA1c 7.6 %\n",
        },
    ],
}


def main():
    visits = adapt_from_simple_history(RAW_PATIENT_DATA)
    analysis = run_longitudinal_analysis(visits)
    print(json.dumps(analysis, indent=2, default=str))

    assert "Hypertension" in analysis["diagnosis_longitudinal"]
    assert analysis["diagnosis_longitudinal"]["Hypertension"]["status"] == "persistent"
    assert "Type 2 diabetes mellitus" in analysis["diagnosis_longitudinal"]
    assert "HbA1c" in analysis["lab_trends"]
    assert analysis["lab_trends"]["HbA1c"]["direction"] == "decreasing"
    assert "Random blood glucose" in analysis["lab_trends"]
    print("\nAll smoke-test assertions passed.")


if __name__ == "__main__":
    main()
