from analysis import _evaluate_lab_status
from medication_journey import _extract_explicit_discontinuations, _normalize_name, compute_medication_journey
from review_rules import REQUIRED_ENDING, sanitize_review


def test_lab_value_with_unit_is_evaluated_deterministically():
    assert _evaluate_lab_status("162.99 mg/dL", "<150 mg/dL") == "Abnormal (High)"


def test_medication_normalization_ignores_form_prefix_and_strength():
    assert _normalize_name("Tab. Metformin 500mg") == _normalize_name("Metformin")


def test_stop_smoking_is_not_a_medication_discontinuation():
    visit = {"documents": [{"clinical_summary": {"clinical_notes": "Stop smoking."}}]}
    known = {"metformin": {"name": "Metformin"}}
    assert _extract_explicit_discontinuations(visit, known) == []


def test_restarted_medication_and_active_list():
    patient = {"visits": [
        {"visit_id": "1", "visit_date": "2026-01-01", "db_medications": [{"name": "Metformin", "dosage": "500mg"}]},
        {"visit_id": "2", "visit_date": "2026-02-01", "db_instructions": ["Stop Metformin"]},
        {"visit_id": "3", "visit_date": "2026-03-01", "db_medications": [{"name": "Tab. Metformin 500mg"}]},
    ]}
    history = compute_medication_journey(patient)
    assert any(change["type"] == "restarted" for change in history[-1]["changes"])
    assert history[-1]["all_active_meds"] == ["Metformin"]


def test_review_sanitization_preserves_section_newlines_and_lab_values():
    raw = "\n".join([
        "# AI Clinical Review", "## Overall Pattern", "The record includes a glucose value of 162 mg/dL.",
        "## Key Insights", "- **Lab value:** 162 mg/dL appears in the documented record.",
        "## Review Points", "- Verify extracted details with the source document.",
        "## Evidence", "- Visit 1, document 2.", REQUIRED_ENDING,
    ])
    result = sanitize_review(raw, {"patient_id": "p", "visits": []})
    assert "## Overall Pattern\n" in result
    assert "## Key Insights\n" in result
    assert "162 mg/dL" in result
    assert result.endswith(REQUIRED_ENDING)
