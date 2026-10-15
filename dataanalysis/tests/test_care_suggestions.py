import pytest

from review_rules import strip_care_suggestions

MEDS, LABS = ["Levothyroxine", "Metformin"], ["TSH", "HbA1c"]

STRIPPED = [
    "- Consider repeating TSH in 6 weeks.",
    "- Levothyroxine dose may need review.",
    "- A lipid panel is recommended.",
    "- Patient may need to start atorvastatin therapy.",
    "- Metformin should be stopped if eGFR falls.",
    "- An HbA1c every 3 months is advisable.",
]

KEPT = [
    "- You should start walking 30 minutes daily.",
    "- Metformin should be taken with food to reduce nausea.",
    "- Patients with diabetes should limit refined carbohydrates to keep glucose levels stable.",
    "- HbA1c fell from 8.4% to 6.9% (Encounters #1-#4).",
    "- The nephrology consult advised a repeat renal function test in 3 months (Encounter #2).",
    "- Repeat renal function in 3 months is documented (Encounter #2).",
    "- For clinician review: TSH still high at Encounter #2.",
    "- High TSH suggests under-replacement.",
    "- HbA1c of 8.4% warrants clinician review.",
]


@pytest.mark.parametrize("line", STRIPPED)
def test_suggestions_are_removed(line):
    assert strip_care_suggestions(line, MEDS, LABS) == ""


@pytest.mark.parametrize("line", KEPT)
def test_record_facts_and_lifestyle_advice_are_kept(line):
    assert strip_care_suggestions(line, MEDS, LABS) == line


def test_only_the_suggesting_sentence_is_dropped_from_a_bullet():
    text = "### Answer\n- TSH 9.8 (High) at Encounter #1. Consider repeating TSH.\n- Hypothyroidism since Encounter #1."
    assert strip_care_suggestions(text, MEDS, LABS) == "### Answer\n- TSH 9.8 (High) at Encounter #1.\n- Hypothyroidism since Encounter #1."
