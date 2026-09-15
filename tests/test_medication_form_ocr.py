"""Dose association for flattened OPD-prescription OCR."""
from clinical_nlp.medications import extract_medications


def _by_name(meds):
    return {m["name"]: m for m in meds}


def test_visit1_column_lag_dose():
    text = """
TAB ECOPSRIN AV
AFTER FOOD
METFORMIN
500 mg
20mg
EMPTY SYOMACH
B COMPLEX
1cap
AFTER FOOD
CALCIUM D3
1000mg
AFTER FOOD
TAB METHOCOBALMIN
1500 ug
AFTER FOOD
10ml
AFTER FOOD
CHIEF COMPLAINT
Headache, upper abdomen pain, dizziness
"""
    meds = _by_name(extract_medications(text))
    assert meds["Ecosprin AV"]["dose"] == 20.0
    assert meds["Ecosprin AV"]["unit"] == "mg"
    assert meds["Ecosprin AV"]["frequency"] in ("empty_stomach", "after_food")
    assert meds["Metformin"]["dose"] == 500.0
    assert meds["Metformin"]["unit"] == "mg"
    assert meds["Vitamin B Complex"]["dose"] == 1.0
    assert meds["Vitamin B Complex"]["unit"] == "cap"
    assert meds["Calcium + Vitamin D3"]["dose"] == 1000.0
    assert meds["Methylcobalamin"]["dose"] == 1500.0
    assert meds["Methylcobalamin"]["unit"] == "ug"


def test_visit3_keeps_metformin_500_not_1mg():
    text = """
TAB ECOPSRIN AV
AFTER FOOD
METFORMIN
500 Mg
BEFORE FOOD
1mg
BEFORE FOOD
20mg
EMPTY STOMACH
B COMPLEX
1cap
AFTER FOOD
CALCIUM D3
1000mg
AFTER FOOD
TAB METHOCOBALMIN
1500 ug
AFTER FOOD
CHIEF COMPLAINT
Rt shoulder pain
"""
    meds = _by_name(extract_medications(text))
    assert meds["Metformin"]["dose"] == 500.0
    assert meds["Metformin"]["frequency"] == "before_food"
    assert meds["Ecosprin AV"]["dose"] == 20.0
    assert meds["Ecosprin AV"]["frequency"] == "empty_stomach"
    assert meds["Vitamin B Complex"]["dose"] == 1.0


def test_visit4_pending_20mg_goes_to_ecosprin_not_omeprazole():
    text = """
TELMISARTAN
40mg
METFORMIN
500 Mg
CAP. OMEPRAZOLE 20 MG
20 Mg
TA. ECOPSRIN AV
TAB NEXITO PLUS
CHIEF COMPLAINT
Insomnia, headache
"""
    meds = _by_name(extract_medications(text))
    assert meds["Omeprazole"]["dose"] == 20.0
    assert meds["Ecosprin AV"]["dose"] == 20.0
    assert meds["Telmisartan"]["dose"] == 40.0
    assert meds["Metformin"]["dose"] == 500.0
    assert "Nexito Plus" in meds
    assert "Nexito" not in meds


def test_single_line_still_works():
    meds = extract_medications("Tab metformin 500 mg bd\nTab telmisartan 40 mg od\n")
    by_name = _by_name(meds)
    assert by_name["Metformin"]["dose"] == 500.0
    assert by_name["Metformin"]["frequency"] == "twice_daily"
    assert by_name["Telmisartan"]["dose"] == 40.0


if __name__ == "__main__":
    test_visit1_column_lag_dose()
    test_visit3_keeps_metformin_500_not_1mg()
    test_visit4_pending_20mg_goes_to_ecosprin_not_omeprazole()
    test_single_line_still_works()
    print("ok")
