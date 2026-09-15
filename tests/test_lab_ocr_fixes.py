import pytest
from model.paddel import sort_reading_order
from clinical_nlp.labs import extract_labs
from normalization.laboratory import normalize_lab_name


def test_ocr_row_sorting():
    # Simulate detected OCR boxes in a 4-column CBC table
    # Key format in merged dict: (box_tuple, text, score)
    merged = {
        # Row 1 (y around 100)
        (50, 100, 200, 100, 200, 120, 50, 120): ( (50, 100, 200, 100, 200, 120, 50, 120), "Hemoglobin (Hb)", 0.99 ),
        (300, 100, 380, 100, 380, 120, 300, 120): ( (300, 100, 380, 100, 380, 120, 300, 120), ": 12.7", 0.98 ),
        (400, 100, 480, 100, 480, 120, 400, 120): ( (400, 100, 480, 100, 480, 120, 400, 120), "gm/dl", 0.97 ),
        (500, 100, 580, 100, 580, 120, 500, 120): ( (500, 100, 580, 100, 580, 120, 500, 120), "12-15", 0.95 ),
        # Row 2 (y around 140)
        (50, 140, 220, 140, 220, 160, 50, 160): ( (50, 140, 220, 140, 220, 160, 50, 160), "Total RBC Count", 0.99 ),
        (300, 140, 380, 140, 380, 160, 300, 160): ( (300, 140, 380, 140, 380, 160, 300, 160), ": 4.59", 0.98 ),
        (400, 140, 510, 140, 510, 160, 400, 160): ( (400, 140, 510, 140, 510, 160, 400, 160), "Millions/Cumm", 0.97 ),
        (530, 140, 600, 140, 600, 160, 530, 160): ( (530, 140, 600, 140, 600, 160, 530, 160), "3.8-4.8", 0.95 ),
    }

    ordered = sort_reading_order(merged)
    texts = [item["text"] for item in ordered]

    expected = [
        "Hemoglobin (Hb)", ": 12.7", "gm/dl", "12-15",
        "Total RBC Count", ": 4.59", "Millions/Cumm", "3.8-4.8",
    ]
    assert texts == expected, f"Expected row-by-row reading order, got: {texts}"


def test_cbc_extraction_and_units():
    sample_text = """
    COMPLETE BLOOD COUNT
    Hemoglobin (Hb) : 12.7 gm/dl 12-15
    Total RBC Count : 4.59 Millions/Cumm 3.8-4.8
    PCV : 37.5 % 40-50
    MCV : 81.7 fL 83-101
    MCH : 27.7 Pg 27-32
    MCHC : 33.9 g/dL 31.5-34.5
    RDW-CV : 13.4 % 11.6-14.0
    Total Leucocyte Count(TLC) : 8040 Cells/Cumm 4000-10000
    DIFFERENTIAL COUNT
    Polymorphs : 49 % 40-80
    Lymphocytes : 38 % 20-40
    Monocytes : 10 % 2-10
    Eosinophils : 3 % 1-6
    Basophils : 0 % 0-2
    Platelet Count : 2.98 Lakhs/Cumm 1.5-4.1
    •Method: Fully automated Hematology analyzer.
    Hb: Colorimetric, Total WBC: Impedence, Diff count: Calculated.RBC: Impedence
    HCT,MCV,MCHC,RDW-CV calculated. Platelets: Impedence Method.
    --End Of Report--
    """

    labs = extract_labs(sample_text)
    by_name = {l["name"]: l for l in labs}

    # Verify key CBC extractions
    assert "Hemoglobin" in by_name
    assert by_name["Hemoglobin"]["value"] == 12.7
    assert by_name["Hemoglobin"]["unit"] in ("gm/dl", "g/dl")

    assert "RBC" in by_name
    assert by_name["RBC"]["value"] == 4.59
    assert by_name["RBC"]["unit"].lower() == "millions/cumm"

    assert "Packed cell volume" in by_name
    assert by_name["Packed cell volume"]["value"] == 37.5
    assert by_name["Packed cell volume"]["abnormal"] is True  # 37.5 < 40

    assert "MCV" in by_name
    assert by_name["MCV"]["value"] == 81.7
    assert by_name["MCV"]["unit"].lower() == "fl"

    assert "MCH" in by_name
    assert by_name["MCH"]["value"] == 27.7
    assert by_name["MCH"]["unit"].lower() == "pg"

    assert "MCHC" in by_name
    assert by_name["MCHC"]["value"] == 33.9
    assert by_name["MCHC"]["unit"].lower() in ("g/dl", "gm/dl", "%")

    assert "Lymphocytes" in by_name
    assert by_name["Lymphocytes"]["value"] == 38.0
    assert by_name["Lymphocytes"]["unit"] == "%"  # Not cells/cumm

    assert "Eosinophils" in by_name
    assert by_name["Eosinophils"]["value"] == 3.0
    assert by_name["Eosinophils"]["unit"] == "%"  # Not lakhs/cumm

    # Ensure no phantom extractions
    assert "Triglycerides" not in by_name
    assert "Urine pH" not in by_name


def test_normalization_synonyms():
    # bare 'ph' in non-urine context should NOT normalize to Urine pH
    norm_ph, matched_ph = normalize_lab_name("ph", in_urine_context=False)
    assert norm_ph != "Urine pH" or matched_ph is False

    # bare 'ph' in urine context SHOULD normalize to Urine pH
    norm_u_ph, matched_u_ph = normalize_lab_name("ph", in_urine_context=True)
    assert norm_u_ph == "Urine pH" and matched_u_ph is True

    # bare 'pg' or 'tg' without lipid context should not accidentally map to Triglycerides
    norm_pg, _ = normalize_lab_name("pg", in_urine_context=False)
    assert norm_pg != "Triglycerides"


def test_form_labels_and_deduplication():
    from model.paddel import finalize_entries
    # 1. Test vertically stacked form labels with values to the right
    # (Checking that stacked left labels do not snowball into a single row)
    merged = {
        (50, 100, 150, 100, 150, 120, 50, 120): ((50, 100, 150, 100, 150, 120, 50, 120), "Time:", 0.99),
        (200, 100, 300, 100, 300, 120, 200, 120): ((200, 100, 300, 100, 300, 120, 200, 120), "10:30 AM", 0.99),
        (50, 130, 150, 130, 150, 150, 50, 150): ((50, 130, 150, 130, 150, 150, 50, 150), "Diagnosis:", 0.99),
        (200, 130, 320, 130, 320, 150, 200, 150): ((200, 130, 320, 130, 320, 150, 200, 150), "Hypertension", 0.99),
    }

    ordered_texts = [e["text"] for e in finalize_entries(merged)]
    assert ordered_texts == ["Time:", "10:30 AM", "Diagnosis:", "Hypertension"]

    # 2. Test repetitive watermark token filtering in handwritten model output
    from model.handwritten import text_to_lines
    raw_handwritten_output = "\n".join(["10 DAYS", "ATRIUM FOODS"] * 10)
    lines = text_to_lines(raw_handwritten_output)
    assert lines.count("10 DAYS") <= 3
    assert lines.count("ATRIUM FOODS") <= 3

