"""
patient_timeline.py

Formats patient history date-wise for AI longitudinal analysis & dashboard display.

Rules this module enforces:
  - Visits are grouped under their DATE. If more than one visit shares a
    date, they're shown as a hierarchy (Date -> Visit A, Visit B, ...)
    instead of a flat list.
  - If a visit has no supporting document (clean_text was null for every
    document_extraction row), that's stated explicitly and ONLY the
    structured database fields are used — nothing is borrowed from any
    other visit's document.
  - If a visit has more than one document, each document is kept fully
    separate with its own mini clinical summary, labeled Document 1,
    Document 2, etc. Documents are never merged together.
"""


def _format_document_block(doc, indent="    "):
    """Render one document's own extraction as a self-contained block."""
    lines = []
    label = doc.get("document_label", "Document")
    cs = doc.get("clinical_summary") or {}

    lines.append(f"{indent}--- {label} ---")

    age_sex = cs.get("age_sex")
    if age_sex:
        lines.append(f"{indent}Age/Sex (document): {age_sex}")

    impression = cs.get("clinical_impression")
    prov = cs.get("provisional_diagnosis")
    conf = cs.get("confirmed_diagnosis")
    if impression:
        lines.append(f"{indent}Clinical Impression ({label}): {impression}")
    if prov:
        lines.append(f"{indent}Provisional Diagnosis ({label}): {prov}")
    if conf:
        lines.append(f"{indent}Confirmed Diagnosis ({label}): {conf}")

    cc = cs.get("chief_complaint")
    if cc:
        lines.append(f"{indent}Chief Complaint ({label}): {cc}")

    symptoms = cs.get("symptoms") or []
    if symptoms:
        lines.append(f"{indent}Symptoms ({label}): {', '.join(symptoms)}")

    meds = doc.get("medications_found") or []
    if meds:
        lines.append(f"{indent}Medicines — from {label} (document, separate from prescription DB): {', '.join(meds)}")

    labs = doc.get("lab_results") or []
    if labs:
        lines.append(f"{indent}Lab Results ({label}):")
        for l in labs:
            t_name = l.get("test_name", "Test")
            val = l.get("value", "")
            ref = l.get("reference", "")
            status = l.get("status", "")
            lines.append(f"{indent}  • {t_name}: {val} (Ref: {ref}) [{status}]")

    fu = cs.get("follow_up")
    if fu:
        lines.append(f"{indent}Follow-up ({label}): {fu}")

    stats = cs.get("extraction_stats") or {}
    if stats.get("ai_assisted"):
        lines.append(f"{indent}[Note: some fields in {label} were AI-assisted — "
                      f"not caught by pattern matching, verify against source.]")

    return lines


def _format_visit_block(visit, indent="  "):
    """Render one visit: DB fields (always) + its document(s) or a clear
    'no supporting document' notice."""
    lines = []
    v_id = visit.get("visit_id", "Unknown")
    lines.append(f"{indent}VISIT ID: {v_id}")

    db_cc   = visit.get("db_chief_complaint") or visit.get("chief_complaint")
    db_prov = visit.get("db_provisional_dx") or visit.get("provisional_diagnosis")
    db_conf = visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis")

    lines.append(f"{indent}Chief Complaint (database): {db_cc or 'Not Documented'}")
    lines.append(f"{indent}Provisional Diagnosis (database): {db_prov or 'Not Documented'}")
    lines.append(f"{indent}Confirmed Diagnosis (database): {db_conf or 'Not Documented'}")

    instructions = visit.get("db_instructions") or []
    if instructions:
        lines.append(f"{indent}Prescription Instructions (database): {'; '.join(instructions)}")

    vitals = visit.get("db_vitals_rows") or []
    if vitals:
        v_str = ", ".join(f"{v['label']}: {v['value']}" for v in vitals)
        lines.append(f"{indent}Vitals (database): {v_str}")

    rx_items = visit.get("db_medications") or []
    if rx_items:
        lines.append(f"{indent}Medicines — Prescription (database):")
        for m in rx_items:
            parts = [m.get("name", "Unknown")]
            if m.get("dosage"):    parts.append(f"dosage {m['dosage']}")
            if m.get("frequency"): parts.append(f"frequency {m['frequency']}")
            if m.get("duration"):  parts.append(f"duration {m['duration']}")
            lines.append(f"{indent}  • " + ", ".join(parts))
    else:
        lines.append(f"{indent}Medicines — Prescription (database): none recorded")

    if not visit.get("has_document"):
        lines.append(f"{indent}Supporting Document: NONE — "
                      f"no document was uploaded/extracted for this visit. "
                      f"Analysis for this visit uses database fields only.")
    else:
        documents = visit.get("documents") or []
        lines.append(f"{indent}Supporting Document(s): {len(documents)} found")
        for doc in documents:
            lines.extend(_format_document_block(doc, indent=indent + "  "))

    discrepancies = visit.get("discrepancies") or []
    if discrepancies:
        lines.append(f"{indent}Discrepancies flagged:")
        for d in discrepancies:
            lines.append(f"{indent}  ⚠ {d}")

    return lines


def create_patient_timeline(patient_data):
    """
    Constructs a structured, date-grouped chronological timeline string
    from patient visit records, for AI longitudinal analysis & dashboard
    display. See module docstring for the grouping/document rules.
    """
    if not patient_data or "visits" not in patient_data or not patient_data["visits"]:
        return "No medical visits recorded for this patient."

    timeline_parts = []
    patient_id = patient_data.get("patient_id", "Unknown")
    timeline_parts.append(f"PATIENT ID: {patient_id}")
    timeline_parts.append(f"TOTAL VISITS RECORDED: {patient_data.get('total_visits', len(patient_data['visits']))}")
    timeline_parts.append("=" * 60)

    # ---- Group visits by date, preserving chronological order ----
    date_groups = []          # list of (date_label, [visits])
    date_index = {}           # date_label -> index into date_groups

    for visit in patient_data["visits"]:
        v_date = visit.get("visit_date")
        date_label = v_date[:10] if v_date else "Date Unspecified"
        if date_label not in date_index:
            date_index[date_label] = len(date_groups)
            date_groups.append((date_label, []))
        date_groups[date_index[date_label]][1].append(visit)

    for date_label, visits_on_date in date_groups:
        timeline_parts.append(f"\nDATE: {date_label}")
        timeline_parts.append("-" * 50)

        if len(visits_on_date) == 1:
            timeline_parts.extend(_format_visit_block(visits_on_date[0], indent="  "))
        else:
            # Multiple visits on the same date — show as an explicit hierarchy.
            timeline_parts.append(f"  {len(visits_on_date)} separate visits recorded on this date:")
            for i, visit in enumerate(visits_on_date, start=1):
                timeline_parts.append(f"\n  Visit {chr(64 + i)} of {len(visits_on_date)} (same date):")
                timeline_parts.extend(_format_visit_block(visit, indent="    "))

    return "\n".join(timeline_parts)