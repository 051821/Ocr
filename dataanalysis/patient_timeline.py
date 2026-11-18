"""
patient_timeline.py

Formats patient history date-wise for AI longitudinal analysis & dashboard display (GET /patients/{id}/timeline).

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
  - Nothing is shown as something it is not: a prescription line the OCR read as a lab result is not listed as a lab;
    a vital an outpatient could not have is marked as an entry error; a vital the record and a document give
    differently is shown with both values, not dropped; a document with nothing readable says so.
  - The prescriber is named by specialization only, never by name.
"""
from core.encounters import dose_not_specified, unclear_dose


def _medicine_line_checker(visit):
    """is_medicine_line for this visit: a 'Not Evaluated' lab row that is a prescription line (dose form, or the
    name of a medicine prescribed at this visit)."""
    from clinical_snapshot import _first_word, is_medicine_line
    words = {_first_word(m.get("name")) for m in visit.get("db_medications") or []}
    return lambda lab: is_medicine_line(lab, words)


def _vital_text(label, value):
    from clinical_snapshot import _celsius_note, vital_implausible
    if not vital_implausible(label, value):
        return f"{label}: {value}"
    return f"{label}: {value} (implausible{_celsius_note(label, value) or ' — likely an entry error'}; not used)"


def _clean(text):
    """Record text as written, without the trailing comma / semicolon the entry form often leaves."""
    return str(text).strip().rstrip(",;").strip() if text else ""


def _age_sex(text):
    """(age in years, 'male' / 'female') from '36 years / Female', missing parts None."""
    import re
    age = re.search(r"(\d{1,3})\s*(?:y|yr|yrs|year|years)?\b", str(text or ""), re.I)
    sex = re.search(r"\b(male|female|m|f)\b", str(text or ""), re.I)
    return (int(age.group(1)) if age else None,
            {"m": "male", "f": "female"}.get(sex.group(1).casefold(), sex.group(1).casefold()) if sex else None)


def _document_checks(doc, visit_no, context):
    """Notes that a document may not belong where it is: another patient's age / sex, or another visit's complaint
    (often an earlier prescription brought to this visit)."""
    notes = []
    cs = doc.get("clinical_summary") or {}
    label = doc.get("document_label", "Document")
    d_age, d_sex = _age_sex(cs.get("age_sex"))
    p_age, p_sex = _age_sex(context.get("age_sex"))
    if (d_sex and p_sex and d_sex != p_sex) or (d_age is not None and p_age is not None and abs(d_age - p_age) > 2):
        notes.append(f"{label}: age/sex on the document ({cs.get('age_sex')}) differs from the patient's record "
                     f"({context.get('age_sex')}); check it belongs to this patient before using it.")
    cc = _clean(cs.get("chief_complaint")).casefold()
    own = context["complaints"].get(visit_no, "")
    other = [n for n, c in context["complaints"].items() if n != visit_no and c and c == cc]
    if cc and other and cc != own:
        n = other[0]
        notes.append(f"{label}: its complaint matches visit #{n} ({context['dates'].get(n, 'date unknown')}), not this "
                     f"visit; it may be that visit's prescription brought again.")
    return notes


def _format_document_block(doc, indent="    ", is_medicine=lambda lab: False):
    """Render one document's own extraction as a self-contained block."""
    lines = []
    label = doc.get("document_label", "Document")
    cs = doc.get("clinical_summary") or {}

    if doc.get("document_date"):  # read from the document itself (label + date only)
        lines.append(f"{indent}Report date ({label}): {doc['document_date']}"
                     + (f" (written as '{doc['document_date_label']}')" if doc.get("document_date_label") else ""))

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

    vitals = [v for v in cs.get("vitals") or [] if v.get("label") and v.get("value")]
    if vitals:
        lines.append(f"{indent}Vitals ({label}): " + ", ".join(_vital_text(v["label"], v["value"]) for v in vitals))

    meds = doc.get("medications_found") or []
    if meds:
        lines.append(f"{indent}Medicines — from {label} (document, separate from prescription DB): {', '.join(meds)}")

    all_labs = doc.get("lab_results") or []
    labs = [l for l in all_labs if not is_medicine(l)]
    if labs:
        lines.append(f"{indent}Lab Results ({label}):")
        for l in labs:
            t_name = l.get("test_name", "Test")
            val = l.get("value", "")
            ref = l.get("reference", "")
            status = l.get("status", "")
            lines.append(f"{indent}  • {t_name}: {val} (Ref: {ref}) [{status}]")
    if len(labs) < len(all_labs):
        lines.append(f"{indent}Note ({label}): {len(all_labs) - len(labs)} prescription line(s) the scan read as lab "
                     f"results; not lab tests, not listed.")

    fu = cs.get("follow_up")
    if fu:
        lines.append(f"{indent}Follow-up ({label}): {fu}")

    stats = cs.get("extraction_stats") or {}
    if not lines:
        garbled = stats.get("garbled_lines") or 0
        lines.append(f"{indent}No clinical fields could be read from {label}"
                     + (f" ({garbled} unreadable line(s))." if garbled else "."))
    if stats.get("ai_assisted"):
        lines.append(f"{indent}[Note: some fields in {label} were AI-assisted — "
                      f"not caught by pattern matching, verify against source.]")

    return [f"{indent}--- {label} ---"] + lines


def _medicine_text(m):
    """'LOBET GM CREAM — 1-0-1 · After Food · 10 days', with the zero-dose cases said plainly."""
    parts = [str(x).strip() for x in (m.get("dosage"), m.get("frequency"), m.get("duration")) if str(x or "").strip()]
    text = f"{m.get('name', 'Unknown')}" + (f" — {' · '.join(parts)}" if parts else "")
    if unclear_dose(m):
        text += " (0-0-0 with no duration: no dose or course stated; not counted as taken)"
    elif dose_not_specified(m):
        text += " (dose not entered)"
    return text


def _followup_lines(followups, indent):
    """One line per scheduled date: every status / resolution written for that date (two prescriptions of a visit
    can each schedule the same follow-up)."""
    by_date = {}
    for fu in followups:
        by_date.setdefault(str(fu.get("scheduled_date") or "?")[:10], []).append(fu)
    statuses = [next((f.get("status") for f in fus if f.get("status") == "MISSED"), fus[0].get("status", ""))
                for fus in by_date.values()]
    lines = [f"{indent}Follow-up Schedule ({len(by_date)} scheduled | {statuses.count('COMPLETED')} completed | "
             f"{statuses.count('MISSED')} missed):"]
    for date, fus in by_date.items():
        status = " / ".join(dict.fromkeys(str(f.get("status") or "?") for f in fus))
        emerg = " 🚨 EMERGENCY" if any(f.get("is_emergency") for f in fus) else ""
        res = "; ".join(dict.fromkeys(
            str(f["resolution"]) + (f" [{f['resolution_notes']}]" if f.get("resolution_notes") else "")
            for f in fus if f.get("resolution")))
        lines.append(f"{indent}  • {date} | {status}{emerg}" + (f" → {res}" if res else ""))
    return lines


def _format_visit_block(visit, indent="  ", visit_no=None, context=None):
    """Render one visit: DB fields (always) + its document(s) or a clear
    'no supporting document' notice."""
    lines = []
    v_id = visit.get("visit_id", "Unknown")
    lines.append(f"{indent}VISIT ID: {v_id}")

    db_cc   = visit.get("db_chief_complaint") or visit.get("chief_complaint")
    db_prov = visit.get("db_provisional_dx") or visit.get("provisional_diagnosis")
    db_conf = visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis")

    lines.append(f"{indent}Chief Complaint (database): {_clean(db_cc) or 'Not Documented'}")
    lines.append(f"{indent}Provisional Diagnosis (database): {_clean(db_prov) or 'Not Documented'}")
    lines.append(f"{indent}Confirmed Diagnosis (database): {_clean(db_conf) or 'Not Documented'}")

    specs = visit.get("prescriber_specializations") or []
    if specs:
        lines.append(f"{indent}Prescribed by (specialization): {', '.join(specs)}")

    instructions = visit.get("db_instructions") or []
    if instructions:
        lines.append(f"{indent}Prescription Instructions (database): {'; '.join(instructions)}")

    vitals = visit.get("db_vitals_rows") or []
    if vitals:
        lines.append(f"{indent}Vitals (database): " + ", ".join(_vital_text(v["label"], v["value"]) for v in vitals))
    # Readings the record and a document give differently: both shown, neither used by the rules (data_quality).
    for c in visit.get("conflicting_vitals") or []:
        doc = c.get("document_value")
        doc = f"{doc:g}" if isinstance(doc, (int, float)) else doc
        lines.append(f"{indent}Vitals conflict: {c.get('label')} {c.get('value')} in the database, {doc} in the "
                     f"document; not used until checked.")

    rx_items = visit.get("db_medications") or []
    if rx_items:
        lines.append(f"{indent}Medicines — Prescription (database):")
        for m in rx_items:
            lines.append(f"{indent}  • " + _medicine_text(m))
    else:
        lines.append(f"{indent}Medicines — Prescription (database): none recorded")

    if not visit.get("has_document"):
        lines.append(f"{indent}Supporting Document: NONE — "
                      f"no document was uploaded/extracted for this visit. "
                      f"Analysis for this visit uses database fields only.")
    else:
        documents = visit.get("documents") or []
        lines.append(f"{indent}Supporting Document(s): {len(documents)} found")
        is_medicine = _medicine_line_checker(visit)
        for doc in documents:
            lines.extend(_format_document_block(doc, indent=indent + "  ", is_medicine=is_medicine))

    discrepancies = list(visit.get("discrepancies") or [])
    if context and visit.get("has_document"):
        for doc in visit.get("documents") or []:
            discrepancies += _document_checks(doc, visit_no, context)
    if discrepancies:
        lines.append(f"{indent}Discrepancies flagged:")
        for d in discrepancies:
            lines.append(f"{indent}  ⚠ {d}")

    followups = visit.get("followups") or []
    if followups:
        lines.extend(_followup_lines(followups, indent))

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
    visits = patient_data["visits"]
    context = {"age_sex": patient_data.get("age_sex"),
               "complaints": {n: _clean(v.get("db_chief_complaint") or v.get("chief_complaint")).casefold()
                              for n, v in enumerate(visits, 1)},
               "dates": {n: str(v.get("visit_date") or "")[:10] for n, v in enumerate(visits, 1)}}
    number = {id(v): n for n, v in enumerate(visits, 1)}

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
            v = visits_on_date[0]
            timeline_parts.extend(_format_visit_block(v, indent="  ", visit_no=number[id(v)], context=context))
        else:
            # Multiple visits on the same date — show as an explicit hierarchy.
            timeline_parts.append(f"  {len(visits_on_date)} separate visits recorded on this date:")
            for i, visit in enumerate(visits_on_date, start=1):
                timeline_parts.append(f"\n  Visit {chr(64 + i)} of {len(visits_on_date)} (same date):")
                timeline_parts.extend(_format_visit_block(visit, indent="    ", visit_no=number[id(visit)], context=context))

    return "\n".join(timeline_parts)
