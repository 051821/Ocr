import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (PROJECT_ROOT, DATAANALYSIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import json
import pandas as pd
import streamlit as st
from analysis import fetch_patient_history, fetch_all_patient_ids
from Zai_analysis import analyze_patient_with_ai

st.set_page_config(
    page_title="Patient Clinical Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Typography */
.main-header   { font-size:2.1rem; font-weight:700; color:#1E3A8A; margin-bottom:.3rem; }
.visit-header  { font-size:1.25rem; font-weight:700; color:#1E3A8A; margin-bottom:.2rem; }
.section-label { font-size:.82rem; font-weight:600; color:#64748B;
                 text-transform:uppercase; letter-spacing:.06em; margin-bottom:.15rem; }

/* Disclaimer */
.disclaimer-box {
    background:#FEF3C7; border-left:5px solid #F59E0B;
    padding:10px 16px; border-radius:4px; margin-bottom:18px;
    font-size:.9rem; color:#78350F;
}

/* Discrepancy alert */
.discrepancy-box {
    background:#FFF7ED; border-left:4px solid #F97316;
    padding:8px 14px; border-radius:4px; margin-top:8px;
    font-size:.88rem; color:#9A3412;
}

/* Medication pill */
.med-pill {
    background:#E0F2FE; color:#0369A1;
    padding:4px 11px; border-radius:20px;
    font-weight:600; font-size:.88rem;
    display:inline-block; margin:3px 4px 3px 0;
}

/* Field label + value blocks */
.field-label { font-weight:600; color:#374151; font-size:.9rem; }
.field-value { color:#111827; font-size:.95rem; }
.field-na    { color:#9CA3AF; font-style:italic; font-size:.9rem; }

/* Summary narrative card */
.summary-card {
    background:#F0F9FF; border:1px solid #BAE6FD;
    border-radius:8px; padding:16px; margin-top:10px;
    font-size:.93rem; color:#0C4A6E;
}

/* Debug box */
.debug-box {
    background:#1E1E2E; color:#A6E3A1; border-radius:6px;
    padding:12px; font-size:.78rem; font-family:monospace;
}

/* Summary card polish */
.summary-card { box-shadow:0 1px 3px rgba(0,0,0,.06); }

/* Medication pill hover */
.med-pill { box-shadow:0 1px 2px rgba(0,0,0,.05); }

/* KPI metric cards */
div[data-testid="stMetric"] {
    background:#FFFFFF; border:1px solid #E5E7EB; border-radius:10px;
    padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,.05);
}

/* Expander (visit card) polish */
div[data-testid="stExpander"] {
    border:1px solid #E5E7EB !important; border-radius:10px !important;
    box-shadow:0 1px 2px rgba(0,0,0,.04); margin-bottom:10px;
}

/* Tabs — a bit more breathing room between the label row and content */
.stTabs [data-baseweb="tab-list"] { gap:4px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _nd(val, label="Not documented"):
    """Return value or a styled 'not documented' span."""
    return val if (val and str(val).strip() not in ("", "—", "-")) else label


def render_field(label: str, value, empty_msg="Not documented"):
    """Render a labelled clinical field row."""
    if value and str(value).strip():
        st.markdown(f"<span class='field-label'>{label}:</span> "
                    f"<span class='field-value'>{value}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='field-label'>{label}:</span> "
                    f"<span class='field-na'>{empty_msg}</span>", unsafe_allow_html=True)


def render_discrepancy(msg: str):
    st.markdown(f"<div class='discrepancy-box'>⚠️ <b>Data discrepancy:</b> {msg}</div>",
                unsafe_allow_html=True)


def vitals_table(rows: list):
    """Render vitals as a clean dataframe."""
    if not rows:
        st.markdown("<span class='field-na'>No vitals documented.</span>", unsafe_allow_html=True)
        return
    df = pd.DataFrame(rows, columns=["Vital", "Value", "Date", "Data Source"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def medications_table(med_rows: list):
    """Render medication table."""
    if not med_rows:
        st.markdown("<span class='field-na'>No medications identified in document.</span>",
                    unsafe_allow_html=True)
        return
    rows = []
    for m in med_rows:
        rows.append({
            "Medication":  m.get("name", "—"),
            "Strength":    _nd(m.get("strength"), "—"),
            "Frequency":   _nd(m.get("frequency"), "—"),
            "Duration":    _nd(m.get("duration"), "—"),
            "Notes":       _nd(m.get("notes"), "—"),
            "Source":      m.get("source", "Document"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def db_medications_table(rx_rows: list):
    """Render medicines from the prescriptionitem table (database only)."""
    if not rx_rows:
        st.markdown("<span class='field-na'>No prescription items recorded in database.</span>",
                    unsafe_allow_html=True)
        return
    rows = [{
        "Medication": m.get("name", "—"),
        "Dosage":     _nd(m.get("dosage"), "—"),
        "Frequency":  _nd(m.get("frequency"), "—"),
        "Duration":   _nd(m.get("duration"), "—"),
    } for m in rx_rows]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def lab_table(lab_rows: list):
    if not lab_rows:
        return
    df = pd.DataFrame(lab_rows)
    df.columns = ["Test Name", "Value", "Reference Interval", "Status"]

    def _style(v):
        if "Abnormal" in str(v):
            return "color:red;font-weight:600"
        if "Normal" in str(v):
            return "color:green;font-weight:600"
        return ""

    st.dataframe(df.style.map(_style, subset=["Status"]),
                 use_container_width=True, hide_index=True)


def build_supporting_summary(cs: dict, db_cc: str, db_prov: str, db_conf: str) -> str:
    """Build a short human-readable narrative from extracted clinical summary."""
    parts = []

    # Diagnosis
    dx = cs.get("clinical_impression") or db_prov or db_conf
    if dx:
        parts.append(f"The document records **{dx}** as the clinical impression.")

    # Chief complaint / symptoms
    cc = cs.get("chief_complaint") or db_cc
    syms = cs.get("symptoms", [])
    if cc:
        parts.append(f"The chief complaint documented is: *{cc}*.")
    if syms:
        sym_text = "; ".join(syms[:4])
        parts.append(f"Documented symptoms include: {sym_text}.")

    # Medications summary
    meds = cs.get("medications", [])
    if meds:
        med_names = ", ".join(m["name"] for m in meds[:6])
        parts.append(f"Medications listed: {med_names}.")

    # Follow-up
    fu = cs.get("follow_up")
    if fu:
        parts.append(f"Follow-up indicated: {fu}.")

    # Notes
    notes = cs.get("clinical_notes", [])
    if notes:
        parts.append(" | ".join(notes[:2]))

    if not parts:
        return "No additional supporting document information could be reliably extracted."
    return " ".join(parts)


# ── Page Header ──────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🏥 Patient Clinical & Longitudinal Dashboard</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
  <b>⚠️ Clinical Information Notice:</b> This dashboard presents structured retrospective
  patient history for informational review only.
  <b>It does NOT provide medical advice, prescribe treatment, or suggest changing medication regimens.</b>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Patient Lookup")

    try:
        patient_ids = fetch_all_patient_ids()
    except Exception as e:
        patient_ids = []
        st.error(f"DB connection error: {e}")

    selected_patient = "-- Select --"
    if patient_ids:
        selected_patient = st.selectbox(
            "Select Patient ID",
            ["-- Select --"] + patient_ids,
            index=1,
        )

    st.markdown("---")
    st.write(f"**Total patients in DB:** {len(patient_ids)}")
    st.markdown("---")

    # Developer / debug mode toggle (hidden in sidebar, off by default)
    debug_mode = st.checkbox("🔧 Developer Debug Mode", value=False,
                              help="Shows raw OCR, DB mappings, and extraction details. Off by default.")

# ── Patient ID Input ─────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    default_id = selected_patient if selected_patient != "-- Select --" else (patient_ids[0] if patient_ids else "")
    patient_id_input = st.text_input(
        "Enter Patient ID (UUID):",
        value=st.session_state.get("active_patient_id", default_id),
        placeholder="e.g. 0003e20a-7c64-410b-b0e2-9cb8c9005423",
    ).strip()
with c2:
    st.write(""); st.write("")
    btn_fetch = st.button("🔍 Fetch Records", type="primary", use_container_width=True)

# ── Data Fetch ────────────────────────────────────────────────────────────────
if (btn_fetch or patient_id_input) and patient_id_input:
    if (st.session_state.get("active_patient_id") != patient_id_input
            or "patient_data" not in st.session_state):
        with st.spinner(f"Fetching records for {patient_id_input[:18]}…"):
            try:
                pd_data = fetch_patient_history(patient_id_input)
                st.session_state["patient_data"] = pd_data
                st.session_state["active_patient_id"] = patient_id_input
                st.session_state.pop("ai_report", None)
            except Exception as e:
                st.error(f"Error fetching records: {e}")

patient_data = st.session_state.get("patient_data")

# ── Main Dashboard ────────────────────────────────────────────────────────────
if not patient_data:
    st.info("Enter a patient ID above and click **Fetch Records** to begin.")
    st.stop()

if patient_data["total_visits"] == 0:
    st.warning(f"No records found for `{patient_id_input}`.")
    st.stop()

st.markdown("---")

# KPI row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Patient ID", patient_data["patient_id"][:20] + "…")
k2.metric("Total Visits", patient_data["total_visits"])
k3.metric("Medicines (Prescription DB)", len(patient_data.get("all_db_medications", [])))
k4.metric("Medicines (Documents)", len(patient_data.get("all_document_medications", [])))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Date-wise visit cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📋 Date-Wise Clinical Visit Records")
st.caption("Visits are grouped by date. A visit with no uploaded document shows database-only fields; "
           "a visit with more than one document shows each document as its own separate summary.")


def render_document_section(doc: dict, db_cc, db_prov, db_conf, show_label=True):
    """Render ONE document's structured clinical summary. Never mixes in
    another document's data."""
    cs = doc.get("clinical_summary", {}) or {}
    label = doc.get("document_label", "Document")

    if show_label:
        st.markdown(f"##### 📄 {label}")

    age_sex = cs.get("age_sex")
    if age_sex:
        st.markdown("<div class='section-label'>Patient Demographics (Document)</div>",
                    unsafe_allow_html=True)
        render_field("Age / Sex", age_sex)
        st.write("")

    ocr_imp = cs.get("clinical_impression")
    ocr_prov = cs.get("provisional_diagnosis")
    ocr_conf = cs.get("confirmed_diagnosis")
    st.markdown("<div class='section-label'>Diagnosis (Document)</div>", unsafe_allow_html=True)
    render_field("Clinical Impression", ocr_imp or "Not clearly documented")
    if ocr_prov:
        render_field("Provisional Dx (Document)", ocr_prov)
    if ocr_conf:
        render_field("Confirmed Dx (Document)", ocr_conf)
    st.write("")

    ocr_cc = cs.get("chief_complaint")
    syms = cs.get("symptoms", [])
    if ocr_cc or syms:
        st.markdown("<div class='section-label'>Complaint & Symptoms (Document)</div>",
                    unsafe_allow_html=True)
        if ocr_cc:
            render_field("Chief Complaint", ocr_cc)
        for s in syms[:5]:
            st.markdown(f"  • {s}")
        st.write("")

    ocr_vitals = cs.get("vitals", [])
    if ocr_vitals:
        st.markdown("<div class='section-label'>Vitals (Document Extraction)</div>",
                    unsafe_allow_html=True)
        v_rows = []
        for vt in ocr_vitals:
            row = [vt["label"], vt["value"],
                   vt.get("date") or "—", vt.get("source", "Document")]
            if vt.get("discrepancy"):
                row[1] += " ⚠️"
            v_rows.append(row)
        vitals_table(v_rows)
        st.write("")

    doc_meds = cs.get("medications", [])
    if doc_meds:
        st.markdown("<div class='section-label'>Medication Regimen (Document)</div>",
                    unsafe_allow_html=True)
        medications_table(doc_meds)
        st.write("")

    labs = doc.get("lab_results", [])
    if labs:
        st.markdown("<div class='section-label'>Lab Test Evaluations (Document)</div>",
                    unsafe_allow_html=True)
        lab_table(labs)
        st.write("")

    fu = cs.get("follow_up")
    notes = cs.get("clinical_notes", [])
    if fu or notes:
        st.markdown("<div class='section-label'>Notes & Follow-Up</div>", unsafe_allow_html=True)
        if fu:
            render_field("Follow-up", fu)
        for n in notes[:3]:
            st.caption(f"• {n}")
        st.write("")

    summary_text = build_supporting_summary(cs, db_cc, db_prov, db_conf)
    st.markdown(
        f'<div class="summary-card">📝 <b>{label} Summary</b><br>{summary_text}</div>',
        unsafe_allow_html=True,
    )

    stats = cs.get("extraction_stats", {})
    if stats:
        total_l = stats.get("total_lines", 0)
        garbled = stats.get("garbled_lines", 0)
        extracted = stats.get("extracted_fields", 0)
        readable = total_l - garbled if total_l else 0
        with st.expander(f"🔎 View extraction / source information — {label}"):
            st.write(f"- **Source:** Uploaded clinical document (OCR pipeline)")
            st.write(f"- **Total lines in document:** {total_l}")
            st.write(f"- **Readable lines processed:** {readable}")
            st.write(f"- **Garbled / unreadable lines discarded:** {garbled}")
            st.write(f"- **Clinical fields successfully extracted:** {extracted}")
            if stats.get("ai_full_pass"):
                st.caption("🤖 This document's layout wasn't recognized by pattern "
                           "matching at all, so the full record above was read by an "
                           "AI-assisted pass instead of being left blank. Please verify "
                           "against the source document.")
            elif stats.get("ai_assisted"):
                st.caption("ℹ️ Some fields above were filled in by an AI-assisted pass "
                           "because they weren't caught by pattern matching. Please verify "
                           "against the source document.")
            if garbled > total_l * 0.4:
                st.warning("Extraction quality: **Partially readable** — "
                           "document may have significant OCR noise.")
            elif extracted > 0:
                st.success("Extraction quality: **Good**")
            else:
                st.info("Extraction quality: **Low** — no structured fields found.")


def render_visit_card(visit: dict, idx: int, total: int, expanded: bool):
    v_date = (visit["visit_date"] or "")
    date_disp = v_date[:10] if len(v_date) >= 10 else "Date unspecified"
    v_id_short = visit["visit_id"][:8]

    db_prov = visit.get("db_provisional_dx")
    db_conf = visit.get("db_confirmed_dx")
    db_cc   = visit.get("db_chief_complaint")
    documents = visit.get("documents", [])
    first_doc_imp = documents[0]["clinical_summary"].get("clinical_impression") if documents else None
    headline_dx = db_conf or db_prov or first_doc_imp or "Diagnosis not documented"

    with st.expander(
        f"📅 Visit #{idx}  ·  {date_disp}  ·  {headline_dx}  ·  ID: {v_id_short}…",
        expanded=expanded,
    ):
        for disc in visit.get("discrepancies", []):
            render_discrepancy(disc)

        col_db, col_doc = st.columns([1, 1], gap="large")

        # ── LEFT: Structured Database Records (always shown, always DB-only) ──
        with col_db:
            st.markdown("#### 🏥 Structured Database Record")

            st.markdown("<div class='section-label'>Diagnosis</div>", unsafe_allow_html=True)
            render_field("Provisional Dx", db_prov)
            render_field("Confirmed Dx",   db_conf)
            st.write("")

            st.markdown("<div class='section-label'>Chief Complaint</div>", unsafe_allow_html=True)
            render_field("Complaint", db_cc)
            st.write("")

            instructions = visit.get("db_instructions", [])
            if instructions:
                st.markdown("<div class='section-label'>Prescription Instructions</div>",
                            unsafe_allow_html=True)
                for instr in instructions:
                    st.markdown(f"  • {instr}")
                st.write("")

            st.markdown("<div class='section-label'>Vitals (Structured Record)</div>",
                        unsafe_allow_html=True)
            db_vital_rows = visit.get("db_vitals_rows", [])
            if db_vital_rows:
                vitals_table([[r["label"], r["value"], r["date"], r["source"]]
                              for r in db_vital_rows])
            else:
                st.markdown("<span class='field-na'>No structured vitals available.</span>",
                            unsafe_allow_html=True)

            st.write("")
            st.markdown("<div class='section-label'>💊 Medicines — Prescription (Database)</div>",
                        unsafe_allow_html=True)
            db_medications_table(visit.get("db_medications", []))

        # ── RIGHT: Document(s) — kept fully separate, or a clear "none" notice ──
        with col_doc:
            if not visit.get("has_document"):
                st.markdown("#### 📄 Supporting Document")
                st.info(
                    "**No supporting document for this visit.** No document was "
                    "uploaded/extracted here — the analysis for this visit uses "
                    "**database fields only** (left column). Nothing from any other "
                    "visit's document is used."
                )
            else:
                st.markdown(f"#### 📄 Supporting Document(s) — {len(documents)} found")
                if len(documents) > 1:
                    # Tabs instead of stacking every document — with several
                    # documents (this dataset has visits with 6), a stacked
                    # layout turns one visit card into a very long scroll.
                    tab_labels = [d.get("document_label", f"Document {i+1}")
                                  for i, d in enumerate(documents)]
                    for tab, doc in zip(st.tabs(tab_labels), documents):
                        with tab:
                            render_document_section(doc, db_cc, db_prov, db_conf, show_label=False)
                else:
                    render_document_section(documents[0], db_cc, db_prov, db_conf)

            if debug_mode:
                with st.expander("🔧 Developer Debug"):
                    for doc in documents:
                        st.markdown(f"**{doc['document_label']} raw text:** not stored post-extraction "
                                    f"(only structured fields are kept in memory).")
                    st.markdown("**DB vitals_json:**")
                    st.json(visit.get("db_vitals_json") or {})
                    st.markdown("**Discrepancies detected:**")
                    for d in visit.get("discrepancies", []):
                        st.markdown(f"- {d}")


# ── Group visits by date, render as a hierarchy when a date has >1 visit ──
_date_groups = []
_date_index = {}
for _visit in patient_data["visits"]:
    _v_date = _visit.get("visit_date")
    _date_label = _v_date[:10] if _v_date else "Date unspecified"
    if _date_label not in _date_index:
        _date_index[_date_label] = len(_date_groups)
        _date_groups.append((_date_label, []))
    _date_groups[_date_index[_date_label]][1].append(_visit)

_running_idx = 0
_total_visits = len(patient_data["visits"])
for _date_label, _visits_on_date in _date_groups:
    if len(_visits_on_date) > 1:
        st.markdown(f"#### 🗓️ {_date_label} — {len(_visits_on_date)} visits recorded")
    for _v in _visits_on_date:
        _running_idx += 1
        render_visit_card(_v, _running_idx, _total_visits,
                           expanded=(_running_idx == _total_visits))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Quick Executive Summary
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📌 Quick Clinical & Longitudinal Executive Summary")

# Trajectory table
st.markdown("#### 📈 Visit Trajectory Overview")
trajectory = []
for i, v in enumerate(patient_data["visits"], start=1):
    vd    = (v.get("visit_date") or "")[:10] or "—"
    docs  = v.get("documents") or []
    _cs   = docs[0]["clinical_summary"] if docs else {}
    pv    = v.get("db_provisional_dx") or _cs.get("clinical_impression") or _cs.get("provisional_diagnosis") or "—"
    cv    = v.get("db_confirmed_dx") or _cs.get("confirmed_diagnosis") or "—"
    cc    = v.get("db_chief_complaint") or _cs.get("chief_complaint") or "—"
    mrx   = len(v.get("db_medications", []))
    mdoc  = len(v.get("document_medications", []))
    lbs   = len(v.get("lab_results", []))
    n_doc = len(docs)
    trajectory.append({
        "Visit #": f"V{i}", "Date": vd,
        "Chief Complaint": cc[:60] + ("…" if len(cc) > 60 else ""),
        "Provisional Dx": pv, "Confirmed Dx": cv,
        "Docs": n_doc if n_doc else "None", "Meds (Rx DB)": mrx, "Meds (Docs)": mdoc, "Labs": lbs,
    })
st.dataframe(pd.DataFrame(trajectory), use_container_width=True, hide_index=True)

# AI Analysis Note
st.markdown("#### 🤖 AI Retrospective Clinical Analysis Note")
st.caption("Analysis is based on past documented data only. "
           "No medical advice or prescribing recommendations are generated.")

c_ai, _ = st.columns([1, 3])
with c_ai:
    btn_ai = st.button("✨ Generate / Refresh AI Analysis", type="secondary",
                       use_container_width=True)

if btn_ai or "ai_report" not in st.session_state:
    with st.spinner("Generating retrospective clinical analysis…"):
        try:
            report = analyze_patient_with_ai(patient_data)
            st.session_state["ai_report"] = report
        except Exception as e:
            st.error(f"AI analysis failed: {e}")

if "ai_report" in st.session_state:
    st.info(st.session_state["ai_report"])