import os
import sys
import re
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (PROJECT_ROOT, DATAANALYSIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import streamlit as st
from analysis import fetch_patient_history, fetch_all_patient_ids
from storage.ai_analysis_csv import get_patient_row
from ai_review_service import get_or_generate_review, is_review_current
from patient_chat import answer_patient_question
from medication_journey import compute_medication_journey
from neo4j_client import is_neo4j_configured, sync_patient_to_graph
from ui_components import render_clinical_encounter, render_doctor_friendly_ai_review
try:
    from config import OPENROUTER_API_KEY as CHAT_API_KEY
except Exception:
    CHAT_API_KEY = None

st.set_page_config(
    page_title="Patient Intelligence | Clinical Longitudinal Record",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean Clinical SaaS Stylesheet ───────────────────────────────────────────
st.markdown("""
<style>
/* Base Theme & Layout */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
}

/* Typography Hierarchy */
.app-brand {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin-bottom: 0.15rem;
}
.app-subtitle {
    font-size: 0.92rem;
    color: #64748B;
    margin-bottom: 0.8rem;
    font-weight: 400;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 650;
    color: #1E293B;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
    letter-spacing: -0.01em;
}

/* Clinical Notices & Banners */
.clinical-notice {
    background-color: #F8FAFC;
    border-left: 3px solid #0284C7;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: #334155;
    margin-bottom: 1rem;
    line-height: 1.45;
}

/* Overview Metric Cards */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 12px 14px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    height: 100%;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.18rem;
    font-weight: 650;
    color: #0F172A;
    line-height: 1.25;
}
.metric-sub {
    font-size: 0.75rem;
    color: #94A3B8;
    margin-top: 3px;
}

/* Discrepancy Alert Box */
.discrepancy-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #D97706;
    border-radius: 5px;
    padding: 10px 14px;
    margin-bottom: 0.6rem;
    font-size: 0.86rem;
    color: #92400E;
    line-height: 1.45;
}
.discrepancy-title {
    font-weight: 650;
    color: #B45309;
    margin-bottom: 3px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Field & Detail Styling with high contrast & theme support */
.field-row {
    margin-bottom: 0.45rem;
    font-size: 0.92rem;
    line-height: 1.5;
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 6px;
}
.field-name {
    font-weight: 650;
    color: #0284C7; /* Clean clinical blue for clear recognition */
    min-width: 140px;
}
.field-val {
    color: inherit;
    font-weight: 500;
}
.field-na {
    color: #94A3B8;
    font-style: italic;
}
.section-subhead {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #0369A1;
    margin-top: 0.7rem;
    margin-bottom: 0.35rem;
    display: block;
}

/* Clean Timeline Tree Styling */
.timeline-card {
    border-left: 2px solid #0284C7;
    padding-left: 14px;
    margin-left: 6px;
    margin-bottom: 1.2rem;
    position: relative;
}
.timeline-date {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0284C7;
    margin-bottom: 6px;
}
.diff-badge-added {
    background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0;
    padding: 2px 7px; border-radius: 4px; font-size: 0.76rem; font-weight: 600;
}
.diff-badge-removed {
    background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA;
    padding: 2px 7px; border-radius: 4px; font-size: 0.76rem; font-weight: 600;
}
.diff-badge-changed {
    background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A;
    padding: 2px 7px; border-radius: 4px; font-size: 0.76rem; font-weight: 600;
}
.diff-badge-stable {
    background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0;
    padding: 2px 7px; border-radius: 4px; font-size: 0.76rem; font-weight: 600;
}

/* Clean AI Callout Container */
.ai-narrative-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 14px 16px;
    font-size: 0.90rem;
    color: #1E293B;
    line-height: 1.55;
    margin-bottom: 0.75rem;
}

/* Traceability Source Badge */
.trace-badge {
    display: inline-block;
    padding: 2px 7px;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 4px;
    background: #F1F5F9;
    color: #475569;
    border: 1px solid #CBD5E1;
}

/* Tab bar polish */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    padding-top: 6px;
    padding-bottom: 8px;
    font-weight: 500;
    font-size: 0.90rem;
}

/* Streamlit native component refinements */
div[data-testid="stExpander"] {
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    margin-bottom: 0.5rem;
}

/* ── Doctor-Friendly Clinical Timeline Styles ── */
.badge-confirmed {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 3px 9px;
    border-radius: 4px;
    font-size: 0.80rem;
    font-weight: 650;
    display: inline-block;
    margin-right: 4px;
    margin-bottom: 4px;
}
.badge-provisional {
    background: rgba(59, 130, 246, 0.15);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.35);
    padding: 3px 9px;
    border-radius: 4px;
    font-size: 0.80rem;
    font-weight: 650;
    display: inline-block;
    margin-right: 4px;
    margin-bottom: 4px;
}
.badge-missed {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 650;
    display: inline-block;
}
.badge-completed {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 650;
    display: inline-block;
}
.badge-pending {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 650;
    display: inline-block;
}
.encounter-subheading {
    font-size: 0.80rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #0284C7;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}
.doc-clean-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 6px;
    padding: 9px 12px;
    margin-bottom: 6px;
}
.doc-label-badge {
    font-weight: 650;
    font-size: 0.83rem;
    color: #0284C7;
}
.doc-id-badge {
    font-size: 0.72rem;
    color: #94A3B8;
    font-family: monospace;
}
.symptom-chip {
    display: inline-block;
    background: rgba(59, 130, 246, 0.12);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 0.74rem;
    margin-right: 4px;
    margin-bottom: 2px;
}
.lab-abnormal-card {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-left: 3px solid #EF4444;
    border-radius: 4px;
    padding: 5px 10px;
    margin-bottom: 4px;
    font-size: 0.85rem;
}
.lab-status-tag {
    background: rgba(239, 68, 68, 0.2);
    color: #F87171;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.72rem;
    font-weight: 700;
}

/* ── Doctor-Friendly AI Clinical Review Styles ── */
.ai-hero-card {
    background: rgba(2, 132, 199, 0.08);
    border: 1px solid rgba(2, 132, 199, 0.25);
    border-left: 4.5px solid #0284C7;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 1.2rem;
}
.ai-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0284C7;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.ai-hero-text {
    font-size: 0.95rem;
    line-height: 1.6;
    color: inherit;
}
.ai-section-hdr {
    font-size: 1.02rem;
    font-weight: 700;
    color: #0284C7;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ai-insight-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-left: 3.5px solid #3B82F6;
    border-radius: 6px;
    padding: 11px 15px;
    margin-bottom: 0.55rem;
}
.ai-tag {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 5px;
}
.ai-insight-text {
    font-size: 0.92rem;
    line-height: 1.5;
    color: inherit;
}
.ai-review-box {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-left: 4px solid #F59E0B;
    border-radius: 6px;
    padding: 12px 18px;
    margin-bottom: 1.2rem;
}
.ai-review-title {
    font-weight: 700;
    color: #B45309;
    font-size: 0.88rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.ai-review-list {
    margin: 0;
    padding-left: 1.2rem;
    font-size: 0.90rem;
    line-height: 1.5;
}
.ai-review-list li {
    margin-bottom: 4px;
}
.ai-consideration-card {
    background: rgba(16, 185, 129, 0.06);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-left: 3px solid #10B981;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: flex-start;
    min-height: 68px;
}
.evidence-chip {
    display: inline-block;
    background: rgba(148, 163, 184, 0.12);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 4px;
    padding: 4px 9px;
    margin: 3px;
    font-size: 0.78rem;
}
.ai-declaration-footer {
    font-size: 0.78rem;
    color: #94A3B8;
    border-top: 1px solid rgba(148, 163, 184, 0.2);
    padding-top: 8px;
    margin-top: 1.2rem;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ── Robust Safe Formatting Helpers ───────────────────────────────────────────
def safe_val(val, default="Not available"):
    """Safely format values, avoiding None, NaN, empty strings or placeholders."""
    if val is None or pd.isna(val):
        return default
    s = str(val).strip()
    if s in ("", "—", "-", "None", "nan", "NaN", "null", "[]", "{}"):
        return default
    return s


def render_field(label: str, value, default="Not available"):
    """Render a clean, responsive labelled field row."""
    s = safe_val(value, default)
    if s == default:
        st.markdown(f"<div class='field-row'><span class='field-name'>{label}:</span> "
                    f"<span class='field-na'>{default}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='field-row'><span class='field-name'>{label}:</span> "
                    f"<span class='field-val'>{s}</span></div>", unsafe_allow_html=True)


st.markdown('<div class="app-brand">Patient Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Longitudinal Clinical Record & Retrospective AI Analysis</div>', unsafe_allow_html=True)

st.markdown("""
<div class="clinical-notice">
    <strong>Clinical Reference Notice:</strong> This platform synthesizes structured database entries, OCR-extracted clinical documents,
    and retrospective AI observations for historical review only. <strong>It does not provide confirmed medical advice, diagnosis, or treatment recommendations.</strong>
</div>
""", unsafe_allow_html=True)


# ── Sidebar Lookup ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Patient Selector")
    try:
        patient_ids = fetch_all_patient_ids()
    except Exception as e:
        patient_ids = []
        st.error(f"Database connection error: {e}")

    selected_patient = "-- Select Patient --"
    if patient_ids:
        default_index = 1 if len(patient_ids) > 1 else 0
        selected_patient = st.selectbox(
            "Registered Patients in Database",
            ["-- Select Patient --"] + patient_ids,
            index=default_index,
            label_visibility="collapsed"
        )
        st.caption(f"Total cohort in database: **{len(patient_ids)}** patients")
    st.markdown("---")
    st.markdown("<small style='color:#64748B;'><strong>Pipeline:</strong> Medical Records → OCR → Structured Schema → Retrospective AI</small>", unsafe_allow_html=True)


# ── Search & Patient Bar ─────────────────────────────────────────────────────
top_c1, top_c2 = st.columns([4, 1])
with top_c1:
    default_id = selected_patient if selected_patient != "-- Select Patient --" else (patient_ids[0] if patient_ids else "")
    patient_id_input = st.text_input(
        "Patient ID (UUID):",
        value=st.session_state.get("active_patient_id", default_id),
        placeholder="Enter patient UUID...",
        label_visibility="collapsed"
    ).strip()
with top_c2:
    btn_fetch = st.button("Fetch Record", type="primary", use_container_width=True)

# ── Data Fetching & Caching Management ───────────────────────────────────────
if (btn_fetch or patient_id_input) and patient_id_input:
    if (btn_fetch
            or st.session_state.get("active_patient_id") != patient_id_input
            or "patient_data" not in st.session_state):
        with st.spinner("Retrieving longitudinal patient records..."):
            try:
                pd_data = fetch_patient_history(patient_id_input)
                prior_patient_id = st.session_state.get("active_patient_id")
                if prior_patient_id and prior_patient_id != patient_id_input:
                    st.session_state.pop("patient_chat_messages", None)
                    st.session_state.pop("patient_chat_patient_id", None)
                st.session_state["patient_data"] = pd_data
                st.session_state["active_patient_id"] = patient_id_input
                st.session_state.pop("ai_report", None)
                st.session_state.pop("ai_metadata", None)
                if is_neo4j_configured():
                    try:
                        sync_patient_to_graph(pd_data)
                    except Exception:
                        pass

            except Exception as e:
                st.error(f"Error retrieving clinical record: {e}")

patient_data = st.session_state.get("patient_data")

if not patient_data:
    st.info("Please select or enter a Patient ID above to view clinical intelligence.")
    st.stop()

if patient_data.get("total_visits", 0) == 0:
    st.warning(f"No clinical encounters found for Patient ID `{patient_id_input}`.")
    st.stop()


# ── 2. PATIENT OVERVIEW (Compact Metric Cards) ───────────────────────────────
visits = patient_data.get("visits", [])
total_visits = patient_data.get("total_visits", 0)

# Extract Age / Sex from documents if available
patient_age_sex = "Not available"
for v in visits:
    for doc in v.get("documents", []):
        as_val = doc.get("clinical_summary", {}).get("age_sex")
        if as_val and str(as_val).strip():
            patient_age_sex = str(as_val).strip()
            break
    if patient_age_sex != "Not available":
        break

# Latest visit date
latest_visit_date = "Not available"
if visits and visits[-1].get("visit_date"):
    latest_visit_date = visits[-1]["visit_date"][:10]

# Active / chronic conditions (collected from provisional and confirmed diagnoses)
all_conditions = set()
for v in visits:
    for dx in (v.get("confirmed_diagnosis"), v.get("provisional_diagnosis"), v.get("db_confirmed_dx"), v.get("db_provisional_dx")):
        if dx and str(dx).strip() and str(dx).strip().lower() not in ("—", "-", "none", "not documented"):
            all_conditions.add(str(dx).strip())

conditions_display = ", ".join(sorted(all_conditions)[:3]) if all_conditions else "Not documented"
if len(all_conditions) > 3:
    conditions_display += f" (+{len(all_conditions)-3} more)"

# Medication count
all_meds_count = len(patient_data.get("all_medications", []))
db_meds_count = len(patient_data.get("all_db_medications", []))

# Status pill
# Follow-up Adherence summary
all_fus = [fu for v in visits for fu in v.get("followups", [])]
fu_completed = sum(1 for fu in all_fus if str(fu.get("status", "")).upper() == "COMPLETED")
fu_missed = sum(1 for fu in all_fus if str(fu.get("status", "")).upper() == "MISSED")
fu_adherence_str = f"{fu_completed}/{len(all_fus)}" if all_fus else "Not scheduled"

ov1, ov2, ov3, ov4, ov5, ov6, ov7 = st.columns([1.6, 1.1, 0.8, 1.0, 1.5, 1.1, 1.1])
with ov1:
    pid_disp = patient_data["patient_id"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Patient ID</div>
        <div class="metric-value" style="font-size:0.95rem; word-break:break-all;">{pid_disp[:14]}...{pid_disp[-6:]}</div>
        <div class="metric-sub">Verified Clinical Record</div>
    </div>
    """, unsafe_allow_html=True)
with ov2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Demographics</div>
        <div class="metric-value" style="font-size:1.05rem;">{patient_age_sex}</div>
        <div class="metric-sub">Age / Gender</div>
    </div>
    """, unsafe_allow_html=True)
with ov3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Encounters</div>
        <div class="metric-value">{total_visits}</div>
        <div class="metric-sub">Recorded visits</div>
    </div>
    """, unsafe_allow_html=True)
with ov4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Latest Visit</div>
        <div class="metric-value" style="font-size:1.05rem;">{latest_visit_date}</div>
        <div class="metric-sub">Most recent entry</div>
    </div>
    """, unsafe_allow_html=True)
with ov5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Primary Conditions</div>
        <div class="metric-value" style="font-size:0.92rem;">{conditions_display}</div>
        <div class="metric-sub">Diagnoses recorded</div>
    </div>
    """, unsafe_allow_html=True)
with ov6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Medications</div>
        <div class="metric-value">{all_meds_count} <span style="font-size:0.75rem; color:#64748B;">({db_meds_count} Rx)</span></div>
        <div class="metric-sub">Total distinct drugs</div>
    </div>
    """, unsafe_allow_html=True)
with ov7:
    fu_sub = f"{fu_missed} missed" if fu_missed else "Completed / Scheduled"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Follow-up Adherence</div>
        <div class="metric-value" style="font-size:1.05rem;">{fu_adherence_str}</div>
        <div class="metric-sub">{fu_sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ── Primary Navigation Tabs ──────────────────────────────────────────────────
tab_timeline, tab_meds, tab_labs, tab_ai, tab_chat, tab_discrepancies, tab_sources = st.tabs([
    "📅 Clinical Timeline",
    "💊 Medication Journey",
    "🧪 Lab Results",
    "🤖 Retrospective AI Analysis",
    "💬 Patient Chat",
    "⚠️ Record Discrepancies",
    "📄 Source Traceability"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: CLINICAL TIMELINE (Deterministic "What happened over time?")
# ═════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.markdown('<div class="section-title">Longitudinal Clinical Timeline</div>', unsafe_allow_html=True)
    st.caption("Deterministic chronology of encounters, complaints, diagnoses, document findings, labs, medication shifts, and follow-ups.")

    med_journey_data = compute_medication_journey(patient_data)
    shifts_by_vid = {enc["visit_id"]: enc["changes"] for enc in med_journey_data}

    for idx, v in enumerate(visits, start=1):
        v_id = str(v.get("visit_id", f"V{idx}"))
        render_clinical_encounter(idx, v, shifts_by_vid.get(v_id, []))

st.write("")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: MEDICATION JOURNEY (Changes over time)
# ═════════════════════════════════════════════════════════════════════════════
with tab_meds:
    st.markdown('<div class="section-title">Medication Regimen & Longitudinal Changes</div>', unsafe_allow_html=True)
    st.caption("Chronological tracking of medication changes: started, continued, dose/frequency adjustments, and document mismatches.")

    med_journey_data = compute_medication_journey(patient_data)
    if not med_journey_data or not any(enc.get("changes") for enc in med_journey_data):
        st.info("No medications recorded in either prescription database or extracted clinical documents.")
    else:
        for enc in med_journey_data:
            st.markdown(f"##### 🗓️ {enc['visit_date']} — Encounter #{enc['visit_index']} (Visit {enc['visit_id']})")
            changes = enc.get("changes", [])
            if not changes:
                st.markdown("<p style='color:#64748B; font-style:italic; margin-left:1rem;'>No medication changes documented for this visit.</p>", unsafe_allow_html=True)
            else:
                for ch in changes:
                    ch_type = ch.get("type", "")
                    ch_text = ch.get("text", "")
                    if ch_type == "started":
                        badge = '<span class="diff-badge-added">Started</span>'
                    elif ch_type == "continued":
                        badge = '<span class="diff-badge-stable">Continued</span>'
                    elif ch_type == "dose_changed":
                        badge = '<span class="diff-badge-changed">Dose / Frequency Changed</span>'
                    elif ch_type == "discontinued":
                        badge = '<span class="diff-badge-removed">Discontinued</span>'
                    elif ch_type == "mismatch":
                        badge = '<span style="background:#FEF3C7; color:#92400E; border:1px solid #FCD34D; padding:2px 7px; border-radius:4px; font-size:0.76rem; font-weight:600;">Mismatch</span>'
                    else:
                        badge = '<span class="diff-badge-stable">Documented</span>'

                    st.markdown(f"<div style='margin-left: 1rem; margin-bottom: 0.35rem;'>{badge} <span style='font-size:0.92rem; font-weight:500; margin-left:6px;'>{ch_text}</span></div>", unsafe_allow_html=True)
            st.write("")

        possible_side_effects = patient_data.get("possible_side_effects", {})
        if possible_side_effects:
            with st.expander("ℹ️ Medication Side Effect Reference", expanded=False):
                st.caption("Common side effects from the local reference list or an AI-assisted lookup for other medicine names. OCR and brand names can be ambiguous; verify medicine identity and information with a pharmacist or clinician. Educational reference only.")
                se_rows = []
                for med_k, se_v in possible_side_effects.items():
                    value = str(se_v)
                    if "AI-assisted general reference" in value:
                        source = "AI-assisted; verify"
                    elif value.startswith("Unable to identify"):
                        source = "Not identified"
                    elif value.startswith("This medicine isn't in our local reference list"):
                        source = "Lookup unavailable"
                    else:
                        source = "Local reference"
                    se_rows.append({"Medication": med_k, "Monitored Considerations": value, "Source": source})
                side_effects_df = pd.DataFrame(se_rows)
                st.dataframe(
                    side_effects_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(600, 40 + 38 * len(side_effects_df)),
                    column_config={
                        "Medication": st.column_config.TextColumn(width="medium"),
                        "Monitored Considerations": st.column_config.TextColumn(width="large"),
                        "Source": st.column_config.TextColumn(width="small"),
                    },
                )
                st.download_button(
                    "Download as CSV",
                    side_effects_df.to_csv(index=False).encode("utf-8"),
                    file_name="medication_reference.csv",
                    mime="text/csv",
                    key="medication_reference_csv",
                )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: LAB RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_labs:
    st.markdown('<div class="section-title">Laboratory Investigations & Evaluations</div>', unsafe_allow_html=True)
    st.caption("Quantitative laboratory test reports extracted across clinical encounters.")

    all_labs = []
    for idx, v in enumerate(visits, start=1):
        v_date = (v.get("visit_date") or "Unspecified")[:10]
        for lab in v.get("lab_results", []):
            all_labs.append({
                "Date": v_date,
                "Encounter": f"Visit #{idx}",
                "Test Name": safe_val(lab.get("test_name")),
                "Result": safe_val(lab.get("value")),
                "Reference Range": safe_val(lab.get("reference")),
                "Status": safe_val(lab.get("status"), "Evaluated")
            })

    if not all_labs:
        st.info("No laboratory investigations documented for this patient.")
    else:
        df_labs = pd.DataFrame(all_labs)

        # Visual indicator styling for normal vs abnormal
        def highlight_status(val):
            s = str(val).lower()
            if "abnormal" in s or "high" in s or "low" in s:
                return "color: #DC2626; font-weight: 600;"
            elif "normal" in s:
                return "color: #16A34A; font-weight: 500;"
            return "color: #475569;"

        styled_labs = df_labs.style.map(highlight_status, subset=["Status"])
        st.dataframe(styled_labs, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: AI RETROSPECTIVE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown('<div class="section-title">AI Clinical Review</div>', unsafe_allow_html=True)

    # Status Bar & Controls
    ai_c1, ai_c2 = st.columns([3, 1])
    cache_current = is_review_current(patient_data)
    with ai_c2:
        force_refresh = st.button("Force Refresh", type="secondary", use_container_width=True) if cache_current else False

    try:
        if not cache_current or force_refresh:
            with st.spinner("Updating retrospective longitudinal review..."):
                review = get_or_generate_review(patient_data, force_refresh=force_refresh)
        else:
            review = get_or_generate_review(patient_data)
        raw_ai_text = review.get("raw_markdown", "")
        st.session_state["ai_report"] = raw_ai_text
        st.session_state["ai_metadata"] = {
            "model_name": review.get("model_name", "AI Engine"),
            "generated_at": datetime.utcnow().isoformat(),
            "source_visits": patient_data.get("total_visits", 0),
        }
        visit_label = f"{patient_data.get('total_visits', 0)} visits"
        if review.get("cache_hit"):
            st.caption(f"Cached review • {visit_label}")
        elif review.get("status", "").startswith("Incremental"):
            st.caption(f"Incremental review • {visit_label}")
        else:
            st.caption(f"Fresh full review • {visit_label}")
        if raw_ai_text:
            render_doctor_friendly_ai_review(raw_ai_text)
    except Exception as e:
        st.error(f"Failed to load or generate AI retrospective review: {e}")

# TAB 5: PATIENT CHAT
with tab_chat:
    st.markdown('<div class="section-title">Patient Chat</div>', unsafe_allow_html=True)
    st.caption("Ask questions about this patient's clinical record.")
    active_id = str(patient_data.get("patient_id", ""))
    if st.session_state.get("patient_chat_patient_id") != active_id:
        st.session_state["patient_chat_patient_id"] = active_id
        st.session_state["patient_chat_messages"] = []
    chat_messages = st.session_state.setdefault("patient_chat_messages", [])
    for message in chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.caption("LLM-generated response")
            st.markdown(message["content"])
            if message["role"] == "assistant" and "retrieved_evidence" in message:
                with st.expander(f"RAG retrieved {len(message['retrieved_evidence'])} item(s)", expanded=False):
                    if message["retrieved_evidence"]:
                        for item in message["retrieved_evidence"]:
                            st.markdown(
                                f"**{item.get('date', 'Date not recorded')} · {item.get('category', 'Evidence')}**  "
                                f"\nVisit: {item.get('visit_id', 'Unknown')} · Source: {item.get('source_id', 'Unknown')}"
                            )
                            st.write(item.get("content", ""))
                    else:
                        st.caption("No relevant evidence was retrieved for this question.")
    question = st.chat_input("Ask a question...")
    if question and question.strip():
        question = question.strip()
        prior_history = list(chat_messages)
        chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving patient record evidence..."):
                try:
                    answer, retrieved = answer_patient_question(
                        patient_data, active_id, question, history=prior_history,
                        api_key=CHAT_API_KEY,
                    )
                    st.caption("LLM-generated response")
                    st.markdown(answer)
                    with st.expander(f"RAG retrieved {len(retrieved)} item(s)", expanded=True):
                        if retrieved:
                            for item in retrieved:
                                st.markdown(
                                    f"**{item.get('date', 'Date not recorded')} · {item.get('category', 'Evidence')}**  "
                                    f"\nVisit: {item.get('visit_id', 'Unknown')} · Source: {item.get('source_id', 'Unknown')}"
                                )
                                st.write(item.get("content", ""))
                        else:
                            st.caption("No relevant evidence was retrieved for this question.")
                    chat_messages.append({
                        "role": "assistant", "content": answer,
                        "retrieved_evidence": retrieved,
                    })
                except Exception as exc:
                    st.error(f"Patient record Q&A failed: {exc}")

# TAB 6: DATA QUALITY / DISCREPANCIES
with tab_discrepancies:
    st.markdown('<div class="section-title">Record Quality & Potential Discrepancies</div>', unsafe_allow_html=True)
    st.caption("Comparative cross-checks between structured database records and OCR-extracted source documentation.")

    all_discrepancies = []
    for idx, v in enumerate(visits, start=1):
        v_date = (v.get("visit_date") or f"Visit {idx}")[:10]
        for d in v.get("discrepancies", []):
            all_discrepancies.append({"Visit Date": v_date, "Encounter": f"Visit #{idx}", "Discrepancy Detail": d})

    if not all_discrepancies:
        st.success("No discrepancies detected between structured records and extracted documentation.")
    else:
        st.markdown(f"""
        <div class="discrepancy-card">
            <div class="discrepancy-title">⚠️ Potential Record Discrepancies Identified ({len(all_discrepancies)})</div>
            The automated pipeline detected conflicting entries between the structured database record and extracted clinical documents.
            These items are flagged for clinical audit and verification.
        </div>
        """, unsafe_allow_html=True)

        for item in all_discrepancies:
            st.markdown(f"- **{item['Visit Date']} ({item['Encounter']}):** {item['Discrepancy Detail']}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7: SOURCE TRACEABILITY
# ═════════════════════════════════════════════════════════════════════════════
with tab_sources:
    st.markdown('<div class="section-title">Source Documentation & Extraction Traceability</div>', unsafe_allow_html=True)
    st.caption("Audit trail linking structured clinical fields back to underlying documents and OCR extraction statistics.")

    source_records = []
    for idx, v in enumerate(visits, start=1):
        v_date = (v.get("visit_date") or f"Visit {idx}")[:10]
        docs = v.get("documents", [])
        if not docs:
            source_records.append({
                "Visit Date": v_date,
                "Encounter": f"Visit #{idx}",
                "Document": "None (Database Entry)",
                "Document ID": "N/A",
                "Lines Processed": "N/A",
                "Readable Lines": "N/A",
                "Extracted Fields": "N/A",
                "Quality Status": "Pure DB Record"
            })
        else:
            for d in docs:
                cs = d.get("clinical_summary", {}) or {}
                stats = cs.get("extraction_stats", {})
                tot_l = stats.get("total_lines", 0)
                garb = stats.get("garbled_lines", 0)
                ext = stats.get("extracted_fields", 0)
                read = tot_l - garb if tot_l else 0

                status_str = "Good"
                if garb > tot_l * 0.4:
                    status_str = "Partially Readable"
                elif ext == 0:
                    status_str = "Low Yield"

                source_records.append({
                    "Visit Date": v_date,
                    "Encounter": f"Visit #{idx}",
                    "Document": d.get("document_label", "Document"),
                    "Document ID": str(d.get("doc_id", "N/A")),
                    "Lines Processed": tot_l,
                    "Readable Lines": read,
                    "Extracted Fields": ext,
                    "Quality Status": status_str
                })

    df_sources = pd.DataFrame(source_records)
    st.dataframe(df_sources, use_container_width=True, hide_index=True)
