import os
import sys
import re
from datetime import datetime

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
from storage.ai_analysis_csv import append_ai_analysis, get_patient_row

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
    border: 1px solid #E2E8F0 !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    margin-bottom: 0.5rem;
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


def parse_ai_markdown_sections(raw_md: str) -> dict:
    """
    Parses retrospective AI markdown note into clean categorized sections
    so clinical users can scan each component rather than facing a wall of text.
    """
    if not raw_md or not raw_md.strip():
        return {}

    lines = raw_md.splitlines()
    sections = {}
    current_key = "Narrative"
    sections[current_key] = []

    keyword_map = [
        ("Patient Past History Summary", "Past History Summary"),
        ("Identified Medications", "Medications & Indication Match"),
        ("Date-Wise Vitals", "Vital Trends & Analysis"),
        ("Laboratory Test Evaluation", "Laboratory Evaluation"),
        ("Clinical Condition Trajectory", "Clinical Trajectory & Outlook"),
        ("General Lifestyle", "Lifestyle & Dietary Guidance"),
        ("Notable discrepancies", "Potential Record Discrepancies"),
    ]

    for line in lines:
        line_clean = line.strip()
        matched_section = None
        for trigger, title in keyword_map:
            if trigger.lower() in line_clean.lower() and len(line_clean) < 110 and not line_clean.startswith("|"):
                matched_section = title
                break

        if matched_section:
            current_key = matched_section
            if current_key not in sections:
                sections[current_key] = []
        else:
            sections[current_key].append(line)

    clean_sections = {}
    for k, v in sections.items():
        joined = "\n".join(v).strip()
        if joined:
            clean_sections[k] = joined

    return clean_sections


# ── Header ───────────────────────────────────────────────────────────────────
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
    if (st.session_state.get("active_patient_id") != patient_id_input
            or "patient_data" not in st.session_state):
        with st.spinner("Retrieving longitudinal patient records..."):
            try:
                pd_data = fetch_patient_history(patient_id_input)
                st.session_state["patient_data"] = pd_data
                st.session_state["active_patient_id"] = patient_id_input
                st.session_state.pop("ai_report", None)
                st.session_state.pop("ai_metadata", None)

                # Check if a persistent retrospective analysis is already cached in CSV
                cached_row = get_patient_row(patient_id_input)
                if cached_row and cached_row.get("analysis"):
                    try:
                        parsed_cached = json.loads(cached_row["analysis"])
                        st.session_state["ai_report"] = parsed_cached.get("raw_markdown", "")
                        st.session_state["ai_metadata"] = {
                            "model_name": cached_row.get("model_name", "AI Engine"),
                            "generated_at": cached_row.get("generated_at", ""),
                            "source_visits": cached_row.get("source_visit_count", "")
                        }
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
has_cached_analysis = "ai_report" in st.session_state and bool(st.session_state["ai_report"])

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
tab_timeline, tab_trends, tab_meds, tab_labs, tab_ai, tab_discrepancies, tab_sources = st.tabs([
    "📅 Clinical Timeline",
    "📈 Clinical Trends",
    "💊 Medication Journey",
    "🧪 Lab Results",
    "🤖 Retrospective AI Analysis",
    "⚠️ Record Discrepancies",
    "📄 Source Traceability"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: CLINICAL TIMELINE & JOURNEY
# ═════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.markdown('<div class="section-title">Patient Journey & Clinical Event Timeline</div>', unsafe_allow_html=True)
    st.caption("Structured chronological tree of clinical complaints, diagnoses, medications, and vitals with direct source traceability.")

    # ── 1. PATIENT JOURNEY EVENT TREE ─────────────────────────────────────────
    st.markdown("##### 📍 Clinical Event Tree")

    for idx, v in enumerate(visits, start=1):
        v_date = safe_val(v.get("visit_date"), "Date Unspecified")[:10]
        v_type = "Initial Encounter" if idx == 1 else f"Follow-Up #{idx - 1}"
        db_cc = safe_val(v.get("db_chief_complaint"), "")
        db_dx = safe_val(v.get("db_confirmed_dx") or v.get("db_provisional_dx"), "")
        docs = v.get("documents", [])
        first_doc = docs[0] if docs else {}
        doc_cs = first_doc.get("clinical_summary", {}) if first_doc else {}

        # Resolve primary headline items
        complaint = db_cc or safe_val(doc_cs.get("chief_complaint"), "")
        diagnosis = db_dx or safe_val(doc_cs.get("clinical_impression") or doc_cs.get("provisional_diagnosis"), "")
        med_names = [m.get("name") for m in v.get("db_medications", []) if m.get("name")]
        if not med_names:
            med_names = [m.get("name") for m in doc_cs.get("medications", []) if m.get("name")]
        vitals_list = [f"{vr['label']} {vr['value']}" for vr in v.get("db_vitals_rows", [])[:2]]
        if not vitals_list:
            vitals_list = [f"{vr.get('label')} {vr.get('value')}" for vr in doc_cs.get("vitals", [])[:2]]
        labs_list = [f"{l.get('test_name')} {l.get('value')}" for l in v.get("lab_results", [])[:2]]

        # Clean event pills for tree display
        event_items = []
        if complaint: event_items.append(("Complaint", complaint, "primary"))
        if diagnosis: event_items.append(("Diagnosis", diagnosis, "dx"))
        for m in med_names[:3]: event_items.append(("Rx", m, "med"))
        for vt in vitals_list: event_items.append(("Vital", vt, "vital"))
        for lb in labs_list: event_items.append(("Lab", lb, "lab"))

        with st.container():
            st.markdown(f"""
            <div class="timeline-card">
                <div class="timeline-date">{v_date} &nbsp;·&nbsp; <span style="font-size:0.8rem; font-weight:500; color:#64748B;">{v_type}</span></div>
            </div>
            """, unsafe_allow_html=True)

            tree_cols = st.columns([1] * min(len(event_items) if event_items else 1, 5))
            for c_idx, (cat, val, style_t) in enumerate(event_items[:5]):
                with tree_cols[c_idx]:
                    btn_label = f"{'├─' if c_idx < len(event_items[:5])-1 else '└─'} {val}"
                    if st.button(btn_label, key=f"tree_btn_{idx}_{c_idx}", help=f"Click to trace {cat}: {val} to source record"):
                        st.session_state["selected_trace"] = {
                            "visit_num": idx,
                            "date": v_date,
                            "category": cat,
                            "item": val,
                            "has_doc": bool(docs),
                            "doc_label": first_doc.get("document_label", "Document 1") if docs else "None (Database Entry)",
                            "doc_id": first_doc.get("doc_id", "N/A") if docs else "N/A"
                        }

    # Traceability modal / banner when doctor clicks an event
    if "selected_trace" in st.session_state:
        tr = st.session_state["selected_trace"]
        st.markdown(f"""
        <div class="clinical-notice" style="background:#F0FDF4; border-left:4px solid #16A34A; margin-top:0.6rem;">
            <strong>🔍 Traceability Audit:</strong> <code>{tr['category']}</code> <strong>{tr['item']}</strong> (Visit #{tr['visit_num']} on {tr['date']})<br>
            <strong>Source:</strong> {tr['doc_label']} (Document ID: <code>{tr['doc_id']}</code>) &nbsp;|&nbsp; 
            <strong>Record Type:</strong> {'OCR Extracted Clinical Document' if tr['has_doc'] else 'Direct Structured Database Entry'}
        </div>
        """, unsafe_allow_html=True)
        if st.button("Close Trace Inspector", key="btn_close_trace"):
            st.session_state.pop("selected_trace", None)
            st.rerun()

    st.markdown("---")

    # ── 2. "WHAT CHANGED?" CLINICAL DIFF ENGINE ───────────────────────────────
    st.markdown("##### ⚡ 'What Changed?' Clinical Diff Engine")
    st.caption("Automatic state comparison between consecutive encounters: tracks diagnosis shifts, medication escalations/de-escalations, and lab changes.")

    if len(visits) < 2:
        st.info("Single encounter recorded. Clinical diff engine activates when 2 or more encounters are documented.")
    else:
        diff_encounters = []
        for i in range(1, len(visits)):
            prev_v = visits[i - 1]
            curr_v = visits[i]
            prev_date = safe_val(prev_v.get("visit_date"), f"V{i}")[:10]
            curr_date = safe_val(curr_v.get("visit_date"), f"V{i+1}")[:10]

            # 1. Diagnosis diff
            prev_dx = set(filter(None, [
                prev_v.get("db_confirmed_dx"), prev_v.get("db_provisional_dx"),
                (prev_v.get("documents", [{}])[0].get("clinical_summary", {}).get("clinical_impression") if prev_v.get("documents") else None)
            ]))
            curr_dx = set(filter(None, [
                curr_v.get("db_confirmed_dx"), curr_v.get("db_provisional_dx"),
                (curr_v.get("documents", [{}])[0].get("clinical_summary", {}).get("clinical_impression") if curr_v.get("documents") else None)
            ]))
            new_dx = curr_dx - prev_dx
            res_dx = prev_dx - curr_dx
            stable_dx = curr_dx & prev_dx

            # 2. Medication diff
            prev_m = set(m.lower().strip() for m in prev_v.get("medications_found", []))
            curr_m = set(m.lower().strip() for m in curr_v.get("medications_found", []))
            added_meds = [m.title() for m in (curr_m - prev_m)]
            removed_meds = [m.title() for m in (prev_m - curr_m)]
            continued_meds = [m.title() for m in (curr_m & prev_m)]

            # 3. Chief Complaint diff
            p_cc = safe_val(prev_v.get("db_chief_complaint"), "")
            c_cc = safe_val(curr_v.get("db_chief_complaint"), "")
            cc_status = "Unchanged" if (p_cc.lower() == c_cc.lower() and p_cc) else (f"{p_cc} → {c_cc}" if p_cc and c_cc else "New complaint documented")

            diff_encounters.append({
                "transition": f"Visit #{i} ({prev_date}) → Visit #{i+1} ({curr_date})",
                "new_dx": list(new_dx),
                "res_dx": list(res_dx),
                "stable_dx": list(stable_dx),
                "added_meds": added_meds,
                "removed_meds": removed_meds,
                "continued_meds": continued_meds,
                "cc_status": cc_status
            })

        for diff in diff_encounters:
            with st.expander(f"🔄 State Diff: {diff['transition']}", expanded=(diff == diff_encounters[-1])):
                d_c1, d_c2, d_c3 = st.columns(3)
                with d_c1:
                    st.markdown("<span class='section-subhead'>Diagnoses</span>", unsafe_allow_html=True)
                    if diff["new_dx"]:
                        st.markdown(f"<span class='diff-badge-added'>New</span> {', '.join(diff['new_dx'])}", unsafe_allow_html=True)
                    if diff["res_dx"]:
                        st.markdown(f"<span class='diff-badge-removed'>Resolved/Shifted</span> {', '.join(diff['res_dx'])}", unsafe_allow_html=True)
                    if diff["stable_dx"]:
                        st.markdown(f"<span class='diff-badge-stable'>Persistent</span> {', '.join(diff['stable_dx'])}", unsafe_allow_html=True)
                    if not (diff["new_dx"] or diff["res_dx"] or diff["stable_dx"]):
                        st.caption("No specific diagnosis change documented")

                with d_c2:
                    st.markdown("<span class='section-subhead'>Medications</span>", unsafe_allow_html=True)
                    if diff["added_meds"]:
                        st.markdown(f"<span class='diff-badge-added'>Added (+{len(diff['added_meds'])})</span> {', '.join(diff['added_meds'][:4])}", unsafe_allow_html=True)
                    if diff["removed_meds"]:
                        st.markdown(f"<span class='diff-badge-removed'>Stopped (-{len(diff['removed_meds'])})</span> {', '.join(diff['removed_meds'][:4])}", unsafe_allow_html=True)
                    if diff["continued_meds"]:
                        st.markdown(f"<span class='diff-badge-stable'>Maintained</span> {', '.join(diff['continued_meds'][:4])}", unsafe_allow_html=True)
                    if not (diff["added_meds"] or diff["removed_meds"] or diff["continued_meds"]):
                        st.caption("No medication modifications documented")

                with d_c3:
                    st.markdown("<span class='section-subhead'>Complaint Evolution</span>", unsafe_allow_html=True)
                    st.write(diff["cc_status"])

    st.markdown("---")
    st.markdown("##### 📋 Detailed Encounter Verification Records")
    st.caption("Dual-source validation per visit (Structured Database Record vs. Extracted Clinical Document).")

    for idx, visit in enumerate(visits, start=1):
        v_date = safe_val(visit.get("visit_date"), "Date not documented")
        v_date_str = v_date[:10] if len(v_date) >= 10 else v_date

        v_type = "Initial Visit" if idx == 1 else f"Follow-Up #{idx - 1}"
        db_prov = safe_val(visit.get("db_provisional_dx"), "")
        db_conf = safe_val(visit.get("db_confirmed_dx"), "")
        db_cc = safe_val(visit.get("db_chief_complaint"), "")

        docs = visit.get("documents", [])
        first_doc_imp = ""
        if docs:
            first_doc_imp = safe_val(docs[0].get("clinical_summary", {}).get("clinical_impression"), "")

        dx_display = db_conf or db_prov or first_doc_imp or "Diagnosis not specified"
        expander_title = f"{v_date_str} — {v_type} | {dx_display} (Visit #{idx})"

        is_latest = (idx == total_visits)
        with st.expander(expander_title, expanded=False):
            col_l, col_r = st.columns([1, 1], gap="medium")

            # Left Column: Structured Database Record
            with col_l:
                st.markdown("##### 🏥 Structured Database Record")
                render_field("Chief Complaint", db_cc)
                render_field("Provisional Diagnosis", db_prov)
                render_field("Confirmed Diagnosis", db_conf)

                db_vitals = visit.get("db_vitals_rows", [])
                if db_vitals:
                    st.markdown("<span class='section-subhead'>Vitals (Recorded)</span>", unsafe_allow_html=True)
                    v_df = pd.DataFrame([{"Vital": v["label"], "Value": v["value"], "Date": v["date"]} for v in db_vitals])
                    st.dataframe(v_df, use_container_width=True, hide_index=True)
                else:
                    render_field("Recorded Vitals", None, "No structured vitals entered")

                rx_items = visit.get("db_medications", [])
                if rx_items:
                    st.markdown("<span class='section-subhead'>Prescriptions (Database)</span>", unsafe_allow_html=True)
                    rx_df = pd.DataFrame([{
                        "Medicine": safe_val(m.get("name")),
                        "Dosage": safe_val(m.get("dosage")),
                        "Frequency": safe_val(m.get("frequency")),
                        "Duration": safe_val(m.get("duration"))
                    } for m in rx_items])
                    st.dataframe(rx_df, use_container_width=True, hide_index=True)
                else:
                    render_field("Prescriptions", None, "No prescription entries recorded")

                # Follow-up Schedule from followup_schedule table
                fu_items = visit.get("followups", [])
                if fu_items:
                    st.markdown("<span class='section-subhead'>Follow-up Schedule (Database)</span>", unsafe_allow_html=True)
                    fu_rows = []
                    for fu in fu_items:
                        s_date = safe_val(fu.get("scheduled_date"))
                        if len(s_date) >= 10: s_date = s_date[:10]
                        fu_status = safe_val(fu.get("status"), "SCHEDULED")
                        emerg = " 🚨 Emergency" if fu.get("is_emergency") else ""
                        fu_rows.append({
                            "Scheduled Date": s_date,
                            "Status": f"{fu_status}{emerg}",
                            "Resolution": safe_val(fu.get("resolution"), "—"),
                            "Notes": safe_val(fu.get("resolution_notes"), "—")
                        })
                    st.dataframe(pd.DataFrame(fu_rows), use_container_width=True, hide_index=True)

            # Right Column: Extracted Document Details
            with col_r:
                st.markdown("##### 📄 Document Extraction")
                if not visit.get("has_document") or not docs:
                    st.info("No supporting clinical document uploaded for this visit. Encounters evaluated purely on structured database records.")
                else:
                    for doc_idx, doc in enumerate(docs, start=1):
                        d_label = doc.get("document_label", f"Document {doc_idx}")
                        cs = doc.get("clinical_summary", {}) or {}

                        if len(docs) > 1:
                            st.markdown(f"**{d_label}**")

                        render_field("Clinical Impression", cs.get("clinical_impression"))
                        render_field("Chief Complaint (Doc)", cs.get("chief_complaint"))
                        render_field("Extracted Demographics", cs.get("age_sex"))

                        doc_vitals = cs.get("vitals", [])
                        if doc_vitals:
                            st.markdown("<span class='section-subhead'>Vitals (Document)</span>", unsafe_allow_html=True)
                            dv_df = pd.DataFrame([{"Vital": v.get("label"), "Value": v.get("value")} for v in doc_vitals])
                            st.dataframe(dv_df, use_container_width=True, hide_index=True)

                        doc_meds = cs.get("medications", [])
                        if doc_meds:
                            st.markdown("<span class='section-subhead'>Medicines (Document)</span>", unsafe_allow_html=True)
                            dm_df = pd.DataFrame([{
                                "Medicine": safe_val(m.get("name")),
                                "Strength": safe_val(m.get("strength")),
                                "Frequency": safe_val(m.get("frequency")),
                                "Duration": safe_val(m.get("duration"))
                            } for m in doc_meds])
                            st.dataframe(dm_df, use_container_width=True, hide_index=True)

                        doc_labs = doc.get("lab_results", [])
                        if doc_labs:
                            st.markdown("<span class='section-subhead'>Labs Evaluated</span>", unsafe_allow_html=True)
                            dl_df = pd.DataFrame([{
                                "Test": l.get("test_name"),
                                "Result": l.get("value"),
                                "Reference": l.get("reference"),
                                "Status": l.get("status")
                            } for l in doc_labs])
                            st.dataframe(dl_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: CLINICAL TRENDS
# ═════════════════════════════════════════════════════════════════════════════
with tab_trends:
    st.markdown('<div class="section-title">Longitudinal Vitals & Clinical Trends</div>', unsafe_allow_html=True)
    st.caption("Structured vital values plotted chronologically across documented encounters.")

    vital_series = []
    for idx, v in enumerate(visits, start=1):
        v_date = (v.get("visit_date") or f"Visit {idx}")[:10]
        # Combine database vitals and document vitals with preference for structured DB
        v_dict = {"Date": v_date, "Visit": f"V{idx}"}

        # Scan DB vitals rows
        for vr in v.get("db_vitals_rows", []):
            label = vr.get("label", "").lower()
            val_raw = vr.get("value", "")
            nums = re.findall(r"[\d.]+", str(val_raw))
            if "bp" in label:
                bp_match = re.search(r"(\d+)\s*/\s*(\d+)", str(val_raw))
                if bp_match:
                    v_dict["Systolic BP"] = float(bp_match.group(1))
                    v_dict["Diastolic BP"] = float(bp_match.group(2))
            elif "pulse" in label or "heart" in label:
                if nums: v_dict["Pulse (bpm)"] = float(nums[0])
            elif "spo" in label:
                if nums: v_dict["SpO2 (%)"] = float(nums[0])
            elif "weight" in label:
                if nums: v_dict["Weight (kg)"] = float(nums[0])
            elif "temp" in label:
                if nums: v_dict["Temperature"] = float(nums[0])

        # If any missing, supplement from document vitals
        for doc in v.get("documents", []):
            for dv in doc.get("clinical_summary", {}).get("vitals", []):
                lbl = dv.get("label", "").lower()
                val_raw = dv.get("value", "")
                nums = re.findall(r"[\d.]+", str(val_raw))
                if "bp" in lbl and "Systolic BP" not in v_dict:
                    bp_match = re.search(r"(\d+)\s*/\s*(\d+)", str(val_raw))
                    if bp_match:
                        v_dict["Systolic BP"] = float(bp_match.group(1))
                        v_dict["Diastolic BP"] = float(bp_match.group(2))
                elif ("pulse" in lbl or "heart" in lbl) and "Pulse (bpm)" not in v_dict:
                    if nums: v_dict["Pulse (bpm)"] = float(nums[0])
                elif "spo" in lbl and "SpO2 (%)" not in v_dict:
                    if nums: v_dict["SpO2 (%)"] = float(nums[0])
                elif "weight" in lbl and "Weight (kg)" not in v_dict:
                    if nums: v_dict["Weight (kg)"] = float(nums[0])
                elif "temp" in lbl and "Temperature" not in v_dict:
                    if nums: v_dict["Temperature"] = float(nums[0])

        vital_series.append(v_dict)

    df_vitals = pd.DataFrame(vital_series)

    # Check which numeric vital columns have at least one valid reading
    has_bp = "Systolic BP" in df_vitals.columns and df_vitals["Systolic BP"].notna().any()
    has_pulse = "Pulse (bpm)" in df_vitals.columns and df_vitals["Pulse (bpm)"].notna().any()
    has_spo2 = "SpO2 (%)" in df_vitals.columns and df_vitals["SpO2 (%)"].notna().any()
    has_wt = "Weight (kg)" in df_vitals.columns and df_vitals["Weight (kg)"].notna().any()
    has_temp = "Temperature" in df_vitals.columns and df_vitals["Temperature"].notna().any()

    if not (has_bp or has_pulse or has_spo2 or has_wt or has_temp):
        st.info("No numerical vital trends documented across visits for this patient.")
    else:
        # Render clean trend visualisations
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            if has_bp:
                st.markdown("##### Blood Pressure Trend (mmHg)")
                bp_df = df_vitals[["Date", "Systolic BP", "Diastolic BP"]].dropna(subset=["Systolic BP"])
                st.line_chart(bp_df.set_index("Date"), use_container_width=True)
            elif has_pulse:
                st.markdown("##### Pulse Rate (bpm)")
                p_df = df_vitals[["Date", "Pulse (bpm)"]].dropna(subset=["Pulse (bpm)"])
                st.line_chart(p_df.set_index("Date"), use_container_width=True)

        with t_col2:
            if has_wt:
                st.markdown("##### Body Weight Trend (kg)")
                wt_df = df_vitals[["Date", "Weight (kg)"]].dropna(subset=["Weight (kg)"])
                st.line_chart(wt_df.set_index("Date"), use_container_width=True)
            elif has_spo2:
                st.markdown("##### Oxygen Saturation - SpO2 (%)")
                spo_df = df_vitals[["Date", "SpO2 (%)"]].dropna(subset=["SpO2 (%)"])
                st.line_chart(spo_df.set_index("Date"), use_container_width=True)

        # Tabular Summary of Vitals
        st.markdown("##### Recorded Vitals Summary Across Visits")
        display_v_cols = [c for c in ["Date", "Visit", "Systolic BP", "Diastolic BP", "Pulse (bpm)", "SpO2 (%)", "Weight (kg)", "Temperature"] if c in df_vitals.columns]
        summary_df = df_vitals[display_v_cols].copy().fillna("—")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: MEDICATION JOURNEY
# ═════════════════════════════════════════════════════════════════════════════
with tab_meds:
    st.markdown('<div class="section-title">Medication Regimen & Longitudinal Journey</div>', unsafe_allow_html=True)
    st.caption("Chronological record of prescription and extracted medications by visit date.")

    med_journey = []
    for idx, v in enumerate(visits, start=1):
        v_date = (v.get("visit_date") or "Unspecified")[:10]
        # Database Prescriptions
        for rx in v.get("db_medications", []):
            med_journey.append({
                "Visit Date": v_date,
                "Visit #": f"V{idx}",
                "Medication Name": safe_val(rx.get("name")),
                "Dosage / Strength": safe_val(rx.get("dosage")),
                "Frequency": safe_val(rx.get("frequency")),
                "Duration": safe_val(rx.get("duration")),
                "Source": "Prescription (Database)"
            })
        # Document Extracted Medications
        for doc in v.get("documents", []):
            for m in doc.get("clinical_summary", {}).get("medications", []):
                med_journey.append({
                    "Visit Date": v_date,
                    "Visit #": f"V{idx}",
                    "Medication Name": safe_val(m.get("name")),
                    "Dosage / Strength": safe_val(m.get("strength")),
                    "Frequency": safe_val(m.get("frequency")),
                    "Duration": safe_val(m.get("duration")),
                    "Source": "Extracted Document"
                })

    if not med_journey:
        st.info("No medications recorded in either prescription database or extracted clinical documents.")
    else:
        df_journey = pd.DataFrame(med_journey)
        # Deduplicate identical rows
        df_journey = df_journey.drop_duplicates(subset=["Visit Date", "Medication Name", "Dosage / Strength", "Source"])
        st.dataframe(df_journey, use_container_width=True, hide_index=True)

        # Medication Reference and Side Effects Panel
        possible_side_effects = patient_data.get("possible_side_effects", {})
        if possible_side_effects:
            with st.expander("ℹ️ Clinical Reference: Documented Medication Effects & Monitored Indications", expanded=False):
                st.caption("Standard reference database indications and side-effect profile for identified medications (educational reference only).")
                se_rows = []
                for med_k, se_v in possible_side_effects.items():
                    se_rows.append({"Medication": med_k, "Monitored Considerations": se_v})
                st.dataframe(pd.DataFrame(se_rows), use_container_width=True, hide_index=True)


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
    st.markdown('<div class="section-title">AI Retrospective Clinical Intelligence</div>', unsafe_allow_html=True)
    st.caption("Longitudinal syntheses of treatment evolution, diagnostic pattern observation, and clinical trajectory.")

    # Status Bar & Controls
    ai_c1, ai_c2 = st.columns([3, 1])
    with ai_c1:
        if has_cached_analysis:
            meta = st.session_state.get("ai_metadata", {})
            m_name = meta.get("model_name", "AI Engine")
            st.markdown(f"<span class='trace-badge'>Status: Analysis Available ({m_name})</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='trace-badge'>Status: No Analysis Generated Yet</span>", unsafe_allow_html=True)
    with ai_c2:
        btn_generate_ai = st.button("Generate / Refresh AI Note", type="secondary", use_container_width=True)

    if btn_generate_ai:
        with st.spinner("Generating retrospective longitudinal analysis with clinical reasoning..."):
            try:
                report = analyze_patient_with_ai(patient_data)
                raw_markdown = report.get("raw_markdown", "")
                st.session_state["ai_report"] = raw_markdown
                st.session_state["ai_metadata"] = {
                    "model_name": report.get("model_name", "AI Engine"),
                    "generated_at": datetime.utcnow().isoformat(),
                    "source_visits": patient_data.get("total_visits", 0)
                }

                # Persist to CSV storage
                try:
                    append_ai_analysis(
                        patient_data=patient_data,
                        analysis={"raw_markdown": raw_markdown},
                        model_name=report.get("model_name", "AI Engine"),
                        model_version=report.get("model_version", ""),
                    )
                except Exception as csv_err:
                    pass
                st.rerun()
            except Exception as e:
                st.error(f"Failed to complete AI retrospective analysis: {e}")

    raw_ai_text = st.session_state.get("ai_report", "")

    if not raw_ai_text:
        st.info("Click **'Generate / Refresh AI Note'** above to run AI retrospective analysis on this patient's longitudinal record.")
    else:
        # Parse into organized clinical sections
        ai_sections = parse_ai_markdown_sections(raw_ai_text)

        if not ai_sections:
            st.markdown(raw_ai_text)
        else:
            # 1. Past History & Clinical Narrative
            if "Past History Summary" in ai_sections:
                st.markdown("##### 📋 Past History & Clinical Summary")
                st.markdown(ai_sections["Past History Summary"])

            # 2. Notable Discrepancies Callout (if in AI text)
            if "Potential Record Discrepancies" in ai_sections:
                st.markdown("""
                <div class="discrepancy-card">
                    <div class="discrepancy-title">⚠️ Potential Record Discrepancies (AI Identified)</div>
                    Please review conflicting entries across medical records.
                </div>
                """, unsafe_allow_html=True)
                st.markdown(ai_sections["Potential Record Discrepancies"])

            # 3. Medications Analysis
            if "Medications & Indication Match" in ai_sections:
                with st.expander("💊 Medication Indication & Indicated Complaint Match", expanded=True):
                    st.markdown(ai_sections["Medications & Indication Match"])

            # 4. Vital Trends
            if "Vital Trends & Analysis" in ai_sections:
                with st.expander("📈 Vital Trends & Clinical Evaluation", expanded=False):
                    st.markdown(ai_sections["Vital Trends & Analysis"])

            # 5. Laboratory Evaluation
            if "Laboratory Evaluation" in ai_sections:
                with st.expander("🧪 Laboratory Evaluation (Normal vs Abnormal)", expanded=False):
                    st.markdown(ai_sections["Laboratory Evaluation"])

            # 6. Clinical Condition Trajectory
            if "Clinical Trajectory & Outlook" in ai_sections:
                st.markdown("##### 🔮 Clinical Condition Trajectory & Retrospective Outlook")
                st.markdown(f'<div class="ai-narrative-card">{ai_sections["Clinical Trajectory & Outlook"]}</div>', unsafe_allow_html=True)

            # 7. General Lifestyle Suggestions
            if "Lifestyle & Dietary Guidance" in ai_sections:
                with st.expander("🥗 General Lifestyle & Dietary Guidance (Non-Prescriptive)", expanded=False):
                    st.markdown(ai_sections["Lifestyle & Dietary Guidance"])

            # Render any unmapped remaining sections
            for s_name, s_content in ai_sections.items():
                if s_name not in ["Past History Summary", "Potential Record Discrepancies", "Medications & Indication Match", "Vital Trends & Analysis", "Laboratory Evaluation", "Clinical Trajectory & Outlook", "Lifestyle & Dietary Guidance", "Narrative"]:
                    with st.expander(f"📌 {s_name}", expanded=False):
                        st.markdown(s_content)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6: DATA QUALITY / DISCREPANCIES
# ═════════════════════════════════════════════════════════════════════════════
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