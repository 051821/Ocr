import os
import sys
import re
import html
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (PROJECT_ROOT, DATAANALYSIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import streamlit as st
from core.fetching import fetch_patient_history, fetch_all_patient_ids
from storage.cache_db import get_patient_row, get_top_questions, record_chat_question
from services.review_service import get_or_generate_review, is_review_current
from services.chat.answers import answer_patient_question
from services.chat.retrieval import build_patient_index
from services.medication_journey import compute_medication_journey
from services.review_facts import build_review_facts, compute_source_hash, visit_hashes
from services.graph_service import is_neo4j_configured, sync_patient_to_graph
import importlib
import core.config
import core.llm_client
importlib.reload(core.config)
importlib.reload(core.llm_client)
import ui_components
importlib.reload(ui_components)
from ui_components import esc, render_clinical_course, render_clinical_encounter, render_clinical_snapshot, render_doctor_friendly_ai_review, render_unlinked_documents, unlinked_link_text
import clinical_snapshot
importlib.reload(clinical_snapshot)
try:
    from config import OPENROUTER_API_KEY as CHAT_API_KEY
except Exception:
    CHAT_API_KEY = None

logger = logging.getLogger(__name__)


# Bump when the shape of fetched patient data changes so cached sessions refetch.
DATA_VERSION = 2


@st.cache_data(ttl=300)
def _fetch_patient_history_cached(patient_id, data_version=DATA_VERSION):
    return fetch_patient_history(patient_id)

st.set_page_config(
    page_title="Patient Intelligence | Clinical Longitudinal Record",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme: light clinical palette (ui/styles.css, .streamlit/config.toml) ──────
with open(os.path.join(DATAANALYSIS_DIR, "ui", "styles.css"), encoding="utf-8") as _css:
    st.markdown(f"<style>{_css.read()}</style>", unsafe_allow_html=True)


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
    label = html.escape(str(label))
    s = html.escape(safe_val(value, default))
    default = html.escape(str(default))
    if s == default:
        st.markdown(f"<div class='field-row'><span class='field-name'>{label}:</span> "
                    f"<span class='field-na'>{default}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='field-row'><span class='field-name'>{label}:</span> "
                    f"<span class='field-val'>{s}</span></div>", unsafe_allow_html=True)


def _chat_label(meta):
    if not meta or not isinstance(meta, dict):
        return "LLM-generated response"
    path = meta.get("path")
    if path == "llm":
        total_tokens = meta.get("total_tokens", 0)
        model = meta.get("model") or "unknown"
        return f"LLM-generated · {total_tokens} tokens ({model})"
    if path == "llm_cache":
        return "Cached LLM answer · 0 new tokens"
    if path == "deterministic":
        return "Deterministic from record · no LLM"
    if path == "fallback":
        return "Rule-based fallback · LLM unavailable"
    return "No LLM used"


st.markdown('<div class="app-brand">Patient Clinical Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Longitudinal Clinical Record & Retrospective AI Analysis</div>', unsafe_allow_html=True)

st.markdown("""
<div class="clinical-notice">
    <strong>Clinical Reference Notice:</strong> This platform synthesizes structured database entries, OCR-extracted clinical documents,
    and retrospective AI observations for historical review only. <strong>It does not provide confirmed medical advice, diagnosis, or treatment recommendations.</strong>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: branding, navigation, patient selector ─────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-logo">🩺</div><div>'
        '<div class="sb-title">Clinical Intelligence Platform</div>'
        '<div class="sb-sub">Longitudinal patient records</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
    nav_page = st.radio("Navigation", ["Dashboard", "Patient Selector", "Reports", "Settings"],
                        key="nav_page", label_visibility="collapsed")
    st.markdown('<div class="sb-section">Patient Selector</div>', unsafe_allow_html=True)
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
        st.caption(f"{len(patient_ids)} patients in database")
        # A new pick in the sidebar switches the dashboard to that patient.
        last_pick = st.session_state.get("_sidebar_pick")
        if last_pick is not None and selected_patient != last_pick and selected_patient != "-- Select Patient --":
            st.session_state["active_patient_id"] = selected_patient
            st.session_state.pop("patient_data", None)
        st.session_state["_sidebar_pick"] = selected_patient


def _open_patient(pid):
    st.session_state["active_patient_id"] = pid
    st.session_state.pop("patient_data", None)
    st.session_state["nav_page"] = "Dashboard"


if nav_page == "Settings":
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
    st.caption("Read-only. Values come from environment variables (.env); change them there and restart the app.")
    flag_names = ["ENABLE_HARDENED_CONFIG", "ENABLE_EXTERNAL_PROMPTS", "ENABLE_PROMPT_INJECTION_SANDBOX",
                  "ENABLE_ENCRYPTED_CACHE", "ENABLE_CHAT_OUTPUT_SANITIZER", "ENABLE_ANONYMIZED_AUDIT_LOGS",
                  "ENABLE_STRICT_HTML_SANITIZATION"]
    rows = "".join(
        f'<div class="field-row"><span class="field-name">{name}</span>'
        f'<span class="status-badge {"status-ok" if core.config.flag(name) else "status-info"}">'
        f'{"On" if core.config.flag(name) else "Off"}</span></div>' for name in flag_names)
    st.markdown(f'<div class="metric-card"><div class="metric-label">Feature flags</div>{rows}</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(
        '<div class="metric-card"><div class="metric-label">AI models &amp; services</div>'
        f'<div class="field-row"><span class="field-name">Chat model</span><span class="field-val">{html.escape(core.config.CHAT_MODEL)}</span></div>'
        f'<div class="field-row"><span class="field-name">Review models</span><span class="field-val">{html.escape(", ".join(core.config.DEFAULT_MODELS))}</span></div>'
        f'<div class="field-row"><span class="field-name">Graph (Neo4j)</span><span class="status-badge {"status-ok" if is_neo4j_configured() else "status-info"}">'
        f'{"Connected" if is_neo4j_configured() else "Not configured"}</span></div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

if nav_page == "Patient Selector":
    st.markdown('<div class="section-title">Patient Selector</div>', unsafe_allow_html=True)
    query = st.text_input("Search patients", placeholder="Search by patient ID...", label_visibility="collapsed").strip().lower()
    shown = [pid for pid in patient_ids if query in pid.lower()]
    st.caption(f"{len(shown)} of {len(patient_ids)} patients · select a row, then open it")
    picked = st.dataframe(pd.DataFrame({"Patient ID": shown}), use_container_width=True, hide_index=True,
                          on_select="rerun", selection_mode="single-row", key="patient_table")
    rows_sel = picked.selection.rows if picked and picked.selection else []
    if rows_sel:
        pid = shown[rows_sel[0]]
        st.button(f"Open {pid[:8]}… in Dashboard", type="primary", on_click=_open_patient, args=(pid,))
    st.stop()


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
            or "patient_data" not in st.session_state
            or st.session_state.get("patient_data_version") != DATA_VERSION):
        with st.spinner("Retrieving longitudinal patient records..."):
            try:
                pd_data = _fetch_patient_history_cached(patient_id_input)
                prior_patient_id = st.session_state.get("active_patient_id")
                if prior_patient_id and prior_patient_id != patient_id_input:
                    st.session_state.pop("patient_chat_messages", None)
                    st.session_state.pop("patient_chat_patient_id", None)
                st.session_state["patient_data"] = pd_data
                st.session_state["patient_data_version"] = DATA_VERSION
                st.session_state["active_patient_id"] = patient_id_input
                st.session_state.pop("medication_journey_data", None)
                st.session_state["medication_journey_patient_id"] = patient_id_input
                for cache_key in ("review_facts", "review_visit_hashes", "review_source_hash"):
                    st.session_state.pop(cache_key, None)
                st.session_state.pop("patient_chat_index", None)
                st.session_state.pop("ai_report", None)
                st.session_state.pop("ai_metadata", None)
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
if (st.session_state.get("medication_journey_patient_id") != str(patient_data.get("patient_id", ""))
        or "medication_journey_data" not in st.session_state):
    st.session_state["medication_journey_patient_id"] = str(patient_data.get("patient_id", ""))
    st.session_state["medication_journey_data"] = compute_medication_journey(patient_data)
patient_data["_medication_journey_cache"] = st.session_state["medication_journey_data"]
review_cache_matches = st.session_state.get("review_facts_patient_id") == str(patient_data.get("patient_id", ""))
if not review_cache_matches or "review_source_hash" not in st.session_state:
    facts = build_review_facts(patient_data)
    hashes = visit_hashes(patient_data, facts)
    st.session_state["review_facts"] = facts
    st.session_state["review_visit_hashes"] = hashes
    st.session_state["review_source_hash"] = compute_source_hash(patient_data.get("patient_id", ""), hashes)
    st.session_state["review_facts_patient_id"] = str(patient_data.get("patient_id", ""))
patient_data["_review_facts"] = st.session_state["review_facts"]
patient_data["_review_visit_hashes"] = st.session_state["review_visit_hashes"]
patient_data["_review_source_hash"] = st.session_state["review_source_hash"]
if is_neo4j_configured():
    sync_col, _ = st.columns([1, 5])
    with sync_col:
        if st.button("Sync graph", key=f"sync_graph_{patient_data['patient_id']}"):
            with st.spinner("Syncing this patient's graph..."):
                if sync_patient_to_graph(patient_data):
                    st.success("Graph is current.")
                else:
                    st.error("Graph sync was unavailable. Check the Neo4j connection.")
chat_index_matches = st.session_state.get("patient_chat_index_hash") == st.session_state["review_source_hash"]
if not chat_index_matches or "patient_chat_index" not in st.session_state:
    st.session_state["patient_chat_index"] = build_patient_index(patient_data)
    st.session_state["patient_chat_index_hash"] = st.session_state["review_source_hash"]

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
    pid_disp = html.escape(str(patient_data["patient_id"]))
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Patient ID</div>
        <div class="metric-value" style="font-size:0.95rem; word-break:break-all;">{pid_disp[:14]}...{pid_disp[-6:]}</div>
        <div class="metric-sub"><span class="status-badge status-ok">✓ Verified Clinical Record</span></div>
    </div>
    """, unsafe_allow_html=True)
with ov2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Demographics</div>
        <div class="metric-value" style="font-size:1.05rem;">{html.escape(str(patient_age_sex))}</div>
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
        <div class="metric-value" style="font-size:1.05rem;">{html.escape(str(latest_visit_date))}</div>
        <div class="metric-sub">Most recent entry</div>
    </div>
    """, unsafe_allow_html=True)
with ov5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Primary Conditions</div>
        <div class="metric-value" style="font-size:0.92rem;">{html.escape(conditions_display)}</div>
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
    fu_sub = (f'<span class="status-badge status-danger">{fu_missed} missed</span>' if fu_missed
              else '<span class="status-badge status-ok">On track</span>')
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Follow-up Adherence</div>
        <div class="metric-value" style="font-size:1.05rem;">{fu_adherence_str}</div>
        <div class="metric-sub">{fu_sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

unlinked_docs = patient_data.get("unlinked_documents") or []
if unlinked_docs:
    ul_abnormal = sum(1 for d in unlinked_docs for l in d.get("lab_results") or []
                      if any(k in str(l.get("status", "")).lower() for k in ("abnormal", "high", "low")))
    ul_matched = sum(1 for d in unlinked_docs if d.get("likely_visit") or d.get("candidate_visits"))
    st.markdown(f"""
    <div class="discrepancy-card">
        <div class="discrepancy-title">📎 {len(unlinked_docs)} document(s) are not linked to a visit in the record</div>
        {ul_matched} matched to likely/possible visits · {ul_abnormal} abnormal lab result(s).
        Their findings are shown in the Timeline, Lab Results, Medication Journey, AI Analysis and Source Traceability tabs, and are searchable in the AI Care Assistant.
    </div>
    """, unsafe_allow_html=True)

def _lab_table_rows():
    rows = []
    for idx, v in enumerate(visits, start=1):
        v_date = (v.get("visit_date") or "Unspecified")[:10]
        for lab in v.get("lab_results", []):
            rows.append({"Date": v_date, "Encounter": f"Visit #{idx}", "Test Name": safe_val(lab.get("test_name")),
                         "Result": safe_val(lab.get("value")), "Reference Range": safe_val(lab.get("reference")),
                         "Status": safe_val(lab.get("status"), "Evaluated")})
    for d in unlinked_docs:
        match = d.get("likely_visit")
        for lab in d.get("lab_results", []):
            rows.append({"Date": match["visit_date"] if match else "Unknown (not linked)",
                         "Encounter": f"{d.get('document_label')} → {unlinked_link_text(d)}",
                         "Test Name": safe_val(lab.get("test_name")), "Result": safe_val(lab.get("value")),
                         "Reference Range": safe_val(lab.get("reference")), "Status": safe_val(lab.get("status"), "Evaluated")})
    return rows


if nav_page == "Reports":
    st.markdown('<div class="section-title">Reports</div>', unsafe_allow_html=True)
    st.caption(f"Downloads for patient {patient_data.get('patient_id', '')}")
    rep_c1, rep_c2 = st.columns(2)
    with rep_c1:
        snap = clinical_snapshot.build_snapshot(patient_data) or {}
        snap_md = "\n".join(
            [f"# Clinical snapshot · {patient_data.get('patient_id', '')}",
             f"{snap.get('age_sex') or 'Age/sex not recorded'} · {snap.get('encounters', 0)} encounters ({snap.get('span', '')})",
             "Active: " + (", ".join(snap.get("active", [])) or "none documented"), "", "## Concerns"]
            + [f"- [{f['level']}] {f['title']}: {f['evidence']}" for f in snap.get("flags", [])])
        st.markdown('<div class="metric-card"><div class="metric-label">Clinical snapshot</div>'
                    '<div class="metric-sub">Concerns and active conditions (Markdown)</div></div>', unsafe_allow_html=True)
        st.download_button("Download snapshot", snap_md.encode("utf-8"), file_name="clinical_snapshot.md",
                           mime="text/markdown", use_container_width=True)
        review_row = get_patient_row(str(patient_data.get("patient_id", ""))) or {}
        st.markdown('<div class="metric-card"><div class="metric-label">AI narrative review</div>'
                    f'<div class="metric-sub">{"Last generated " + html.escape(str(review_row.get("generated_at", ""))[:16]) if review_row.get("review_md") else "Not generated yet"}</div></div>',
                    unsafe_allow_html=True)
        if review_row.get("review_md"):
            st.download_button("Download AI review", review_row["review_md"].encode("utf-8"), file_name="ai_review.md",
                               mime="text/markdown", use_container_width=True)
    with rep_c2:
        lab_rows = _lab_table_rows()
        st.markdown('<div class="metric-card"><div class="metric-label">Lab results</div>'
                    f'<div class="metric-sub">{len(lab_rows)} result(s), including documents not linked to a visit</div></div>',
                    unsafe_allow_html=True)
        st.download_button("Download labs (CSV)", pd.DataFrame(lab_rows).to_csv(index=False).encode("utf-8"),
                           file_name="lab_results.csv", mime="text/csv", use_container_width=True, disabled=not lab_rows)
        journey_rows = [{"Date": enc["visit_date"], "Encounter": enc["visit_index"], "Change": ch.get("type"), "Detail": ch.get("text")}
                        for enc in st.session_state["medication_journey_data"] for ch in enc.get("changes", [])]
        st.markdown('<div class="metric-card"><div class="metric-label">Medication journey</div>'
                    f'<div class="metric-sub">{len(journey_rows)} documented change(s)</div></div>', unsafe_allow_html=True)
        st.download_button("Download medication journey (CSV)", pd.DataFrame(journey_rows).to_csv(index=False).encode("utf-8"),
                           file_name="medication_journey.csv", mime="text/csv", use_container_width=True, disabled=not journey_rows)
    st.stop()

# ── Primary Navigation Tabs (capsule style via ui/styles.css) ────────────────
tab_timeline, tab_meds, tab_labs, tab_ai, tab_discrepancies, tab_sources = st.tabs([
    "Timeline",
    "Medication Journey",
    "Lab Results",
    "Retrospective AI Analysis",
    "Record Discrepancies",
    "Source Traceability",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: CLINICAL TIMELINE (Deterministic "What happened over time?")
# ═════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.markdown('<div class="section-title">Longitudinal Clinical Timeline</div>', unsafe_allow_html=True)
    st.caption("Deterministic chronology of encounters, complaints, diagnoses, document findings, labs, medication shifts, and follow-ups.")

    med_journey_data = st.session_state["medication_journey_data"]
    shifts_by_vid = {enc["visit_id"]: enc["changes"] for enc in med_journey_data}

    for idx, v in enumerate(visits, start=1):
        v_id = str(v.get("visit_id", f"V{idx}"))
        render_clinical_encounter(idx, v, shifts_by_vid.get(v_id, []))
        likely_here = [d for d in unlinked_docs if (d.get("likely_visit") or {}).get("visit_index") == idx]
        if likely_here:
            render_unlinked_documents(
                likely_here, heading=f"📎 Likely related document(s) for Encounter #{idx} (not linked in the record)",
                caption="", expanded=False)

    other_unlinked = [d for d in unlinked_docs if not d.get("likely_visit")]
    if other_unlinked:
        render_unlinked_documents(other_unlinked, heading=f"📎 Documents Not Linked to Any Visit ({len(other_unlinked)})")

st.write("")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: MEDICATION JOURNEY (Changes over time)
# ═════════════════════════════════════════════════════════════════════════════
with tab_meds:
    st.markdown('<div class="section-title">Medication Regimen & Longitudinal Changes</div>', unsafe_allow_html=True)
    st.caption("Chronological tracking of medication changes: started, continued, dose/frequency adjustments, and document mismatches.")

    side_effects_cache_key = f"side_effects:v2:{patient_data.get('patient_id', '')}:{st.session_state['review_source_hash']}"
    if side_effects_cache_key in st.session_state:
        patient_data["possible_side_effects"] = st.session_state[side_effects_cache_key]
    pending_lookup = any(
        "isn't in our local reference list" in str(value)
        or str(value).startswith("Unable to identify")
        for value in patient_data.get("possible_side_effects", {}).values()
    )
    if pending_lookup:
        if st.button("Look up unlisted medicines", key="lookup_unlisted_medicines"):
            with st.spinner("Looking up unlisted medicine names..."):
                from analysis import get_possible_side_effects
                looked_up = get_possible_side_effects(patient_data.get("all_medications", []), use_ai=True)
                st.session_state[side_effects_cache_key] = looked_up
                patient_data["possible_side_effects"] = looked_up
                st.session_state["patient_chat_index"] = build_patient_index(patient_data)
                st.session_state["patient_chat_index_hash"] = st.session_state["review_source_hash"]

    med_journey_data = st.session_state["medication_journey_data"]
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
                    ch_text = html.escape(str(ch.get("text", "")))
                    if ch_type == "started":
                        badge = '<span class="diff-badge-added">Started</span>'
                    elif ch_type == "restarted":
                        badge = '<span class="diff-badge-added">Restarted</span>'
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

                    st.markdown(f"<div style='margin-left: 1rem; margin-bottom: 0.35rem;'>{badge} <span style='font-size:0.92rem; font-weight:500; margin-left:6px;'>{esc(ch_text)}</span></div>", unsafe_allow_html=True)
            st.write("")

    ul_with_meds = [d for d in unlinked_docs if d.get("medications_found")]
    if ul_with_meds:
        st.markdown("##### 📎 Medicines in documents not linked to a visit")
        for d in ul_with_meds:
            st.markdown(f"<div style='margin-left: 1rem; margin-bottom: 0.35rem;'><span class='diff-badge-changed'>Unlinked document</span> "
                        f"<span style='font-size:0.92rem; font-weight:500; margin-left:6px;'>{html.escape(d.get('document_label', ''))} "
                        f"({html.escape(unlinked_link_text(d))}): {html.escape(', '.join(d['medications_found']))}</span></div>",
                        unsafe_allow_html=True)

    if med_journey_data and any(enc.get("changes") for enc in med_journey_data):
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
                for row in se_rows:
                    st.markdown(f"**{html.escape(str(row['Medication']))}**  ")
                    st.write(row["Monitored Considerations"])
                    st.caption(row["Source"])
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

    all_labs = _lab_table_rows()

    if not all_labs:
        st.info("No laboratory investigations documented for this patient.")
    else:
        df_labs = pd.DataFrame(all_labs)

        # Visual indicator styling for normal vs abnormal
        def highlight_status(val):
            s = str(val).lower()
            if "abnormal" in s or "high" in s or "low" in s:
                return "background-color: #fdecec; color: #a61b1b; font-weight: 700;"
            elif "normal" in s:
                return "background-color: #e6f6ec; color: #1b873f; font-weight: 600;"
            return "color: #5b6474;"

        styled_labs = df_labs.style.map(highlight_status, subset=["Status"])
        st.dataframe(styled_labs, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: AI RETROSPECTIVE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
# Fragment: Generate/Force Refresh and the toggles rerun this tab only, not the whole page.
@st.fragment
def _render_ai_tab():
    st.markdown('<div class="section-title">Clinical Snapshot</div>', unsafe_allow_html=True)
    try:
        render_clinical_snapshot(clinical_snapshot.build_snapshot(patient_data))
    except Exception as e:
        st.warning(f"Clinical snapshot unavailable: {e}")

    st.markdown('<div class="section-title">Clinical Course</div>', unsafe_allow_html=True)
    try:
        render_clinical_course(clinical_snapshot.build_clinical_course(patient_data))
    except Exception as e:
        st.warning(f"Clinical course unavailable: {e}")

    st.markdown('<div class="section-title">AI Narrative Review</div>', unsafe_allow_html=True)
    # Status Bar & Controls
    ai_c1, ai_c2 = st.columns([3, 1])
    cache_current = is_review_current(patient_data)
    with ai_c2:
        review_action = st.button(
            "Force Refresh" if cache_current else "Generate review",
            type="secondary" if cache_current else "primary",
            use_container_width=True,
        )

    try:
        if cache_current:
            review = get_or_generate_review(
                patient_data, force_refresh=bool(review_action)
            )
        elif review_action:
            with st.spinner("Updating retrospective longitudinal review..."):
                review = get_or_generate_review(patient_data)
        else:
            review = None
            st.info("No current review is cached. Generate one when you’re ready.")
        if review:
            raw_ai_text = review.get("raw_markdown", "")
            st.session_state["ai_report"] = raw_ai_text
            st.session_state["ai_metadata"] = {
                "model_name": review.get("model_name", "AI Engine"),
                "generated_at": review.get("generated_at", ""),
                "source_visits": patient_data.get("total_visits", 0),
            }
            visit_label = f"{patient_data.get('total_visits', 0)} visits"
            if review.get("cache_hit"):
                st.caption(f"Cached review • {visit_label} • generated {review.get('generated_at', 'date unavailable')}")
            elif review.get("status", "").startswith("Incremental"):
                st.caption(f"Incremental review • {visit_label}")
            else:
                st.caption(f"Fresh full review • {visit_label}")
            if raw_ai_text:
                # Toggles, not expanders: the rendered sections contain expanders of their own.
                if st.toggle("Show detailed AI narrative review", key="show_ai_narrative"):
                    render_doctor_friendly_ai_review(raw_ai_text)
    except Exception as e:
        st.error(f"Failed to load or generate AI retrospective review: {e}")
    if unlinked_docs:
        if st.toggle(f"Show documents not linked to a visit ({len(unlinked_docs)})", key="show_ai_unlinked"):
            render_unlinked_documents(unlinked_docs, heading="", expanded=False)


with tab_ai:
    _render_ai_tab()

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

    if any(v.get("documents") for v in visits):
        if st.button("AI assist low-confidence document extraction", key="ai_assist_extraction"):
            from analysis import _augment_with_ai_if_gaps, extract_medications_from_text, get_possible_side_effects
            with st.spinner("Checking low-confidence documents..."):
                document_meds = set()
                for visit in visits:
                    visit_meds = set()
                    for doc in visit.get("documents", []) or []:
                        lines = doc.get("extracted_text") or []
                        summary = doc.get("clinical_summary") or {}
                        summary = _augment_with_ai_if_gaps(lines, summary)
                        doc["clinical_summary"] = summary
                        doc["lab_results"] = summary.get("lab_results") or doc.get("lab_results", [])
                        names = set(extract_medications_from_text(lines))
                        names.update(m.get("name") for m in summary.get("medications", []) if m.get("name"))
                        doc["medications_found"] = sorted(names)
                        visit_meds.update(names)
                    visit["document_medications"] = sorted(visit_meds)
                    document_meds.update(visit_meds)
                    visit_db_meds = {m.get("name") for m in visit.get("db_medications", []) if m.get("name")}
                    visit["medications_found"] = sorted(visit_db_meds | visit_meds)
                patient_data["all_document_medications"] = sorted(document_meds)
                all_meds = set(patient_data.get("all_db_medications", [])) | document_meds
                patient_data["all_medications"] = sorted(all_meds)
                patient_data["possible_side_effects"] = get_possible_side_effects(sorted(all_meds), use_ai=False)
                st.session_state.pop(f"side_effects:{patient_data.get('patient_id', '')}", None)
                patient_data.pop("_medication_journey_cache", None)
                st.session_state.pop("medication_journey_data", None)
                for cache_key in ("review_facts", "review_visit_hashes", "review_source_hash"):
                    st.session_state.pop(cache_key, None)
                    patient_data.pop(f"_{cache_key}", None)
            st.rerun()

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
                if cs.get("ai_assisted_fields"):
                    status_str = "AI-assisted; verify"

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

    for d in unlinked_docs:
        stats = (d.get("clinical_summary") or {}).get("extraction_stats", {})
        tot_l, garb = stats.get("total_lines", 0), stats.get("garbled_lines", 0)
        source_records.append({
            "Visit Date": (d.get("likely_visit") or {}).get("visit_date", "Unknown"),
            "Encounter": f"Not linked → {unlinked_link_text(d)}",
            "Document": d.get("document_label", "Unlinked document"),
            "Document ID": str(d.get("doc_id", "N/A")),
            "Lines Processed": tot_l,
            "Readable Lines": tot_l - garb if tot_l else 0,
            "Extracted Fields": stats.get("extracted_fields", 0),
            "Quality Status": "Not linked to a visit",
        })

    # Rows mix "N/A" with counts; strings keep the Arrow conversion clean.
    df_sources = pd.DataFrame(source_records).astype(str)
    st.dataframe(df_sources, use_container_width=True, hide_index=True)


# ── Floating AI Care Assistant (bottom-right) ────────────────────────────────
def _toggle_assistant():
    st.session_state["assistant_open"] = not st.session_state.get("assistant_open", False)


def _render_chat_message(message, expanded=False):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            meta = message.get("meta")
            st.caption(_chat_label(meta) if meta is not None else "LLM-generated response")
        st.markdown(message["content"])
        if message["role"] == "assistant" and "retrieved_evidence" in message:
            with st.expander(f"RAG retrieved {len(message['retrieved_evidence'])} item(s)", expanded=expanded):
                if message["retrieved_evidence"]:
                    for item in message["retrieved_evidence"]:
                        st.markdown(
                            f"**{item.get('date', 'Date not recorded')} · {item.get('category', 'Evidence')}**  "
                            f"\nVisit: {item.get('visit_id', 'Unknown')} · Source: {item.get('source_id', 'Unknown')}"
                        )
                        st.write(item.get("content", ""))
                else:
                    st.caption("No relevant evidence was retrieved for this question.")


def _toggle_assistant_size():
    st.session_state["assistant_large"] = not st.session_state.get("assistant_large", False)


# Fragment: chatting, switching mode or resizing reruns only the assistant.
@st.fragment
def _render_care_assistant():
    active_id = str(patient_data.get("patient_id", ""))
    if st.session_state.get("patient_chat_patient_id") != active_id:
        st.session_state["patient_chat_patient_id"] = active_id
        st.session_state["patient_chat_messages"] = []
    chat_messages = st.session_state.setdefault("patient_chat_messages", [])

    if not st.session_state.get("assistant_open", False):
        with st.container(key="care_launcher"):
            st.button("💬 AI Care Assistant", on_click=_toggle_assistant, key="open_assistant")
    else:
        large = st.session_state.get("assistant_large", False)
        with st.container(key="care_assistant_large" if large else "care_assistant"):
            st.markdown('<div class="ca-header"><span>🩺</span><div><div class="ca-title">AI Care Assistant</div>'
                        '<div class="ca-sub">Answers from this patient\'s record only</div></div></div>', unsafe_allow_html=True)
            mode_col, size_col, close_col = st.columns([5, 1, 1])
            with mode_col:
                mode = st.segmented_control("Mode", ["Summary", "Ask AI"], default="Summary",
                                            key="assistant_mode", label_visibility="collapsed") or "Summary"
            with size_col:
                st.button("⤡" if large else "⤢", on_click=_toggle_assistant_size, key="resize_assistant",
                          help="Shrink" if large else "Expand to a larger window")
            with close_col:
                st.button("✕", on_click=_toggle_assistant, key="close_assistant", help="Close assistant")

            if mode == "Summary":
                snap = clinical_snapshot.build_snapshot(patient_data) or {}
                st.markdown(
                    f'<div class="ca-summary-line"><b>{html.escape(snap.get("age_sex") or "Age/sex not recorded")}</b> · '
                    f'{snap.get("encounters", 0)} encounters · Active: {html.escape(", ".join(snap.get("active", [])) or "none documented")}</div>'
                    + "".join(f'<div class="ca-flag ca-flag-{f["level"]}"><b>{html.escape(f["title"])}</b><br>{html.escape(f["evidence"])}</div>'
                              for f in snap.get("flags", []) if f["level"] in ("high", "medium")),
                    unsafe_allow_html=True,
                )
                st.caption("Rule-based from the record · no AI tokens. Full snapshot in Retrospective AI Analysis.")
            else:
                try:
                    faq = get_top_questions(active_id)
                except Exception:
                    faq = []
                if faq:
                    st.caption("Frequently asked")
                    for i, q in enumerate(faq):
                        if st.button(q, key=f"faq_{active_id}_{i}", use_container_width=True):
                            st.session_state["faq_pick"] = q
                            st.rerun(scope="fragment")
                history = st.container(height=560 if large else 320)
                with history:
                    if not chat_messages:
                        st.caption("Ask about diagnoses, labs, medicines, trends or documents for this patient.")
                    for message in chat_messages:
                        _render_chat_message(message)
                st.caption(f"Session LLM tokens: {sum(m.get('meta', {}).get('total_tokens', 0) for m in chat_messages)}")
                question = (st.chat_input("Ask a question about this patient's records...", key="assistant_input")
                            or st.session_state.pop("faq_pick", None))
                if question and question.strip():
                    question = question.strip()
                    prior_history = list(chat_messages)
                    user_message = {"role": "user", "content": question}
                    chat_messages.append(user_message)
                    with history:
                        _render_chat_message(user_message)
                        with st.spinner("Retrieving patient record evidence..."):
                            try:
                                chat_meta = {}
                                answer, retrieved = answer_patient_question(
                                    patient_data, active_id, question, history=prior_history,
                                    api_key=CHAT_API_KEY,
                                    retrieval_index=st.session_state["patient_chat_index"],
                                    source_hash=st.session_state["review_source_hash"],
                                    meta=chat_meta,
                                )
                                if retrieved:
                                    try:
                                        record_chat_question(active_id, question)
                                    except Exception:
                                        pass
                                reply = {"role": "assistant", "content": answer,
                                         "retrieved_evidence": retrieved, "meta": chat_meta}
                                chat_messages.append(reply)
                                _render_chat_message(reply, expanded=False)
                            except Exception as exc:
                                st.error(f"Patient record Q&A failed: {exc}")


_render_care_assistant()
