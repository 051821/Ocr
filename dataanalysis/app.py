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
import importlib
import api_client
import lab_validation
importlib.reload(api_client)
import ui_components
importlib.reload(ui_components)
import insight_tabs
importlib.reload(insight_tabs)
from ui_components import esc, strip_remote_images, render_clinical_encounter, render_unlinked_documents, unlinked_link_text

logger = logging.getLogger(__name__)


# Bump when the shape of fetched patient data changes so cached sessions refetch.
DATA_VERSION = 4  # 3: lab validation (ordered-test lists, impossible values, critical flags); 4: no panel-name rows

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
st.markdown('<div class="app-subtitle">Longitudinal Clinical Record · AI Analysis & Summary</div>', unsafe_allow_html=True)

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
        patient_ids = api_client.list_patients()
    except api_client.ApiError as e:
        patient_ids = []
        st.error(f"API error: {e.detail}")

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


def _api_settings():
    """Server settings (flags, models, Neo4j), fetched once per session."""
    if "api_settings" not in st.session_state:
        try:
            st.session_state["api_settings"] = api_client.get_settings()
        except api_client.ApiError:
            return {"flags": {}, "chat_model": "", "review_models": [], "neo4j_configured": False}
    return st.session_state["api_settings"]


def _load_patient(pid, refresh=False):
    """Fetch the record and its derived views from the API into session state."""
    if refresh:
        api_client.refresh_patient(pid)
    st.session_state["patient_data"] = api_client.get_patient(pid)
    st.session_state["medication_journey_data"] = api_client.get_medication_journey(pid)
    st.session_state["clinical_views"] = api_client.get_snapshot(pid)
    try:
        st.session_state["referrals"] = api_client.get_referrals(pid)
    except api_client.ApiError as e:
        logger.warning("Referrals unavailable for %s: %s", pid, e.detail)
        st.session_state["referrals"] = []
    st.session_state.pop("insight_view", None)


_TIERS = {"quick": "Quick", "detailed": "Detailed"}


def _ai_tier():
    """Tier that opens: ?view= in the address, else Quick."""
    for v in (st.query_params.get("view"), "quick"):
        if str(v or "").casefold() in _TIERS:
            return str(v).casefold()
    return "quick"


def _insight_view(refresh=False):
    """Data for the AI Analysis & Summary and Lab Results tabs (merged brief, concerns with raw data and rules,
    visits, labs), fetched once per patient and again after the AI brief is generated."""
    if refresh or "insight_view" not in st.session_state:
        st.session_state["insight_view"] = api_client.get_insight_view(st.session_state["active_patient_id"])
    return st.session_state["insight_view"]


if nav_page == "Settings":
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
    st.caption("Read-only. Values come from the API server's environment (.env); change them there and restart the API.")
    st.session_state.pop("api_settings", None)
    settings = _api_settings()
    rows = "".join(
        f'<div class="field-row"><span class="field-name">{html.escape(name)}</span>'
        f'<span class="status-badge {"status-ok" if on else "status-info"}">'
        f'{"On" if on else "Off"}</span></div>' for name, on in settings["flags"].items())
    st.markdown(f'<div class="metric-card"><div class="metric-label">Feature flags</div>{rows}</div>', unsafe_allow_html=True)
    st.write("")
    neo4j_on = settings["neo4j_configured"]
    st.markdown(
        '<div class="metric-card"><div class="metric-label">AI models &amp; services</div>'
        f'<div class="field-row"><span class="field-name">Chat model</span><span class="field-val">{html.escape(settings["chat_model"])}</span></div>'
        f'<div class="field-row"><span class="field-name">Review models</span><span class="field-val">{html.escape(", ".join(settings["review_models"]))}</span></div>'
        f'<div class="field-row"><span class="field-name">Graph (Neo4j)</span><span class="status-badge {"status-ok" if neo4j_on else "status-info"}">'
        f'{"Connected" if neo4j_on else "Not configured"}</span></div></div>',
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
                prior_patient_id = st.session_state.get("active_patient_id")
                if prior_patient_id and prior_patient_id != patient_id_input:
                    st.session_state.pop("patient_chat_messages", None)
                    st.session_state.pop("patient_chat_patient_id", None)
                _load_patient(patient_id_input, refresh=bool(btn_fetch))
                st.session_state["patient_data_version"] = DATA_VERSION
                st.session_state["active_patient_id"] = patient_id_input
                st.session_state.pop("ai_report", None)
                st.session_state.pop("ai_metadata", None)
            except api_client.ApiError as e:
                if e.status_code == 404:
                    st.warning(f"No clinical visits found for Patient ID `{patient_id_input}`.")
                else:
                    st.error(f"Error retrieving clinical record: {e.detail}")
                # Never keep showing the previous patient under the newly entered ID.
                for stale_key in ("patient_data", "patient_data_version",
                                  "patient_chat_messages", "patient_chat_patient_id",
                                  "medication_journey_data", "clinical_views", "insight_view",
                                  "ai_report", "ai_metadata"):
                    st.session_state.pop(stale_key, None)

patient_data = st.session_state.get("patient_data")

if not patient_data:
    st.info("Please select or enter a Patient ID above to view clinical intelligence.")
    st.stop()

if patient_data.get("total_visits", 0) == 0:
    st.warning(f"No clinical visits found for Patient ID `{patient_id_input}`.")
    st.stop()


# ── 2. PATIENT OVERVIEW (Compact Metric Cards) ───────────────────────────────
visits = patient_data.get("visits", [])
total_visits = patient_data.get("total_visits", 0)
active_pid = str(patient_data.get("patient_id", ""))
if "medication_journey_data" not in st.session_state or "clinical_views" not in st.session_state:
    try:
        _load_patient(active_pid)
    except api_client.ApiError as e:
        st.error(f"Error retrieving clinical record: {e.detail}")
        st.stop()
if _api_settings()["neo4j_configured"]:
    sync_col, _ = st.columns([1, 5])
    with sync_col:
        if st.button("Sync graph", key=f"sync_graph_{patient_data['patient_id']}"):
            with st.spinner("Syncing this patient's graph..."):
                if api_client.graph_sync(active_pid):
                    st.success("Graph is current.")
                else:
                    st.error("Graph sync was unavailable. Check the Neo4j connection.")

# Age / Sex from the patient table; the documents only when the table has none
patient_age_sex = patient_data.get("age_sex") or "Not available"
if patient_age_sex == "Not available":
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

# The snapshot's active conditions (named, de-duplicated), comma-separated; raw diagnosis text when it has none.
_active = (st.session_state["clinical_views"].get("snapshot") or {}).get("active") or sorted(all_conditions)
conditions_display = ", ".join(_active[:3]) if _active else "Not documented"
if len(_active) > 3:
    conditions_display += f", +{len(_active)-3} more"

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
        <div class="metric-label">Visits</div>
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

# ── Long-term conditions (patient_condition -> long_term_condition) ──────────
long_term_conditions = patient_data.get("long_term_conditions") or []
if long_term_conditions:
    ltc_chips = "".join(
        f'<span class="status-badge status-info" style="margin:0 6px 4px 0;" '
        f'title="{html.escape("Diagnosed " + c["diagnosed_at"] if c.get("diagnosed_at") else "Diagnosis date not recorded")}">'
        f'{html.escape(" · ".join(str(x) for x in (c["name"], c.get("severity"), c.get("status")) if x))}</span>'
        for c in long_term_conditions)
    st.markdown(f"""
    <div class="metric-card" style="margin-top:10px;">
        <div class="metric-label">Long-term Conditions</div>
        <div class="metric-sub" style="margin-top:6px;">{ltc_chips}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Health history (patient.health_history_json items marked present) ────────
health_history = patient_data.get("health_history_present") or []
if health_history:
    hh_chips = "".join(
        f'<span class="status-badge status-warn" style="margin:0 6px 4px 0;">'
        f'{html.escape(h["item"] + (" · " + h["detail"] if h.get("detail") else ""))}</span>'
        for h in health_history)
    st.markdown(f"""
    <div class="metric-card" style="margin-top:10px;">
        <div class="metric-label">Health History</div>
        <div class="metric-sub" style="margin-top:6px;">{hh_chips}</div>
        <div class="metric-sub">Patient-reported at registration</div>
    </div>
    """, unsafe_allow_html=True)

# ── Patient type (body-system grouping from diagnoses + complaints) ──────────
ptypes = st.session_state["clinical_views"].get("patient_types") or []
visit_clusters = st.session_state["clinical_views"].get("visit_clusters") or []
if ptypes:
    chips = "".join(
        f'<span class="status-badge {"status-ok" if t["primary"] else "status-info"}" style="margin:0 6px 4px 0;" '
        f'title="{html.escape(", ".join(t["terms"]))}">{html.escape(t["type"])} · {t["visits"]} visit{"s" if t["visits"] != 1 else ""}</span>'
        for t in ptypes)
    exercise = st.session_state["clinical_views"].get("exercise") or {"exercise": "—", "reason": ""}
    ex_badge = (f'<span class="status-badge {"status-ok" if exercise["exercise"] == "Yes" else "status-warn"}" '
                f'title="{html.escape(exercise["reason"])}">Exercise benefit: {html.escape(exercise["exercise"])}</span>')
    st.markdown(f"""
    <div class="metric-card" style="margin-top:10px;">
        <div class="metric-label">Patient Type</div>
        <div class="metric-value" style="font-size:1.05rem;">{html.escape(ptypes[0]["type"])} {ex_badge}</div>
        <div class="metric-sub" style="margin-top:4px;"><b>Why this type:</b> {html.escape(exercise.get("type_reason", ""))}</div>
        <div class="metric-sub" style="margin-top:2px;"><b>Exercise:</b> {html.escape(exercise["reason"])}</div>
        <div class="metric-sub" style="margin-top:6px;">{chips}</div>
    </div>
    """, unsafe_allow_html=True)
if visit_clusters:
    with st.expander(f"Cluster & exercise by visit ({len(visit_clusters)})", expanded=False):
        st.dataframe(pd.DataFrame([{
            "Patient ID": r["patient_id"],
            "Visit ID": r["visit_id"],
            "Visit Date": r["visit_date"],
            "Cluster": r["cluster"],
            "Why this cluster": r.get("type_reason", ""),
            "Also matched": ", ".join(r["also_matched"]),
            "Can Exercise": r["exercise"],
            "Why": r.get("exercise_reason", ""),
            "Matched terms": ", ".join(r["terms"]),
        } for r in visit_clusters]), use_container_width=True, hide_index=True)


def _referral_line(r):
    """'Psychiatrist → Behavior Healthcare Center' and its sub-line (date · centre · status)."""
    head = " → ".join(esc(x) for x in (r.get("referred_for"), r.get("referred_to")) if x) or "Referred to another provider"
    date = (r.get("referral_date") or "")[:10]
    status = "⏳ Pending" if r.get("is_open") else "✓ Resolved"
    if r.get("referral_status"):
        status = esc(str(r["referral_status"]).title())
    bits = [date, esc(r.get("center_name") or ""), status]
    if r.get("referral_reason") and r.get("specialty_name"):
        bits.append("Reason: " + esc(r["referral_reason"]))
    if r.get("expected_review_date") and r.get("is_open"):
        bits.append("Review by " + str(r["expected_review_date"])[:10])
    return head, " · ".join(b for b in bits if b)


referrals = st.session_state.get("referrals") or []
if referrals:
    latest = referrals[0]
    head, sub = _referral_line(latest)
    badge_cls = "status-warn" if latest.get("is_open") else "status-ok"
    more = f'<span class="metric-sub" style="margin-left:8px;">+{len(referrals)-1} more</span>' if len(referrals) > 1 else ""
    st.markdown(f"""
    <div class="metric-card" style="margin-top:8px;">
        <div><span class="status-badge {badge_cls}">🔀 Referred out</span>
             <span style="font-weight:600; margin-left:8px;">{head}</span>{more}</div>
        <div class="metric-sub" style="margin-top:4px;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)
    if len(referrals) > 1:
        with st.expander(f"All referrals ({len(referrals)})", expanded=False):
            for r in referrals:
                h, sb = _referral_line(r)
                st.markdown(f"<b>{h}</b><br><span style='font-size:0.85rem;'>{sb}</span>", unsafe_allow_html=True)

st.write("")

unlinked_docs =patient_data.get("unlinked_documents") or []
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
            rows.append({"Date": v_date, "Visit": f"Visit #{idx}", "Test Name": safe_val(lab.get("test_name")),
                         "Result": safe_val(lab.get("value")), "Reference Range": safe_val(lab.get("reference")),
                         "Status": safe_val(lab.get("status"), "Evaluated"), "Note": lab.get("note") or "",
                         "_dated": v_date if v.get("visit_date") else None})
    for d in unlinked_docs:
        match = d.get("likely_visit")
        for lab in d.get("lab_results", []):
            rows.append({"Date": match["visit_date"] if match else "Unknown (not linked)",
                         "Visit": f"{d.get('document_label')} → {unlinked_link_text(d)}",
                         "Test Name": safe_val(lab.get("test_name")), "Result": safe_val(lab.get("value")),
                         "Reference Range": safe_val(lab.get("reference")), "Status": safe_val(lab.get("status"), "Evaluated"),
                         "Note": lab.get("note") or "", "_dated": None})  # undated: matched only with undated rows
    # Same test, same value, same date: shown once (a report and its image / typed copy), with where the copy was.
    rows = lab_validation.drop_same_day_duplicates(
        rows, date=lambda r: r["_dated"], test=lambda r: r["Test Name"], value=lambda r: r["Result"],
        source=lambda r: r["Visit"])
    for r in rows:
        dups = r.pop("duplicate_sources", None)
        r.pop("_dated", None)
        if dups:
            r["Note"] = "; ".join(x for x in (r["Note"], "Same result also in: " + ", ".join(dict.fromkeys(dups))) if x)
    return rows


if nav_page == "Reports":
    st.markdown('<div class="section-title">Reports</div>', unsafe_allow_html=True)
    st.caption(f"Downloads for patient {patient_data.get('patient_id', '')}")
    rep_c1, rep_c2 = st.columns(2)
    with rep_c1:
        snap = st.session_state["clinical_views"].get("snapshot") or {}
        snap_md = "\n".join(
            [f"# Clinical snapshot · {patient_data.get('patient_id', '')}",
             f"{snap.get('age_sex') or 'Age/sex not recorded'} · {snap.get('encounters', 0)} visits ({snap.get('span', '')})",
             "Active: " + (", ".join(snap.get("active", [])) or "none documented"), "", "## Concerns"]
            + [f"- [{f['level']}] {f['title']}: {f['evidence']}" for f in snap.get("flags", [])])
        st.markdown('<div class="metric-card"><div class="metric-label">Clinical snapshot</div>'
                    '<div class="metric-sub">Concerns and active conditions (Markdown)</div></div>', unsafe_allow_html=True)
        st.download_button("Download snapshot", snap_md.encode("utf-8"), file_name="clinical_snapshot.md",
                           mime="text/markdown", use_container_width=True)
        try:
            review_row = api_client.get_review(active_pid) or {}
        except api_client.ApiError:
            review_row = {}
        st.markdown('<div class="metric-card"><div class="metric-label">AI narrative review</div>'
                    f'<div class="metric-sub">{"Last generated " + html.escape(str(review_row.get("generated_at", ""))[:16]) if review_row.get("raw_markdown") else "Not generated yet"}</div></div>',
                    unsafe_allow_html=True)
        if review_row.get("raw_markdown"):
            st.download_button("Download AI review", review_row["raw_markdown"].encode("utf-8"), file_name="ai_review.md",
                               mime="text/markdown", use_container_width=True)
    with rep_c2:
        lab_rows = _lab_table_rows()
        st.markdown('<div class="metric-card"><div class="metric-label">Lab results</div>'
                    f'<div class="metric-sub">{len(lab_rows)} result(s), including documents not linked to a visit</div></div>',
                    unsafe_allow_html=True)
        st.download_button("Download labs (CSV)", pd.DataFrame(lab_rows).to_csv(index=False).encode("utf-8"),
                           file_name="lab_results.csv", mime="text/csv", use_container_width=True, disabled=not lab_rows)
        journey_rows = [{"Date": enc["visit_date"], "Visit": enc["visit_index"], "Change": ch.get("type"), "Detail": ch.get("text")}
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
    "AI Analysis & Summary",
    "Record Discrepancies",
    "Source Traceability",
], default="AI Analysis & Summary")  # the tab that opens first


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: CLINICAL TIMELINE (Deterministic "What happened over time?")
# ═════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.markdown('<div class="section-title">Longitudinal Clinical Timeline</div>', unsafe_allow_html=True)
    st.caption("Deterministic chronology of visits, complaints, diagnoses, document findings, labs, medication shifts, and follow-ups.")

    med_journey_data = st.session_state["medication_journey_data"]
    shifts_by_vid = {enc["visit_id"]: enc["changes"] for enc in med_journey_data}

    for idx, v in enumerate(visits, start=1):
        v_id = str(v.get("visit_id", f"V{idx}"))
        render_clinical_encounter(idx, v, shifts_by_vid.get(v_id, []))
        likely_here = [d for d in unlinked_docs if (d.get("likely_visit") or {}).get("visit_index") == idx]
        if likely_here:
            render_unlinked_documents(
                likely_here, heading=f"📎 Likely related document(s) for Visit #{idx} (not linked in the record)",
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

    pending_lookup = any(
        "isn't in our local reference list" in str(value)
        or str(value).startswith("Unable to identify")
        for value in patient_data.get("possible_side_effects", {}).values()
    )
    if pending_lookup:
        if st.button("Look up unlisted medicines", key="lookup_unlisted_medicines"):
            with st.spinner("Looking up unlisted medicine names..."):
                try:
                    # Server keeps the result, so chat and review see it too.
                    patient_data["possible_side_effects"] = api_client.get_side_effects(active_pid, use_ai=True)
                except api_client.ApiError as e:
                    st.error(f"Side-effect lookup failed: {e.detail}")

    med_journey_data = st.session_state["medication_journey_data"]
    if not med_journey_data or not any(enc.get("changes") for enc in med_journey_data):
        st.info("No medications recorded in either prescription database or extracted clinical documents.")
    else:
        # Specialization of the doctor(s) who wrote each visit's prescriptions; never a name.
        specs_by_vid = {str(v.get("visit_id")): v.get("prescriber_specializations") or [] for v in visits}
        for enc in med_journey_data:
            st.markdown(f"##### 🗓️ {enc['visit_date']} — Visit #{enc['visit_index']} (ID {enc['visit_id']})")
            specs = specs_by_vid.get(str(enc["visit_id"])) or []
            if specs:
                st.markdown('<div style="margin:-6px 0 6px 0; font-size:0.85rem; color:#475569;">Prescribed by: '
                            + " · ".join(f'<span style="font-weight:600; color:#0f2a5f;">{html.escape(s)}</span>'
                                         for s in specs) + "</div>", unsafe_allow_html=True)
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
    # Same view as the static Patient Insight Review: every result once (copies merged), status badges, filter.
    try:
        insight_tabs.render_labs_tab(_insight_view())
    except api_client.ApiError as e:
        st.error(f"Lab results unavailable: {e.detail}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: AI ANALYSIS & SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
# Fragment: Generate/Force Refresh reruns this tab only, not the whole page.
@st.fragment
def _render_ai_tab():
    try:
        view = _insight_view()
    except api_client.ApiError as e:
        st.error(f"AI analysis unavailable: {e.detail}")
        return
    # Quick (bedside, one screen) or Detailed (everything, each fact once); see summary_tiers.py.
    tier = st.segmented_control("View", list(_TIERS), default=_ai_tier(), format_func=_TIERS.get,
                                key=f"ai_tier_{active_pid}", label_visibility="collapsed") or "quick"
    if st.query_params.get("view") != tier:
        st.query_params["view"] = tier
    insight_tabs.render_view_help()
    # Status tiles first (click one to highlight its points below), then the Patient Brief heading and controls.
    insight_tabs.render_ai_header(view)
    has_review = bool(view["patient"].get("ai_markdown"))
    ai_c1, ai_c2 = st.columns([4, 1], vertical_alignment="bottom")
    with ai_c1:
        insight_tabs.render_section_title("Patient Brief")
    with ai_c2:
        review_action = st.button(
            "Force Refresh" if has_review else "Generate AI brief",
            type="secondary" if has_review else "primary",
            use_container_width=True,
        )
    status = ""
    if review_action:
        try:
            with st.spinner("Updating retrospective longitudinal review..."):
                review = api_client.generate_review(active_pid, force_refresh=has_review)
                status = ("cached" if review.get("cache_hit")
                          else "incremental update" if str(review.get("status", "")).startswith("Incremental")
                          else "fresh")
                view = _insight_view(refresh=True)
        except api_client.ApiError as e:
            st.error(f"Failed to generate AI retrospective review: {e.detail}")
    if status:
        st.caption(f"AI brief: {status}")
    if tier == "quick":
        insight_tabs.render_quick_body(view)
    else:
        # Detailed: brief (points link to concerns), checks, concerns, visit grid, Clinical Course, reference.
        insight_tabs.render_ai_body(view, status)


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
            all_discrepancies.append({"Visit Date": v_date, "Visit": f"Visit #{idx}", "Discrepancy Detail": d})

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
            st.markdown(f"- **{item['Visit Date']} ({item['Visit']}):** {item['Discrepancy Detail']}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7: SOURCE TRACEABILITY
# ═════════════════════════════════════════════════════════════════════════════
with tab_sources:
    st.markdown('<div class="section-title">Source Documentation & Extraction Traceability</div>', unsafe_allow_html=True)
    st.caption("Audit trail linking structured clinical fields back to underlying documents and OCR extraction statistics.")

    if any(v.get("documents") for v in visits):
        if st.button("AI assist low-confidence document extraction", key="ai_assist_extraction"):
            with st.spinner("Checking low-confidence documents..."):
                try:
                    api_client.ai_assist(active_pid)
                    _load_patient(active_pid)
                except api_client.ApiError as e:
                    st.error(f"AI assist failed: {e.detail}")
                    st.stop()
            st.rerun()

    source_records = []
    for idx, v in enumerate(visits, start=1):
        v_date = (v.get("visit_date") or f"Visit {idx}")[:10]
        docs = v.get("documents", [])
        if not docs:
            source_records.append({
                "Visit Date": v_date,
                "Visit": f"Visit #{idx}",
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
                    "Visit": f"Visit #{idx}",
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
            "Visit": f"Not linked → {unlinked_link_text(d)}",
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


def _split_evidence(content):
    """Split an answer into its body and its '### Evidence' section (shown only on request)."""
    parts = re.split(r"(?im)^#+\s*evidence\s*$", content or "", maxsplit=1)
    return parts[0].rstrip(), (parts[1].strip() if len(parts) > 1 else "")


def _render_chat_message(message, expanded=False):
    with st.chat_message(message["role"]):
        if message["role"] != "assistant":
            st.markdown(strip_remote_images(message["content"]))
            return
        meta = message.get("meta")
        st.caption(_chat_label(meta) if meta is not None else "LLM-generated response")
        body, evidence_md = _split_evidence(message["content"])
        st.markdown(strip_remote_images(body))
        retrieved = message.get("retrieved_evidence")
        if not evidence_md and retrieved is None:
            return
        # Evidence stays collapsed so the answer reads cleanly; open it to check the sources.
        with st.expander("Evidence", expanded=expanded):
            if evidence_md:
                st.markdown(strip_remote_images(evidence_md))
            if retrieved is not None:
                st.caption(f"RAG retrieved {len(retrieved)} item(s)")
                if not retrieved:
                    st.caption("No relevant evidence was retrieved for this question.")
                for item in retrieved:
                    st.markdown(
                        f"**{item.get('date', 'Date not recorded')} · {item.get('category', 'Evidence')}**  "
                        f"\nVisit: {item.get('visit_id', 'Unknown')} · Source: {item.get('source_id', 'Unknown')}"
                    )
                    st.markdown(strip_remote_images(item.get("content", "")))


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
                # Concise: one short line per concern (snapshot_view.concern_line), grouped by topic, most serious first,
                # in a box that scrolls inside the assistant only.
                snap = st.session_state["clinical_views"].get("snapshot") or {}
                flags = snap.get("flags", [])
                groups = snap.get("concern_groups") or [{"label": "", "concerns": [
                    i for i, f in enumerate(flags) if f["level"] in ("high", "verify", "medium")]}]
                rows = []
                for g in groups:
                    if g["label"]:
                        rows.append(f'<div class="ca-group">{html.escape(g["label"])}</div>')
                    for i in g["concerns"]:
                        f = flags[i]
                        rows.append(f'<div class="ca-line ca-line-{html.escape(f["level"])}">'
                                    f'{html.escape(f.get("summary") or f["title"])}</div>')
                with st.container(height=520 if large else 300, key="ca_summary_box"):
                    st.markdown(
                        f'<div class="ca-summary-line"><b>{html.escape(snap.get("age_sex") or "Age/sex not recorded")}</b> · '
                        f'{snap.get("encounters", 0)} visits</div>'
                        + ("".join(rows) or '<div class="ca-line">No high or medium concern in the record rules.</div>'),
                        unsafe_allow_html=True,
                    )
                st.caption("Rule-based from the record · no AI tokens. Full snapshot in AI Analysis & Summary.")
            else:
                try:
                    faq = api_client.get_faq(active_id)
                except api_client.ApiError:
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
                                # Server records the question for FAQ when evidence is found.
                                answer, retrieved, chat_meta = api_client.chat(active_id, question, history=prior_history)
                                reply = {"role": "assistant", "content": answer,
                                         "retrieved_evidence": retrieved, "meta": chat_meta}
                                chat_messages.append(reply)
                                _render_chat_message(reply, expanded=False)
                            except api_client.ApiError as exc:
                                st.error(f"Patient record Q&A failed: {exc.detail}")


_render_care_assistant()
