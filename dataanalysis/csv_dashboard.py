"""
csv_dashboard.py
----------------
Dashboard with smart CSV-backed AI analysis:

  Hash matches  → show cached (no LLM call)
  Hash changed, mismatch < 4 → incremental update (old summary + new visits)
  Hash changed, mismatch >= 4 → full re-analysis on all visits
  No record     → full analysis + save

Run:  streamlit run csv_dashboard.py
"""

import os, sys, csv, json, logging
from pathlib import Path

_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_DIR), _DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import streamlit as st

from analysis import fetch_patient_history, fetch_all_patient_ids
from Zai_analysis import analyze_patient_with_ai, analyze_patient_incremental
from storage.ai_analysis_csv import (
    compute_source_hash, get_patient_row, upsert_ai_analysis,
    FULL_RERUN_THRESHOLD, _DEFAULT_CSV_PATH,
)

log = logging.getLogger(__name__)

# ── page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Patient AI Analysis (CSV Cache)", page_icon="🗂️", layout="wide")
st.markdown("""
<style>
.main-header{font-size:2.1rem;font-weight:700;color:#1E3A8A;margin-bottom:.3rem}
.disclaimer-box{background:#FEF3C7;border-left:5px solid #F59E0B;padding:10px 16px;
  border-radius:4px;margin-bottom:18px;font-size:.9rem;color:#78350F}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🗂️ Patient AI Analysis — CSV Cache</div>', unsafe_allow_html=True)
st.markdown('<div class="disclaimer-box"><b>⚠️ Clinical Notice:</b> Retrospective data only. No medical advice.</div>',
            unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Patient Lookup")
    try:
        patient_ids = fetch_all_patient_ids()
    except Exception as e:
        patient_ids = []
        st.error(f"DB error: {e}")
    selected = st.selectbox("Select Patient ID", ["-- Select --"] + patient_ids, index=1) if patient_ids else "-- Select --"
    st.markdown("---")
    st.write(f"**Total patients:** {len(patient_ids)}")

# ── patient input ─────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    default = selected if selected != "-- Select --" else (patient_ids[0] if patient_ids else "")
    patient_id = st.text_input("Patient ID:", value=st.session_state.get("cpid", default)).strip()
with c2:
    st.write(""); st.write("")
    btn_fetch = st.button("🔍 Fetch", type="primary", use_container_width=True)

if not patient_id:
    st.info("Enter a patient ID to begin.")
    st.stop()

st.session_state["cpid"] = patient_id

# ── fetch DB records ──────────────────────────────────────────────────────────
if btn_fetch or st.session_state.get("cpid_loaded") != patient_id:
    with st.spinner("Fetching patient records…"):
        try:
            st.session_state["cpdata"] = fetch_patient_history(patient_id)
            st.session_state["cpid_loaded"] = patient_id
        except Exception as e:
            st.error(f"Error: {e}"); st.stop()

patient_data = st.session_state.get("cpdata")
if not patient_data or patient_data["total_visits"] == 0:
    st.warning(f"No records for `{patient_id}`."); st.stop()

# ── smart cache logic ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🤖 AI Retrospective Clinical Analysis Note")
st.caption("Based on past documented data only. No medical advice.")

current_hash   = compute_source_hash(patient_data)
existing_row   = get_patient_row(patient_id)
report         = None   # will hold {"raw_markdown":..., "model_name":..., "model_version":...}
status_msg     = ""

if existing_row is None:
    # ── NEW PATIENT: full analysis ────────────────────────────────────────────
    with st.spinner("No cached analysis — running full AI analysis…"):
        try:
            report = analyze_patient_with_ai(patient_data)
            status_msg = "✅ First analysis generated and saved."
        except Exception as e:
            st.error(f"AI failed: {e}"); st.stop()

elif existing_row["source_data_hash"] == current_hash:
    # ── HASH MATCH: show cache, no LLM call ──────────────────────────────────
    try:
        cached = json.loads(existing_row["analysis"])
        markdown_text = cached.get("raw_markdown", "")
    except Exception:
        markdown_text = existing_row.get("analysis", "")

    meta = (f"📦 Cached  ·  visits: `{existing_row['source_visit_count']}`  ·  "
            f"model: `{existing_row['model_name']}`  ·  "
            f"saved: `{existing_row['generated_at'][:19]} UTC`")
    st.caption(meta)
    st.info(markdown_text)

    c_btn, _ = st.columns([1, 3])
    with c_btn:
        if st.button("🔄 Force Refresh", type="secondary", use_container_width=True):
            with st.spinner("Running full re-analysis…"):
                try:
                    report = analyze_patient_with_ai(patient_data)
                    status_msg = "✅ Force-refreshed and saved."
                except Exception as e:
                    st.error(f"AI failed: {e}"); st.stop()
        else:
            st.stop()

else:
    # ── HASH MISMATCH: incremental or full ───────────────────────────────────
    prev_mismatch = int(existing_row.get("hash_mismatch_count", 0))
    new_mismatch  = prev_mismatch + 1
    prev_visits   = int(existing_row.get("source_visit_count", 0))

    if new_mismatch >= FULL_RERUN_THRESHOLD:
        # Full re-analysis
        with st.spinner(f"Hash changed {new_mismatch}× — running full re-analysis on all {patient_data['total_visits']} visits…"):
            try:
                report = analyze_patient_with_ai(patient_data)
                new_mismatch = 0   # reset after full run
                status_msg = f"✅ Full re-analysis done (was {prev_mismatch} mismatches). Saved."
            except Exception as e:
                st.error(f"AI failed: {e}"); st.stop()
    else:
        # Incremental update
        try:
            old_markdown = json.loads(existing_row["analysis"]).get("raw_markdown", "")
        except Exception:
            old_markdown = existing_row.get("analysis", "")

        new_visit_count = patient_data["total_visits"] - prev_visits
        with st.spinner(f"Hash changed ({new_mismatch}/{FULL_RERUN_THRESHOLD}) — incremental update for {new_visit_count} new visit(s)…"):
            try:
                report = analyze_patient_incremental(old_markdown, patient_data, prev_visits)
                status_msg = f"✅ Incremental update done (mismatch {new_mismatch}/{FULL_RERUN_THRESHOLD}). Saved."
            except Exception as e:
                st.error(f"AI failed: {e}"); st.stop()

# ── save + display (only reached when a new analysis was generated) ───────────
if report:
    try:
        upsert_ai_analysis(
            patient_data=patient_data,
            analysis={"raw_markdown": report["raw_markdown"]},
            model_name=report["model_name"],
            model_version=report.get("model_version", ""),
            mismatch_count=new_mismatch if "new_mismatch" in dir() else 0,
        )
        st.caption(status_msg)
    except Exception as csv_err:
        log.error("CSV upsert failed: %s", csv_err)
        st.caption("⚠️ CSV save failed (see logs).")

    st.info(report["raw_markdown"])

# ── metadata expander (always shown if row exists) ────────────────────────────
row = get_patient_row(patient_id)
if row:
    with st.expander("📋 Stored record metadata", expanded=False):
        cols = ["id","model_name","prompt_version","source_visit_count",
                "hash_mismatch_count","generated_at","source_data_hash"]
        st.dataframe(pd.DataFrame([{c: row.get(c,"") for c in cols}]),
                     use_container_width=True, hide_index=True)
