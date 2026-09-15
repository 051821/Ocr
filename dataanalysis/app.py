import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if DATAANALYSIS_DIR not in sys.path:
    sys.path.insert(0, DATAANALYSIS_DIR)

import streamlit as st
import pandas as pd

from analysis import fetch_patient_history
from pipeline_integration import run_patient_pipeline
from longitudinal.report_generator import generate_longitudinal_report

st.set_page_config(
    page_title="Clinical Longitudinal Analysis",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Patient Longitudinal Clinical Analysis")

# Helper functions for clean formatting
def fmt_ref(r_low, r_high):
    """Format reference range as 'low – high' or '<high' or '>low'."""
    def _clean(v):
        if v is None:
            return ""
        return str(int(v)) if v == int(v) else str(v)

    if r_low is not None and r_high is not None:
        return f"{_clean(r_low)} – {_clean(r_high)}"
    elif r_high is not None:
        return f"<{_clean(r_high)}"
    elif r_low is not None:
        return f">{_clean(r_low)}"
    return "Not documented"

def status_badge(status):
    """Return a color-coded status indicator."""
    if not status:
        return "⚪ Unknown"
    s = str(status).upper()
    if "CRITIC" in s:
        return f"🚨 {s}"
    if "HIGH" in s or "LOW" in s or "ABNORMAL" in s:
        return f"🔴 {s}"
    if "NORMAL" in s:
        return "🟢 Normal"
    return f"⚪ {status}"

# Search input
patient_id_input = st.text_input(
    "Enter Patient ID",
    value=st.session_state.get("active_patient_id", ""),
    placeholder="e.g. 6e55c6c3-059b-48c5-bbb4-02a0a1eb9b2e"
).strip()

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    btn_analyze = st.button("🔍 Analyze Patient & Lab Trends", type="primary")

if btn_analyze and patient_id_input:
    with st.spinner("Extracting clinical events, test results, reference ranges, and trends..."):
        try:
            patient_data = fetch_patient_history(patient_id_input)
            result = run_patient_pipeline(patient_id_input, generate_report=False)
            st.session_state["patient_data"] = patient_data
            st.session_state["longitudinal_result"] = result
            st.session_state["active_patient_id"] = patient_id_input
            # Clear previous report on new patient query
            st.session_state.pop("longitudinal_report", None)
        except Exception as e:
            st.error(f"Analysis failed: {e}")

patient_data = st.session_state.get("patient_data")
result = st.session_state.get("longitudinal_result")

if patient_data is not None and result is not None:
    if patient_data.get("total_visits", 0) == 0 or result.get("analysis") is None:
        st.warning(f"No records found for Patient ID: {st.session_state.get('active_patient_id')}")
    else:
        analysis = result["analysis"]
        obs_start = analysis['observation_period']['start'][:10] if analysis['observation_period']['start'] else "N/A"
        obs_end = analysis['observation_period']['end'][:10] if analysis['observation_period']['end'] else "N/A"

        # Summary KPIs
        st.markdown("---")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Visits", patient_data["total_visits"])
        kpi2.metric("Observation Period", f"{obs_start} → {obs_end}")
        kpi3.metric("Lab Tests Identified", len(analysis.get("lab_trends", {})))
        kpi4.metric("Medications Documented", len(analysis.get("medication_longitudinal", {})))
        kpi5.metric("Clinical Events Extracted", analysis.get("total_events_after_dedup", 0))

        # =====================================================================
        # 1. LABORATORY FINDINGS & LONGITUDINAL TRENDS
        # =====================================================================
        st.markdown("### 🧪 Laboratory Findings & Longitudinal Trends")
        lab_trends = analysis.get("lab_trends", {})

        if not lab_trends:
            st.info("No laboratory test results found in the clinical records for this patient.")
        else:
            lab_rows = []
            for test_name, trend in lab_trends.items():
                display_name = trend.get("name_raw") or test_name
                if display_name != test_name:
                    display_name = f"{display_name} ({test_name})"

                val = trend.get("latest_value")
                unit = (trend.get("unit") or "").strip()
                r_low = trend.get("ref_low")
                r_high = trend.get("ref_high")
                ref_str = fmt_ref(r_low, r_high)
                if unit and ref_str != "Not documented":
                    ref_str = f"{ref_str} {unit}"

                st_val = trend.get("latest_status") or ("ABNORMAL" if trend.get("num_abnormal", 0) > 0 else "NORMAL")
                badge = status_badge(st_val)

                direction = trend.get("direction", "single_visit")
                trend_desc = trend.get("trend_summary", "")
                if direction == "single_visit":
                    trend_label = f"Baseline ({trend.get('latest_date', '')[:10]})"
                elif direction in ("improving", "worsening", "stable", "gaining", "dropping"):
                    arrow = "📈" if (trend.get("absolute_change") or 0) > 0 else "📉" if (trend.get("absolute_change") or 0) < 0 else "➡️"
                    pct = f" ({trend['percentage_change']}%)" if trend.get("percentage_change") is not None else ""
                    trend_label = f"{arrow} {direction.upper()}{pct}"
                elif direction == "same_date_readings":
                    trend_label = "Multiple same-date readings"
                else:
                    trend_label = direction

                lab_rows.append({
                    "Investigation / Test": display_name,
                    "Latest Result": f"{val} {unit}".strip() if val is not None else "N/A",
                    "Bio. Ref. Interval": ref_str,
                    "Status": badge,
                    "Trend / Pattern": trend_label,
                    "Visits": trend.get("num_visits", 1),
                    "Latest Date": trend.get("latest_date", "")[:10] if trend.get("latest_date") else "N/A",
                })

            df_labs = pd.DataFrame(lab_rows)
            st.dataframe(df_labs, use_container_width=True, hide_index=True)

            # Detailed trend inspection
            with st.expander("🔍 View Detailed Test Trajectory & History"):
                for test_name, trend in lab_trends.items():
                    r_low, r_high = trend.get("ref_low"), trend.get("ref_high")
                    ref_info = fmt_ref(r_low, r_high)
                    u = trend.get("unit", "")
                    st.markdown(f"#### **{trend.get('name_raw') or test_name}** (`{test_name}`)")
                    st.caption(f"Reference Interval: **{ref_info} {u}** | Status: {status_badge(trend.get('latest_status'))} | Summary: {trend.get('trend_summary')}")

                    points = trend.get("all_points", [])
                    if len(points) > 1:
                        p_rows = []
                        for p in points:
                            p_rows.append({
                                "Date": p.get("date", "")[:10],
                                "Value": f"{p.get('value')} {p.get('unit', '')}".strip(),
                                "Status": status_badge(p.get("status")),
                            })
                        st.table(pd.DataFrame(p_rows))
                    st.markdown("---")

        # =====================================================================
        # 2. VISIT-BY-VISIT CLINICAL RECORDS & EXTRACTED FINDINGS
        # =====================================================================
        st.markdown("---")
        st.markdown("### 📋 Visit-by-Visit Medical Records & Extracted Findings")

        # Map timeline visits by visit_id for direct association
        timeline_visits = {v["visit_id"]: v for v in analysis.get("timeline", {}).get("visits", [])}

        for index, visit in enumerate(patient_data["visits"], start=1):
            v_id = visit["visit_id"]
            v_date = visit.get("visit_date")
            date_disp = v_date[:10] if v_date else "Unknown Date"
            v_nlp = timeline_visits.get(v_id, {})
            v_labs = v_nlp.get("laboratory", [])
            v_meds = v_nlp.get("medications", [])
            v_dx = v_nlp.get("diagnoses", [])

            with st.expander(f"Visit {index} | {date_disp} (ID: {v_id[:8]}...)", expanded=(index <= 2)):
                st.write("**Provisional Diagnosis:**", visit.get("provisional_diagnosis") or "Not documented")
                st.write("**Confirmed Diagnosis:**", visit.get("confirmed_diagnosis") or "Not documented")

                # Show Extracted Labs for this specific visit
                if v_labs:
                    st.markdown("##### 🔬 Extracted Laboratory Test Results:")
                    v_lab_rows = []
                    for l in v_labs:
                        disp = l.get("name_raw") or l.get("name")
                        val = l.get("value")
                        u = (l.get("unit") or "").strip()
                        r_lo = l.get("reference_low")
                        r_hi = l.get("reference_high")
                        ref_str = fmt_ref(r_lo, r_hi)
                        if u and ref_str != "Not documented":
                            ref_str = f"{ref_str} {u}"

                        st_flag = "ABNORMAL" if l.get("abnormal") else "NORMAL"
                        badge = status_badge(st_flag)

                        v_lab_rows.append({
                            "Investigation": disp,
                            "Result": f"{val} {u}".strip() if val is not None else str(l.get("qualitative_value")),
                            "Bio. Ref. Interval": ref_str,
                            "Status": badge,
                        })
                    st.table(pd.DataFrame(v_lab_rows))

                # Show Extracted Medications for this specific visit
                if v_meds:
                    st.markdown("##### 💊 Extracted Medications Prescribed:")
                    med_bullets = []
                    for m in v_meds:
                        name = m.get("name") or m.get("name_raw")
                        raw = m.get("name_raw")
                        if raw and str(raw).strip().lower() != str(name).strip().lower():
                            name = f"{name} ({raw})"
                        dose_val = m.get("dose")
                        if dose_val is not None:
                            if isinstance(dose_val, float) and dose_val == int(dose_val):
                                dose_val = int(dose_val)
                            unit = (m.get("unit") or "").strip()
                            dose = f"{dose_val} {unit}".strip()
                        else:
                            dose = ""
                        freq = f"({m.get('frequency')})" if m.get("frequency") else ""
                        med_bullets.append(f"- **{name}** {dose} {freq}".strip())
                    st.markdown("\n".join(med_bullets))

                with st.expander("📄 View Raw Extracted OCR Document Text"):
                    st.text(visit.get("medical_text") or "No clinical text available.")

        # =====================================================================
        # 3. MEDICATION LONGITUDINAL HISTORY
        # =====================================================================
        st.markdown("---")
        st.markdown("### 💊 Medication Longitudinal History")
        med_history = analysis.get("medication_longitudinal", {})

        if not med_history:
            st.info("No medications tracked across visits.")
        else:
            for name, info in med_history.items():
                flag = " ⚠️ Needs Verification" if info.get("any_needs_verification") else ""
                st.write(
                    f"- **{name}** — `{info['status'].upper()}`{flag} "
                    f"({info['num_visits_documented']} visits: {info['first_visit_date'][:10]} → {info['latest_visit_date'][:10]})"
                )
                for change in info.get("changes", []):
                    st.caption(f"    ↳ {change['type']}: {change['from']} → {change['to']} on {change['visit_date'][:10]}")

        # =====================================================================
        # 4. CHRONIC CONDITIONS & DIAGNOSES
        # =====================================================================
        st.markdown("---")
        st.markdown("### 🩺 Chronic Conditions & Diagnoses")
        dx_history = analysis.get("diagnosis_longitudinal", {})

        if not dx_history:
            st.info("No diagnoses documented across visits.")
        else:
            for name, info in dx_history.items():
                st.write(
                    f"- **{name}** — `{info['status'].upper()}` "
                    f"({info['num_visits_documented']} of {info['total_visits_in_record']} visits, "
                    f"{info['first_documented'][:10]} → {info['last_documented'][:10]})"
                )

        # =====================================================================
        # 5. UNCERTAIN FINDINGS
        # =====================================================================
        uncertain = analysis.get("uncertain_findings", {})
        if any(uncertain.values()):
            st.markdown("---")
            st.markdown("### ⚠️ AI-Detected Findings Needing Verification")
            st.caption("Extracted by the biomedical NER model — confirm these against the original document.")

            if uncertain.get("medication"):
                st.write("**Possible Medications:**")
                for item in uncertain["medication"]:
                    st.write(f"- `{item['name_raw']}` → *{item['name']}* (Confidence: {item['confidence']}): \"{item['source_text']}\"")

            if uncertain.get("diagnosis"):
                st.write("**Possible Diagnoses:**")
                for item in uncertain["diagnosis"]:
                    st.write(f"- `{item['name_raw']}` → *{item['name']}* (Confidence: {item['confidence']}): \"{item['source_text']}\"")

        # =====================================================================
        # 6. AI NARRATIVE CLINICAL REPORT
        # =====================================================================
        st.markdown("---")
        st.markdown("### 🤖 Longitudinal Clinical Narrative Report (LLM)")

        if st.button("Generate Narrative Report via Ollama", type="secondary"):
            with st.spinner("Synthesizing longitudinal narrative report with lab trends..."):
                try:
                    report_text = generate_longitudinal_report(analysis)
                    st.session_state["longitudinal_report"] = report_text
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

        if "longitudinal_report" in st.session_state:
            st.markdown("#### Generated Clinical Report")
            st.markdown(st.session_state["longitudinal_report"])

        # Debug: Raw payload
        with st.expander("🛠️ View Raw Structured Analysis Payload (JSON)"):
            st.json(analysis)