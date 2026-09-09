import streamlit as st

from analysis import fetch_patient_history
from Zai_analysis import analyze_patient_with_ai

st.set_page_config(
    page_title="Patient Analysis",
    layout="wide"
)

st.title("Patient Longitudinal Analysis")

patient_id = st.text_input("Enter Patient ID").strip()

if st.button("Get Patient History") and patient_id:
    with st.spinner("Fetching patient history..."):
        patient_data = fetch_patient_history(patient_id)

    if patient_data["total_visits"] == 0:
        st.warning("No records found for this patient.")
    else:
        st.success(f"Found {patient_data['total_visits']} visits")

        for index, visit in enumerate(patient_data["visits"], start=1):
            with st.expander(f"Visit {index} | {visit['visit_date'] or 'Unknown Date'}"):
                st.write("**provisional_diagnosis:**", visit.get("provisional_diagnosis") or "Not documented")
                st.write("**confirmed_diagnosis:**", visit.get("confirmed_diagnosis") or "Not documented")
                st.write("**Clinical Information:**")
                st.text(visit.get("medical_text") or "No clinical information available.")
                st.divider()

        if st.button("View AI Analysis"):
            with st.spinner("Generating AI analysis..."):
                ai_output = analyze_patient_with_ai(patient_data)
            st.code(ai_output)

from pipeline_integration import run_patient_pipeline
st.divider()
st.subheader("Longitudinal Clinical Analysis")

if st.button("Run Longitudinal Clinical NLP Analysis") and patient_id:
    with st.spinner("Extracting clinical events, deduplicating, building timeline..."):
        result = run_patient_pipeline(patient_id, generate_report=False)
    st.session_state["longitudinal_result"] = result
    st.session_state["longitudinal_patient_id"] = patient_id

result = st.session_state.get("longitudinal_result")

if result is not None:
    if result["analysis"] is None:
        st.warning("No records found for this patient.")
    else:
        analysis = result["analysis"]

        st.write(
            f"**Observation period:** {analysis['observation_period']['start']} "
            f"→ {analysis['observation_period']['end']}"
        )
        st.caption(
            f"{analysis['total_events_extracted']} events extracted, "
            f"{analysis['total_events_after_dedup']} after deduplication."
        )

        st.markdown("### Chronic Conditions")
        for name, info in analysis["diagnosis_longitudinal"].items():
            st.write(f"- **{name}** — {info['status']} "
                      f"({info['num_visits_documented']} of {info['total_visits_in_record']} visits, "
                      f"{info['first_documented']} → {info['last_documented']})")

        st.markdown("### Laboratory Trends & Findings")

        def _fmt_ref(v):
            """Show ref bound as int if whole number, else float."""
            if v is None:
                return ""
            return str(int(v)) if v == int(v) else str(v)

        for name, trend in analysis["lab_trends"].items():
            unit = (trend.get("unit") or "").upper()
            r_low = trend.get("ref_low")
            r_high = trend.get("ref_high")
            if r_low is not None and r_high is not None:
                ref_str = f" (Ref: {_fmt_ref(r_low)} – {_fmt_ref(r_high)})"
            elif r_high is not None:
                ref_str = f" (Ref: <{_fmt_ref(r_high)})"
            elif r_low is not None:
                ref_str = f" (Ref: >{_fmt_ref(r_low)})"
            else:
                ref_str = ""

            is_abnormal = trend.get("num_abnormal", 0) > 0
            flag = " 🔴 **Abnormal**" if is_abnormal else " 🟢 Normal"
            date_str = trend.get("latest_date", "")[:10] if trend.get("latest_date") else "Visit"

            if trend["direction"] == "qualitative_observation":
                st.write(f"- **{name}**: `{trend['latest_value']}`{ref_str}{flag} ({date_str})")
            elif trend["direction"] in ("gain", "drop", "increasing", "decreasing", "stable"):
                d1 = trend.get("first_date", "")[:10]
                d2 = trend.get("latest_date", "")[:10]
                pct_str = f" ({trend['percentage_change']}%)" if trend.get("percentage_change") is not None else ""
                arrow = "📈" if trend["direction"] in ("gain", "increasing") else "📉"
                st.write(
                    f"- **{name}**: {trend['first_value']} {unit} ({d1}) → "
                    f"**{trend['latest_value']} {unit}** ({d2}) "
                    f"{arrow} **{trend['direction'].upper()}**{pct_str}{flag}"
                )
            else:  # single_visit / baseline reading
                val_disp = trend['latest_value']
                st.write(
                    f"- **{name}**: **{val_disp}** {unit}{ref_str} — {flag} ({date_str})"
                )



        st.markdown("### Medication History")
        for name, info in analysis["medication_longitudinal"].items():
            flag = " ⚠️ needs verification" if info["any_needs_verification"] else ""
            st.write(f"- **{name}** — {info['status']}{flag} "
                      f"({info['num_visits_documented']} visits, "
                      f"{info['first_visit_date']} → {info['latest_visit_date']})")
            for change in info["changes"]:
                st.caption(f"    {change['type']}: {change['from']} → {change['to']} "
                           f"on {change['visit_date']}")

        uncertain = analysis["uncertain_findings"]
        if any(uncertain.values()):
            st.markdown("### ⚠️ AI-Detected Findings Needing Verification")
            st.caption("Caught by the biomedical NER model, not the dictionary-based "
                       "extractor — confirm these against the source document.")

            if uncertain["medication"]:
                st.write("**Possible medications:**")
                for item in uncertain["medication"]:
                    st.write(f"- `{item['name_raw']}` → *{item['name']}* "
                             f"(confidence {item['confidence']}, visit {item['visit_date']}): "
                             f"\"{item['source_text']}\"")

            if uncertain["diagnosis"]:
                st.write("**Possible diagnoses:**")
                for item in uncertain["diagnosis"]:
                    st.write(f"- `{item['name_raw']}` → *{item['name']}* "
                             f"(confidence {item['confidence']}, visit {item['visit_date']}): "
                             f"\"{item['source_text']}\"")

            if uncertain["symptom"]:
                st.write("**Possible symptoms:**")
                for item in uncertain["symptom"]:
                    st.write(f"- *{item['name']}* "
                             f"(confidence {item['confidence']}, visit {item['visit_date']}): "
                             f"\"{item['source_text']}\"")

        with st.expander("Raw structured payload (for the LLM report)"):
            st.json(analysis)

        if st.button("Generate Narrative Report (LLM)"):
            with st.spinner("Generating longitudinal report..."):
                try:
                    from longitudinal.report_generator import generate_longitudinal_report
                    report_text = generate_longitudinal_report(analysis)
                    st.session_state["longitudinal_report"] = report_text
                except RuntimeError as e:
                    st.error(str(e))

        if "longitudinal_report" in st.session_state:
            st.markdown("### Longitudinal Patient Report")
            st.markdown(st.session_state["longitudinal_report"])