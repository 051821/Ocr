# ---------------------------------------------------------------------------
# ADD to app.py — new section below your existing "View AI Analysis" button.
# This calls the new Stage 5-16 pipeline instead of the old single-shot
# summarizer, and renders the structured lab trends / medication history /
# diagnosis history alongside the LLM narrative report.
# ---------------------------------------------------------------------------

from pipeline_integration import run_patient_pipeline
import streamlit as st
st.divider()
st.subheader("Longitudinal Clinical Analysis")

if st.button("Run Longitudinal Clinical NLP Analysis") and patient_id:
    with st.spinner("Extracting clinical events, deduplicating, building timeline..."):
        result = run_patient_pipeline(patient_id, generate_report=False)
    # persist across reruns instead of a local variable
    st.session_state["longitudinal_result"] = result
    st.session_state["longitudinal_patient_id"] = patient_id

# read from session_state, not from the block above, so it survives reruns
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

        st.markdown("### Laboratory Trends")
        for name, trend in analysis["lab_trends"].items():
            st.write(
                f"- **{name}**: {trend['first_value']} ({trend['first_date']}) → "
                f"{trend['latest_value']} ({trend['latest_date']}) "
                f"— {trend['direction']}"
                + (f", {trend['percentage_change']}%" if trend['percentage_change'] is not None else "")
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

        if analysis["medication_uncertainty"]:
            st.markdown("### ⚠️ Extractions Needing Verification")
            for item in analysis["medication_uncertainty"]:
                st.write(f"- `{item['name_raw']}` (visit {item['visit_date']}): "
                         f"\"{item['source_text']}\"")

        with st.expander("Raw structured payload (for the LLM report)"):
            st.json(analysis)

        # this button is now a sibling, not nested inside the first `if`,
        # and reads `analysis` from session_state — so it survives the rerun
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