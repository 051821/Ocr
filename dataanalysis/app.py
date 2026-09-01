import streamlit as st

from analysis import fetch_patient_history
from patient_timeline import create_patient_timeline


st.set_page_config(
    page_title="Patient Analysis",
    layout="wide"
)

st.title("Patient Longitudinal Analysis")

patient_id = st.text_input(
    "Enter Patient ID"
).strip()

if st.button("Get Patient History") and patient_id:

    with st.spinner("Fetching patient history..."):
        patient_data = fetch_patient_history(patient_id)

    if patient_data["total_visits"] == 0:
        st.warning("No records found for this patient.")

    else:
        st.success(
            f"Found {patient_data['total_visits']} visits"
        )

        for index, visit in enumerate(
            patient_data["visits"],
            start=1
        ):
            with st.expander(
                f"Visit {index} | "
                f"{visit['visit_date'] or 'Unknown Date'}"
            ):
                st.write(
                    "**Chief Complaint:**",
                    visit["chief_complaint"] or "Not documented"
                )

                st.write("**Clinical Information:**")
                st.text(visit["medical_text"])

        st.divider()

        if st.button("View AI Input"):
            timeline = create_patient_timeline(patient_data)
            st.code(timeline)