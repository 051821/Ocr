def create_patient_timeline(patient_data):
    timeline = []

    timeline.append("PATIENT LONGITUDINAL MEDICAL HISTORY")
    timeline.append(
        f"Patient ID: {patient_data['patient_id']}"
    )
    timeline.append(
        f"Total Visits: {patient_data['total_visits']}"
    )
    timeline.append("")

    for index, visit in enumerate(patient_data["visits"], start=1):
        timeline.append("=" * 60)
        timeline.append(f"VISIT {index}")
        timeline.append("=" * 60)
        timeline.append(
            f"Visit Date: {visit['visit_date'] or 'Unknown'}"
        )
        timeline.append(
            f"Chief Complaint: "
            f"{visit['chief_complaint'] or 'Not documented'}"
        )
        timeline.append("")
        timeline.append("Clinical Information:")
        timeline.append(visit["medical_text"])
        timeline.append("")

    return "\n".join(timeline)