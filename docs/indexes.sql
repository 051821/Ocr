-- Apply manually after reviewing the schema and query plan.
CREATE INDEX IF NOT EXISTS idx_visit_patient_id
    ON visit (patient_id);

CREATE INDEX IF NOT EXISTS idx_document_extraction_visit_id
    ON document_extraction (visit_id);

CREATE INDEX IF NOT EXISTS idx_prescription_visit_id
    ON prescription (visit_id);

CREATE INDEX IF NOT EXISTS idx_prescriptionitem_prescription_id
    ON prescriptionitem (prescription_id);

CREATE INDEX IF NOT EXISTS idx_followup_schedule_source_prescription_id
    ON followup_schedule (source_prescription_id);
