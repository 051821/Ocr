Extract facts explicitly present in this clinical document. Do not diagnose or infer. Preserve medicine, diagnosis, lab, and value strings as written; use null or [] when absent. Return only JSON with keys: age_sex, chief_complaint, symptoms, clinical_impression, provisional_diagnosis, confirmed_diagnosis, vitals[{{label,value}}], medications[{{name,strength,frequency,duration}}], lab_results[{{test_name,value,reference}}], follow_up, clinical_notes.

DOCUMENT:
{document}
