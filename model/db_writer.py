"""
model/db_writer.py
"""
import csv
import json
import os
import re
import uuid

import psycopg2
import psycopg2.extras
import psycopg2.pool

import config


def ensure_clean_text_column(conn):
    """Checks if the clean_text column exists in the document_extraction table,
    and adds it as JSONB if missing."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name = 'document_extraction' 
                      AND column_name = 'clean_text'
                )
                """
            )
            exists = cur.fetchone()[0]
            if not exists:
                print("[db] Adding missing column 'clean_text' (JSONB) to table 'document_extraction'...")
                cur.execute("ALTER TABLE document_extraction ADD COLUMN clean_text JSONB;")
                conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[db] Warning during schema check/update: {e}")


def split_text_lines(text_lines):
    if not text_lines:
        return [], []

    if isinstance(text_lines, str):
        text_lines = text_lines.splitlines()

    pii_lines = []
    medical_lines = []
    in_lab_section = False
    expect_age_sex_value = False

    def has_non_english_letters(value):
        return any(char.isalpha() and not char.isascii() for char in value)

    # Standard PII patterns
    phone_pattern = r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b|\b\d{3,5}-\d{6,8}\b'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    url_pattern = r'\b(?:www\.|https?://)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    pincode_pattern = r'\b\d{6}\b|\b\d{3}\s*\d{3}\b'
    
    # Dates: numeric (11/03/2026) and textual (Mar 11, 2026 or 11-Mar-2026)
    date_regex = (
        r'\b\d{1,2}[-/\s.]\d{1,2}[-/\s.]\d{2,4}\b|'
        r'\b\d{4}[-/\s.]\d{1,2}[-/\s.]\d{1,2}\b|'
        r'\b(?:\d{1,2}[-/\s.]?)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[-/\s.]\d{1,2}(?:st|nd|rd|th)?(?:[-/\s,.]+\d{2,4})?\b'
    )

    honorific_pattern = r'\b(?i:mr|mrs|ms|miss|master|dr|prof|sr)\b\.?\s+[A-Za-z]'
    relation_prefix_pattern = r'\b(?i:s|d|w|c)\s*[./]\s*o\b\.?'

    # Doctor credentials and titles
    doctor_credential_pattern = (
        r'\b(?i:mbbs|bams|bhms|bds|dnb|dgo|dch|mrcp|frcs|m\.?d\.?|m\.?s\.?|d\.?m\.?|m\.?ch\.?|diploma|fellowship)\b|'
        r'\b(?i:reg(?:n)?\.?\s*(?:no\.?|number)?\s*:?|bmc\s*\d+|mci\s*\d+|nmc\s*\d+|council\s*reg)\b|'
        r'\b(?i:general\s+md\s+medicine|general\s+medicine|consultant\s+physician|medical\s+officer|treating\s+doctor)\b'
    )

    # Organization / facility / browser artifacts
    org_artifact_pattern = (
        r'\b(?i:foundation|hospital|clinic|dispensary|trust|society|ngo|charitable|health\s*care|'
        r'health\s*centre|nursing\s+home|polyclinic|diagnostics|institute|college|digiswasthya)\b|'
        r'\b(?i:out:blank|about:blank)\b|'
        r'^\s*(?i:page\s*)?\d+\s*(?:/|of)\s*\d+\s*$'
    )

    
    admin_label_pattern = (
        r'(?i)\b(?:patient|patent|pautent|atient)\s*(?:id|name)\b|'
        r'\b(?:sample|specimen)\s*(?:coll(?:ection)?|col1|co11)\s*(?:date|time)?\b|'
        r'\b(?:ref(?:erred)?|ree)\s*[:.-]?\s*(?:by)?\s*doctor\b|'
        r'\b(?:facility|fachlity|facil(?:ity)?)\s*name\b|'
        r'\b(?:reg(?:istration)?|report)\s*(?:date|time|date\s*/\s*time)\b'
    )

    # Administrative and demographic metadata
    pii_keywords = [
        'patient name', 'guardian name', 'father name', 'husband name', 'wife name',
        'daughter name', 'son name', 'patient address', 'patient mobile', 'patient contact',
        'contact number', 'contact no', 'mobile no', 'phone no', 'phone number', 'email id',
        'ref. physician', 'referred by', 'reported by', 'tested by', 'date of collection',
        'date tested', 'date reported', 'date collected', 'hospital reg', 'registration no',
        'reg no', 'reg. no', 'episode id', 'nikshay', 'establishment id', 'laboratory name', 'lab name',
        'signature', 'signed by', 'other contact', 'occupation', 'prescription id', 'prescription no',
        'uhid', 'patient id', 'token no', 'bill no', 'receipt no','patient id', 'REF BY DOCTOR : DR','name'
    ]

    standalone_keywords = [
        'landmark', 'address', 'pincode', 'district', 'state', 'uhid', 'opd', 'ipd',
        'physician', 'referred', 'consultant', 'radiologist', 'guardian',
        'pathologist', 'clinic', 'hospital', 'laboratory', 'institute', 'college'
    ]

    demographic_pattern = (
        r'\b(?i:age\s*[/,&-]?\s*gender|age\s*[/,&-]?\s*sex|gender\s*[/,&-]?\s*age)\b|'
        r'\b\d{1,3}\s*(?i:yrs?|years?|y)?\s*[/|-]\s*(?i:male|female|m|f)\b|'
        r'\b(?i:male|female)\s*[/|-]\s*\d{1,3}\b|'
        r'^\s*(?i:age\s*:\s*\d+)\s*$|'
        r'^\s*(?i:gender|sex)\s*:\s*(?i:male|female|m|f)\s*$'
    )

    # Comprehensive medical vocabulary for vocabulary check & protection
    medical_vocab = {
        # Dosages, units, forms, frequencies
        'tab', 'tabs', 'tablet', 'tablets', 'cap', 'caps', 'capsule', 'capsules',
        'syr', 'syrup', 'inj', 'injection', 'oint', 'ointment', 'drops', 'drop',
        'susp', 'suspension', 'gel', 'cream', 'lotion', 'powder', 'inhaler', 'respules',
        'sachet', 'mg', 'gm', 'g', 'mcg', 'ug', 'ml', 'iu', 'units', 'bpm', 'mmhg',
        'od', 'bd', 'bid', 'tds', 'tid', 'qid', 'hs', 'sos', 'bbf', 'ac', 'pc',
        'po', 'prn', 'stat', 'q4h', 'q6h', 'q8h', 'q12h', 'daily', 'weekly', 'bedtime',
        'morning', 'evening', 'night', 'empty', 'stomach', 'food',

        # Diagnoses, conditions, symptoms
        'htn', 't2dm', 't1dm', 'dm', 'hypertension', 'diabetes', 'diabetic', 'daibetic',
        'asthma', 'copd', 'cad', 'ckd', 'gerd', 'fever', 'cough', 'cold', 'pain', 'ache',
        'knee', 'neck', 'spine', 'joint', 'shoulder', 'leg', 'arm', 'hand', 'foot',
        'head', 'chest', 'back', 'abdomen', 'throat', 'eye', 'ear', 'skin', 'headache',
        'swelling', 'edema', 'infection', 'allergy', 'gastritis', 'ulcer', 'calculus',
        'stones', 'arthritis', 'osteoarthritis', 'weakness', 'vomiting', 'nausea',
        'rash', 'breathlessness', 'dyspnea', 'cs', 'ls', 'cervical', 'lumbar', 'thoracic',

        # Clinical section headers & advice
        'rx', 'dx', 'hx', 'tx', 'diagnosis', 'clinical', 'notes', 'chief', 'complaint',
        'recom', 'recommendation', 'advice', 'diet', 'salt', 'low', 'monitoring',
        'investigation', 'investigations', 'blood', 'test', 'tests', 'urine', 'serum',
        'cbc', 'rbs', 'fbs', 'ppbs', 'kft', 'lft', 'hba1c', 'ecg', 'xray', 'x-ray',
        'usg', 'ultrasound', 'mri', 'ct', 'scan', 'vitals', 'pulse', 'spo2', 'temp',
        'temperature', 'bp', 'medicine', 'instruction', 'frequency', 'dose', 'dosage',
        'duration', 'days', 'weeks', 'months', 'review', 'follow-up', 'followup',

        # Common report findings and analytes.  These are deliberately kept
        # separate from LAB_VOCAB: the latter describes table headings, while
        # these appear in the individual result rows OCR returns.
        'haemoglobin', 'hemoglobin', 'hb', 'rbc', 'wbc', 'tlc', 'dlc', 'esr',
        'platelet', 'platelets', 'pcv', 'hematocrit', 'mcv', 'mch', 'mchc',
        'colour', 'color', 'appearance', 'clarity', 'gravity', 'ph', 'ketone',
        'bacteria', 'crystal', 'crystals', 'pus', 'epithelial', 'cells',
        'microscopic', 'physical', 'chemical', 'quantity', 'volume',
        'glucose', 'sugar', 'urea', 'creatinine', 'uric', 'bilirubin',
        'albumin', 'globulin', 'protein', 'cholesterol', 'triglycerides',
        'hdl', 'ldl', 'vldl', 'sodium', 'potassium', 'calcium', 'chloride',
        'tsh', 'thyroid', 'sgot', 'sgpt', 'alp', 'crp', 'vitamin', 'b12',
        'culture', 'sensitivity', 'positive', 'negative', 'reactive',
        'nonreactive', 'impression', 'assessment', 'provisional', 'findings',
        'history', 'examination', 'palpitation', 'dizziness', 'diarrhea',
        'constipation', 'fatigue', 'anemia', 'anaemia', 'jaundice',

        # Common medications
        'amlodipine', 'telmisartan', 'metformin', 'ecosprin', 'ecopsrin', 'tramadol',
        'calcium', 'd3', 'methocobalmin', 'methylcobalamin', 'paracetamol', 'pantoprazole',
        'omeprazole', 'rabeprazole', 'atorvastatin', 'rosuvastatin', 'azithromycin',
        'amoxicillin', 'clav', 'cefixime', 'ciprofloxacin', 'cetirizine', 'levocetirizine',
        'montelukast', 'ibuprofen', 'diclofenac', 'aceclofenac', 'ranitidine', 'domperidone',
        'ondansetron', 'multivitamin', 'complex', 'zinc', 'iron', 'folic', 'acid',
        'insulin', 'glimepiride', 'vildagliptin', 'teneligliptin', 'dapagliflozin',
        'empagliflozin', 'losartan', 'enalapril', 'ramipril', 'atenolol', 'metoprolol',
        'propranolol', 'clopidogrel', 'heparin', 'warfarin', 'doxycycline', 'metronidazole',
        'salbutamol', 'budesonide', 'formoterol', 'fluticasone', 'levothyroxine', 'thyroxine',
        'prednisolone', 'dexamethasone', 'av'
    }

    # Add config lab vocab
    medical_vocab.update(
        word.lower()
        for phrase in config.LAB_VOCAB
        for word in re.findall(r'[A-Za-z]+', phrase)
    )

    def extract_age_gender(line):
        age_sex = re.search(
            r'(?i)\b(?:age\s*[/,&-]?\s*(?:gender|sex)?\s*[:.-]?\s*)?'
            r'(\d{1,3})\s*(?:yrs?|years?|y)?(?:\s*[/|-]\s*|\s+)'
            # ``N`` is a frequent OCR substitution for the printed ``M``.
            r'(male|female|ml|fl|m|f|n)\b',
            line,
        )
        if age_sex:
            age, gender = age_sex.groups()
            return f"Age/Sex: {age}/{'F' if gender.lower().startswith('f') else 'M'}"

        age = re.search(r'(?i)\bage\s*[:/.,-]?\s*(\d{1,3})\b', line)
        gender = re.search(r'(?i)\b(?:gender|sex)\s*[:/.,-]?\s*(male|female|ml|fl|m|f|n)\b', line)
        if age and gender:
            sex = 'F' if gender.group(1).lower().startswith('f') else 'M'
            return f"Age/Sex: {age.group(1)}/{sex}"
        if age:
            return f"Age: {age.group(1)}"
        if gender:
            sex = 'F' if gender.group(1).lower().startswith('f') else 'M'
            return f"Sex: {sex}"
        return None

    def has_medical_content(line):
        """Whether a non-metadata line is suitable for clean_text.

        clean_text is an allow-list: unrecognised OCR must stay in
        extracted_text rather than being treated as clinical information.
        """
        words = set(re.findall(r'[a-z]+', line.lower()))
        if words & medical_vocab:
            return True
        if is_lab_section_signal(line):
            return True
        return bool(re.search(r'(?i)\b(?:mmol\s*/\s*l|meq\s*/\s*l|µ?mol\s*/\s*l|'
                              r'u?g\s*/\s*(?:dl|ml)|mg\s*/\s*(?:dl|l)|'
                              r'g\s*/\s*(?:dl|l)|%|fl|pg)\b', line))

    def is_lab_section_signal(line):
        return bool(re.search(
            r'(?i)\b(?:investigation|biochemistry|haematology|hematology|'
            r'pathology|result|units?|reference|ref\.?\s*interval|'
            r'interpretation|sample\s*type|meth(?:od|os|oe|ad)|serum|plasma|urine|'
            r'urea(?:se)?|gldh)\b',
            line,
        ))

    def is_lab_value_or_range(line):
        """Match an OCR cell containing only a lab result or reference range."""
        if re.match(
            r'^\s*[:\uff1a]?\s*(?:absent|present|positive|negative|nil|trace|'
            r'clear|turbid|yellow|pale\s+yellow|normal|abnormal|reactive|'
            r'non[-\s]?reactive|few|moderate|many|none|/\s*hpf|/\s*lpf|'
            r'(?:mg|g|ml|mmol|meq)\s*/\s*(?:dl|l)|ml)\s*$',
            line,
            re.IGNORECASE,
        ):
            return True
        return bool(re.match(
            r'^\s*[:：]?\s*(?:[<>≤≥]\s*)?\d+(?:\.\d+)?'
            r'(?:\s*(?:-|–|—|to)\s*(?:[<>≤≥]\s*)?\d*(?:\.\d+)?)?\s*$',
            line,
            re.IGNORECASE,
        ))

    def is_clearly_medical(line):
        """Returns True if the line contains unequivocal medical signals."""
        lower = line.lower()
        # Strength / dosage pattern (e.g. 5 mg, 500 mg, 1500 ug)
        if re.search(r'\b\d+(?:\.\d+)?\s*(?:mg|gm|g|mcg|ug|ml|iu|units?)\b', lower):
            return True
        # Medicine dosage form prefix
        if re.search(r'\b(?:tab(?:\.|\b)|cap(?:\.|\b)|syr(?:\.|\b)|inj(?:\.|\b)|oint(?:\.|\b)|drops?|susp(?:\.|\b)|cream|gel|lotion)\b', lower):
            return True
        # Core medical diagnosis / clinical terms
        if re.search(r'\b(?:htn|t2dm|t1dm|dm|hypertension|diabetes|daibetic|diabetic|knee\s+pain|neck\s+pain|chest\s+pain|back\s+pain|clinical\s+notes?|diagnosis|recom(?::|\b)|blood\s+test|lipid\s+profile|cbc|rbs|fbs|ppbs|kft|lft|hba1c|ecg|xray|x-ray|cs\s+spine|bp\s+monitoring|low\s+salt|ha?emoglobin|\bhb\b|rbc|wbc|tlc|dlc|platelets?|hematocrit|mcv|mchc?|glucose|creatinine|bilirubin|cholesterol|triglycerides?|\b(?:hdl|ldl|vldl|tsh|sgot|sgpt|crp)\b|uric\s+acid|electrolytes?|culture\s*(?:&|and)?\s*sensitivity|impression|provisional\s+diagnosis|clinical\s+findings?)\b', lower):
            return True
        # Table headers in Rx section
        if lower in ('medicine', 'instruction', 'frequency', 'rx', 'diagnosis', 'clinical notes', 'investigations', 'blood test'):
            return True
        return False

    def looks_like_bare_name(s):
        """Identify an unlabeled patient/doctor name on its own line.
        Never flags lines that contain medical terms or dosage digits."""
        if is_lab_section_signal(s):
            return False
        tokens = s.split()
        if not (2 <= len(tokens) <= 4):
            return False
        for tok in tokens:
            core = re.sub(r'[^A-Za-z]', '', tok)
            if not core:
                return False
            if core.lower() in medical_vocab:
                return False
            # Must look like a capitalized human name
            if not ((len(core) == 1 and core.isupper()) or core.isupper() or (core[0].isupper() and core[1:].islower())):
                return False
        return True

    for line in text_lines:
        line_str = str(line).strip().replace(chr(0xFF1A), ':')
        if not line_str:
            continue

        if line_str in ("```", "'''"):
            continue

        line_str = re.sub(r'\[ILLEGIBLE\]', '', line_str, flags=re.IGNORECASE).strip()
        if not line_str:
            continue

        # Keep non-English OCR text for reference in extracted_text (clean_text stays English)
        if has_non_english_letters(line_str):
            pii_lines.append(line_str)
            continue

        # Ignore / filter lines consisting solely of tick marks, checkmarks, dashes, or punctuation
        if not re.search(r'[A-Za-z0-9]', line_str):
            pii_lines.append(line_str)
            continue

        line_lower = line_str.lower()
        is_pii = False

        if is_lab_section_signal(line_str):
            in_lab_section = True

        # Check for pure metadata date/timestamp lines (e.g. "Mar 11, 2026, 11:53 AM")
        is_pure_date = bool(re.match(r'^(?:date\s*[:.-]?\s*)?' + date_regex + r'(?:\s*,\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?)?\s*$', line_str, re.IGNORECASE))
        if is_pure_date:
            pii_lines.append(line_str)
            continue

        # Age and sex are permitted clinical demographics, even if OCR puts
        # them on the same line as a patient's name or other PII.  Store the
        # normalised value in clean_text while retaining the original line in
        # extracted_text for the PII filter.
        demographic_text = extract_age_gender(line_str)
        if demographic_text and re.search(r'\b(?:age|sex|gender)\b', line_lower):
            medical_lines.append(demographic_text)
            pii_lines.append(line_str)
            continue

        # Some report tables put ``AGE/SEX`` in one OCR cell and ``57 YEAR
        # F`` in the following cell.  Treat only that following value as
        # demographic data; it must not make arbitrary numeric rows medical.
        if expect_age_sex_value:
            expect_age_sex_value = False
            demographic_text = extract_age_gender(line_str)
            if demographic_text:
                medical_lines.append(demographic_text)
            pii_lines.append(line_str)
            continue

        # Age and gender are deliberately retained, but all other patient
        # data on the same OCR line is excluded.
        if re.search(demographic_pattern, line_str):
            demographic_text = extract_age_gender(line_str)
            if demographic_text:
                medical_lines.append(demographic_text)
            elif re.fullmatch(r'(?i)\s*age\s*[/,&-]?\s*(?:gender|sex)\s*', line_str):
                expect_age_sex_value = True
            pii_lines.append(line_str)
            continue

        # Check explicitly medical lines first to protect them from false PII matching
        if is_clearly_medical(line_str):
            # Still verify if it has a direct patient prefix (e.g. "Patient Name: ...")
            if not any(line_lower.startswith(prefix) for prefix in ('patient name:', 'patient:', 'pt name:', 'name:')):
                cleaned_line = re.sub(phone_pattern, '', line_str)
                cleaned_line = re.sub(email_pattern, '', cleaned_line)
                cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
                if cleaned_line:
                    medical_lines.append(cleaned_line)
                continue

        # 1. Honorific + name (Mrs. Rekha, Dr. Pankaj Kumar)
        if re.search(honorific_pattern, line_str):
            is_pii = True

        # 2. Doctor credentials & registration numbers (MBBS MD, Reg. No: BMC ...)
        elif re.search(doctor_credential_pattern, line_str):
            is_pii = True

        # 3. Organization headers & browser artifacts (DigiSwasthya Foundation, out:blank)
        elif re.search(org_artifact_pattern, line_str):
            is_pii = True

        # 4. Administrative report labels, including OCR variants
        elif re.search(admin_label_pattern, line_str):
            is_pii = True

        # 5. Phone, email, url
        elif re.search(phone_pattern, line_str) or re.search(email_pattern, line_str) or re.search(url_pattern, line_lower):
            is_pii = True

        # 6. Pure digit / ID lines
        elif re.match(r'^\s*\d{5,}\s*$', line_str):
            is_pii = True

        # 6. Multi-word PII prefixes
        elif any(kw in line_lower for kw in pii_keywords):
            is_pii = True

        # 7. Patient demographics keywords
        elif 'patient' in line_lower:
            is_pii = True

        # 8. Relation prefixes S/O, D/O, W/O, C/O
        elif re.search(relation_prefix_pattern, line_str):
            is_pii = True

        # 9. Standalone keywords (word-boundary matched)
        if not is_pii:
            for kw in standalone_keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', line_lower) or line_lower.startswith(kw):
                    is_pii = True
                    break

        # 10. Date lines (starting with date or matching date pattern without medical context)
        if not is_pii:
            if line_lower.startswith('date') or (re.search(date_regex, line_str) and not any(w in line_lower for w in ('tab', 'mg', 'od', 'bd', 'pain', 'diet', 'test'))):
                is_pii = True

        # 11. Pincode-bearing lines
        if not is_pii:
            if re.search(pincode_pattern, line_str):
                is_pii = True

        # 12. Bare unlabeled name fallback (e.g. "Jijabai Dhoke" on its own line)
        if not is_pii and not in_lab_section:
            if looks_like_bare_name(line_str):
                is_pii = True

        if is_pii:
            pii_lines.append(line_str)
        elif has_medical_content(line_str) or in_lab_section:
            cleaned_line = re.sub(phone_pattern, '', line_str)
            cleaned_line = re.sub(email_pattern, '', cleaned_line)
            cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
            if cleaned_line:
                medical_lines.append(cleaned_line)
        else:
            # Unknown text (including OCR'd patient/facility/ID fragments) is
            # retained only in extracted_text, never promoted to clean_text.
            pii_lines.append(line_str)

    return pii_lines, medical_lines

_pool = None


# ---------------------------------------------------------------------------
# POOL LIFECYCLE
# ---------------------------------------------------------------------------
def init_pool():
    global _pool
    if _pool is not None:
        return
    _pool = psycopg2.pool.ThreadedConnectionPool(
        config.DB_POOL_MIN,
        config.DB_POOL_MAX,
        dsn=config.DATABASE_URL,
    )
    print(f"[db] connection pool ready ({config.DB_POOL_MIN}-{config.DB_POOL_MAX})")

    # Check and add clean_text column if not exists
    conn = _pool.getconn()
    try:
        ensure_clean_text_column(conn)
    finally:
        _pool.putconn(conn)


def close_pool():
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        print("[db] connection pool closed")


def _get_conn():
    if _pool is None:
        raise RuntimeError("db_writer.init_pool() must be called before use")
    return _pool.getconn()


def _put_conn(conn):
    _pool.putconn(conn)


# ---------------------------------------------------------------------------
# SKIP-KEY QUERY
# dedup key = imagename ONLY, exact full-string match
# ---------------------------------------------------------------------------
def load_skip_keys():
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT imagename FROM document_extraction")
            db_keys = {row[0] for row in cur.fetchall()}
    finally:
        _put_conn(conn)

    return db_keys


def retry_pending_linkage():
    """No-op retained for pipeline backwards-compatibility.
    Since patient_id and visit_id are now sourced directly from the database,
    external CSV/legacy ID linkage retry is no longer needed."""
    return


# ---------------------------------------------------------------------------
# INSERT
# ---------------------------------------------------------------------------
def _insert_row(imagename, text_lines, patient_id, visit_id, legacy_id=None, healthcase_id=None):
    """Inserts an extracted document row into document_extraction table.
    Sets healthcase_id and legacy_id to NULL, using patient_id and visit_id directly."""
    if not patient_id or not visit_id:
        print(f"[db] SKIP {imagename}: missing patient_id={patient_id} or visit_id={visit_id}")
        return False

    conn = _get_conn()
    try:
        pii_lines, clean_lines = split_text_lines(text_lines)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_extraction
                    (id, healthcase_id, legacy_id, imagename, extracted_text, clean_text, patient_id, visit_id, priority)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (imagename) DO NOTHING
                """,
                (
                    str(uuid.uuid4()),
                    None,  # healthcase_id left as NULL
                    None,  # legacy_id left as NULL
                    imagename,
                    psycopg2.extras.Json(pii_lines) if pii_lines else None,
                    psycopg2.extras.Json(clean_lines) if clean_lines else None,
                    str(patient_id),
                    str(visit_id),
                    config.DEFAULT_PRIORITY,
                ),
            )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        _put_conn(conn)


def insert_extracted_document(
    imagename,
    text_lines,
    patient_id=None,
    visit_id=None,
    legacy_id=None,
    healthcase_id=None,
    **kwargs,
):
    """Public entry point used by paddel.py / handwritten.py. Never raises --
    a genuine DB error for one image is logged and isolated so it doesn't
    kill the whole run; the image just stays eligible for retry next time."""
    try:
        ok = _insert_row(
            imagename=imagename,
            text_lines=text_lines,
            patient_id=patient_id,
            visit_id=visit_id,
            legacy_id=None,
            healthcase_id=None,
        )
    except Exception as e:
        print(f"[db] ERROR inserting {imagename}: {e}")
        return False

    return ok
