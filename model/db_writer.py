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

    # 1. Disclaimers & Consent Forms (English & Hindi)
    disclaimer_pattern = (
        r'(?i)\b(?:disclaimer|अग्रिम\s*सूचना|अटकल|अस्वीकरण|अस्थिरता|असदिकरण|'
        r'remote\s+consultation|physical\s+examination|unusual\s+symptoms|adverse\s+reactions|'
        r'emergency\s+department|drug\s+allergies|informational\s+purposes\s+only|'
        r'prescriber\s+shall\s+not\s+be\s+liable|health\s+problem\s+or\s+disease|'
        r'दूरस्थ\s+परामर्श|व्यापक\s+मूल्यांकन|आपातकालीन\s+विभाग|टेलीमेडिसिन\s+दिशानिर्देश|'
        r'सहमति\s+कथन|मैं\s+सहमत|थर्ड\s+पार्टी|व्यक्तिगत\s+डेटा|गुमनाम|समग्र|'
        r'DigiSwasthya\s+को\s+केवल|स्वास्थ्य\s+डेटा|अधिकार|अनुमति)\b'
    )

    # 2. Refusal / Error messages from LLM
    refusal_pattern = (
        r'(?i)\b(?:the\s+image\s+appears\s+to\s+be|text\s+is\s+not\s+clearly\s+visible|'
        r'unable\s+to\s+extract\s+any\s+text|no\s+text\s+returned|cannot\s+read)\b'
    )

    # 3. Facility / Hospital / Center / Admin Headers
    facility_header_pattern = (
        r'(?i)\b(?:center\s*id|centre\s*id|facility\s*name|facility|reg\.?\s*lab|'
        r'digiswasthya\s+telemedicine|telemedicine\s+centre|hindlabs?|hindla|hll\s+lifecare|'
        r'mahanagar|diagnostic\s*&\s*research\s*centre|diagnostic\s*centre|research\s*centre|suraj\s+bhavan|'
        r'out\s+patient\s+ticket|all\s+india\s+institute\s+of\s+medical\s+sciences|aiims|institute\s+of\s+national\s+importance|'
        r'official\s+prescription|opd\s+prescription|opd\s+package|patient\s+category|opd\s+days|'
        r'department\s*:|processed\s+at|authorized\s+by|govt\.?\s+of|महालक्ष्मी|nhm|महाराष्ट्र\s+शासन)\b'
    )

    # 4. Doctor / Staff metadata
    doctor_pattern = (
        r'(?i)\b(?:dr\.?\s+dr\.?|dr\.?\s+[a-z]+|consultant\s*:|ref\.?\s*by\s*doctor|'
        r'mbbs|ddv|dermatologist|pathologist|radiologist|doctor\s+signature|signed\s+by|'
        r'reg\.?\s*no\.?\s*:?\s*[a-z0-9\-_]+)\b'
    )

    # 5. PII - Patient Name, Patient ID, Signature, Address, Contact, Language, Timestamps
    pii_pattern = (
        r'(?i)\b(?:patient\s*name|patient\s*id|\bid\s*[:.-]\s*#?|\bname\s*[:.-]?|signature\s*[:.-]?|'
        r'patient\s*information|patient’s\s*preferred\s+language|preferred\s+language|'
        r'registration\s*no|cr\s*no|uhid|episode\s*id|nikshay|patient\s*registration\s*code|'
        r'reg\.?\s*ref\.?\s*id|ref\.?\s*physician|referred\s+by|sample\s*coll|sample\s*col|'
        r'reg\.?\s*date|report\s*date|visit\s*date|\bdate\s*[:.-]?|contact\s*no|mobile\s*no|phone\s*no|email\s*id|address\s*:)\b'
    )

    id_hashtag_pattern = r'^\s*#(?:[0-9A-Fa-f]{6,12})\s*$'
    phone_pattern = r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b|\b\d{3,5}-\d{6,8}\b'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    url_pattern = r'\b(?:www\.|https?://)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    pincode_pattern = r'\b\d{6}\b|\b\d{3}\s*\d{3}\b'

    def extract_age_gender(line):
        line_clean = re.sub(r'[*`|_]+', '', line).strip()
        m = re.search(r'(?i)\b(\d{1,3})\s*(?:y|yrs?|years?)?\s*[-/•,\s]+\s*(male|female|m|f)\b', line_clean)
        if m:
            age, sex = m.groups()
            s_char = 'F' if sex.lower().startswith('f') else 'M'
            return f"Age/Sex: {age}/{s_char}"
        m_age = re.search(r'(?i)\bage\s*[:.-]?\s*(\d{1,3})\b', line_clean)
        m_sex = re.search(r'(?i)\b(?:gender|sex)\s*[:.-]?\s*(male|female|m|f)\b', line_clean)
        if m_age and m_sex:
            s_char = 'F' if m_sex.group(1).lower().startswith('f') else 'M'
            return f"Age/Sex: {m_age.group(1)}/{s_char}"
        if m_age:
            return f"Age: {m_age.group(1)}"
        if m_sex:
            s_char = 'F' if m_sex.group(1).lower().startswith('f') else 'M'
            return f"Sex: {s_char}"
        return None

    def is_bare_patient_name(line):
        clean = re.sub(r'[*`|_]+', '', line).strip()
        tokens = clean.split()
        if not (2 <= len(tokens) <= 4):
            return False
        medical_words = {
            'tab', 'cap', 'syr', 'inj', 'ointment', 'cream', 'lotion', 'mg', 'gm', 'ml',
            'daily', 'days', 'weeks', 'after', 'before', 'food', 'tinea', 'eczema',
            'folliculitis', 'vitals', 'pulse', 'bp', 'temp', 'spo2', 'heels', 'cracked',
            'follow', 'up', 'next', 'fluconazole', 'citrizine', 'miconazole', 'lobet',
            'vasaline', 'topisal', 'augmentin', 'fucibet', 'omnacortil', 'atarax',
            'investigation', 'result', 'units', 'hemoglobin', 'rbc', 'wbc', 'tlc', 'dlc',
            'pcv', 'mcv', 'mch', 'mchc', 'platelet', 'count', 'serum', 't3', 't4', 'tsh',
            'eclia', 'blood', 'glucose', 'rbs', 'hba1c', 'iron', 'bilirubin', 'creatinine',
            'as', 'needed', 'twice', 'week', 'times', 'day', 'chief', 'complaint',
            'provisional', 'diagnosis', 'impression', 'medications', 'strength', 'dosage',
            'frequency', 'duration', 'notes', 'timing', 'history', 'illness', 'present'
        }
        for tok in tokens:
            t_lower = tok.lower()
            if t_lower in medical_words or any(c.isdigit() for c in tok):
                return False
            core = re.sub(r'[^A-Za-z]', '', tok)
            if not core or not core[0].isupper():
                return False
        return True

    for line in text_lines:
        line_str = str(line).strip().replace(chr(0xFF1A), ':')
        if not line_str:
            continue

        if line_str in ("```", "'''", "---", "--"):
            pii_lines.append(line_str)
            continue

        line_str_clean = re.sub(r'\[ILLEGIBLE\]', '', line_str, flags=re.IGNORECASE).strip()
        if not line_str_clean or line_str_clean.upper() in ("[UNCLEAR]", "UNCLEAR"):
            pii_lines.append(line_str)
            continue

        if re.match(r'^\s*\|?\s*[:\-\s|]+\s*\|?\s*$', line_str_clean) or not re.search(r'[A-Za-z0-9]', line_str_clean):
            pii_lines.append(line_str)
            continue

        line_lower = line_str_clean.lower()

        if re.search(refusal_pattern, line_str_clean):
            pii_lines.append(line_str)
            continue

        demo_text = extract_age_gender(line_str_clean)
        if demo_text and not re.search(disclaimer_pattern, line_str_clean) and not re.search(facility_header_pattern, line_str_clean) and not re.search(doctor_pattern, line_str_clean):
            if demo_text not in medical_lines:
                medical_lines.append(demo_text)
            if re.search(r'(?i)\b(?:name|id|patient)\b', line_lower) or is_bare_patient_name(line_str_clean):
                pii_lines.append(line_str)
            continue

        if re.search(disclaimer_pattern, line_str_clean):
            pii_lines.append(line_str)
            continue

        if re.search(facility_header_pattern, line_str_clean):
            pii_lines.append(line_str)
            continue

        if re.search(doctor_pattern, line_str_clean):
            pii_lines.append(line_str)
            continue

        if (re.search(pii_pattern, line_str_clean) or 
            re.search(id_hashtag_pattern, line_str_clean) or 
            re.search(phone_pattern, line_str_clean) or 
            re.search(email_pattern, line_str_clean) or 
            re.search(url_pattern, line_lower) or 
            re.search(pincode_pattern, line_str_clean) or
            is_bare_patient_name(line_str_clean)):
            pii_lines.append(line_str)
            continue

        # Dates / dotted date lines
        if re.match(r'^\s*(?:\*\*|##\s*)?(?:date\s*[:.-]?\s*)?(?:\.|\d{1,2}[-/\s.](?:\d{1,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[-/\s.]\d{2,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{2,4}|\.{3,})\s*(?:\*\*|##)?\s*$', line_str_clean, re.IGNORECASE):
            pii_lines.append(line_str)
            continue

        cleaned_medical = line_str.strip()
        medical_lines.append(cleaned_medical)

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
