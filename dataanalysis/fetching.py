import re
import config
import psycopg2.pool


db_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=config.DB_POOL_MIN,
    maxconn=config.DB_POOL_MAX,
    dsn=config.DATABASE_URL
)

def fetch_batch(limit=2, offset=0):
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT
                    de.patient_id AS pid,
                    de.visit_id AS vid,
                    de.clean_text AS info,
                    v.created_at,
                    v.chief_complaint

                FROM document_extraction de

                LEFT JOIN visit v
                    ON v.id = de.visit_id

                WHERE de.clean_text IS NOT NULL

                ORDER BY de.visit_id

                LIMIT %s OFFSET %s
            """

            cur.execute(query, (limit, offset))

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

            return [
                dict(zip(columns, row))
                for row in rows
            ]

    finally:
        db_pool.putconn(conn)


import re


def clean_text(text):
    if not text:
        return []

    if isinstance(text, list):
        text_lines = text
    else:
        text_lines = str(text).splitlines()

    cleaned_lines = []

    # Standard PII patterns
    phone_pattern = r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b|\b\d{3,5}-\d{6,8}\b'
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    url_pattern = r'\b(?:www\.|https?://)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    pincode_pattern = r'\b\d{6}\b|\b\d{3}\s*\d{3}\b'
    
    # Generic date matches e.g. 10-11-2025 or 10/11/2025
    date_regex = r'\b\d{1,2}[-/\s.]\d{1,2}[-/\s.]\d{2,4}\b|\b\d{4}[-/\s.]\d{1,2}[-/\s.]\d{1,2}\b'

    # Honorifics/titles followed by capitalized letter (names). 
    honorific_pattern = r'\b(?i:mr|mrs|ms|miss|master|dr|prof|sr)\b\.?\s+(?!(?i:tb|tuberculosis|regimen|mono|poly|status)\b)[A-Z]'

    # Keywords that indicate the line is administrative or PII
    pii_keywords = [
        'patient name', 'guardian name', 'father name', 'husband name', 'wife name', 
        'daughter name', 'son name', 'patient address', 'patient mobile', 'patient contact',
        'contact number', 'contact no', 'mobile no', 'phone no', 'phone number', 'email id',
        'ref. physician', 'referred by', 'reported by', 'tested by', 'date of collection',
        'date tested', 'date reported', 'date collected', 'hospital reg', 'registration no',
        'episode id', 'nikshay', 'establishment id', 'laboratory name', 'lab name',
        'signature', 'signed by', 'other contact'
    ]

    # Individual standalone keyword indicators
    standalone_keywords = [
        'landmark', 'address', 'pincode', 'district', 'state', 'uhid', 'opd', 'ipd',
        'physician', 'referred', 'consultant', 'radiologist',
        'pathologist', 'clinic', 'hospital', 'laboratory', 'institute', 'college'
    ]

    for line in text_lines:
        line_str = str(line).strip()
        if not line_str:
            continue

        if line_str in ("```", "'''"):
            continue

        # Remove [ILLEGIBLE] anywhere
        line_str = re.sub(
            r'\[ILLEGIBLE\]',
            '',
            line_str,
            flags=re.IGNORECASE
        ).strip()

        if not line_str:
            continue

        line_lower = line_str.lower()
        is_pii = False

        # 1. Check if line matches honorific (e.g. Mrs. Rekha, Dr. Nayeem)
        if re.search(honorific_pattern, line_str):
            is_pii = True

        # 2. Check if line contains phone, email, url, or pincode
        elif re.search(phone_pattern, line_str) or re.search(email_pattern, line_str) or re.search(url_pattern, line_lower):
            is_pii = True

        # 3. Check if line matches pure digits or ID patterns (e.g. 201007876)
        elif re.match(r'^\s*\d{5,}\s*$', line_str):
            is_pii = True

        # 4. Check for direct multi-word PII prefixes
        elif any(kw in line_lower for kw in pii_keywords):
            is_pii = True

        # 5. Check if it contains patient demographics keywords specifically
        elif 'patient' in line_lower:
            demo_kws = ['name', 'address', 'mobile', 'phone', 'contact', 'id', 'info', 'detail', 'signature', 'thumb']
            if any(dk in line_lower for dk in demo_kws):
                is_pii = True

        # 6. Check for standalone keywords
        if not is_pii:
            for kw in standalone_keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', line_lower) or line_lower.startswith(kw):
                    is_pii = True
                    break

        # 7. Check if line starts with Date or matches generic date
        if not is_pii:
            if line_lower.startswith('date') or re.search(date_regex, line_str):
                is_pii = True

        # 8. Check for common header/footer junk fields containing pincodes or hospital details
        if not is_pii:
            if re.search(pincode_pattern, line_str):
                is_pii = True

        if not is_pii:
            # We still redact any inline phone/emails from medical lines
            cleaned_line = re.sub(phone_pattern, '', line_str)
            cleaned_line = re.sub(email_pattern, '', cleaned_line)
            cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

    return cleaned_lines


# ============================================================
# PREPARE DATA FOR AI ANALYSIS
# ============================================================

def prepare_medical_records(rows):

    medical_records = []

    for row in rows:

        cleaned_ocr_text = clean_text(row["info"])

        record = {
            "visit_id": row["vid"],

            "visit_date": (
                str(row["created_at"])
                if row["created_at"]
                else None
            ),

            "chief_complaint": row["chief_complaint"],

            "medical_document_text": cleaned_ocr_text
        }

        medical_records.append(record)

    return medical_records


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:
        # Fetch records
        rows = fetch_batch(limit=2)

        print("\nFetched Records:")
        print("=" * 70)

        # Clean and prepare records
        medical_records = prepare_medical_records(rows)

        for index, record in enumerate(medical_records, start=1):

            print(f"\nRECORD {index}")
            print("-" * 70)
            print(f"Visit ID: {record['visit_id']}")
            print(f"Visit Date: {record['visit_date']}")
            print(f"Chief Complaint: {record['chief_complaint']}")

            print("\nCleaned Medical OCR Text:")
            print(record["medical_document_text"])

            print("-" * 70)

    finally:
        # Close all connections when the script finishes
        db_pool.closeall()
