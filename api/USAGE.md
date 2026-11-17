# Patient Intelligence API: usage

Send a patient ID and get that patient's clinical analysis back as JSON. You never connect to the database.

**API URL:** `https://cultures-tariff-regular-neither.trycloudflare.com`

## Contents

1. [What you get](#1-what-you-get)
2. [Setup](#2-setup)
3. [Python helper (copy into your project)](#3-python-helper-copy-into-your-project)
4. [Common tasks](#4-common-tasks)
5. [Endpoint reference](#5-endpoint-reference)
6. [Errors](#6-errors)
7. [Good to know](#7-good-to-know)

---

## 1. What you get

- **You get:** patient IDs; doctor IDs and names; age and sex; health history; long-term conditions; patient type and exercise suitability; ongoing medicines; vitals and vital flags; emergency and referral flags per visit; the clinical snapshot; and a rule-based clinical brief.
- **How it's made:** every answer is worked out fresh from the patient database on each call, by fixed rules. No AI model is involved, so the same record always gives the same answer.
- **You don't get:** patient names, phone numbers, addresses, uploaded documents or their OCR text, or the database.

The API has only the 10 endpoints in section 5 (plus `/health`). Anything else returns `404`.

---

## 2. Setup

### Step 1: Get the two things you need

| What | Where it comes from |
|---|---|
| API URL | The top of this file |
| API key | Sent to you separately. Keep it private: don't commit it to git or paste it in shared chats. |

### Step 2: Check the API is up

Open this in a browser. It doesn't need a key:

```
https://cultures-tariff-regular-neither.trycloudflare.com/health
```

You should see `{"status":"ok"}`. If the page doesn't load, the API is offline. See [Good to know](#7-good-to-know).

### Step 3: Try it in the browser (optional)

1. Open `https://cultures-tariff-regular-neither.trycloudflare.com/docs`.
2. Click **Authorize**, paste your key, then click **Authorize** again.
3. Open any endpoint, click **Try it out**, then **Execute**.

### Step 4: Install `requests`

You need Python 3.9 or later.

```bash
pip install requests
```

### Step 5: Store the URL and key as environment variables

Windows PowerShell, for the current window:

```powershell
$env:PATIENT_API_URL = "https://cultures-tariff-regular-neither.trycloudflare.com"
$env:PATIENT_API_KEY = "Qs3Bi7LjeUrQVnRYNbVqEzq2NXnbAv8qOX1np_-6BIA"
```

macOS or Linux:

```bash
export PATIENT_API_URL="https://cultures-tariff-regular-neither.trycloudflare.com"
export PATIENT_API_KEY="Qs3Bi7LjeUrQVnRYNbVqEzq2NXnbAv8qOX1np_-6BIA"
```

Alternatively, put the same two lines in a `.env` file that's listed in `.gitignore`.

### Step 6: Test your key

```bash
curl -H "X-API-Key: https://cultures-tariff-regular-neither.trycloudflare.com>" "https://cultures-tariff-regular-neither.trycloudflare.com/doctors"
```

A list of doctor IDs means everything is working. A `401` means the key is wrong or missing.

---

## 3. Python helper (copy into your project)

Save this as `patient_api.py`. Every function returns the parsed JSON, and raises `PatientApiError` if the request fails.

```python
"""Client for the Patient Intelligence API."""
import os

import requests

API_URL = os.environ.get("PATIENT_API_URL", "https://cultures-tariff-regular-neither.trycloudflare.com").rstrip("/")
API_KEY = os.environ.get("PATIENT_API_KEY", "")

_session = requests.Session()
_session.headers["X-API-Key"] = API_KEY


class PatientApiError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _call(method, path, timeout=60, **kwargs):
    try:
        resp = _session.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise PatientApiError(0, f"API unreachable: {exc}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise PatientApiError(resp.status_code, detail)
    return resp.json()


# ── Health ──────────────────────────────────────────────────────────────────

def health():
    """{"status": "ok"} when the API is up."""
    return _call("GET", "/health", timeout=15)


# ── Lists ───────────────────────────────────────────────────────────────────

def list_patients():
    """All patient IDs: {"patient_ids": [...], "total": n}."""
    return _call("GET", "/patients")


def list_doctors():
    """Doctors on at least one visit, busiest first: [{"doctor_id", "name", "patients"}]."""
    return _call("GET", "/doctors")["doctors"]


def doctor_patients(doctor_id, id_prefix="", limit=200):
    """Patients with 2+ completed visits with this doctor, most recently seen first.

    id_prefix: optional start of the patient ID (hex characters and '-').
    Returns [{"patient_id", "visits", "last_visit"}].
    """
    params = {"limit": limit}
    if id_prefix:
        params["q"] = id_prefix
    return _call("GET", f"/doctors/{doctor_id}/patients", params=params)["patients"]


# ── One patient ─────────────────────────────────────────────────────────────

def get_summary(patient_id):
    """Everything for one patient in one call."""
    return _call("GET", f"/patients/{patient_id}/summary")


def get_demographics(patient_id):
    """Age and sex only. Fast: reads just the patient row."""
    return _call("GET", f"/patients/{patient_id}/demographics")


def get_snapshot(patient_id):
    """Clinical snapshot, clinical course, rule-based brief and patient types."""
    return _call("GET", f"/patients/{patient_id}/snapshot")


def get_patient_type(patient_id):
    """Patient type, exercise suitability and the per-visit breakdown."""
    return _call("GET", f"/patients/{patient_id}/patient-type")


def get_long_term_conditions(patient_id):
    """[{"name", "severity", "status", "diagnosed_at"}]."""
    return _call("GET", f"/patients/{patient_id}/long-term-conditions")["long_term_conditions"]


def get_medications(patient_id):
    """Medicine changes visit by visit, oldest first."""
    return _call("GET", f"/patients/{patient_id}/medications")


def get_timeline(patient_id):
    """Visit-by-visit timeline as plain text."""
    return _call("GET", f"/patients/{patient_id}/timeline")["timeline"]


def get_brief(patient_id):
    """The rule-based clinical brief (Markdown), worked out fresh on every call."""
    return get_snapshot(patient_id)["brief"]
```

---

## 4. Common tasks

### Find a doctor's patients and load each one

```python
import patient_api as api

for doctor in api.list_doctors()[:3]:
    patients = api.doctor_patients(doctor["doctor_id"])
    print(doctor["doctor_id"], len(patients), "patients")
    for p in patients[:5]:
        s = api.get_summary(p["patient_id"])
        print(" ", p["patient_id"], s["age_years"], s["sex"], s["patient_type"], s["exercise_suitable"])
```

### Read one patient's summary

```python
s = api.get_summary("<patient_id>")

print("Age / sex:", s["age_years"], s["sex"])
print("Type:", s["patient_type"], "| exercise suitable:", s["exercise_suitable"])
print("Ongoing medicines:", s["ongoing_medicines"])
print("Long-term conditions:", [c["name"] for c in s["long_term_conditions"]])

# Health history items the patient reported (present = true)
reported = {k: v for k, v in s["health_history"].items() if isinstance(v, dict) and v.get("present")}
print("Health history:", reported)

# Latest vitals
if s["vitals"]:
    latest = s["vitals"][0]
    print("Vitals on", latest["visit_date"], {r["label"]: r["value"] for r in latest["readings"]})

# Doctors who saw the patient (last 10 visits)
print("Doctors:", {v["doctor_id"] for v in s["visits"] if v["doctor_id"]})
```

### Get the clinical brief

```python
print(api.get_brief("<patient_id>"))   # Markdown, rule-based, fresh on every call
```

### Search a doctor's patients by the start of a patient ID

```python
api.doctor_patients("<doctor_id>", id_prefix="0b1c")
```

### Handle errors

```python
try:
    s = api.get_summary("<patient_id>")
except api.PatientApiError as e:
    if e.status_code == 404:
        print("No clinical visits for this patient")
    elif e.status_code == 0:
        print("API is offline:", e.detail)
    else:
        raise
```

---

## 5. Endpoint reference

Every endpoint needs the `X-API-Key` header. Dates are `YYYY-MM-DD` unless noted otherwise.

| # | Method and path | Python helper | What it returns |
|---|---|---|---|
| 1 | `GET /patients/{patient_id}/summary` | `get_summary` | **Start here.** Everything for one patient |
| 2 | `GET /patients` | `list_patients` | All patient IDs |
| 3 | `GET /doctors` | `list_doctors` | Doctor IDs and names with patient counts |
| 4 | `GET /doctors/{doctor_id}/patients` | `doctor_patients` | One doctor's patients |
| 5 | `GET /patients/{patient_id}/snapshot` | `get_snapshot`, `get_brief` | Snapshot, clinical course, rule-based clinical brief |
| 6 | `GET /patients/{patient_id}/patient-type` | `get_patient_type` | Patient type and exercise suitability |
| 7 | `GET /patients/{patient_id}/long-term-conditions` | `get_long_term_conditions` | Long-term conditions |
| 8 | `GET /patients/{patient_id}/medications` | `get_medications` | Medicine changes per visit |
| 9 | `GET /patients/{patient_id}/timeline` | `get_timeline` | Visit timeline as text |
| 10 | `GET /patients/{patient_id}/demographics` | `get_demographics` | Age and sex only (fastest call) |

### 1. Patient summary: `GET /patients/{patient_id}/summary`

| Field | Type | What it is |
|---|---|---|
| `patient_id` | string | The patient ID you asked for |
| `age_years` | number or `null` | Age |
| `sex` | string or `null` | `male` or `female` |
| `health_history` | object | The health-history questionnaire, exactly as stored (see below). `{}` when none was recorded. |
| `long_term_conditions` | list | `[{"name", "severity", "status", "diagnosed_at"}]`. Deleted conditions are left out. |
| `visits` | list | Last 10 visits, newest first: `[{"visit_id", "visit_date", "doctor_id", "is_emergency", "is_referral_recommended", "referral_specialist_name", "referral_treatment_name"}]`. Each visit has its own doctor. The emergency and referral fields come from the visit's prescription: `false` / `null` when there is none. |
| `patient_type` | string | Main body-system type, for example `Musculoskeletal`. `Unclassified` when nothing matched. |
| `exercise_suitable` | `true`, `false` or `null` | Whether exercise suits the main type. `null` when the patient is unclassified. |
| `types` | list | Every matched type, ranked: `[{"type", "score", "visits", "terms", "primary", "exercise"}]` |
| `ongoing_medicines` | list of strings | Medicines still active at the latest visit |
| `vitals` | list | Per visit, newest first: `[{"visit_id", "visit_date", "readings": [{"label", "value", "date", "source"}]}]`. Visits with no recorded vitals are left out. |
| `vital_flags` | list | Out-of-range vitals flagged by the clinic app on the last 5 completed visits, newest first: `[{"visit_id", "date", "metric", "code", "severity", "value"}]`. `severity` is `ABNORMAL` or `CRITICAL`; implausible readings are left out. |
| `clinical_snapshot` | object | Same as `snapshot` in endpoint 5 |

A vitals reading looks like `{"label": "BP", "value": "130/80 mmHg", "date": "2026-01-01", "source": "Structured record"}`.

Readings use these labels, in this order, and only appear when a value was recorded: `BP` (mmHg), `Pulse` (bpm), `SpO₂` (%), `Temperature` (°F or °C, as recorded), `Sugar`, `Fasting Sugar`, `Non-fasting Sugar` (mg/dL), `Hb` (g/dL), `Weight` (kg), `Height` (cm), `BMI` (kg/m²). The same readings appear in the `Vitals` line of `/timeline`.

A vital flag looks like `{"visit_id": "...", "date": "2026-10-03", "metric": "nonfasting_sugar_mg_dl", "code": "NONFASTING_SUGAR_MG_DL_ELEVATED", "severity": "ABNORMAL", "value": 150.0}`.

`health_history` looks like this:

```json
{
  "diabetes":     {"present": true,  "notes": "since 2019"},
  "hypertension": {"present": false, "notes": ""},
  "smoking":      {"present": false, "qty": ""},
  "alcohol":      {"present": false, "qty": ""},
  "other_notes": ""
}
```

| Keys | Fields |
|---|---|
| `asthma`, `diabetes`, `heart_disease`, `high_cholesterol`, `hypertension`, `thyroid`, `tuberculosis` | `present` (true or false), `notes` |
| `surgeries`, `hospitalisations`, `drug_allergies` | `present`, `notes` |
| `smoking`, `alcohol` | `present`, `qty` (amount) |
| `other_notes` | A free-text string, not an object |

Treat an item as recorded only when `present` is `true`. Keys can be missing on older records.

### 2. All patients: `GET /patients`

```json
{"patient_ids": ["0b1c...", "1f2e..."], "total": 2}
```

### 3. Doctors: `GET /doctors`

Doctors are linked per visit, not per patient. This lists doctors assigned to at least one visit, busiest first.

```json
{"doctors": [{"doctor_id": "7a9d...", "name": "Asha Rao", "patients": 42}]}
```

`patients` is the number of distinct patients the doctor has seen.

### 4. A doctor's patients: `GET /doctors/{doctor_id}/patients`

Patients with at least 2 completed visits with this doctor, most recently seen first, up to 200.

| Query parameter | Default | Meaning |
|---|---|---|
| `q` | none | Optional. Start of the patient ID, using hex characters and `-` only |
| `limit` | `200` | 1 to 200 |

```json
{"doctor_id": "7a9d...", "total": 1,
 "patients": [{"patient_id": "0b1c...", "visits": 5, "last_visit": "2026-09-14"}]}
```

A patient seen by several doctors appears under each doctor where they had 2 or more completed visits.

### 5. Clinical snapshot: `GET /patients/{patient_id}/snapshot`

| Field | What it is |
|---|---|
| `snapshot` | `age_sex`, `encounters` (visit count), `span` (first to last visit), `active` (ongoing conditions), `active_meds` (count), `flags`, `timeline`, `dates`, `encounter_columns` |
| `snapshot.encounter_columns` | Grid column headers: `[{"encounter": 1, "date": "2026-08-06"}, ...]`, oldest first |
| `snapshot.timeline` | One row per grid line: `{"label", "kind", "encounters", "cells"}`. `kind` is `condition` (green cell) or `alert` (red cell: raised BP, missed follow-up, abnormal report). `encounters` lists the encounter numbers where it was recorded. `cells` has one entry per encounter, `{"encounter", "date", "recorded"}`; `recorded: false` is a grey "not recorded" cell. |
| `snapshot.flags` | `[{"level", "title", "evidence", "significance", "kind"}]`, most serious first. `level` is `high`, `medium` or `info`. `kind` is `risk`, `condition`, `medication`, `lab`, `followup`, `pattern` or `data`. |
| `clinical_course` | `headline`, `consistent`, `diagnosis_shifts`, `medicine_shifts`, `timeline` |
| `brief` | Clinical brief as Markdown, built by fixed rules on every call. Its heading says "AI Clinical Review", but no AI model is involved. |
| `patient_types` | Same as `types` in the summary |
| `visit_clusters` | Per visit: `[{"patient_id", "visit_id", "visit_date", "cluster", "also_matched", "terms", "exercise"}]` |

### 6. Patient type: `GET /patients/{patient_id}/patient-type`

```json
{
  "patient_id": "0b1c...",
  "patient_type": "Musculoskeletal",
  "exercise_suitable": true,
  "types": [
    {"type": "Musculoskeletal", "score": 4, "visits": 2, "terms": ["oa knee"], "primary": true, "exercise": "Yes"}
  ],
  "visits": [
    {"patient_id": "0b1c...", "visit_id": "v1", "visit_date": "2026-01-01", "cluster": "Musculoskeletal",
     "also_matched": [], "terms": ["oa knee"], "exercise": "Yes"}
  ]
}
```

These results come from fixed rules, not the AI model, so the same record always gives the same answer. A match in a diagnosis counts more than a match only in the chief complaint. `terms` lists the words that matched. `exercise` is `Yes`, `No`, or `—` for an unclassified visit.

### 7. Long-term conditions: `GET /patients/{patient_id}/long-term-conditions`

```json
{"patient_id": "0b1c...",
 "long_term_conditions": [{"name": "Hypertension", "severity": "MODERATE", "status": "ACTIVE", "diagnosed_at": "2024-03-01"}]}
```

`severity` and `status` are returned exactly as they're stored in the clinic database.

### 8. Medication journey: `GET /patients/{patient_id}/medications`

A list with one entry per visit, oldest first:

```json
[{"visit_id": "...", "visit_date": "2026-01-01T10:00:00", "visit_index": 0,
  "changes": [{"type": "started", "medication": "Metformin 500", "text": "..."}],
  "active_rx_count": 2, "all_active_meds": ["Amlodipine 5", "Metformin 500"]}]
```

`type` is one of `started`, `continued`, `dose_changed`, `discontinued`, `newly_documented` or `mismatch`. `visit_date` here includes the time.

### 9. Timeline: `GET /patients/{patient_id}/timeline`

```json
{"timeline": "<plain text, one block per visit>"}
```

### 10. Age and sex: `GET /patients/{patient_id}/demographics`

Reads only the patient row, so it's much faster than `/summary` when you just need age and sex. `patient_id` must be a UUID (otherwise `422`); an unknown ID returns `404`.

```json
{"patient_id": "31db6f20-...", "age_years": 62, "sex": "F", "age_sex": "62 years / Female"}
```

`age_years` and `sex` can be `null` if they weren't recorded; `age_sex` is then `null` too when both are missing.

## 6. Errors

| Status | Meaning | What to do |
|---|---|---|
| `0` (Python helper only) | The API couldn't be reached | It's offline or the URL changed. See section 7. |
| `401` | The key is missing or wrong | Check `PATIENT_API_KEY` |
| `404` | The patient has no clinical visits, or the endpoint doesn't exist | Use only the endpoints in section 5 |
| `422` | A parameter is invalid, for example `q` contains characters other than hex and `-` | Fix the parameter |
| `502` | The server couldn't load the patient's data | Try again; if it persists, contact the API owner |
| `503` | The server isn't configured | Contact the API owner |

---

## 7. Good to know

- **Availability:** the API is hosted and always on. If `/health` doesn't respond, contact the API owner.
- **Always fresh:** nothing is saved between calls, so every answer reflects the patient database at the moment you ask.
- **Speed:** most calls take a few seconds.
- **Patient privacy:** the data is real patients' health records. Store it securely, don't share your key, and use it only for the agreed work.
