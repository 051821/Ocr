# Clinical record dashboard

Doctor-facing dashboard that turns a patient's scanned/OCR clinical records into a timeline, medication journey, validated labs, an AI clinical review and record-grounded chat.

## Run

```powershell
# Streamlit dashboard (from dataanalysis/)
..\venv\Scripts\python.exe -m streamlit run csv_dashboard.py

# Optional FastAPI layer (from repo root)
uvicorn api.main:app --reload
```

`csv_dashboard.py` is the stable entry point and runs `app.py`.

## Workflow

```
PostgreSQL (read-only)
   │  core/fetching.py, analysis.py
   ▼
Patient record ──► extraction/ (meds, labs, summary) ──► lab_validation.py (catalog, ranges, critical/unverified)
   │
   ▼  services/patient_context.py (shared by UI and API)
Deterministic facts (no LLM)
   ├─ clinical_snapshot.py   concerns + condition timeline
   ├─ clinical_insights.py   evidence pack: conditions, meds, abnormal labs, gaps
   └─ services/timeline.py, services/medication_journey.py
   │
   ▼
LLM layer (core/llm_client.py, versioned prompts/)
   ├─ AI review:  ai_review_service.py ──► review_rules.py checks every line against the evidence
   └─ Chat:       services/chat/ (retrieval + answers) ──► answer_verification.py drops unsupported facts
   │
   ▼
UI: app.py / ui/ (Streamlit)   or   api/ (FastAPI, via api_client.py)
```

The AI Analysis & Summary is built in two tiers per patient (`summary_tiers.py`): **Quick** (one bedside screen relative to the latest visit: headline, current and relevant earlier concerns, treatment by condition, tests to check, counts of what is only in Detailed) and **Detailed** (everything, each fact once). A coverage check fails the build if any flag or abnormal lab is in neither. The AI tab opens Quick unless the address has `?view=detailed`. See [docs/AI_REVIEW.md](../docs/AI_REVIEW.md#two-tiers-quick-and-detailed-from-v25).

Key principles:
- **Record first** – values, units, ranges and evidence are rendered from the record; the model only writes the narrative.
- **Verified output** – review and chat lines that cite facts not in the evidence are dropped, including a medicine linked to a condition or class the record does not support and "above range / uncontrolled" claims with no flagged value behind them.
- **Must-not-miss points** – high-level rule flags, medication risks (dose above the usual maximum, steroid or NSAID with a condition it can worsen), long-standing symptoms and long-term conditions are added to the review from the record whenever the model leaves them out (`clinical_insights.ensure_key_points`).
- **Registry-driven** – conditions, medicine classes, maximum daily doses and drug-condition rules live in `data/reference/condition_registry.json`; a new brand or abbreviation is a data change, not a code change.
- **Cache-first** – reviews are keyed by a facts hash (`review_facts.py`) and only regenerate when the record or prompt version changes.

## Layout

| Directory | Responsibility |
| --- | --- |
| `core/` | Config, read-only PostgreSQL access, fetching, LLM client, prompt loading. |
| `extraction/` | Medication, lab and summary extraction. |
| `services/` | Patient context, reviews, timeline, medication journey, chat, optional graph sync. |
| `storage/` | Local SQLite review and LLM cache (`data/cache.db`). |
| `prompts/` | Versioned review and chat prompts. |
| `data/reference/` | Deterministic reference data (lab catalog, lab significance, conditions). |
| `ui/` | Streamlit components and styles. |
| `../api/` | FastAPI routes over the same services. |

## Data and safety

PostgreSQL is never written to. Everything the app persists lives in one local SQLite file, `data/cache.db` at the repo root (WAL mode), managed by `storage/ai_analysis_csv.py`.

### What is cached

| Table | Contents | Key / invalidation |
| --- | --- | --- |
| `ai_review` | One AI clinical review per patient (markdown, model, prompt version, per-visit hashes). | `patient_id`. Reused while the record's `source_hash` and `PROMPT_VERSION` match; changed visits trigger an incremental update, and after 3 incremental updates the next change triggers a full rerun. |
| `llm_cache` | Raw LLM responses for extraction, side-effect lookups and chat answers. | Hash of the task, prompt, model and inputs (for chat: record hash + normalised question + prompt version), so any change is a miss. Stores an HMAC `patient_hash`, never the plain patient ID. Expires after `CACHE_TTL_DAYS` (0 = never). |
| `chat_faq` | Frequently asked questions per patient, used for quick-pick suggestions. | `patient_id` + normalised question; purged with the TTL. |

### Guideline context (ICMR / national programmes)

`clinical_snapshot.guideline_context` reads the whole record (all visits and reports) against Indian guideline
cut-offs chosen for the patient's age, sex and any recorded pregnancy: HbA1c and BP against target, monitoring tests
due by interval, anaemia band, Asian Indian BMI, guideline bands for results (also when the report gave no range),
antibiotics for usually viral illnesses, and a long cough with no TB test. The rows show in the snapshot
(`snapshot["guideline"]`) and reach the AI review as `guideline_context`; guideline flags carry a `source`.

The values live in `condition_registry.json` → `guidelines`, each section with its source and `verified`,
`verified_by`, `verified_on`. Every section produces rows, in the doctor's snapshot and in `/summary`
(`clinical_snapshot.guideline`): each row carries `section`, `verified`, `verified_by`, `verified_on`, verified rows
first, and `guideline.sections` lists every section's state. What verification changes:

| | Draft section | Verified section |
|---|---|---|
| Row in the UI and `/summary` | yes, badged "Draft · not verified" | yes, badged "✓ Verified · name, date" |
| Raises snapshot flags | no | yes |
| Reaches the AI review / no-AI brief | no | yes |
| Replaces the older "no related lab tests" flag | no | yes |

A section counts as verified only with all three fields set, after a clinician has checked its values against the
cited document: `python scripts/verify_guideline.py --show <section>` prints them, `<section> --by "Dr …"` records
the verification (`--revoke` undoes it, `--list` shows every section). `ENABLE_DRAFT_GUIDELINES=true` treats drafts
as verified (testing only). The guideline version and the verified sections are part of `PROMPT_VERSION`, so cached
reviews regenerate when either changes.

### Painkiller safety rules (published guidance)

`medicine_safety.py` checks NSAIDs, opioid painkillers and paracetamol against published guidance (AGS Beers
2023, NICE CKS, KDIGO 2024 / ICMR CKD, the tramadol label, BNF-based paracetamol guidance). The 12 rules are data in
`condition_registry.json` → `medicine_safety_rules`; each concern shows its threshold, source and whether a clinician
has verified the rule (all are draft now). The local NSAID thresholds with no guideline number (`nsaid_repeat`: 2
visit days; `nsaid_cumulative_days`: 30 days) say so on their concern. Details and the full table:
[docs/BRIEF_AND_SNAPSHOT_RULES.md](../docs/BRIEF_AND_SNAPSHOT_RULES.md).

### Security controls

Each control is behind an environment flag so it can be rolled out safely:

| Flag | Effect |
| --- | --- |
| `ENABLE_ENCRYPTED_CACHE` | Encrypts `llm_cache` values with Fernet (`CACHE_ENCRYPTION_KEY`). Old plaintext rows still read; unreadable rows are treated as a cache miss. |
| `ENABLE_ANONYMIZED_AUDIT_LOGS` | Chat logs record a salted HMAC of the patient ID (`AUDIT_HASH_SALT`) instead of the raw ID. |
| `ENABLE_PROMPT_INJECTION_SANDBOX` | Record text goes to the model inside `<clinical_record>` tags and is treated as data, never as instructions. |
| `ENABLE_ANSWER_VERIFICATION` / `ENABLE_CHAT_OUTPUT_SANITIZER` | Before the LLM call, add the record's own entries for any lab, medicine, visit or diagnosis the question names that retrieval missed. After it, drop chat lines citing values, diagnoses or prescriptions missing from the evidence, or calling something "not recorded" that the record has; strip prescribing instructions. Dropped/added kinds go to the chat audit log. |
| `ENABLE_STRICT_HTML_SANITIZATION` | Escapes all dynamic values rendered as HTML in the UI. |
| `ENABLE_HARDENED_CONFIG` | Reads secrets only from environment variables and fails fast when they are missing. |
| `ENABLE_EXTERNAL_PROMPTS` | Loads prompts from `prompts/`; any prompt edit changes the hash and invalidates cached reviews. |

Cache maintenance (prints counts only, never clinical content):

```powershell
..\venv\Scripts\python.exe scripts\cache_admin.py status | purge-expired | purge-patient <id> | purge-all
```

Neo4j is optional and syncs only on the explicit **Sync graph** action. Credentials come from environment variables and are never committed. See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the full design.

## Tests

```powershell
..\venv\Scripts\python.exe -m pytest -q tests
```



### Brief checks and golden regression

Why these exist and how to use them: [docs/PIPELINE_ROBUSTNESS.md](../docs/PIPELINE_ROBUSTNESS.md).

- `brief_checks.py` checks every built view (`insight_view.build_view` → `brief_checks`) against its record: visit
  numbers outside the record, broken citations, "not on the latest prescription" for a medicine written on the latest
  day, "no <test> in the record" while the record has it, and high-level concerns, abnormal results or long-term
  conditions the brief never states. Issues are logged (counts only) and carried in the view.
- `scripts/audit_sample.py --patients 50 [--generate] [--fail-over 0.05]` runs the checks over a random sample and
  writes `../output/brief_audit_<date>.json` (patient IDs and failed checks only). Run it weekly or after a change.
- `tests/golden/` holds de-identified, source-audited patients (IDs replaced, document text removed, dates shifted).
  `tests/test_golden.py` fails when a brief line, snapshot flag or check changes for any of them. After checking a
  deliberate change against the record, accept it with `python scripts/freeze_golden.py --update [fixture]`; freeze a
  newly audited patient with `python scripts/freeze_golden.py <patient_id>`. Add every confirmed bug report this way.
