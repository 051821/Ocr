# AI Clinical Review

How the **AI clinical review** is built: which prompt the current version (`clinical_ai_review_v25`) uses, where the LLM is called, how rule-based code and the LLM split the work, and how the result is saved. Includes a worked example for one patient.

For the fully rule-based brief and the clinical snapshot, see [BRIEF_AND_SNAPSHOT_RULES.md](BRIEF_AND_SNAPSHOT_RULES.md).

| Output | API | Built by |
|---|---|---|
| AI review | `GET` / `POST /patients/{id}/review`, and `ai_review` in `GET /patients/{id}/summary` | `get_or_generate_review()` in [ai_review_service.py](../dataanalysis/ai_review_service.py), exposed to the API via [services/review_service.py](../dataanalysis/services/review_service.py) and [api/routers/patients.py](../api/routers/patients.py) |

The rule-based code assembles the review, and the LLM writes only one section of it. The Overview, Abnormal Results, Evidence and closing line always come from the record. The model writes only the **Clinical Insights** bullets, and rule checks plus a second model pass check every point before it is shown.

## Two tiers: Quick and Detailed (from v25)

The AI Analysis & Summary is built in two tiers for every patient by [summary_tiers.py](../dataanalysis/summary_tiers.py) `build_summary()`, called from `insight_view.build_view()` and returned as `patient.summary` by `GET /patients/{id}/insight-view`:

```
summary = { headline, tiles, quick, detailed, coverage }
```

| Tier | What it holds | Who writes the text |
|---|---|---|
| `quick` | Relative to the **latest visit**. `now`: current concerns, high > verify > medium, one line each (≤25 words) with `as_of`, `area`, flag `id`, `cites` (#n). `earlier`: past high / verify concerns, plus past medium ones that name a medicine on the latest prescription or belong to an active condition. `treatment`: the latest prescription grouped by the condition each medicine treats, courses running but not renewed, medicines stopped within `recent_stop_days`. `tests_to_check`: missing / undated / overdue monitoring tests and missing-test flags, one per test family (guideline monitoring keys, else lab-catalog key), marked guideline-verified, guideline-draft or record rule. `hidden_counts`: what only Detailed shows (past concerns, info, record quality, labs, course) and each capped list's overflow. | Rules build every item. A second, small LLM call ([quick_review.py](../dataanalysis/quick_review.py), [prompts/review_quick.md](../dataanalysis/prompts/review_quick.md), `max_tokens` 500) rewords the headline (≤35 words) and the concern lines. |
| `detailed` | Everything, each fact once. `brief`: Key findings, Conditions, Treatment, Gaps (Course is part of `course`); a point that states a concern is a link (`see: [flag ids]`), not a repeat. `concerns`: every concern grouped by topic, plus info notes. `course`: course brief points, consistent / diagnosis shifts / medicine shifts, condition grid, medicine grid, monthly timeline. `reference`: abnormal lab values with ranges, guideline rows, record-quality notes, unlinked documents, AI-cited visits, AI checks, EHR markdown. | The existing review prompt (Clinical Insights) merged with the rule brief, unchanged. |

Caps, word limits, `recent_stop_days`, `stale_latest_days` (183: the headline starts with "Latest visit …, N months ago" when the latest visit is older) and the fallback `default_view` are in `clinical_thresholds.json` → `summary_tiers`. Nothing is per disease or per drug.

**Coverage check.** Every flag id and every abnormal lab must be in `detailed`, and must be either shown in `quick` or counted once in `quick.hidden_counts`. `build_summary` raises `CoverageError` otherwise, so the insight view fails rather than silently dropping a finding. `summary.coverage` reports the counts.

**AI quick checks** (`quick_review.apply`, run every time the view is built): the id exists in the rule skeleton; ≤25 words; at least one #n, every #n an encounter of the record and one of that concern's own encounters; every number in the concern's facts or the evidence pack; no medicine the record does not have; no advice or care suggestion; no unsupported record claim (`answer_verification.verify_answer`); no flag-level words. The headline: ≤35 words, numbers from the facts, keeps the stale-record note. A line that fails keeps the rule text (`src: rule`); an accepted line is `src: both` (raised by the rules, worded by the AI); `quick.ai` lists what was rejected and why. When the call fails or the review is the deterministic fallback, the quick tier is fully rule-built.

**Default view.** `doctor_settings` table (`doctor_id`, `default_view`, `updated_at`) in `data/cache.db`; `GET` / `PUT /doctors/{doctor_id}/settings`. The Streamlit AI tab has a Quick / Detailed toggle and opens `?view=quick|detailed` if given, else `quick`; it does not select a doctor or save a default. The settings API stays for other clients: `GET /patients/{id}/insight-view?doctor_id=` returns that doctor's saved default as `view`.

## Which prompt `clinical_ai_review_v25` uses

The prompt source depends on the `ENABLE_EXTERNAL_PROMPTS` flag in `.env`. **It is `false` now**, so the review runs on the prompts hardcoded in Python, not on the files in [prompts/](../dataanalysis/prompts/).

| Message | `ENABLE_EXTERNAL_PROMPTS=false` (current) | `ENABLE_EXTERNAL_PROMPTS=true` |
|---|---|---|
| System | `SYSTEM_PROMPT` in [Zai_analysis.py](../dataanalysis/Zai_analysis.py) (lines 25-69) | [prompts/review_system.md](../dataanalysis/prompts/review_system.md) |
| User, full review | Fallback string in `build_ai_review_prompt()`, [Zai_analysis.py](../dataanalysis/Zai_analysis.py) | [prompts/review_user.md](../dataanalysis/prompts/review_user.md) |
| User, update after new or changed visits | Fallback string in `build_incremental_prompt()`, [Zai_analysis.py](../dataanalysis/Zai_analysis.py) | [prompts/review_incremental.md](../dataanalysis/prompts/review_incremental.md) |
| Self-check (second LLM pass) | `_SELF_CHECK_SYSTEM` in [review_rules.py](../dataanalysis/review_rules.py) | Same; there is no file version |

### Where the version string comes from

`PROMPT_VERSION` is set in [review_facts.py](../dataanalysis/review_facts.py):

```
clinical_ai_review_v25  +  +guidelines-<registry version>[-verified-<hash>][-draft]
```

- `clinical_ai_review_v25` is the static base (v25 added the quick tier). When `ENABLE_EXTERNAL_PROMPTS=true`, it is replaced by `REVIEW_PROMPT_VERSION` from `core/prompts.py`, which is derived from the prompt files, so any prompt edit invalidates the cache.
- `+guidelines-…` comes from `guidelines.version` in [condition_registry.json](../dataanalysis/data/reference/condition_registry.json) (now `2026-11-draft1`). Verifying a guideline section adds `-verified-<hash>`, and `ENABLE_DRAFT_GUIDELINES=true` adds `-draft`.

Reviews stored before v25 carry `clinical_ai_review_v24+guidelines-2026-11-draft1`; they no longer match and regenerate (with the quick tier) on the next Generate.

### Differences between the hardcoded prompt and `review_system.md`

The `.md` system prompt is newer. Turning the flag on brings in:

- **`guideline_context`**: reading results and control against ICMR / national-programme cut-offs.
- **Course as one line**: "Presented with … (#n) · recurring … (#n–m)", with no age, sex, visit count or dates. The hardcoded prompt still asks for age/sex, visit count and period.
- **Brief rules**: cite only the visits that support a point, one point per high-level rule flag, record errors only under Gaps to check, and copy durations without doing arithmetic.

Turning the flag on changes `PROMPT_VERSION`, so every cached review regenerates once, on its next load.

### What the prompt contains

- **System**: role, how to read each field of the evidence JSON, rules (only stated facts, exact values with units, no prescribing advice, no generic statements), the five themes (Course, Conditions, Key findings, Treatment, Gaps to check), the output format (`## Clinical Insights` plus 4-5 bullets, under 180 words) and the exact required ending line.
- **User**: "Write the Clinical Insights for this patient…" followed by `PATIENT EVIDENCE (JSON)`, the output of `evidence_for_prompt(build_insight_evidence(patient))` from [clinical_insights.py](../dataanalysis/clinical_insights.py).
- **Incremental user**: the previous review (first ~260 words), which visits changed (`#n`) and the full current evidence JSON.

## Where the LLM is used

Every call goes through `complete()` in [core/llm_client.py](../dataanalysis/core/llm_client.py), which calls OpenRouter. Models are configured in [core/config.py](../dataanalysis/core/config.py).

| Step | Where | Task | Model | Settings |
|---|---|---|---|---|
| Write the Clinical Insights draft | `_call_openrouter()` in [Zai_analysis.py](../dataanalysis/Zai_analysis.py) | `review` | `DEFAULT_MODELS` in order: mistral-small-3.2-24b, llama-3.3-70b, gemini-2.0-flash (`:free` models are removed when `ENABLE_HARDENED_CONFIG` is on) | temperature 0.1, max 600 tokens, output checked by `validate_model_review` |
| Check each point: ok / fix / drop | `self_check()` in [review_rules.py](../dataanalysis/review_rules.py) | `review_check` | `CHAT_MODEL` (gemini-2.5-flash) | temperature 0, JSON output |
| Classify brand names, abbreviations and complaint terms used in the evidence | `_ask_model()` in [clinical_terms.py](../dataanalysis/clinical_terms.py) | `terms` | `CHAT_MODEL` | Answer checked against the registry vocabulary and cached per term |

The self-check and term classification run only when `ENABLE_DYNAMIC_ANALYSIS` is on (default) and an OpenRouter key is set. The model's answer about what a term is gets checked against the registry. Decisions about flags, levels and wording are always made by the rules.

Other LLM features, not part of the review, are patient chat ([patient_chat.py](../dataanalysis/patient_chat.py)) and document extraction and side-effect lookup ([analysis.py](../dataanalysis/analysis.py), prompts `extraction.md` and `side_effects_lookup.md`).

## Pipeline: rule-based and AI steps

```
get_or_generate_review(patient_data, force_refresh)          ai_review_service.py
│
├─ RULE  build_review_facts → visit_hashes → source_hash       review_facts.py
│        Cache hit (same source_hash + same PROMPT_VERSION + not a fallback
│        + valid structure)? → return the stored review. No LLM call.
│
├─ RULE  build_insight_evidence(patient)                       clinical_insights.py
│        conditions + status, medicines + "for" + class, abnormal labs + significance,
│        rule_flags (clinical_snapshot.py), recurring complaints, vitals, follow-up gaps,
│        record checks, guideline rows. Reference data: data/reference/*.json
│
├─ RULE  evidence_for_prompt → compact JSON for the prompt
│
├─ AI    analyze_patient_with_ai           (full review)       Zai_analysis.py
│     or analyze_patient_incremental       (only some visits changed and
│                                           mismatch_count + 1 < FULL_RERUN_THRESHOLD = 4)
│
├─ RULE  validate_model_review: one "## Clinical Insights" section + required ending.
│        If it fails → one retry with a correction note.
│
├─ RULE+AI sanitize_review                                     review_rules.py
│        1. strip_care_suggestions: removes "start / stop / increase …" advice
│        2. verify_answer: drops values, dates, doses or visits not in the evidence  (answer_verification.py)
│        3. drop_unsupported_status_claims, replace_contradicted_points, fix_new_condition_claims
│        4. AI self_check (gemini-2.5-flash): ok / fix / drop per point
│           guards: a fix may not add new numbers or grow much; ignored if it drops > half the points
│        5. drop_unsupported_medicine_links / _control_ / _active_ / _timing_ / _stop_ / implausible cause claims
│        6. drop_filler_points → trim to 5 bullets, 190 words
│        7. ensure_key_points: adds must-not-miss points from the record after trimming
│        (nothing survives → deterministic_insights)
│
├─ RULE  _assemble: final Markdown
│        # AI Clinical Review
│        ## Overview            ← record_overview         (rule)
│        ## Clinical Insights   ← checked model bullets   (AI + rule)
│        ## Abnormal Results    ← render_abnormal_results (rule)
│        ## Evidence            ← render_evidence         (rule)
│        REQUIRED_ENDING        ← constant
│        Before assembly, the bullets get record terms, citations, "encounter" → "visit" and
│        plain wording. If they still break the format (brief_violations), the rule-built brief replaces them.
│
├─ RULE  LLM error or invalid result → deterministic_review   model = "Deterministic fallback"
│
├─ RULE  _one_line_course / refresh_record_lines: Course line rebuilt from the record
│        (runs on fresh and cached reviews)
│
└─ SAVE  upsert_ai_analysis → data/cache.db, table ai_review   storage/ai_analysis_csv.py
```

## How the review is saved

Storage is SQLite at `data/cache.db`, written by [storage/ai_analysis_csv.py](../dataanalysis/storage/ai_analysis_csv.py). The module keeps its old CSV name, and an old `patient_ai_analysis.csv` is imported once if found.

### Table `ai_review` (one row per patient, upserted)

| Column | Content |
|---|---|
| `patient_id` | Primary key |
| `source_hash` | Hash of all visit hashes for the patient. Changes when any clinical fact changes. |
| `prompt_version` | `PROMPT_VERSION` at generation time, e.g. `clinical_ai_review_v25+guidelines-2026-11-draft1` |
| `model` | OpenRouter model that wrote the draft, or `Deterministic fallback` |
| `review_md` | Full final Markdown. Encrypted as `enc:v1:<Fernet token>` when `ENABLE_ENCRYPTED_CACHE=true` (on now), using key `CACHE_ENCRYPTION_KEY`. |
| `visit_count` | Number of visits at generation time |
| `visit_hashes` | JSON `{visit_id: hash}`, plus `unlinked_documents`, `health_history` and `long_term_conditions` when present |
| `mismatch_count` | Incremental updates since the last full rerun. A full rerun happens at `FULL_RERUN_THRESHOLD = 4`. |
| `generated_at` | UTC ISO timestamp |
| `quick_json` | AI wording of the quick tier, JSON `{"headline", "lines": {flag id: text}, "model"}`; encrypted like `review_md`. Empty = rule-built quick tier. |

**What goes into a visit hash** (`build_review_facts`): date, chief complaint, provisional and confirmed diagnosis, medication changes, normalised prescriptions (dose, frequency, duration), vitals, labs (name, value, status, reference), follow-up statuses, discrepancy count and document IDs. Display text and prompts are excluded, so only a clinical change triggers a new review.

### When a stored review is reused

All of these must hold. Otherwise the review is regenerated:

1. `source_hash` matches the current record.
2. `prompt_version` matches the current `PROMPT_VERSION`. A prompt or guideline version change triggers a rerun.
3. `model` is not `Deterministic fallback`. Fallback reviews are retried on the next load so a temporary outage isn't locked in.
4. The decrypted Markdown passes `validate_review`. A wrong encryption key or encryption turned off counts as a miss.
5. `force_refresh` is not set (`POST /review?force_refresh=true` sets it).

If only some visits changed and earlier visits are unchanged, an **incremental** update runs instead of a full one.

### Table `llm_cache`

Each successful LLM reply is also cached under `llm:<blake2b hash of task, messages, models, temperature, max_tokens>`. The row has `created_at` and a `patient_hash` so `purge_patient` can delete it. A truncated reply or one that fails its validator is never stored. `CACHE_TTL_DAYS` > 0 enables expiry and `purge_expired_cache`. The `terms` results from `clinical_terms.py` are stored here too, per term.

## Worked example

Patient `6959e408-…`, 57-year-old male, 20 visits from 2026-05-18 to 2026-10-10. Stored row: `model = mistralai/mistral-small-3.2-24b-instruct`, `prompt_version = clinical_ai_review_v24+guidelines-2026-11-draft1`, `review_md` encrypted.

Decrypted review, with the source of each part:

| Section | Source | Content |
|---|---|---|
| Overview | Rule, `record_overview` | "57 years / Male; 20 visit(s) on 13 clinical days from 2026-05-18 to 2026-10-10. Known diagnoses (K/C/O or long-term list): Hypertension (6 visits). Working (provisional) diagnoses: Tinea (10 visits)." |
| Course | AI, then checked | "57-year-old male, 20 visits over 145 days; recurring complaints: tinea (11 visits), hypertension (6 visits)". Age and visit count follow the hardcoded prompt; the `.md` prompt forbids them. |
| Conditions | AI, then checked | "Known hypertension (6 visits); working diagnosis: tinea (10 visits); BMI 31.2 (obese, Asian Indian cut-off)". Statuses match `conditions[].status`. |
| Key findings | AI, then rule checks and rule-added points | "BP at or above 140/90 target at 6 of 11 readings"; "No Hypertension medicine prescribed since 6 Aug"; "oral azole antifungal with a calcium-channel blocker, also repeated oral azole antifungal without liver tests"; "Steroid for a fungal complaint with no KOH test: Miconaziole cream + Eberfine M Cream (mometasone)"; "+1 more finding in the Clinical Snapshot". These follow `rule_flags` titles, so they were probably added by `ensure_key_points` / `complete_flag_points`. The overflow line comes from `finalise_brief`. |
| Treatment | None | Missing: either the model didn't write it or the validators removed it |
| Gaps to check | AI or rule | "no blood glucose, HbA1c in the record (recurrent fungal skin infection)" |
| Abnormal Results | Rule, `render_abnormal_results` | "No laboratory results are available in the record." plus the missing glucose / HbA1c note |
| Evidence | Rule, `render_evidence` | Visits #1, #2, #11, #17, #19, #20 with complaint, diagnosis and the themes that cite them |
| Ending | Constant `REQUIRED_ENDING` | "Retrospective record-based observations only; clinician review is required for interpretation and decision-making." |

The stored row does not record which step produced each line. The "probably" attributions above are inferred from the wording. The per-run counts (`validator_removed_points`, `format_violations`, `format_rejected`) are kept only in memory in `patient_data["_review_metrics"]` during generation.

## Files

| Role | Files |
|---|---|
| Orchestration and cache logic | [ai_review_service.py](../dataanalysis/ai_review_service.py), [services/review_service.py](../dataanalysis/services/review_service.py) |
| Prompt and LLM call | [Zai_analysis.py](../dataanalysis/Zai_analysis.py), [prompts/review_*.md](../dataanalysis/prompts/), [core/llm_client.py](../dataanalysis/core/llm_client.py), [core/config.py](../dataanalysis/core/config.py) |
| Evidence pack (rule-based) | [clinical_insights.py](../dataanalysis/clinical_insights.py), [clinical_snapshot.py](../dataanalysis/clinical_snapshot.py), [medicine_profile.py](../dataanalysis/medicine_profile.py), [lab_validation.py](../dataanalysis/lab_validation.py), [clinical_terms.py](../dataanalysis/clinical_terms.py), [data/reference/](../dataanalysis/data/reference/) |
| Checks and assembly | [review_rules.py](../dataanalysis/review_rules.py), [answer_verification.py](../dataanalysis/answer_verification.py) |
| Versioning and hashes | [review_facts.py](../dataanalysis/review_facts.py) |
| Storage | [storage/ai_analysis_csv.py](../dataanalysis/storage/ai_analysis_csv.py) → `data/cache.db` |
| Two tiers (quick / detailed) | [summary_tiers.py](../dataanalysis/summary_tiers.py), [quick_review.py](../dataanalysis/quick_review.py), [prompts/review_quick.md](../dataanalysis/prompts/review_quick.md), [insight_tabs.py](../dataanalysis/insight_tabs.py) (`render_quick_body`, `_VIEW_HELP`) |

## Feature flags that affect the review

| Flag | Now | Effect |
|---|---|---|
| `ENABLE_EXTERNAL_PROMPTS` | `false` | `true` loads the `prompts/*.md` files and versions the prompt by file hash |
| `ENABLE_ENCRYPTED_CACHE` | `true` | Encrypts `review_md` (needs `CACHE_ENCRYPTION_KEY`) |
| `ENABLE_DRAFT_GUIDELINES` | off | Runs unverified guideline sections and adds `-draft` to the version |
| `ENABLE_DYNAMIC_ANALYSIS` | on (default) | Enables the LLM self-check and term classification |
| `ENABLE_HARDENED_CONFIG` | `true` | Enforces the model allowlist and removes `:free` models |
