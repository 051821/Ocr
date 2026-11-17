# Pipeline robustness: why new patients break the brief, and how we guard against it

This note records what a source-level audit of the patient brief found, why the same kinds of bugs keep appearing with
new data, and what is in place to catch them. Read it before changing a rule in `clinical_snapshot.py`,
`clinical_insights.py` or `data_quality.py`, or the reference data in `dataanalysis/data/reference/`.

## 1. The audit

30 real patients (1–17 visits each) were run through the full pipeline with fresh AI reviews. Every brief point was
checked against the raw database rows: complaints, diagnoses, prescription lines, vitals, follow-ups, lab reports and
the long-term condition list. The checks covered:

- the visit numbers (`#n`) each point cites;
- the concern numbers Key findings link to (`→ concern N`);
- whether anything in the record was missed or wrongly interpreted.

The audit ran in two rounds. The first 20 patients turned up 25 issues. The 10 that followed turned up 9 more, one of
them a regression from a round-1 fix. All of these are fixed in shared code or reference data, never for one patient,
and each has a test in `dataanalysis/tests/test_review_round7.py`.

## 2. The problem: why each new patient breaks something

The individual bugs differed from patient to patient, but nearly all of them came from six root causes.

| Root cause | Examples found in the audit |
|---|---|
| **Free text is cleaned in many places, each slightly differently** | "K/C/O-HTN-DM" became "o-htn-dm" (the text was split on `/` before the K/C/O check). "Ct. T. Amlo" (*continue* Amlo) was read as a new medicine. "Not recorded" and "Nill" were treated as complaints. Newline-separated diagnoses ran together. "Review With Report" stayed glued to "Eltroxin", so the whole line was read as advice, not a medicine. |
| **The same rule is coded twice** | There were two medicine-name cleaners. Four parsers for `#n` each handled `#1–3` differently: one read it as `#1`, giving "URTI (last #1)" when URTI was at all three visits. "Latest prescription" meant the latest *day* in one function and the last *visit* in another, so medicines from the first of two same-day visits were reported as stopped. |
| **Silent shortcuts** | Evidence lists were cut at 3–4 items with no marker (an impossible-temperature flag listed #1–4 while #5 and #6 had the same reading). Treatment dropped a 4th condition group, which hid the hypertension medicines. Empty vitals saved as `0` became lab results. |
| **Keyword matching without context** | "knee" turned an itching patch into joint pain. "readings" matched an impossible-temperature point to the BP flag. A statin was blamed for shoulder pain recorded 10 months before it was started. Aspirin given with clopidogrel for heart disease was flagged as a risky painkiller. |
| **Gaps in the reference list** | Eltroxin, Thyronorm and Neomercazole were missing from the thyroid-medicine pattern. Glucose and Hb recorded with the vitals were never read as results. The AI term lookup usually hides such gaps, so they only appear when it is absent (tests, outages, new terms). |
| **No check across sections** | The brief said "no glucose in the record" while the vitals held 171 and 260 mg/dL. "Not on the latest prescription" appeared for a medicine written that same day. Nothing compared one section with another. |

One more class appeared only once the regression tests existed: **output that differs between runs of the same
record**. One sentence was built from a Python `set`, so "Aten + Telma 40 added" sometimes read "Telma 40 + Aten
added". This also changes the review's facts hash, which can trigger an unnecessary AI regeneration.

## 3. The solution

Points 1 and 2 are in place. Points 3–7 are the plan.

### 3.1 Automatic checks on every brief (done)

`dataanalysis/brief_checks.py` checks each built patient view against its own record. `insight_view.build_view` runs it
for the dashboard and the API, and the view carries the result as `brief_checks`.

| Check | Severity | Catches |
|---|---|---|
| `cite_out_of_range` | error | a `#n` the patient does not have |
| `garbled_text` | error | broken fragments: `(–2)`, `(Visit )`, `(#1, )`, a sentence ending "noted at" |
| `stale_latest_rx` | error | "not on the latest prescription" for a medicine written on the latest day |
| `contradicts_record` | error | "no / without <test> in the record" while the record holds that test's result |
| `not_covered` | warning | a high-level concern, an abnormal result or a long-term condition the brief never states |

Each check reads only the view, never blocks it, and logs counts only (no patient content).

Not yet done:

- **Checking that each cited visit actually holds what the point says.** This is not in `brief_checks`, because a
  word-level check was too noisy. The citation corrector (`clinical_insights.cite_points`) still re-checks and
  re-cites every point as the review is built.
- **Showing `brief_checks` in the UI.** The issues are carried in the view and the API, but no "needs review" badge
  is rendered yet.

**Sampling audit.** `scripts/audit_sample.py` runs the same checks over a random sample of patients and writes
`output/brief_audit_<date>.json`, which holds patient IDs and failed checks only. Run from `dataanalysis/`:

```powershell
..\venv\Scripts\python.exe scripts\audit_sample.py --patients 50                   # cached AI reviews only, no LLM calls
..\venv\Scripts\python.exe scripts\audit_sample.py --patients 50 --generate        # also generate missing reviews
..\venv\Scripts\python.exe scripts\audit_sample.py --patients 50 --fail-over 0.05  # exit 1 above 5% (cron / CI)
```

Run it weekly, and after any rule or reference-data change. A rising error rate is the early signal that new data
contains a pattern the rules do not handle.

### 3.2 Golden regression fixtures (done)

`dataanalysis/tests/golden/` holds the 30 audited patients as fixtures. Each one contains:

- the record;
- the term lookups it had;
- the verified rule-based brief, snapshot flags and check results.

`tests/test_golden.py` fails with a line-by-line diff when any of these change. This stops a fix for one patient from
silently changing another.

- **De-identification** (`evals/golden.py`): patient and visit IDs are replaced, and doctor and referral names
  removed. Scanned-document text is removed, while the lab results, medicines and diagnoses extracted from it stay.
  Every date is shifted by a fixed number of days per patient. Clinical text such as complaints and diagnoses
  remains, so treat the fixtures as sensitive.
- **Determinism**: term lookups are replayed from the fixture, so tests need no model. The fixtures give identical
  results under different `PYTHONHASHSEED` values.

| When | Command (from `dataanalysis/`) |
|---|---|
| A test fails after a deliberate change, and the new output is checked against the record | `python scripts\freeze_golden.py --update tests\golden\<fixture>.json` |
| A patient has been audited (or a confirmed bug report is fixed) | `python scripts\freeze_golden.py <patient_id>` |
| Check for order-dependent output | run `pytest tests\test_golden.py` with two different `PYTHONHASHSEED` values |

### 3.3 Next steps (not done yet)

3. **Clean the text once, upstream.** Move all free-text cleaning into one normalisation step that runs before any
   rule. Today it is spread across `data_quality.py`, `_complaint_phrases`, `_tidy`, `_first_word` and
   `_med_display`. The step would:
   - strip prefixes (Tab, T., Ct./Cont.);
   - split K/C/O history, placeholders and status notes (pregnancy, LMP) away from complaints;
   - separate trailing advice from medicine names;
   - treat zeros as empty;
   - split on newlines;
   - treat same-day visits as one day.

   Rules then read only clean, structured fields. `data_quality.prepare` is the natural home, and the fixtures and
   checks will catch any change in output.
4. **One definition per shared idea.** Keep one `#n`/range parser, one "latest prescription day", one "is this
   condition current" and one "medicine is for this condition", imported everywhere. Several are already shared
   (`clinical_insights._expand_ranges`, `_is_current`, `_prescribed_for`). The remaining copies should be replaced.
5. **No silent fallbacks.** Anything cut, skipped or unmatched says so ("+N more", "route unclear", "unrecognised
   medicine"). Review the unmapped-terms reports (`coverage.py`, the unmapped-brands report) weekly. Each entry is
   usually a one-line reference-data fix, like adding Eltroxin.
6. **Meaning from the AI term lookup, keywords only as fallback.** Rules that read the cached `clinical_terms`
   resolution (Disprin: aspirin, antiplatelet, for coronary artery disease) are much sturdier than site or word lists
   such as `knee`. Where a keyword list must stay, give it an `exclude` (see the joint-pain entry in
   `condition_registry.json`).
7. **A clinician reviews a weekly sample.** Someone with clinical training checks 10 random briefs against the record.
   Each confirmed error becomes a golden fixture (3.2).

## 4. Workflow for a new bug report

1. Reproduce it on the patient, and find which of the root causes in section 2 it belongs to.
2. Fix it where that class of bug lives: shared code, normalisation or reference data. Never add a check on one
   patient's values.
3. Add a small synthetic test (see `tests/test_review_round7.py`).
4. Run `pytest tests`. If golden fixtures change, check every diff against the source records before running
   `freeze_golden.py --update`.
5. Freeze the reported patient: `python scripts\freeze_golden.py <patient_id>`.
6. Run `scripts\audit_sample.py` on a fresh sample to confirm nothing new appears.

## 5. Changes from the audit (reference)

| Area | Change | Where |
|---|---|---|
| Latest prescription | Every encounter on the latest day counts | `clinical_snapshot.regimen`, `clinical_insights._treatment_bullet` |
| Medicine effects | Only symptom visits within the window after the medicine count (statin muscle pain) | `clinical_snapshot` symptom–medicine links |
| Aspirin | Not counted as an NSAID risk when the record has a condition it is the antiplatelet treatment for | `clinical_snapshot.classes_of` |
| Vitals | Glucose and Hb in the vitals become validated lab results; zeros are skipped | `data_quality.vitals_as_labs` |
| Prescription lines | Trailing advice is split off; "Ct./Cont." lines are the same medicine, continued | `data_quality.split_trailing_advice`, `core/encounters.py`, `_first_word`, `_med_display` |
| Follow-ups | One miss per due date; only a later day can attend a follow-up | `clinical_snapshot`, `core/encounters.recheck_followups` |
| Condition ↔ medicine | A medicine is for a condition only when the record ties them; one label per condition | `clinical_insights._prescribed_for` |
| Current conditions | A condition whose medicine is on the latest prescription stays active | `clinical_insights._is_current` |
| Course line | K/C/O, placeholders, doses, LMP and pregnancy notes are skipped; mixed wordings are named by their concept | `clinical_insights._complaint_phrases`, `_recurring_complaints` |
| Citations | Ranges expanded and re-compressed; record checks and conditions keep their own visits; no dangling words | `clinical_insights.cite_points`, `_flag_score` |
| AI claims | Wrong "last #n" corrected; "missed follow-up" dropped when the record has none | `clinical_insights.fix_record_contradictions` |
| Evidence lists | Repeated readings grouped; cut lists end "(+N more)" | `clinical_snapshot._listed`, `_grouped_readings` |
| Reference data | Thyroid brands added; joint pain excludes skin terms; red-flag title renamed "Possibly cardiac symptoms" | `condition_registry.json` |
| Data quality | Long-term list contradicting the visits (hyper- vs hypo-) is flagged | `clinical_snapshot` |
| Determinism | Treatment-change text keeps prescription order | `clinical_snapshot` BP rule |

Rule-level details are in [BRIEF_AND_SNAPSHOT_RULES.md](BRIEF_AND_SNAPSHOT_RULES.md). The AI review pipeline is
described in [AI_REVIEW.md](AI_REVIEW.md).
