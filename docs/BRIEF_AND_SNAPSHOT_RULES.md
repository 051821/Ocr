# Rule-based Patient Brief and Clinical Snapshot

How the API builds the **clinical snapshot** and the **rule-based patient brief**, which rules they apply, and a worked example for one real patient.

Both come from fixed rules over the patient's record. There is no AI writing, no randomness, and the same record always gives the same output.

| Output | API field | Built by |
|---|---|---|
| Clinical snapshot | `snapshot` in `GET /patients/{id}/snapshot`, `clinical_snapshot` in `GET /patients/{id}/summary` | `build_snapshot()` in [clinical_snapshot.py](../dataanalysis/clinical_snapshot.py) |
| Patient brief | `brief` in `GET /patients/{id}/snapshot` | `deterministic_review()` in [review_rules.py](../dataanalysis/review_rules.py), using [clinical_insights.py](../dataanalysis/clinical_insights.py) |

The AI review (`POST /patients/{id}/review`, and `ai_review` in `/summary`) is a different output. It reuses the brief's Overview, Abnormal Results and Evidence sections unchanged, and only its Clinical Insights section is written by the model.

## Where the rules and thresholds live

- **Rule logic** is Python code in the files above.
- **Lists and thresholds** (which conditions count as NSAID risks, maximum daily doses, symptom patterns, control-lab patterns and so on) are in [condition_registry.json](../dataanalysis/data/reference/condition_registry.json). Change a list or threshold there, not in the code.
- **Recognising terms.** To place a brand name, abbreviation or misspelling in the registry's vocabulary (for example "Ezact" is etoricoxib, an NSAID; "HTN" is hypertension), `clinical_terms.py` asks the model once per new term. It checks the answer against the registry vocabulary and caches it for every patient. When a term isn't resolved (offline, `ENABLE_DYNAMIC_ANALYSIS` off, no API key), the registry's regex patterns are used. The model only says what a term is. Every decision about flags, levels and wording is made by the rules.
- **Painkiller safety rules** (NSAIDs, opioid painkillers, paracetamol) are in `medicine_safety_rules` in the same file, each with its source, threshold and verified / draft state; see "Painkiller safety rules (published guidance)" below. The local NSAID thresholds are `nsaid_repeat` (2 visit days) and `drug_rules.nsaid_cumulative_days` (30 days).
- **Guideline rows (ICMR / national programmes)** run only for sections marked `"verified": true` in the registry, or when `ENABLE_DRAFT_GUIDELINES` is on. All of them are unverified right now and the flag is off, so `snapshot.guideline.rows` is empty for every patient.

## Shared first step: finding conditions

Both outputs start by turning the record into a list of conditions, using `build_condition_trend()` in [patient_chat.py](../dataanalysis/patient_chat.py).

1. Every visit's confirmed and provisional diagnosis is matched against the registry `condition_map` patterns. For example "eczema" becomes **Skin condition** and "OA knee" becomes **Osteoarthritis**. Free text the registry doesn't know is kept as written.
2. Chief complaints add a condition only when they name a known one (for example "K/C/O HTN DM"). Symptoms alone never add a condition.
3. Back, neck and joint pain at the same patient are merged into one "Musculoskeletal pain" row.
4. A condition seen at the latest visit is **ongoing**. Otherwise it is "last noted <date>".

Conditions on the patient's **long-term list** or **health history** count as present for the risk rules, even if no visit names them.

Encounters are numbered `#1, #2, …` from oldest to newest. Two visits on the same day are separate encounters.

---

## Part 1: Clinical snapshot

Before any rule, glucose and Hb recorded with the vitals (fasting / non-fasting sugar, Hb) are added to that visit's lab
results as validated results marked "point-of-care reading" (`data_quality.vitals_as_labs`), so control, monitoring and
guideline rules see them. Medicines count as "on the latest prescription" when written at any encounter on the latest day.

### Fields

| Field | How it's computed |
|---|---|
| `age_sex` | From the patient record, else the first document that states it |
| `encounters` | Number of visits |
| `span` | First visit date to latest visit date |
| `active` | Conditions marked ongoing (seen at the latest visit) |
| `active_meds` | Number of distinct medicines at the latest visit |
| `flags` | Highlighted concerns (rules below), sorted `high`, then `medium`, then `info` |
| `guideline` | Guideline rows (currently empty, see above) |
| `timeline` | The condition grid (see below) |
| `dates`, `encounter_columns` | Visit date for each encounter column |

Each flag is `{level, title, evidence, significance, kind}`. `kind` is one of `lab`, `condition`, `risk`, `medication`, `pattern`, `followup` or `data`. Flags state what the record shows and why it matters. They never suggest treatment.

### Timeline grid

Each row has one cell per encounter, and each cell is either recorded or not recorded.

| Row | `kind` | A cell is recorded when… |
|---|---|---|
| The top 5 conditions (most visits first) | `condition` | That condition is in the encounter's diagnosis or complaint |
| `BP ≥140/90` (only if any BP was recorded) | `alert` | That encounter's BP has systolic ≥ 140 or diastolic ≥ 90 |
| `Missed follow-up` (only if any) | `alert` | A follow-up after that encounter has status MISSED |
| `Abnormal report results` (only if any) | `alert` | A document at that encounter has a lab marked abnormal, high or low |

### Flag rules

Readings that can't be real are dropped before any rule runs: BP outside 60–260 / 30–160 or with systolic ≤ diastolic, temperature outside 34.5–42.5 °C or 93–108.5 °F, SpO₂ < 70, pulse outside 30–220, weight outside 2–300 kg, height outside 40–230 cm. These are reported separately as data flags.

**Lab results**

| Flag | Level | Fires when |
|---|---|---|
| Critical lab value | high | A result's status is Critical (from `lab_validation`) |
| N lab result(s) need checking against the source document | medium | Status is Unverified (likely OCR misread). These results aren't interpreted. |
| N medicine line(s) read as lab results | info | A "lab" is really a medicine line (tab, cream, a prescribed name) |
| N lab result(s) not evaluated | info | No reference range in the report and none on file |
| `<Condition>`: `<marker>` out of range | per registry (diabetes high, others medium) | A recorded condition's control lab (`control_labs`: HbA1c/glucose, TSH, Hb, uric acid, lipids, creatinine) is out of range in that condition's direction |
| `<Condition>` without `<marker>` in the record | medium | A recorded condition has a control lab defined but none exists. Hypertension with BP readings instead gets an info flag: "no related lab tests in the record". |
| `<Condition>` pattern result before it was documented / with no matching diagnosis | per registry | An out-of-range control lab appears before the condition is diagnosed, or the condition is never diagnosed |
| Dyslipidaemia, Low haemoglobin / PCV, Raised / Low white cell count, Abnormal platelet count, Kidney / electrolyte, Liver test, Thyroid test, Glucose / HbA1c out of range | medium (info + "check OCR" if every value is far outside its range) | Abnormal document results, grouped one line per problem. Anything not grouped goes under "Other abnormal results" (info). |
| Creatinine raised | high | Creatinine status is high |
| Creatinine at upper limit of normal | medium | Creatinine ≥ 95% of the upper reference limit |

**Symptoms (from chief complaints)**

| Flag | Level | Fires when |
|---|---|---|
| Condition-linked symptoms (`condition_symptoms`) | per rule | Diabetes: foot or leg wound (high), visual symptoms (high), thirst, weakness or fatigue (medium, only if glucose or HbA1c is out of range). Hypertension: headache, dizziness, palpitations or chest pain (medium, only at visits where BP ≥ 140/90 or no BP was taken). Also asthma/COPD, thyroid, anaemia, heart failure and epilepsy rules. |
| Possibly cardiac symptoms with cardiovascular risk | medium | Chest pain, breathlessness, palpitations or syncope, and either age ≥ 50 or diabetes, hypertension, dyslipidaemia, CAD, CKD or heart failure |
| Long-standing symptom | medium | A complaint states a duration ≥ 90 days that is longer than the record's own span |
| Persistent ulcer / lump / bleeding | high | Age ≥ 18 and an ulcer, lump, bleeding, weight loss or hoarseness lasting ≥ 21 days (stated, or seen at visits that far apart) |
| Possible neuropathic symptoms | medium (info if no cause condition and no nerve medicine) | Numbness, tingling, pins and needles, or burning in hands or feet. Ignored at visits diagnosed only as a skin condition, and when the phrase is about itch, rash or wounds. Names any cause condition (diabetes, thyroid, CKD, anaemia) and nerve medicines (B12, pregabalin and similar). |

**Vitals**

| Flag | Level | Fires when |
|---|---|---|
| BP above target at latest visit | medium (info if no hypertension diagnosis and the readings are only mildly raised) | Latest BP ≥ 140/90, or ≥ 130/80 with diabetes, CKD, CAD or heart failure. Says whether it improved since the first reading, and whether there is no hypertension diagnosis at all. |
| BP improved; earlier readings raised | info | Latest BP is under target, earlier readings were ≥ 140/90, and the latest systolic is below the first |
| BP ≥140/90 at N of M readings | medium (info without hypertension) | Latest BP is under target but earlier readings were raised |
| Low BP reading(s) | medium | Adult with systolic < 90, or systolic < 100 with diastolic < 60 |
| Dizziness / weakness with BP not raised, on BP-lowering medicine | medium | At the same visit: a BP-lowering medicine, dizziness or weakness, and BP < 120/80 |
| BMI overweight / obese / underweight | info / medium / medium | Adult, height 120–220 cm, using the latest weight and the most common height. WHO cut-offs 18.5 / 25 / 30 (Asian Indian cut-offs once that guideline is verified). |
| Weight readings inconsistent | medium | Weight changes ≥ 15% within 90 days |
| Vitals data quality | info | Recorded heights differ by ≥ 5 cm |
| Pulse above 100 | info | Adult with pulse > 100 |
| Impossible vital reading in the structured record | medium | Any reading outside the possible ranges above (for example 35 °F). Identical readings are grouped with every encounter they were recorded at; a list cut for length ends with `(+N more)`. |

**Medicines**

| Flag | Level | Fires when |
|---|---|---|
| Repeated oral NSAIDs with `<conditions>` | high if 2+ risk conditions, else medium | Oral NSAID on 2+ different visit days (several encounters on one date count once), plus one of `nsaid_risk_conditions`: hypertension, diabetes, CKD, heart failure, CAD or gout. Both numbers are in `nsaid_repeat` (`min_visit_days`, `high_when_conditions_at_least`). Counts days with a prescription, not days of intake. **Local threshold:** no guideline gives a visit count, so the concern says so and is marked draft. |
| Oral NSAIDs for about N days in total | medium | The stated course durations of oral NSAIDs add up to ≥ `drug_rules.nsaid_cumulative_days` (30), one value per clinical day. **Local threshold:** no guideline gives a day count; shown as draft on the concern. |
| Oral NSAID with `<conditions>` | medium | Oral NSAID at 1 visit, plus one of those conditions. Aspirin (an NSAID that is also an antiplatelet) is not counted when the record has a condition it is the antiplatelet treatment for (e.g. CAD). |
| Oral NSAID with raised BP reading | medium | No risk condition, but BP ≥ 140/90 at the NSAID visit |
| Topical NSAIDs in record | info | Only gels or creams, with a risk condition |
| Daily dose above the usual maximum | high | Strength × the morning-noon-night count is above `max_daily_dose` (for example etoricoxib 120 mg, diclofenac 150 mg, paracetamol 4000 mg). Skipped when the strength isn't written, for combination products, and for topicals. |
| Oral steroid with `<condition or reading>` | medium | A systemic steroid plus hypertension, diabetes, or BP ≥ 140/90 at that visit |
| No `<condition>` medicine prescribed since #N | medium | A chronic condition's own medicine stops appearing for 2+ later prescribing visits, or for 1 visit ≥ 60 days later. Skipped if the latest note says "continue" or "same treatment", and when that medicine was never prescribed in the record. |
| Medicine for a condition not recorded | medium | A hypertension, diabetes or thyroid medicine with no matching diagnosis, long-term condition or history |
| Symptom–medicine links | medium | Leg or ankle swelling on a calcium-channel blocker (amlodipine, nifedipine, felodipine, cilnidipine) or pioglitazone, cough on an ACE inhibitor, muscle pain on a statin. For a medicine-effect flag only the symptom visits within the window after the medicine are shown (pain before the statin started is not "on a statin"). |
| Recurrent `<skin dx>` with repeated steroids | medium | Skin condition at 2+ visits and either any oral steroid or a potent topical steroid at 2+ visits. The notes cover a return after oral courses, a fungal diagnosis, hypertension or diabetes, leg-predominant rash with a venous diagnosis, and potent topical at 3+ visits. |

**Painkiller safety rules (published guidance)**

NSAIDs, opioid painkillers (tramadol, tapentadol) and paracetamol are also checked against published guidance by [medicine_safety.py](../dataanalysis/medicine_safety.py). The rules are data, in `medicine_safety_rules` in [condition_registry.json](../dataanalysis/data/reference/condition_registry.json); the code has no drug or disease names. Medicines match by registry class (`NSAID (pain / anti-inflammatory)`, `Opioid analgesic`, `Antiplatelet`, …), route, ingredient or name pattern. A name pattern also checks the mapped ingredients, so a brand counts as its generic ("Alprax" is alprazolam). Only lines written as oral count for the oral rules; gels and creams never do.

| Flag | Level | Threshold (as shown on the concern) | Source |
|---|---|---|---|
| Oral NSAID with an antiplatelet, anticoagulant or steroid | medium; high with an anticoagulant | Any oral NSAID on the same day as, or during a course of, an antiplatelet (e.g. aspirin), anticoagulant or oral steroid; even short-term use. Aspirin is not counted as the NSAID (aspirin + clopidogrel is dual antiplatelet therapy). | AGS Beers Criteria 2023 |
| Diclofenac / etoricoxib with cardiovascular risk | medium | Diclofenac or etoricoxib (any dose) with hypertension, diabetes, dyslipidaemia, coronary disease or heart failure | NICE CKS: NSAIDs – prescribing issues |
| High-dose ibuprofen with cardiovascular risk | medium | Ibuprofen above 2400 mg/day with one of those conditions | NICE CKS |
| Repeated oral NSAIDs with no kidney function or BP check | medium | Oral NSAID on `nsaid_repeat.min_visit_days`+ visit days with no creatinine / eGFR result, or no BP reading, in the record. The guidance gives no visit count, so the local repeat threshold is used. | NICE CKS; SPS NSAID monitoring |
| Repeated oral NSAIDs in an adult aged 65 or over | medium | Age 65+ with an oral NSAID on `min_visit_days`+ visit days. Beers says "avoid chronic use" without a day count, so the local repeat threshold is used. | AGS Beers Criteria 2023 |
| Oral NSAID with eGFR below 30 | high | Any oral NSAID while the latest eGFR is below 30 mL/min/1.73 m² | KDIGO 2024 (NKF algorithm); ICMR STW Chronic Kidney Disease |
| Tramadol above 300 mg/day over age 75 | high | Tramadol above 300 mg/day in a patient older than 75 (adult maximum 400 mg/day, already checked by "Daily dose above the usual maximum") | Tramadol prescribing information (FDA label) |
| Opioid painkiller with a benzodiazepine | high | Tramadol / tapentadol on the same day as, or during a course of, a benzodiazepine (or zolpidem) | AGS Beers Criteria 2023; FDA boxed warning |
| Opioid painkiller with pregabalin / gabapentin | medium | Tramadol / tapentadol on the same day as, or during a course of, pregabalin or gabapentin | AGS Beers Criteria 2023 |
| Tramadol with an SSRI / SNRI | medium | Tramadol on the same day as, or during a course of, an SSRI or SNRI | Tramadol prescribing information (FDA label) |
| Tramadol in an adult aged 65 or over with no sodium result | medium | Tramadol at age 65+ with no sodium result in the record | AGS Beers Criteria 2023 (use with caution) |
| Paracetamol above 3 g/day with a liver-toxicity risk factor | medium | Paracetamol above 3 g/day with weight under 50 kg, liver disease, or alcohol use in the health history (usual maximum 4 g/day, already checked by "Daily dose above the usual maximum") | BNF-based NHS guidance (BASL paracetamol position statement 2022) |

How these rules behave:

- **"Same day or during a course":** two medicines count together when they are prescribed on the same date, or when one's course (start date + stated duration) overlaps the other's.
- **Daily dose:** strength × the morning-noon-night count, for single-ingredient lines only. A combination product's strength is not split between its parts, so dose rules skip it.
- **Threshold on every concern:** each concern shows a highlighted "Threshold:" line with the rule, the patient's own figures where relevant, the source, and whether the rule is verified, e.g. *"Threshold: any oral NSAID on the same day as, or during a course of, an antiplatelet (e.g. aspirin)…. Source: AGS Beers Criteria 2023 (draft: not yet clinically verified)."* The Repeated-NSAID and cumulative-days concerns say plainly that their numbers are local, not from a guideline.
- **Levels are local choices:** the guidance says "avoid" or "use with caution"; High / Medium is our choice and is reviewed with the thresholds.
- **One brief point per problem:** each rule has a `problem` (`NSAID use`, `Opioid painkiller use`, `Paracetamol use`), so its concern groups with the patient's other concerns about the same medicines and the brief makes one point for them.
- **Draft until verified:** every rule has `verified`, `verified_by` and `verified_on`. All are draft now. When a clinician has checked a rule against the source, set the three fields on that rule; its concern then reads "verified by … on …" instead of "draft". Unlike draft guideline sections, these rules raise concerns while draft, because they report a prescription combination in the record, not a guideline target.
- **Adding a rule:** add an entry with an `id`, a `type`, the medicine spec, `level`, `title`, `threshold`, `significance`, `source`, `url` and the three verification fields. Types: `combo` (two medicine groups on the same day or overlapping), `with_conditions`, `daily_dose_with_conditions`, `repeat_without_tests`, `age_repeat`, `lab_below`, `age_dose`, `age_without_lab`, `dose_with_risk`. A threshold written as `{min_visit_days}` is filled from `nsaid_repeat`.
- **Tests:** `tests/test_medicine_safety.py` checks every rule type, the aspirin exclusion, topical exclusion, brand matching and that every rule carries a source, threshold and verification state.

Sources: [AGS Beers Criteria 2023](https://doi.org/10.1111/jgs.18372) · [NICE CKS: NSAIDs – prescribing issues](https://cks.nice.org.uk/topics/nsaids-prescribing-issues/) · [SPS: NSAID monitoring](https://www.sps.nhs.uk/monitorings/nsaids-monitoring/) · [KDIGO 2024 CKD guideline](https://kdigo.org/wp-content/uploads/2024/03/KDIGO-2024-CKD-Guideline.pdf) · [ICMR STW nephrology](https://icmr.gov.in/icmrobject/uploads/STWs/1725952346_nephrology_all.pdf) · [Tramadol label (DailyMed)](https://dailymed.nlm.nih.gov/dailymed/search.cfm?query=tramadol) · [BASL paracetamol position statement 2022](https://basl.org.uk/uploads/BHPG/Paracetamol%20Position%20Statement_Mar%202022.pdf). The wording of each source should be checked against the original when a rule is verified.

**Course across visits**

| Flag | Level | Fires when |
|---|---|---|
| `<Condition>` not settling | medium if 4+ visits, else info | A non-chronic condition at 3+ visits, still at the latest, with the regimen changing between returns |
| Visits becoming more frequent / spacing out | medium / info | 5+ visits, and recent gaps are ≤ 60% (or ≥ 170%) of early gaps |
| Medicine load rising | info | 3+ visits, and the latest visit has at least 2 more medicines than the first |
| Diagnosis evolution | info | The diagnosis wording changes 2–5 times for the same problem across 3+ visits |

**Record quality and follow-up**

| Flag | Level | Fires when |
|---|---|---|
| Diagnosis differs between document and record | medium | The scanned document names a different diagnosis than the structured record |
| Vitals differ between document and record | info | They differ by ≥ 15% |
| Out-of-range vitals in documents | medium | Document vitals: temperature outside 35–38 °C, BP ≥ 140/90, SpO₂ < 94, pulse > 100 |
| Implausible vital reading | info | A document vital that can't be real |
| Symptoms only in documents / Medicines only in documents | info | They appear in a document but not in the structured record |
| N missed follow-up(s) | high if a follow-up booked on the latest day is missed, else medium | A follow-up has status MISSED. One per due date (two same-day encounters booking one review are one miss). Only a visit on a later day than the booking can attend it off-date. |
| Long-term condition list conflicts with the visit diagnoses | info (data quality) | A long-term list entry `hyper…` / `hypo…` while the visits only write the opposite (e.g. list 'Hyperthyroidism', visits 'hypothyroidism') |
| Recurrent musculoskeletal pain | info | A pain condition at 5+ visits |
| N report(s) not linked to a visit | info | Documents with no visit, so their lab dates are unknown |

---

## Part 2: Patient brief

`deterministic_review()` returns Markdown with four fixed sections.

```
# AI Clinical Review
## Overview
## Clinical Insights
## Abnormal Results
## Evidence
<fixed disclaimer line>
```

It first builds an **evidence pack** (`build_insight_evidence`): the patient's age, sex and dates, conditions with their status, medicines and what each one treats, lab results, abnormal lab groups, the snapshot flags (except lab flags), follow-up gaps and a per-visit timeline. Every line of the brief is built from this pack.

### Condition status

Each condition gets the strongest status the record supports, in this order:

1. **confirmed**: in a visit's confirmed diagnosis
2. **working diagnosis**: in a visit's provisional diagnosis
3. **known history**: written as "K/C/O" or "H/O" in a complaint, or on the long-term list or health history
4. **suspected**: written with "?", "r/o", "suspected", "probable" and similar
5. **document only**: only in a scanned document

### Sections

| Section | Content |
|---|---|
| **Overview** | Age and sex, number of encounters, period, then conditions grouped by status, plus "Possible but not diagnosed" when a control lab points to an undiagnosed condition |
| **Clinical Insights** | Up to 5 theme bullets, described below. A theme with nothing to say is left out. |
| **Abnormal Results** | Up to 10 out-of-range tests: value, status, reference range, date, encounter, what it means (from `lab_significance.json`), earlier and later values, and the recorded condition and medicine. Also lists results outside the range in a direction that doesn't matter, and unverified results. |
| **Evidence** | The encounters cited by abnormal labs plus the last 2 encounters (up to 6), each with its complaint, diagnosis and number of documents |

### Clinical Insights themes

| Theme | What goes in |
|---|---|
| **Course** | Age and sex, number of visits and period. Complaints raised at 2+ visits (top 3). Each visit's complaint for 3 visits or fewer; otherwise only the first and latest. |
| **Conditions** | **Active**: seen at the latest visit, known history, or long-term (up to 4). **Not recorded since**: older ones (up to 3). Then up to 2 course-pattern flags. |
| **Key findings** | Guideline rows (when enabled). Out-of-range control labs against their condition and medicine ("above range … despite …"). Possible undiagnosed conditions. Other abnormal labs, up to 3 points in total. Then up to 2 symptom or vital flags. |
| **Treatment** | Medicines at the latest visit grouped by the condition they treat, then "Also:" for the rest. Medicines with a condition link that weren't continued. The top high or medium medicine flag, marked ⚠. |
| **Gaps to check** | Follow-up flags, pending or unsuccessful follow-up calls, advised tests with no result, out-of-range values never repeated, high or medium data-quality flags, and uninterpretable results (up to 3) |

When fewer than 2 themes have content, the section reads "The record holds too little clinical data for patient-specific insights."

---

## Worked example

Patient `0c2a35b2-7b0f-4599-9ca5-66909bfe6285`: the patient in the dashboard screenshot. Output below is from a run on 2026-11-06.

### Record input

Patient level: 65 years / Female. Long-term conditions: **Diabetes Mellitus** (active), **Hypertension** (active).

| # | Date | Complaint | Diagnosis | Medicines | BP |
|---|---|---|---|---|---|
| 1 | 2026-08-06 | Both hand and fingers itching and crack | eczema | Clop cream, Levocet, Omega 3, Vitamin D3 60000, Paraffin cream | 128/82 |
| 2 | 2026-08-06 | Both leg pain, tingling sensation | OA knee | Calcium + D3, Diclofenac Gel, **Ibuprofen Tab** 1-1-0, Tramadol 50 mg 0-0-1, Vitamin B-Complex | 128/82 |
| 3 | 2026-09-23 | White discoloration, cracks in skin, itching… and bleeding | f/u/c/o eczema | Cetirizine syrup, Clop cream, Parasoft/vaseline, **Prednisolone 10mg** 1-0-0 | 116/68 |

There are no lab results and no documents. Both follow-ups were completed ("treatment going on"). Weight is 52 → 54.8 kg, height 155–156 cm, pulse 89–90.

### Snapshot output and the rule behind each part

| Output | Rule that produced it |
|---|---|
| `active: ["Skin condition"]` | "eczema" (#1) and "f/u/c/o eczema" (#3) map to Skin condition, which is seen at the latest visit, so it's ongoing. Osteoarthritis was last seen at #2. |
| `active_meds: 4` | 4 distinct medicines at #3 |
| **Oral NSAID with Hypertension and Type 2 diabetes mellitus** (medium) | Ibuprofen is an oral NSAID at 1 visit (#2), and hypertension and diabetes are on the long-term list. Diclofenac gel is topical, so it doesn't count. Tramadol isn't an NSAID. |
| **Oral steroid with Hypertension and Type 2 diabetes mellitus** (medium) | Prednisolone is a systemic steroid at #3, with both conditions present, so "raise BP and raise blood glucose" |
| **Possible neuropathic symptoms** (medium) | "tingling sensation" at #2. #2's diagnosis is OA knee, not a skin condition, so it's kept. Diabetes is a cause condition, so the level is medium. |
| **Recurrent eczema with repeated steroids** (medium) | Skin condition at 2 visits and an oral steroid at #3. The note "can worsen Hypertension and Type 2 diabetes mellitus" comes from oral steroid + those conditions. There's no "returned after steroid course" note because no skin visit comes after #3. |
| Timeline `Skin condition` cells ✔ ✘ ✔ | Encounters [1, 3] |
| Timeline `Osteoarthritis` cells ✘ ✔ ✘ | Encounter [2] |
| Timeline `BP ≥140/90` cells ✘ ✘ ✘ | The row appears because BP was recorded. No reading is ≥ 140/90. |

Rules checked that didn't fire, and why:

- **BP**: target is < 130/80 because of diabetes. The latest reading, 116/68, is under it, and no reading is ≥ 140/90.
- **BMI**: 54.8 / 1.56² = 22.5, which is in the normal range.
- **Height spread**: 1 cm, below the 5 cm threshold.
- **Weight change**: about 5%, below 15%.
- **Pulse**: never over 100.
- **Dose maximum**: Ibuprofen has no written strength. Tramadol is 50 mg/day, under 400.
- **No hypertension or diabetes medicine since #N**: these medicines are never prescribed in the record, so the rule has no last prescription to measure from.
- **Missed follow-up**: none.
- **Lab rules**: no labs.

### Brief output

```markdown
# AI Clinical Review
## Overview
65 years / Female; 3 encounter(s) from 2026-08-06 to 2026-09-23. Working (provisional) diagnoses: Eczema (2 visits) and Oa knee. Known history: Diabetes Mellitus and Hypertension.
## Clinical Insights
- **Course:** 65 years / Female, 3 visit(s) from 2026-08-06 to 2026-09-23 (48 days); #1: both hand, fingers itching, crack; #2: both leg pain, tingling sensation; #3: white discoloration, cracks in skin, itching on both ….
- **Conditions:** Active: Eczema (working diagnosis, 2 visits), Diabetes Mellitus (known history), Hypertension (known history); Not recorded since: Oa knee (last #2).
- **Key findings:** Possible neuropathic symptoms: tingling sensation (#2).
- **Treatment:** Eczema: Clop cream; Also: Cetirizine Dihydrochloride Syrup, Parasoft cream/vaseline jelly, Prednisolone 10mg; ⚠ Oral NSAID with Hypertension and Type 2 diabetes mellitus: NSAIDs can raise BP, worsen fluid retention and affect kidney function.
## Abnormal Results
- No laboratory results are available in the record.
## Evidence
- #2 2026-08-06: complaint 'Both leg pain, tingling sensation'; diagnosis 'OA knee'.
- #3 2026-09-23: complaint 'White discoloration,cracks in skin and itching on both hand, legs and whole Body and bleed'; diagnosis 'F/u/c/o eczema'.

Retrospective record-based observations only; clinician review is required for interpretation and decision-making.
```

How each part was built:

- **Overview**: Eczema and OA knee are provisional diagnoses, so they're working diagnoses. The record's own spelling is kept ("Oa knee"). Diabetes and hypertension come from the long-term list, so they're known history.
- **Course**: no complaint repeats across visits, so there is no "Recurring complaints" point. With only 3 visits, each visit's complaint is listed, clipped to 8 words.
- **Conditions**: Eczema was seen at the latest visit, and the long-term conditions always count as active. OA knee was last seen at #2.
- **Key findings**: there are no labs and no guideline rows, so the only point is the symptom flag (neuropathic).
- **Treatment**: of the medicines at #3, only Clop cream is linked to a recorded condition (eczema). The rest go under "Also". The first medium medicine flag is the NSAID one, so it gets the ⚠ point. Ibuprofen has no condition link, so it doesn't appear under "Not continued".
- **Gaps to check**: left out. The follow-ups were routine and there are no data-quality or follow-up flags.
- **Evidence**: no labs cite an encounter, so it shows the last 2 encounters.
