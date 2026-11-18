"""Insight view: the data behind the AI Analysis & Summary and Lab Results tabs.

Every insight carries the raw rows it was built from and the rule (paraphrased from the pipeline code) that produced
it; the AI brief and the rule-based brief are merged into one Patient Brief with each point marked by who made it.
Read-only over the pipeline: no LLM call (the AI part is the cached review, when one is current).
"""
import logging
import re

import lab_validation

logger = logging.getLogger(__name__)
NSAID =re.compile(r"diclo|dicof|nucox|etor|etosh|ezact|retoz|arcox|etody|acecl|hifenac|zerodo|zerodal|ibup|brufen|"
                   r"combiflam|naprox|indometh|pirox|mefen|meftal|nimes|nise|voveran|dynapar", re.I)
TOPICAL = re.compile(r"\b(?:gel|cream|oint(?:ment)?|spray|lotion|patch)\b", re.I)


def date10(v):
    return str(v or "")[:10]


# ── Raw data ────────────────────────────────────────────────────────────────

def raw_visits(record):
    out = []
    for i, v in enumerate(record.get("visits") or [], 1):
        out.append({
            "enc": i,
            "date": date10(v.get("visit_date")),
            "complaint": (v.get("db_chief_complaint") or v.get("chief_complaint") or "").strip(),
            "provisional": (v.get("db_provisional_dx") or v.get("provisional_diagnosis") or "").strip(),
            "confirmed": (v.get("db_confirmed_dx") or v.get("confirmed_diagnosis") or "").strip(),
            "vitals": [{"label": r.get("label"), "value": r.get("value")} for r in v.get("db_vitals_rows") or []],
            "meds": [{"name": m.get("name"), "dose": m.get("dosage"), "freq": m.get("frequency"),
                      "dur": m.get("duration"), "spec": m.get("prescriber_specialization") or ""}
                     for m in v.get("db_medications") or []],
            "specialties": list(v.get("prescriber_specializations") or []),
            "instructions": [str(a).strip() for a in v.get("db_instructions") or [] if str(a).strip()],
            "followups": [{"date": date10(f.get("scheduled_date")), "status": f.get("status"),
                           "resolution": f.get("resolution")} for f in v.get("followups") or []],
            "labs": [lab_row(l) for l in v.get("lab_results") or []],
            "documents": len(v.get("documents") or []),
            "discrepancies": v.get("discrepancies") or [],
            "emergency": bool(v.get("is_emergency")),
            "referral": v.get("referral_specialist_name") if v.get("is_referral_recommended") else None,
        })
    return out


def lab_row(l, source=""):
    return {"test": l.get("test_name"), "value": l.get("value"), "reference": l.get("reference"),
            "status": l.get("status"), "note": l.get("note") or "", "source": source,
            "report_date": l.get("report_date") or ""}  # the date read from the report itself


def raw_reports(record):
    out = []
    for d in record.get("unlinked_documents") or []:
        cs = d.get("clinical_summary") or {}
        likely = d.get("likely_visit") or {}
        out.append({
            "label": d.get("document_label"),
            "likely_visit": (f"#{likely.get('visit_index')} ({likely.get('visit_date')})" if likely else ""),
            "likely_reasons": likely.get("reasons") or [],
            "impression": cs.get("clinical_impression") or "",
            "complaint": cs.get("chief_complaint") or "",
            "tests_advised": cs.get("tests_advised") or [],
            "labs": [lab_row(l, d.get("document_label")) for l in d.get("lab_results") or []],
        })
    return out


def all_labs(visits, reports):
    """Every lab row, the same test with the same value on the same date shown once (lab_validation, shared with the
    app's Lab Results tab); undated rows are matched only with undated rows."""

    rows = []
    # Two dates per row: the report's own (read from the document) and the visit it is filed under.
    for v in visits:
        rows += [{**l, "source": f"Visit #{v['enc']}", "visit_date": v["date"] or "", "_date": v["date"] or None}
                 for l in v["labs"]]
    for r in reports:
        rows += [{**l, "source": f"{r['label']} (not linked to a visit" + ("" if l.get("report_date") else ", undated") + ")",
                  "visit_date": "", "_date": l.get("report_date") or None} for l in r["labs"]]
    rows = lab_validation.drop_same_day_duplicates(rows, date=lambda r: r["_date"], test=lambda r: r["test"],
                                                   value=lambda r: r["value"], source=lambda r: r["source"])
    for r in rows:
        r.pop("_date", None)
        dups = r.pop("duplicate_sources", None)
        if dups:
            r["note"] = "; ".join(x for x in (r.get("note"), "Same result also in: " + ", ".join(dict.fromkeys(dups))) if x)
    return rows


# ── Raw-data selectors per insight ──────────────────────────────────────────

def encs_in(text):
    from clinical_insights import _expand_ranges
    return sorted({int(n) for n in re.findall(r"#(\d+)", _expand_ranges(text or ""))})  # '#1–3' is #1, #2, #3


def visit_rows(visits, encs, field):
    return [{"enc": v["enc"], "date": v["date"], "complaint": v["complaint"],
             "detail": field(v)} for v in visits if v["enc"] in encs]


def meds_text(v):
    return "; ".join(f"{m['name']} {m['dose'] or ''} {m['freq'] or ''} {m['dur'] or ''}".strip() for m in v["meds"])


def vital(v, label):
    return next((r["value"] for r in v["vitals"] if r["label"] == label), "")


def labs_matching(labs, pattern):
    return [l for l in labs if re.search(pattern, str(l["test"] or ""), re.I)]


STEROID = re.compile(r"omnacort|wysolon|predni|deflaz|defcort|dzacort|defza|methylpred|medrol|dexameth|dexona|betnesol|"
                     r"lobet|clobet|tenovate|betnovate|betameth|halobet|mometas", re.I)
BP_MEDS = r"telmi|amlod|losar|olmes|ramip|enala|metop|aten|nebiv|chlorthal|hydrochlor|cilnid|nifed"


def labs_named_in(labs, text):
    low = (text or "").casefold()
    return [l for l in labs if l["test"] and str(l["test"]).casefold() in low]


def raw_for_flag(flag, visits, labs, reports):
    t, ev = flag["title"], flag.get("evidence", "")
    if "pattern result" in t or flag.get("kind") == "lab":
        found = labs_named_in(labs, ev)
        if found:
            return {"labs": found}
    if "read as lab results" in t:
        pat = r"\b(?:tab|tablet|cap|capsule|cream|gel|oint(?:ment)?|lotion|syp|syrup|inj|drops?|vaseline|vasaline|soap|powder)\b"
        return {"labs": [l for l in labs if re.search(pat, str(l["test"]), re.I) or "not evaluated" in str(l["status"]).casefold()]}
    if "steroid" in t.casefold():
        encs = [v["enc"] for v in visits if any(STEROID.search(str(m["name"])) for m in v["meds"])]
        return {"visits": visit_rows(visits, encs, lambda v: (v["provisional"] or v["confirmed"]) + " · " + "; ".join(
            m["name"] for m in v["meds"] if STEROID.search(str(m["name"]))))}
    if t.endswith("not settling"):
        return {"visits": visit_rows(visits, [v["enc"] for v in visits],
                                     lambda v: (v["provisional"] or v["confirmed"] or "-") + " · " + meds_text(v))}
    if "out of range" in t and ":" in t:
        return {"labs": labs_matching(labs, r"hba1c|glyco|glucose|sugar")}
    if re.match(r"No .+ medicine prescribed since #(\d+)", t):
        last = int(re.search(r"#(\d+)", t).group(1))
        return {"visits": visit_rows(visits, [v["enc"] for v in visits if v["enc"] >= last], meds_text)}
    if t.endswith("no related lab tests in the record"):
        return {"visits": [{"enc": v["enc"], "date": v["date"], "complaint": "", "detail": f"BP {vital(v, 'BP')}"}
                           for v in visits if vital(v, "BP")],
                "note": f"Lab results in the whole record: {len(labs)}."}
    if t == "Oral NSAID with raised BP reading" or t.startswith("Daily dose above"):
        return {"visits": visit_rows(visits, encs_in(ev), lambda v: f"BP {vital(v, 'BP') or '-'} · " + meds_text(v))}
    if t.startswith("Repeated oral NSAIDs") or t.startswith("Oral NSAID"):
        encs = [v["enc"] for v in visits if any(NSAID.search(str(m["name"])) for m in v["meds"])]
        return {"visits": visit_rows(visits, encs, lambda v: "; ".join(
            f"{m['name']} {m['dose'] or ''}" + (" (topical)" if TOPICAL.search(str(m['name'])) else "")
            for m in v["meds"] if NSAID.search(str(m["name"]))))}
    if "missed follow-up" in t:
        return {"visits": [{"enc": v["enc"], "date": v["date"], "complaint": v["complaint"],
                            "detail": "; ".join(f"{f['date']}: {f['status']} / {f['resolution'] or '-'}" for f in v["followups"])}
                           for v in visits if v["followups"]]}
    if t.startswith("BP "):
        return {"visits": [{"enc": v["enc"], "date": v["date"], "complaint": v["complaint"],
                            "detail": f"BP {vital(v, 'BP')}" + (f" · on {', '.join(m['name'] for m in v['meds'] if re.search(BP_MEDS, str(m['name']), re.I))}" if v['meds'] else "")}
                           for v in visits if vital(v, "BP")]}
    if t.startswith("BMI") or t == "Vitals data quality":
        return {"visits": [{"enc": v["enc"], "date": v["date"], "complaint": "",
                            "detail": f"Weight {vital(v, 'Weight') or '-'} · Height {vital(v, 'Height') or '-'}"}
                           for v in visits if vital(v, "Weight") or vital(v, "Height")]}
    if t.startswith("Creatinine"):
        encs = [v["enc"] for v in visits if any(re.search("metf", str(m["name"]), re.I) for m in v["meds"])]
        return {"labs": labs_matching(labs, "creatinine|urea|egfr"),
                "note": f"Metformin (renal-caution medicine) prescribed at {len(encs)} visit(s)."}
    if t.startswith("Impossible vital"):
        return {"visits": [{"enc": v["enc"], "date": v["date"], "complaint": "",
                            "detail": f"Temperature {vital(v, 'Temperature')}"}
                           for v in visits if vital(v, "Temperature")]}
    if t.startswith("Pulse"):
        return {"visits": [{"enc": v["enc"], "date": v["date"], "complaint": "", "detail": f"Pulse {vital(v, 'Pulse')}"}
                           for v in visits if vital(v, "Pulse")]}
    if t.startswith("Medicine load"):
        return {"visits": [{"enc": v["enc"], "date": v["date"], "complaint": "",
                            "detail": f"{len(v['meds'])} medicine(s): {', '.join(m['name'] for m in v['meds'])}"}
                           for v in visits]}
    if t.startswith("Recurrent musculoskeletal"):
        pat = r"pain|ache|spondyl|strain|cervical|shoulder|knee|back|joint|neck"
        return {"visits": visit_rows(visits, [v["enc"] for v in visits if re.search(pat, v["complaint"] + " " + v["provisional"] + " " + v["confirmed"], re.I)],
                                     lambda v: v["provisional"] or v["confirmed"])}
    if "not linked to a visit" in t:
        return {"reports": [{"label": r["label"], "likely_visit": r["likely_visit"] or "no match",
                             "labs": len(r["labs"])} for r in reports]}
    if t.startswith("Leg / ankle"):
        encs = set(encs_in(ev)) | {v["enc"] for v in visits if any(re.search("amlod", str(m["name"]), re.I) for m in v["meds"])}
        return {"visits": visit_rows(visits, sorted(encs), meds_text)}
    encs = encs_in(ev)
    if t.startswith("Temperature below"):
        return {"visits": visit_rows(visits, encs, lambda v: f"Temperature {vital(v, 'Temperature') or '-'}")}
    if t.startswith("No ") and "in the record (" in t:
        return {"visits": visit_rows(visits, encs, lambda v: (v["provisional"] or v["confirmed"] or "-")
                                     + (f" · Temperature {vital(v, 'Temperature')}" if vital(v, "Temperature") else "")),
                "note": f"Lab results in the whole record: {len(labs)}."}
    if encs and flag.get("kind") == "medication":
        return {"visits": visit_rows(visits, encs, meds_text)}
    if encs:
        return {"visits": visit_rows(visits, encs, lambda v: v["provisional"] or v["confirmed"])}
    return {}


# ── Rules (paraphrased from the pipeline code) ──────────────────────────────

SNAP = "dataanalysis/clinical_snapshot.py · build_snapshot()"
RULES = [
    (r": .* out of range$", "Condition control marker",
     "For each recorded condition that has a control marker in condition_registry.json → control_labs, every result of "
     "that marker is checked. Type 2 diabetes: tests whose name matches HbA1c / glyco / glucose / sugar, flagged when "
     "out of range in the HIGH direction. Values that are physically impossible are excluded. Level comes from the "
     "registry (high for diabetes). The condition's medicines are shown as \"on X\" only when they are on the latest prescription; otherwise as \"X last prescribed at #n, not on the latest prescription\". \"despite current therapy\" is added only when every shown result is dated after the medicines were first prescribed; a result on or before that date says so, and an undated result is not placed against the treatment.",
     SNAP),
    (r"pattern result (before it was documented|with no matching diagnosis)", "Document result ahead of the diagnosis",
     "For each condition with a control marker (registry control_labs: HbA1c/glucose for diabetes, triglycerides/LDL/"
     "cholesterol for dyslipidaemia, TSH for thyroid ...), a document result of that marker out of range in the adverse "
     "direction on a DATE before the condition was first documented (same-day visits are one day, so not "
     "'before'; 'before it was documented'), or when the record "
     "never names the condition ('no matching diagnosis'). Impossible values excluded. Level from the registry.",
     SNAP + " · registry control_labs"),
    (r"^Oral steroid with", "Medicine risk: systemic steroid",
     "A systemic steroid (registry steroids.systemic: Omnacortil, Wysolon, prednisolone, deflazacort, "
     "dexamethasone ...) at any visit, AND Hypertension or Type 2 diabetes is recorded, or the BP at that visit was "
     ">=140/90. Significance names the effect (raise BP / raise blood glucose). Level: medium.", SNAP + " · registry steroids"),
    (r"^Possible neuropathic", "Symptom: possible neuropathy",
     "A complaint mentions numbness / tingling / pins and needles / burning of the feet, hands, legs or palms "
     "(skin words such as itch, rash, patch are excluded, and visits diagnosed only as a skin condition are skipped). "
     "Medium when a cause condition (e.g. diabetes) or a nerve medicine is in the record, else info.",
     SNAP + " · registry neuropathic_symptoms"),
    (r"^Recurrent .* with repeated steroids", "Course pattern: skin condition and steroids",
     "A skin condition recorded at 2+ visits with steroids prescribed: oral (registry steroids.systemic) or potent "
     "topical (Lobet, clobetasol, betamethasone, mometasone ...). 'Steroid-dependent' is said only after 2 or more oral "
     "courses (one per clinical day) with the condition returning after them; after one course the note says only that "
     "it returned. Other notes: a potent topical at 3+ visits, oral steroids with Hypertension / diabetes. "
     "Level: medium.", SNAP + " · registry steroids"),
    (r"^(Dyslipidaemia|Low haemoglobin|Raised white cell|Low white cell|Abnormal platelet|Kidney|Liver test|Thyroid test|Glucose / HbA1c)",
     "Lab group: out-of-range results",
     "Out-of-range lab results not already reported by another flag, grouped by test family: triglycerides / LDL / "
     "VLDL / cholesterol high = dyslipidaemia; haemoglobin / PCV / RBC / HCT low = possible anaemia; TLC / WBC high = "
     "raised white cell count; similarly for platelets, kidney, liver, thyroid and glucose tests. Every report is read with "
     "the patient's recorded sex (not each scan's own age/sex line), so copies of one report get the same reading; the "
     "same report linked to two visits then counts once. Medium; info with "
     "'(check OCR)' when every value is physically implausible.", SNAP),
    (r"^Diagnosis differs between document and record", "Record quality: diagnosis mismatch",
     "At a visit, the diagnosis read from the scanned document neither contains nor is contained in the structured "
     "diagnosis, and the two are not one diagnosis family (registry diagnosis_refinements: LSC is a form of eczema, "
     "T2DM = diabetes, HTN = hypertension). Garbled OCR lines and vitals / signature lines are ignored. Level: medium.", SNAP),
    (r"^BP improved; earlier readings raised", "Blood pressure target",
     "Only for hypertension / diabetes / CKD / CAD / heart failure. Target 130/80 with diabetes / CKD / CAD / HF, else "
     "140/90. The latest reading is below target, at least one earlier reading was at or above it, and the last 3 "
     "readings are lower than the earlier ones (mean systolic down 5 mmHg or more). Level: info.",
     SNAP + " · registry bp_lower_target_conditions"),
    (r"not settling$", "Course pattern: condition not settling",
     "A non-chronic condition recorded at 3+ visits and still recorded at the latest visit. Reports how many different "
     "treatments were tried (antihistamines and emollients are supportive care and not counted) and at how many return "
     "visits the regimen changed. Medium at 4+ visits, else info.",
     SNAP + " · _course_patterns()"),
    (r"^Medicines only in documents", "Record quality: medicine not prescribed",
     "A medicine name found in a scanned document with no close match (75% or more similar) to any medicine in the "
     "structured prescriptions. Level: info.", SNAP),
    (r"medicine line\(s\) read as lab results", "Record quality: OCR filed medicines as labs",
     "Lab rows that could not be evaluated and whose name is really a medicine line (tab / cap / cream / gel / "
     "ointment / lotion / syrup / injection / drops / soap / powder, or the first word of a prescribed medicine). "
     "They are ignored in every analysis. Level: info.", SNAP),
    (r"^Diagnosis evolution", "Course pattern: diagnosis wording over time",
     "The visit diagnoses in order, with consecutive near-identical wordings (80% or more similar) merged into one run "
     "and garbled OCR text skipped. Shown when there are 2 to 5 runs across 3+ visits and they belong to the same "
     "problem. Level: info.", SNAP + " · _course_patterns()"),
    (r"^Implausible vital reading$", "Record quality: implausible vital in a document",
     "A vital read from a scanned document that an outpatient could not have (same limits as the structured check: "
     "temperature 93-108.5 F or 34.5-42.5 C, BP 60-260 / 30-160, pulse 30-220, SpO2 70-100). Treated as a recording "
     "or OCR error, not a finding. Level: info.", "dataanalysis/clinical_snapshot.py · vital_implausible()"),
    (r"^Daily dose above the usual maximum", "Medicine risk: dose above maximum",
     "For each prescribed oral, single-ingredient medicine: tablets per day from the regimen (e.g. 1-0-1 = 2, a half "
     "counts 0.5) x strength in mg (from the name or the resolved medicine) = daily dose. Flagged when it exceeds the "
     "usual adult maximum in registry max_daily_dose (e.g. etoricoxib 120 mg/day). SOS, combination and topical "
     "medicines are skipped. The evidence says whether the medicine is still on the latest prescription. Level: high.",
     "dataanalysis/clinical_snapshot.py · daily_dose_over_max()"),
    (r"^Oral NSAID with raised BP reading", "Medicine risk: NSAID with raised BP (retired)",
     "An oral NSAID was prescribed at a visit whose own BP reading was raised (systolic >=140 or diastolic >=90), and "
     "the record has no NSAID-risk condition (Hypertension, diabetes, CKD, heart failure, CAD, gout); with such a "
     "condition the stronger NSAID rule applies instead. Level: medium.", SNAP + " · registry nsaid_risk_conditions"),
    (r"^No .+ medicine prescribed since #", "Medicine gap: chronic treatment stopped appearing",
     "For a chronic condition that needs ongoing treatment (registry chronic_treatment_conditions), the condition's own "
     "medicines (registry condition_meds / class_treats) were last prescribed at an earlier visit, and after it there "
     "are 2+ visits with other prescriptions, or 1 such visit 60+ days later. Skipped when the latest visit's advice "
     "says to continue the previous medicines. For hypertension the BP readings recorded after that date are listed "
     "(or 'no BP reading recorded since'). Level: medium.", SNAP + " · registry chronic_treatment_conditions"),
    (r"no related lab tests in the record$", "Monitoring: no related lab tests",
     "A condition with related monitoring tests (registry control_labs / condition_labs) has none of them in the "
     "record and no guideline monitoring row covers it. For Hypertension with BP readings, BP shows control, so this "
     "is info only: kidney function, electrolytes and lipids are simply absent.", SNAP),
    (r"^Two NSAIDs on the same day \(route unclear\)", "Duplication: two NSAIDs, route not written (verify)",
     "As 'two oral NSAIDs on one day', but at least one line has no written form ('Diclofenac': tablet or gel?) and no "
     "identical prescription elsewhere in the record shows its form. Not assumed oral: reported for checking. Level: verify.",
     SNAP + " · medicine_profile.py (route, template inference)"),
    (r"\(route unclear\)$", "Route-dependent rule, route not written (verify)",
     "A rule that needs an oral route fired only when lines with no written form were counted as oral. Reported for "
     "checking; a high level becomes verify.", SNAP + " · route_rule()"),
    (r"^Two oral NSAIDs on the same day", "Duplication: two NSAIDs on one clinical day",
     "All visits on one date are one clinical day (two departments on the same day). Classes come from each "
     "product's active ingredients (brands expanded). Flagged when one day holds two different oral NSAIDs, or the same NSAID prescribed at two visits that day. Topical gels are not counted. "
     "The evidence says on how many days, and whether the medicines are on the latest prescription (all of them; "
     "'not all on the latest prescription' when only some are). Level: high.", SNAP + " · registry drug_rules.duplicate_classes"),
    (r"^Same-class medicines on the same day", "Duplication: same class on one clinical day",
     "On one clinical day, two medicines of a class in registry drug_rules.duplicate_classes (acid suppressants / PPIs, "
     "calcium / vitamin D, antihistamines, systemic steroids), or one of them prescribed at two visits that day. "
     "Spelling variants of one drug are merged. Ranked by risk: calcium / vitamin D twice and a PPI with a plain "
     "antacid are listed last, and when they are the only duplicates the flag is info, not medium.",
     SNAP + " · registry drug_rules.duplicate_classes"),
    (r"^Oral NSAIDs for about .* days in total", "Cumulative NSAID exposure",
     "The stated durations of oral NSAID prescriptions are added across the record, one value per clinical day (the "
     "longest that day). Flagged at 30 days or more (registry drug_rules.nsaid_cumulative_days). Level: medium.",
     SNAP + " · registry drug_rules"),
    (r"^Oral azole antifungal with a calcium-channel blocker", "Interaction: azole + calcium-channel blocker",
     "An oral azole (itraconazole, ketoconazole, fluconazole ...) on a day when a calcium-channel blocker (amlodipine, "
     "nifedipine ...) was prescribed the same day or within the 30 days before. Azoles raise CCB levels. 'On the latest "
     "prescription' only when the azole AND the CCB are both on it. Level: medium.",
     SNAP + " · registry drug_rules.systemic_azole, ccb"),
    (r"^Repeated oral azole antifungal without liver tests", "Monitoring: azole without liver tests",
     "An oral azole on 2+ clinical days, or for 14+ days in total by stated duration, and no liver test (bilirubin, "
     "AST, ALT, ALP, GGT or an LFT panel) anywhere in the record. Level: medium.", SNAP + " · registry drug_rules.azole_long_term"),
    (r"^Oral NSAID with ARB / ACE inhibitor in diabetes", "Interaction: NSAID + ARB / ACE inhibitor in diabetes",
     "Type 2 diabetes is recorded, and one clinical day holds an oral NSAID and an ARB / ACE inhibitor (telmisartan, "
     "losartan, ramipril ...). Medium; high when a diuretic is also prescribed that day ('triple whammy').",
     SNAP + " · registry drug_rules.arb_acei, diuretic"),
    (r"^Temperature below 95", "Vital: temperature below 95 °F",
     "A temperature written in °F that is possible (93 °F or more) but below 95 °F, in the structured record or a "
     "document: hypothermia range, or more often a measuring / entry error. Level: medium.",
     "dataanalysis/clinical_snapshot.py · _low_temp()"),
    (r"^No .+ in the record \(", "Missing investigation",
     "A finding normally followed by a test, with no such test anywhere in the record (registry expected_investigations): "
     "urinary frequency with no urine test; recurrent fungal infection on 2+ days with no glucose / HbA1c; fever (or "
     "temperature 100.4 °F+) with no blood "
     "count; recurring weakness with no haemoglobin; tingling / numbness with no glucose or B12. A record gap, not advice.",
     SNAP + " · registry expected_investigations"),
    (r"^Weight entry error", "Record quality: weight entry error",
     "A weight that goes A → B → about A (within 2%) within 30 days: B is a recording error, not a change, and is left "
     "out of BMI and the weight-change rule.", SNAP + " · clinical_thresholds.json weight_entry_error"),
    (r"^Height varies by", "Record quality: height entry error",
     "An adult's recorded height spread by more than 3 cm: a recording error; BMI uses the most frequent height.",
     SNAP + " · clinical_thresholds.json height_entry_error"),
    (r"attended off-date", "Record quality: follow-up marked missed but attended",
     "The source marked the follow-up MISSED, but the patient had another visit within ±7 days of the due date.",
     SNAP + " · core/encounters.recheck_followups"),
    (r"^Weight change of 5% or more within 30 days", "Vital: weight change",
     "Two consecutive weight readings 30 days or less apart that differ by 5% or more (15% or more is treated as an "
     "entry error by the 'weight readings inconsistent' rule instead). Level: medium.", SNAP),
    (r"^Oral NSAIDs with gastritis / acidity", "Medicine risk: NSAID with gastritis / acidity",
     "An oral NSAID on a clinical day whose complaint or diagnosis records gastritis, acidity, APD, GERD, reflux, "
     "dyspepsia or epigastric pain. Says on how many days and whether the NSAID is on the latest prescription. "
     "Level: medium.", SNAP),
    (r"^Steroid for a fungal complaint", "Medicine risk: steroid on a fungal complaint",
     "Any visit whose complaint or diagnosis says fungal / tinea / ringworm / candida, where ANY prescribed product "
     "contains a topical or systemic corticosteroid by its ingredients (combination creams such as Lobet GM or "
     "Eberfine M included). States whether a KOH / fungal test is in the record. Level: medium.", SNAP + " · registry steroids"),
    (r"^Diagnosis changed for the same complaint", "Course pattern: diagnosis changed for one complaint",
     "Consecutive visits whose complaints share a word (the same symptom or site) but whose diagnoses are neither the "
     "same, a near spelling, nor one refinement family (registry diagnosis_refinements). Level: medium.",
     SNAP + " · _course_patterns()"),
    (r"^Visual symptoms with diabetes", "Condition-linked symptom",
     "Type 2 diabetes is recorded, and a chief complaint mentions \"vision\" or \"eye\" (registry condition_symptoms, "
     "tag visual_change). Level: high.", SNAP + " · registry condition_symptoms"),
    (r"^Symptoms consistent with hyperglycaemia", "Condition-linked symptom (needs abnormal control)",
     "Type 2 diabetes is recorded, a chief complaint mentions dry mouth / thirst / weakness / fatigue / urination, AND "
     "the diabetes control marker (HbA1c / glucose) is out of range. Weakness alone, without an abnormal glucose, "
     "does not raise this flag. Level: medium.", SNAP + " · registry condition_symptoms"),
    (r"^Repeated oral NSAIDs", "Medicine risk: NSAIDs",
     "An oral NSAID (name matches the registry NSAID list: diclofenac, etoricoxib/Nucoxia/Etoshine, ibuprofen, "
     "aceclofenac …) is prescribed at 2 or more visits (the number of clinical days is shown too) AND the patient has an NSAID-risk condition (Hypertension, "
     "Type 2 diabetes, CKD, heart failure, CAD, gout). High when 2+ risk conditions, else medium. Topical gels are "
     "listed separately and do not count towards the visit total.", SNAP + " · registry nsaid_pattern, nsaid_risk_conditions"),
    (r"missed follow-up", "Follow-up adherence",
     "A follow-up the source marks MISSED is re-checked: if the patient had any other visit within ±7 days of the due "
     "date it is 'attended (off-date)' (shown as a data note), and only the rest are missed visits. Calls that did not reach the patient (call not "
     "connected / not answered) are NOT missed visits and are listed separately under the gaps. Level is high when the "
     "follow-up after the latest visit was missed, else medium.", SNAP),
    (r"^BP (?!improved)", "Blood pressure target",
     "Applied only to patients with hypertension, or diabetes / CKD / CAD / heart failure (anyone else's raised "
     "readings are not judged against a target). Target is 130/80 with diabetes, CKD, CAD or heart failure (registry "
     "bp_lower_target_conditions), else 140/90. The trend is read from the last 3 readings against the earlier ones "
     "(mean systolic, 5 mmHg or more counts as lower / higher), with the last change in BP treatment and the readings "
     "since it, not first against last. Flag when the LATEST reading has systolic ≥ target OR diastolic ≥ "
     "target. Impossible readings are excluded. \"Improved\" when the latest reading is lower than the first. "
     "Antihypertensives are shown as \"on X\" only when on the latest prescription, else \"X last prescribed at #n\". "
     "Medium when hypertension is a known diagnosis.", SNAP + " · registry bp_lower_target_conditions"),
    (r"^BMI", "Body mass index",
     "BMI = latest weight ÷ (most common recorded height in m)², adults only; weights marked as entry errors are left "
     "out. Population cut-offs (default Asian Indian: overweight ≥23, obese ≥25; BMI_POPULATION) with the WHO category "
     "alongside. Obese → medium; overweight → info.", SNAP + " · clinical_thresholds.json bmi"),
    (r"^Creatinine at upper limit", "Kidney function",
     "A creatinine result not already covered by another flag, not marked High, but at or above 95% of the report's "
     "upper limit. Context added: renal-caution medicines in the record (metformin, lithium, spironolactone …) and "
     "NSAIDs at 2+ visits. Significance names the kidney-risk conditions present.",
     SNAP + " · registry renal_caution_meds, kidney_risk_conditions"),
    (r"^Leg / ankle swelling", "Symptom that a medicine can cause",
     "A complaint mentions swelling at the leg / ankle / foot. A medicine known to cause it (calcium-channel blockers: "
     "amlodipine, nifedipine …; pioglitazone) is named only if it was first prescribed on or before the date of the "
     "symptom; one started later cannot explain it. Level: medium.",
     SNAP + " · registry symptom_med_links"),
    (r"^Impossible vital reading", "Record quality: impossible vital",
     "A structured vital outside what an outpatient could have: temperature in °F must be 93–108.5 (a bare 35–42 is "
     "Celsius); BP systolic 60–260 and diastolic 30–160; pulse 30–220; SpO₂ 70–100; weight 2–300 kg; height "
     "40–230 cm. A °F temperature of 34–42 is marked 'likely °C entered as °F' with its °F equivalent. These readings "
     "are excluded from every other rule.", "dataanalysis/clinical_snapshot.py · vital_implausible()"),
    (r"^Vitals data quality", "Record quality: height varies",
     "Recorded heights differ by 5 cm or more across visits.", SNAP),
    (r"^Pulse above 100", "Pulse",
     "Adult patient, a plausible pulse reading above 100 bpm at any visit. Level: info.", SNAP),
    (r"^Medicine load rising", "Course pattern: medicine count",
     "3 or more clinical days, and the latest day has at least 2 more distinct medicines than the first day (same-day "
     "visits counted together; 0-0-0 lines not counted). Level: info.",
     SNAP + " · _course_patterns()"),
    (r"^Other abnormal results", "Lab: remaining out-of-range results",
     "Out-of-range lab results not already shown under a condition or kidney flag. Level: info.", SNAP),
    (r"^Recurrent musculoskeletal pain", "Chronic pain",
     "A pain / musculoskeletal condition recorded at 5 or more visits. Level: info.", SNAP),
    (r"not linked to a visit", "Record quality: unlinked reports",
     "Documents that could not be matched to a visit, so their lab results have no date. Level: info.", SNAP),
]

KIND_LABEL = {"condition": "Condition", "risk": "Risk", "medication": "Medication", "followup": "Follow-up",
              "data": "Record quality", "lab": "Lab", "pattern": "Course pattern"}


def rule_for(title):
    for pattern, name, text, source in RULES:
        if re.search(pattern, title):
            return {"name": name, "text": text, "source": source}
    return {"name": "Snapshot rule", "text": "Rule-based flag from the snapshot.", "source": SNAP}


# ── Briefs ──────────────────────────────────────────────────────────────────

BRIEF_RULES = {
    "Course": ("_course_bullet()", "Patient age/sex, visit count with clinical days (same-day visits are one day) and span; recurring complaints (same symptom at 2+ "
               "visits, spelling variants merged at ≥80% similarity); what the first and latest visit were for."),
    "Conditions": ("_conditions_bullet()", "Active = recorded at the latest visit, known history, or on the long-term "
                   "list. Status by strongest evidence: confirmed → working diagnosis → known history → suspected → "
                   "document only. Others are listed as \"not recorded since\". Course-pattern flags are added."),
    "Key findings": ("_findings_bullet()", "Verified guideline rows; abnormal labs grouped by the condition they "
                     "control (registry control_labs, adverse direction only); \"despite X\" only when X started before "
                     "the result's date, otherwise the dates are stated; "
                     "possible undiagnosed conditions; then symptom and vital flags."),
    "Treatment": ("_treatment_bullet()", "Medicines at the latest visit grouped by the condition they treat (registry "
                  "class_treats / condition_meds only; never guessed); medicines for a condition not continued; the "
                  "top high/medium medicine-risk flag (⚠)."),
    "Gaps to check": ("_gaps_bullet()", "Follow-up flags and gaps (pending / missed / not-reached calls, tests advised "
                      "with no later result, abnormal results never repeated); high/medium record-quality flags; "
                      "unverified labs."),
}
THEME_EVIDENCE = {
    "Course": ["patient", "recurring_complaints", "timeline"],
    "Conditions": ["conditions", "health_history"],
    "Key findings": ["abnormal_labs", "possible_conditions", "latest_vitals"],
    "Treatment": ["medicines"],
    "Gaps to check": ["followup_gaps", "record_checks", "unverified_labs"],
}
AI_CHECKS = [
    "Removed if it suggests a medicine change (strip_care_suggestions).",
    "Removed if it states a number, date, dose or visit that is not in the evidence pack (verify_answer).",
    "Removed if it gives a condition a diagnosis status the record does not support.",
    "A second model pass checks each point against the evidence and fixes or drops it (self_check).",
    "Removed if it links a medicine or lab to a condition without registry support, or is generic filler.",
    "Removed if it calls a condition active / current when it was last recorded before the latest visit and is "
    "neither known history nor on the long-term list (drop_unsupported_active_claims).",
    "Ordering is checked against the dates: 'despite X' only when X started before the result (else rewritten to "
    "what the dates show); 'X stopped / started after Y' dropped unless the dates show that order; any stated cause "
    "('because', 'due to') dropped (drop_unsupported_timing_claims, and the self-check pass).",
    "Numbers must belong to the same fact: a lab value must sit with its own test in the evidence ('creatinine 7.6' is "
    "dropped when 7.6 is the HbA1c), and a span ('over 4 months') must be one the code computed (verify_answer).",
    "Every high and medium rule flag the model left out is added back, most important first (overdoses, dual NSAIDs "
    "and interactions before calcium or PPI duplicates; all high, at most two medium per theme), rewritten in the "
    "brief's own style; a point the model already made in other words is not added again (ensure_key_points).",
    "Citations are exact: a point from a rule flag cites exactly that flag's visits; otherwise only visits "
    "whose record holds what the point names. Citations the model wrote are removed where the visit does not "
    "hold the claim, and a lab from an undated / unlinked report is marked 'undated report', never given a #n "
    "(cite_points).",
    "Every high flag is its own point: a related point ('repeated NSAIDs') does not stand in for another high flag "
    "('two NSAIDs on the same day'); multi-day findings say on how many days.",
    "'Stopped' / 'not continued' is removed when the medicine's last course (date + duration) was still running at the "
    "next visit or at the latest visit (drop_unsupported_stop_claims).",
    "A medicine is named as a possible cause of a lab result only if it was taken within that drug class's window "
    "before the test (e.g. a steroid within 7 days for white cells); other cause clauses are removed "
    "(drop_implausible_cause_claims).",
    "A medicine already present at the first recorded visit, or with a known (K/C/O) condition, is 'first recorded on "
    "<date>', never 'started' (use_first_recorded_wording).",
    "A citation must point at an visit holding the stated fact: a stated value (7.6, 150/93) pins it to the "
    "visits holding that value; otherwise any record medicine, diagnosis or complaint word; wrong #n are replaced "
    "or removed (cite_points).",
    "Key findings hold clinical findings only; implausible temperatures (below 95 °F / above 108 °F) and other record or entry errors (impossible readings, unit errors, document "
    "mismatches) are moved to Gaps to check (route_record_issues).",
    "Conditions are named in the record's own words ('Gastritis', not 'GERD / acidity'), and a bare health-history "
    "item that repeats a condition is not listed again (use_record_terms). Spelling variants ('Reecalcitrant Tinea') "
    "are merged into one condition before visits are counted.",
    "'Improved on X' is kept only if X was prescribed before the earlier reading; otherwise 'on X' is removed.",
    "Citations: from a same-day pair, only the visit that holds the drug or reading; a 'latest' claim cites the "
    "latest visit; a missed-follow-up point cites every missed follow-up.",
    "The BMI flag is always kept, Conditions names every confirmed and known diagnosis, and Gaps to check names each "
    "issue once (dedupe_gaps).",
]


def split_bullets(lines):
    out = []
    for line in lines:
        m = re.match(r"^\s*[-*]\s*\*\*(.+?):\*\*\s*(.*)$", line.strip())
        if m:
            points = (p.strip().rstrip(".").strip() for p in re.split(r";\s+(?![^()]*\))", m.group(2)))
            points = [p for p in points if p]  # "." alone is no point; a theme with none is not shown
            if points:
                out.append({"theme": m.group(1), "points": points})
    return out


def kf_links(points, flags):
    """For each Key-finding point, the indices of the snapshot flags it stands for: the flag it states, plus every
    high / verify / medium flag of the same clinical problem (clinical_insights.finalise_brief groups them)."""
    from clinical_insights import _point_flag
    out = []
    for pt in points:
        f = _point_flag(pt, flags) if not pt.startswith("+") else None
        idx = []
        if f:
            idx = [next(i for i, g in enumerate(flags) if g is f)]
            if f.get("problem"):
                idx += [i for i, g in enumerate(flags) if g is not f and g.get("problem") == f["problem"]
                        and g.get("level") in ("high", "verify", "medium")]
        out.append(idx)
    return out


def ai_section(markdown, heading):
    m = re.search(rf"## {heading}\s+(.+?)(?=\n## |\n\nRetrospective|\Z)", markdown or "", re.S)
    return [l for l in (m.group(1).splitlines() if m else []) if l.strip()]


def theme_evidence(evidence, theme, flags, kinds):
    data = {k: evidence.get(k) for k in THEME_EVIDENCE.get(theme, []) if evidence.get(k)}
    related = [f["title"] for f in flags if f.get("kind") in kinds]
    return {"evidence": data, "flags": related}


THEME_FLAG_KINDS = {"Course": {"pattern"}, "Conditions": {"pattern", "condition"},
                    "Key findings": {"condition", "risk", "lab"}, "Treatment": {"medication"},
                    "Gaps to check": {"followup", "data"}}


def corrections(metrics):
    """{'total', 'text'} from the validator's counts (review_rules.sanitize_review)."""
    names = {"citations_corrected": "citations corrected", "validator_removed_points": "unsupported points removed",
             "flags_added": "missing flags added to Key findings"}
    parts = [(names.get(k, k.replace("_", " ")), v) for k, v in metrics.items() if isinstance(v, int) and v]
    return {"total": sum(v for _, v in parts), "text": ", ".join(f"{v} {k}" for k, v in parts) or "none needed"} if metrics else None


def with_areas(themes, tagged):
    """Each theme's point areas from the pipeline tags (same split), so a tile click finds them."""
    by_theme = {t["theme"].casefold(): t["points"] for t in tagged or []}
    for b in themes:
        pts = by_theme.get(b["theme"].casefold()) or []
        b["areas"] = [pts[k]["areas"] if k < len(pts) else ["general"] for k in range(len(b["points"]))]
    return themes


def merged_brief(ai, brief, flags, rule_flags):
    """One Patient Brief from the AI brief and the rule-based brief, in the standard theme order: every AI point (checked
    against the evidence), then each rule point the AI did not already make. A rule point counts as made when any AI
    point, in any theme, states the same rule flag, or one in its theme states the same fact in other words (most of
    the content words, or several of them); that AI point is marked 'both'. Without an AI brief, the rule brief alone."""
    from clinical_insights import _THEMES, _content_words, _point_flag
    strip = lambda t: re.sub(r"\s*\((?:#\d+(?:,\s*)?)+\)|#\d+", "", t)

    def same_fact(a, r):
        wa, wr = _content_words(strip(a)), _content_words(strip(r))
        shared = len(wa & wr)
        return bool(wa and wr) and (shared / len(wa | wr) >= 0.5 or (shared >= 3 and shared / len(wa | wr) >= 0.25))

    def joined(points):  # a rule point split mid-sentence ('result undated, so …') rejoins the point before it
        out = []
        for t in points:
            if out and t[:1].islower():
                out[-1] = (out[-1][0] + "; " + t, out[-1][1])
            else:
                out.append((t, len(out)))
        return [t for t, _ in out]

    rank = {t.casefold(): i for i, t in enumerate(_THEMES)}
    themes = sorted({b["theme"] for b in ai} | {b["theme"] for b in brief}, key=lambda t: rank.get(t.casefold(), 99))
    get = lambda src, t: next((b for b in src if b["theme"].casefold() == t.casefold()), None)
    merged = {}
    for t in themes:
        a = get(ai, t)
        merged[t] = [{"text": x, "src": "ai", "areas": (a.get("areas") or [[]] * len(a["points"]))[k],
                      "flag": _point_flag(x, rule_flags) if not x.startswith("+") else None}
                     for k, x in enumerate(a["points"])] if a else []
    for t in themes:
        r = get(brief, t)
        if not r:
            continue
        r_areas = dict(zip(r["points"], r.get("areas") or []))
        for x in joined(r["points"]):
            if re.fullmatch(r"\+\d+ more", x.strip()):
                continue  # the rule list's own '+N more' means nothing in the merged list
            fr = _point_flag(x, rule_flags) if not x.startswith("+") else None
            hit = next((p for ps in merged.values() for p in ps if p["src"] != "rule" and fr is not None and p["flag"] is fr), None)
            hit = hit or next((p for p in merged[t] if p["src"] != "rule" and (
                (x.startswith("+") and p["text"].startswith("+")) or same_fact(p["text"], x))), None)
            if hit:
                hit["src"] = "both"
            else:
                merged[t].append({"text": x, "src": "rule", "areas": r_areas.get(x.split("; ")[0], []), "flag": fr})
    out = []
    for t in themes:
        base, r = get(ai, t) or get(brief, t), get(brief, t)
        ps = merged[t]
        if t.casefold() == "key findings":
            # A rule finding whose concerns an AI finding already links to is that finding (the AI one is kept).
            links = kf_links([p["text"] for p in ps], flags)
            keep = []
            for p, l in zip(ps, links):
                cover = next((q for q, ql in zip(ps, links) if q is not p and q["src"] != "rule" and ql and set(l) <= set(ql)), None)
                if p["src"] == "rule" and l and cover:
                    cover["src"] = "both"
                    continue
                keep.append(p)
            ps = keep
        ps = [p for p in ps if not p["text"].startswith("+")] + [p for p in ps if p["text"].startswith("+")]  # '+N more' last
        m = {"theme": t, "points": [p["text"] for p in ps], "src": [p["src"] for p in ps], "areas": [p["areas"] for p in ps],
             "rule": (r or {}).get("rule"), "evidence": base.get("evidence"), "flags": base.get("flags")}
        if t.casefold() == "key findings":
            m["links"] = kf_links(m["points"], flags)
        out.append(m)
    return out


def build_patient(d):
    import area_tiles
    import medicine_grid
    record, ev, snap = d["record"], d["evidence"], d["snapshot"]
    visits, reports = raw_visits(record), raw_reports(record)
    all_rows = all_labs(visits, reports)
    # Prescription lines the OCR filed as lab results stay out of the Lab Results tab (they are not tests); the
    # "medicine line(s) read as lab results" flag still shows them as its evidence.
    import clinical_snapshot
    rx_words = {clinical_snapshot._first_word(m["name"]) for v in visits for m in v["meds"]}
    labs = [l for l in all_rows if not clinical_snapshot.is_medicine_line(l, rx_words)]
    flags = []
    for f in snap.get("flags") or []:
        flags.append({**f, "kind_label": KIND_LABEL.get(f.get("kind"), f.get("kind")),
                      "rule": rule_for(f["title"]), "raw": raw_for_flag(f, visits, all_rows, reports)})
    brief = []
    for b in split_bullets(d["deterministic_brief"]):
        fn, text = BRIEF_RULES.get(b["theme"], ("", ""))
        if b["theme"].casefold() == "key findings":
            b["links"] = kf_links(b["points"], flags)
        brief.append({**b, "rule": {"name": b["theme"], "text": text,
                                    "source": f"dataanalysis/clinical_insights.py · {fn}"},
                      **theme_evidence(ev, b["theme"], flags, THEME_FLAG_KINDS.get(b["theme"], set()))})
    with_areas(brief, d.get("brief_tags"))
    review = d.get("review") or {}
    md = review.get("raw_markdown", "")
    ai = []
    for b in split_bullets(ai_section(md, "Clinical Insights")):
        if b["theme"].casefold() == "key findings":
            b["links"] = kf_links(b["points"], flags)
        ai.append({**b, **theme_evidence(ev, b["theme"], flags, THEME_FLAG_KINDS.get(b["theme"], set()))})
    with_areas(ai, (d.get("area_tags") or {}).get("insights"))
    abnormal = []
    for g in ev.get("abnormal_labs") or []:
        abnormal.append({**g, "raw": [l for l in labs if str(l["test"]).casefold() == str(g["test_as_written"]).casefold()
                                      or (l["value"] or "").split(" ")[0] == g["value"].split(" ")[0] and g["test"] in str(l["test"])]})
    journey = [{"enc": j.get("visit_index"), "date": j.get("visit_date"),
                "changes": [c.get("text") for c in j.get("changes") or []],
                "active": j.get("all_active_meds") or []} for j in d.get("medication_journey") or []]
    hh = record.get("health_history") or {}
    return {
        "id": d["patient_id"],
        "age_sex": ev["patient"]["age_sex"],
        "encounters": ev["patient"]["encounters"],
        "clinical_days": ev["patient"].get("clinical_days") or ev["patient"]["encounters"],
        "first_visit": ev["patient"]["first_visit"],
        "latest_visit": ev["patient"]["latest_visit"],
        "active": snap.get("active") or [],
        "long_term": [c.get("name") for c in record.get("long_term_conditions") or []],
        "health_history": [k.replace("_", " ").capitalize() + (f" ({v.get('notes') or v.get('qty')})" if (v.get('notes') or v.get('qty')) else "")
                           for k, v in hh.items() if isinstance(v, dict) and v.get("present")],
        "latest_vitals": ev.get("latest_vitals") or {},
        # Prescribed on the latest date, or an earlier course still running then; 0-0-0 + duration noted.
        "current_meds": [c["name"] + (f" ({c['note']})" if c.get("note") else "") for c in snap.get("current_medicines") or []]
                        or [m["name"] for m in ev.get("medicines") or [] if m.get("at_latest_visit")],
        "visits": visits,
        "reports": reports,
        "labs": labs,
        "journey": journey,
        "side_effects": record.get("possible_side_effects") or {},
        "flags": flags,
        "guideline": (snap.get("guideline") or {}).get("rows") or [],
        "guideline_version": (snap.get("guideline") or {}).get("version", ""),
        "abnormal_labs": abnormal,
        "brief": brief,
        "ai": ai,
        "merged": merged_brief(ai, brief, flags, ev.get("rule_flags") or []),
        "ai_overview": " ".join(ai_section(md, "Overview")),
        "ai_model": review.get("model_name", ""),
        "ai_generated": review.get("generated_at", "")[:16].replace("T", " "),
        "corrections": corrections(d.get("review_metrics") or {}),
        "citation_log": (d.get("review_metrics") or {}).get("citation_log") or [],
        "lab_counts": ev.get("lab_counts") or {},
        "evidence_pack": ev,
        "course": d.get("clinical_course") or {},
        "snap_span": snap.get("span", ""),
        "snap_dates": snap.get("dates") or [],
        "snap_tiles": snap.get("tiles") or [],
        "snap_concerns": snap.get("concerns"),
        "snap_concern_groups": snap.get("concern_groups") or [],
        "snap_timeline": [{"label": r["label"], "kind": r.get("kind", ""),
                           "encs": sorted(int(n) for n in re.findall(r"\d+", str(r.get("encounters"))))}
                          for r in snap.get("timeline") or []],
        "ai_abnormal": [re.sub(r"^\s*-\s*", "", l) for l in ai_section(md, "Abnormal Results")],
        "ai_abnormal_areas": [t["areas"] for t in (d.get("area_tags") or {}).get("labs") or []],
        "tiles_html": area_tiles.tiles_html(snap.get("tiles") or []),
        # Prescriber specialization -> tag colour, the same in the grid and in the AI tab (no names).
        "spec_colours": medicine_grid.spec_palette(snap.get("specialties")),
        "med_grid_html": medicine_grid.medicine_grid_html(snap.get("medicine_timeline") or [], snap.get("dates") or [],
                                                          specialties=snap.get("specialties")),
        "ai_evidence": [re.sub(r"^\s*-\s*", "", l) for l in ai_section(md, "Evidence")],
        "ai_markdown": md,
        "discrepancies": [{"enc": v["enc"], "date": v["date"], "text": t} for v in visits for t in v["discrepancies"]],
    }


def build_view(patient_data, review=None, today=None):
    """The view for one patient: build_patient over the live pipeline output (review: the cached review or None),
    plus the two-tier summary (summary_tiers.build_summary: quick + detailed, coverage-checked)."""
    from clinical_insights import build_insight_evidence, deterministic_insights
    from clinical_snapshot import build_clinical_course, build_snapshot
    import snapshot_view

    evidence = build_insight_evidence(patient_data)
    snapshot = build_snapshot(patient_data) or {}
    brief = deterministic_insights(evidence)
    flags = snapshot.get("flags") or []
    course = build_clinical_course(patient_data)
    view = build_patient({
        "patient_id": str(patient_data.get("patient_id", "")),
        "record": {k: v for k, v in patient_data.items() if not str(k).startswith("_")},
        "medication_journey": patient_data.get("_medication_journey_cache"),
        "evidence": evidence,
        "snapshot": snapshot,
        "clinical_course": course,
        "deterministic_brief": brief,
        "review": review,
        "review_metrics": patient_data.get("_review_metrics") or {},
        "area_tags": snapshot_view.tag_review((review or {}).get("raw_markdown", ""), flags),
        "brief_tags": snapshot_view.tag_brief(brief, flags),
    })
    view.pop("evidence_pack", None)  # the themes carry the parts they use; the whole pack is not shown
    import summary_tiers
    # Raises summary_tiers.CoverageError when a flag or abnormal lab would be lost from either tier.
    view["summary"] = summary_tiers.build_summary(snapshot, evidence, course, view, today=today,
                                                  ai_quick=(review or {}).get("quick_ai"))
    view["summary"]["detailed"]["reference"]["ai_checks"]["checks"] = AI_CHECKS if view.get("ai") else []
    # Automatic checks of the brief against the record (brief_checks); never fails the view.
    try:
        import brief_checks
        view["brief_checks"] = brief_checks.check_view(view)
        if view["brief_checks"]:
            logger.warning("brief checks: %s", brief_checks.summarise(view["brief_checks"]))
    except Exception as exc:
        logger.warning("brief checks not run: %s", exc)
        view["brief_checks"] = []
    return {"patient": view, "ai_checks": AI_CHECKS}
