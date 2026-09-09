"""
Stage 11-13: numerical trend analysis, done entirely in Python — never
delegated to the LLM (per spec, Step 11).
"""
from __future__ import annotations
from collections import defaultdict


def analyze_lab_trends(timeline: dict) -> dict:
    """
    For each normalized lab test name, computes first/last/min/max,
    absolute + percentage change, direction ('gain' | 'drop' | 'stable' | 'single_visit'),
    abnormal counts, units, and reference intervals.
    Handles both quantitative and qualitative tests.
    """
    series = defaultdict(list)       # name -> [(date, numeric_value, abnormal), ...]
    qual_series = defaultdict(list)  # name -> [(date, qual_value, abnormal), ...]
    units = {}
    ref_ranges = {}

    for visit in timeline["visits"]:
        for lab in visit["laboratory"]:
            date = lab["event_date"] or visit["visit_date"]
            name = lab["name"]

            if lab.get("unit") and name not in units:
                units[name] = lab["unit"]
            if (lab.get("reference_low") is not None or lab.get("reference_high") is not None) and name not in ref_ranges:
                ref_ranges[name] = (lab.get("reference_low"), lab.get("reference_high"))

            if lab.get("value") is not None:
                series[name].append((date, lab["value"], lab.get("abnormal")))
            elif lab.get("qualitative_value") is not None:
                qual_series[name].append((date, lab["qualitative_value"], lab.get("abnormal")))

    trends = {}

    # 1. Numeric series trends
    for name, points in series.items():
        dated = [p for p in points if p[0]]
        dated.sort(key=lambda p: p[0])
        if not dated:
            continue

        values = [p[1] for p in dated]
        first_val, last_val = values[0], values[-1]
        abs_change = round(last_val - first_val, 4)
        pct_change = round((abs_change / first_val) * 100, 2) if first_val else None

        unit_str = units.get(name, "")
        if len(dated) < 2:
            direction = "single_visit"
            trend_summary = "Single documented reading"
        elif abs_change > 0:
            direction = "gain"
            pct_str = f" (+{pct_change}%)" if pct_change is not None else ""
            trend_summary = f"gain of +{abs_change} {unit_str}{pct_str}"
        elif abs_change < 0:
            direction = "drop"
            pct_str = f" ({pct_change}%)" if pct_change is not None else ""
            trend_summary = f"drop of -{abs(abs_change)} {unit_str}{pct_str}"
        else:
            direction = "stable"
            trend_summary = "stable (no change)"

        r_low, r_high = ref_ranges.get(name, (None, None))

        trends[name] = {
            "first_value": first_val,
            "first_date": dated[0][0],
            "latest_value": last_val,
            "latest_date": dated[-1][0],
            "min_value": min(values),
            "max_value": max(values),
            "absolute_change": abs_change,
            "percentage_change": pct_change,
            "direction": direction,
            "trend_summary": trend_summary,
            "unit": unit_str,
            "ref_low": r_low,
            "ref_high": r_high,
            "num_measurements": len(dated),
            "num_abnormal": sum(1 for p in dated if p[2]),
            "all_points": [{"date": d, "value": v, "abnormal": a} for d, v, a in dated],
        }

    # 2. Qualitative series (e.g. Urine routine findings: ABSENT, CLEAR, 0-1 /HPF)
    for name, q_points in qual_series.items():
        if name in trends:
            continue  # prefer numeric if both exist
        dated_q = [p for p in q_points if p[0]]
        dated_q.sort(key=lambda p: p[0])
        if not dated_q:
            continue

        first_q = dated_q[0][1]
        latest_q = dated_q[-1][1]
        r_low, r_high = ref_ranges.get(name, (None, None))
        unit_str = units.get(name, "")

        trends[name] = {
            "first_value": first_q,
            "first_date": dated_q[0][0],
            "latest_value": latest_q,
            "latest_date": dated_q[-1][0],
            "min_value": None,
            "max_value": None,
            "absolute_change": None,
            "percentage_change": None,
            "direction": "qualitative_observation",
            "trend_summary": f"Observed: {latest_q}",
            "unit": unit_str,
            "ref_low": r_low,
            "ref_high": r_high,
            "num_measurements": len(dated_q),
            "num_abnormal": sum(1 for p in dated_q if p[2]),
            "all_points": [{"date": d, "value": v, "abnormal": a} for d, v, a in dated_q],
        }

    return trends



def analyze_medication_longitudinal(timeline: dict) -> dict:
    """
    Tracks each medication across visits: first appearance, continued,
    dose/frequency changes. Never claims a medication was "discontinued"
    unless it appeared in an earlier visit and the record explicitly
    shows it stopped (this pipeline only has positive evidence of
    prescriptions, so absence in a later visit is reported as
    "not documented in later visit(s)", not asserted as stopped, per
    spec Step 12/15).
    """
    by_med = defaultdict(list)  # name -> [(visit_date, dose, unit, frequency), ...]

    for visit in timeline["visits"]:
        for med in visit["medications"]:
            by_med[med["name"]].append({
                "visit_id": visit["visit_id"],
                "visit_date": visit["visit_date"],
                "dose": med["dose"],
                "unit": med["unit"],
                "frequency": med["frequency"],
                "needs_verification": med["needs_verification"],
            })

    result = {}
    for name, records in by_med.items():
        records.sort(key=lambda r: (r["visit_date"] is None, r["visit_date"] or ""))
        changes = []
        for prev, curr in zip(records, records[1:]):
            if prev["dose"] != curr["dose"] and None not in (prev["dose"], curr["dose"]):
                changes.append({
                    "type": "dose_change",
                    "from": f"{prev['dose']}{prev['unit'] or ''}",
                    "to": f"{curr['dose']}{curr['unit'] or ''}",
                    "at_visit": curr["visit_id"],
                    "visit_date": curr["visit_date"],
                })
            if prev["frequency"] != curr["frequency"] and None not in (prev["frequency"], curr["frequency"]):
                changes.append({
                    "type": "frequency_change",
                    "from": prev["frequency"],
                    "to": curr["frequency"],
                    "at_visit": curr["visit_id"],
                    "visit_date": curr["visit_date"],
                })

        result[name] = {
            "first_visit_date": records[0]["visit_date"],
            "latest_visit_date": records[-1]["visit_date"],
            "num_visits_documented": len(records),
            "status": "continued" if len(records) > 1 else "single_visit_only",
            "changes": changes,
            "any_needs_verification": any(r["needs_verification"] for r in records),
        }

    return result


def analyze_diagnosis_longitudinal(timeline: dict) -> dict:
    """
    Tracks each diagnosis across visits. Only reports "persistent" when
    the diagnosis is documented as present at more than one visit; a
    diagnosis missing from a later visit's documents is reported as
    "not re-documented" rather than "resolved" (spec Step 13/15 — never
    infer resolution from absence).
    """
    all_visit_dates = [v["visit_date"] for v in timeline["visits"]]
    by_dx = defaultdict(list)

    for visit in timeline["visits"]:
        present_here = set()
        for dx in visit["diagnoses"]:
            if dx["assertion"] in ("present", "historical"):
                present_here.add(dx["name"])
        for name in present_here:
            by_dx[name].append(visit["visit_date"])

    result = {}
    for name, visit_dates in by_dx.items():
        result[name] = {
            "first_documented": min([d for d in visit_dates if d], default=None),
            "last_documented": max([d for d in visit_dates if d], default=None),
            "num_visits_documented": len(visit_dates),
            "total_visits_in_record": len(all_visit_dates),
            "status": "persistent" if len(visit_dates) > 1 else "documented_once",
        }

    return result
