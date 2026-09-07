"""
Stage 11-13: numerical trend analysis, done entirely in Python — never
delegated to the LLM (per spec, Step 11).
"""
from __future__ import annotations
from collections import defaultdict


def analyze_lab_trends(timeline: dict) -> dict:
    """
    For each normalized lab test name, computes first/last/min/max,
    absolute + percentage change, direction, count, abnormal count, and
    first/latest measurement dates.

    Returns: {lab_name: {...trend stats...}}
    """
    series = defaultdict(list)  # name -> [(date, value, abnormal), ...]

    for visit in timeline["visits"]:
        for lab in visit["laboratory"]:
            if lab["value"] is None:
                continue
            date = lab["event_date"] or visit["visit_date"]
            series[lab["name"]].append((date, lab["value"], lab.get("abnormal")))

    trends = {}
    for name, points in series.items():
        # sort by date; None dates go last and don't determine direction
        dated = [p for p in points if p[0]]
        dated.sort(key=lambda p: p[0])
        if not dated:
            continue

        values = [p[1] for p in dated]
        first_val, last_val = values[0], values[-1]
        abs_change = round(last_val - first_val, 4)
        pct_change = round((abs_change / first_val) * 100, 2) if first_val else None

        if len(dated) < 2:
            direction = "insufficient_data"
        elif abs_change > 0:
            direction = "increasing"
        elif abs_change < 0:
            direction = "decreasing"
        else:
            direction = "stable"

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
            "num_measurements": len(dated),
            "num_abnormal": sum(1 for p in dated if p[2]),
            "all_points": [{"date": d, "value": v, "abnormal": a} for d, v, a in dated],
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
