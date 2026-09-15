"""
Stage 11-13: numerical trend analysis, done entirely in Python — never
delegated to the LLM.

Rules enforced here:
  Rule 10  — lab abnormality is a 5-state status (LOW/NORMAL/HIGH/CRITICAL/UNKNOWN),
             never inferred from generic interpretation text.
  Rule 11  — longitudinal comparison only when test name, unit, and visit date
             all match; same-date measurements from a SINGLE visit are never
             compared as a trend.
  Rule 12  — direction vocabulary: improving/worsening/stable/
             single_visit/insufficient_data (not gain/drop).
  Rule 13  — diagnosis status uses full spec vocabulary.
"""
from __future__ import annotations
from collections import defaultdict
from clinical_nlp.labs import _DEFAULT_REFERENCE_INTERVALS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lab_status(value: float, ref_low, ref_high) -> str:
    """
    Rule 10: compute LOW / NORMAL / HIGH / CRITICAL / UNKNOWN.
    CRITICAL thresholds are ±50 % beyond the reference boundary.
    Returns UNKNOWN when the reference range is completely absent.
    """
    if value is None:
        return "UNKNOWN"
    if ref_low is None and ref_high is None:
        return "UNKNOWN"

    if ref_high is not None and value > ref_high:
        critical_hi = ref_high + 0.50 * (ref_high - (ref_low or 0))
        return "CRITICALLY_HIGH" if value >= critical_hi else "HIGH"

    if ref_low is not None and value < ref_low:
        critical_lo = ref_low - 0.50 * ((ref_high or ref_low) - ref_low)
        return "CRITICALLY_LOW" if value <= critical_lo else "LOW"

    return "NORMAL"


def _units_compatible(u1: str, u2: str) -> bool:
    """
    Rule 11: require units to be the same (case-insensitive) before
    comparing values across visits.  A missing unit on one side is
    treated as compatible (benefit of the doubt — the OCR may have
    dropped it) but two *different* non-empty units are incompatible.
    """
    if not u1 or not u2:
        return True
    return u1.strip().lower() == u2.strip().lower()


# ---------------------------------------------------------------------------
# Stage 11: Laboratory trends
# ---------------------------------------------------------------------------

def analyze_lab_trends(timeline: dict) -> dict:
    """
    For each normalized lab test name build a per-visit series, then
    compute cross-visit trends only.

    Rule 11 — same-visit / same-date measurements are collapsed to a
    single data point (highest-confidence one wins) so they can NEVER
    produce a spurious percentage-change trend.

    Rule 11 — if units across two visits are incompatible the trend is
    reported as insufficient_data rather than silently comparing numbers.

    Rule 10 — abnormality is returned as the 5-state status string.

    Rule 12 — direction vocabulary: improving / worsening / stable /
    single_visit / insufficient_data.
    """
    # Collect per-test, per-visit best measurement
    # Structure: {test_name: {visit_id: best_point_dict}}
    by_test_visit: dict[str, dict[str, dict]] = defaultdict(dict)
    units_seen: dict[str, set] = defaultdict(set)
    ref_ranges: dict[str, tuple] = {}
    raw_names: dict[str, str] = {}
    qual_series: dict[str, dict[str, dict]] = defaultdict(dict)

    for visit in timeline["visits"]:
        visit_id   = visit["visit_id"]
        visit_date = visit["visit_date"]

        for lab in visit["laboratory"]:
            name = lab["name"]
            date = lab.get("event_date") or visit_date
            if name not in raw_names and lab.get("name_raw"):
                raw_names[name] = lab["name_raw"]

            r_low  = lab.get("reference_low")
            r_high = lab.get("reference_high")
            if name not in ref_ranges and (r_low is not None or r_high is not None):
                ref_ranges[name] = (r_low, r_high)

            unit = (lab.get("unit") or "").strip()
            if unit:
                units_seen[name].add(unit.lower())

            if lab.get("value") is not None:
                point = {
                    "date":       date,
                    "visit_id":   visit_id,
                    "value":      lab["value"],
                    "unit":       unit,
                    "confidence": lab.get("confidence", 0),
                    "ref_low":    r_low,
                    "ref_high":   r_high,
                }
                existing = by_test_visit[name].get(visit_id)
                if existing is None or point["confidence"] > existing["confidence"]:
                    by_test_visit[name][visit_id] = point

            elif lab.get("qualitative_value") is not None:
                qpoint = {
                    "date":     date,
                    "visit_id": visit_id,
                    "value":    lab["qualitative_value"],
                    "abnormal": lab.get("abnormal"),
                    "unit":     unit,
                    "confidence": lab.get("confidence", 0),
                }
                existing = qual_series[name].get(visit_id)
                if existing is None or qpoint["confidence"] > existing["confidence"]:
                    qual_series[name][visit_id] = qpoint

    trends: dict[str, dict] = {}

    # --- Numeric trends ---
    for name, visit_map in by_test_visit.items():
        points = sorted(visit_map.values(),
                        key=lambda p: (p["date"] is None, p["date"] or ""))
        if not points:
            continue

        r_low, r_high = ref_ranges.get(name, (None, None))
        if r_low is None and r_high is None and name in _DEFAULT_REFERENCE_INTERVALS:
            r_low, r_high = _DEFAULT_REFERENCE_INTERVALS[name]
        unit_str = next(iter(units_seen[name]), "") if len(units_seen[name]) == 1 else \
                   " | ".join(sorted(units_seen[name]))

        # Rule 11: check unit compatibility across all collected points
        all_units = [p["unit"] for p in points if p["unit"]]
        units_incompatible = len(set(u.lower() for u in all_units)) > 1

        first_p, last_p = points[0], points[-1]
        first_val, last_val = first_p["value"], last_p["value"]

        # Compute status for first and last
        ref_lo_f, ref_hi_f = first_p.get("ref_low") or r_low, first_p.get("ref_high") or r_high
        ref_lo_l, ref_hi_l = last_p.get("ref_low") or r_low, last_p.get("ref_high") or r_high
        first_status = _lab_status(first_val, ref_lo_f, ref_hi_f)
        latest_status = _lab_status(last_val, ref_lo_l, ref_hi_l)

        all_points_out = []
        for p in points:
            rl = p.get("ref_low") or r_low
            rh = p.get("ref_high") or r_high
            all_points_out.append({
                "date":   p["date"],
                "value":  p["value"],
                "unit":   p["unit"],
                "status": _lab_status(p["value"], rl, rh),
            })

        dates_distinct = (
            first_p["date"] is not None
            and last_p["date"] is not None
            and first_p["date"] != last_p["date"]
        )

        if len(points) < 2 or units_incompatible or not dates_distinct:
            if len(points) < 2:
                direction    = "single_visit"
                trend_summary = "Single documented reading"
            elif not dates_distinct:
                direction    = "same_date_readings"
                trend_summary = f"Multiple measurements on same date ({first_p['date']}): {first_val} and {last_val} {unit_str}"
            else:
                direction    = "insufficient_data"
                trend_summary = "Trend calculation blocked: incompatible units across visits"
            pct_change   = None
            abs_change   = None
        else:
            abs_change = round(last_val - first_val, 4)
            pct_change = round((abs_change / first_val) * 100, 2) if first_val else None

            # Rule 12: direction relative to reference range, not just numeric direction
            if abs_change == 0:
                direction = "stable"
                trend_summary = "stable (no change)"
            elif _lab_status(first_val, r_low, r_high) in ("HIGH", "CRITICALLY_HIGH") and abs_change < 0:
                direction = "improving"
                trend_summary = f"improving: dropped by {abs(abs_change)} {unit_str} ({pct_change}%)"
            elif _lab_status(first_val, r_low, r_high) in ("LOW", "CRITICALLY_LOW") and abs_change > 0:
                direction = "improving"
                trend_summary = f"improving: rose by {abs_change} {unit_str} (+{pct_change}%)"
            elif _lab_status(first_val, r_low, r_high) == "NORMAL" and \
                 _lab_status(last_val, r_low, r_high) != "NORMAL":
                direction = "worsening"
                trend_summary = f"worsening: moved out of reference range ({latest_status})"
            elif abs_change > 0:
                direction = "worsening" if _lab_status(last_val, r_low, r_high) in ("HIGH", "CRITICALLY_HIGH") \
                            else "gaining"
                pct_str = f" (+{pct_change}%)" if pct_change is not None else ""
                trend_summary = f"{direction}: +{abs_change} {unit_str}{pct_str}"
            else:
                direction = "worsening" if _lab_status(last_val, r_low, r_high) in ("LOW", "CRITICALLY_LOW") \
                            else "dropping"
                pct_str = f" ({pct_change}%)" if pct_change is not None else ""
                trend_summary = f"{direction}: -{abs(abs_change)} {unit_str}{pct_str}"

        trends[name] = {
            "name":              name,
            "name_raw":          raw_names.get(name, name),
            "first_value":       first_val,
            "first_date":        first_p["date"],
            "first_status":      first_status,
            "latest_value":      last_val,
            "latest_date":       last_p["date"],
            "latest_status":     latest_status,
            "min_value":         min(p["value"] for p in points),
            "max_value":         max(p["value"] for p in points),
            "absolute_change":   abs_change,
            "percentage_change": pct_change,
            "direction":         direction,
            "trend_summary":     trend_summary,
            "unit":              unit_str,
            "units_incompatible": units_incompatible,
            "ref_low":           r_low,
            "ref_high":          r_high,
            "num_visits":        len(points),
            "num_abnormal":      sum(
                1 for p in all_points_out
                if p["status"] not in ("NORMAL", "UNKNOWN")
            ),
            "all_points":        all_points_out,
        }

    # --- Qualitative trends ---
    for name, visit_map in qual_series.items():
        if name in trends:
            continue
        q_points = sorted(visit_map.values(),
                          key=lambda p: (p["date"] is None, p["date"] or ""))
        if not q_points:
            continue

        r_low, r_high = ref_ranges.get(name, (None, None))
        unit_str = next(iter(units_seen[name]), "")

        def _qual_status(p):
            ab = p.get("abnormal")
            val = str(p.get("value") or "").strip().upper()
            if ab is True or val in ("PRESENT", "POSITIVE", "REACTIVE", "DETECTED", "TRACE"):
                return "ABNORMAL"
            if ab is False or val in ("ABSENT", "NIL", "NEGATIVE", "CLEAR", "NORMAL"):
                return "NORMAL"
            return "UNKNOWN"

        first_st = _qual_status(q_points[0])
        latest_st = _qual_status(q_points[-1])

        trends[name] = {
            "first_value":       q_points[0]["value"],
            "first_date":        q_points[0]["date"],
            "first_status":      first_st,
            "latest_value":      q_points[-1]["value"],
            "latest_date":       q_points[-1]["date"],
            "latest_status":     latest_st,
            "min_value":         None,
            "max_value":         None,
            "absolute_change":   None,
            "percentage_change": None,
            "direction":         "qualitative_observation",
            "trend_summary":     f"Observed: {q_points[-1]['value']}",
            "unit":              unit_str,
            "units_incompatible": False,
            "ref_low":           r_low,
            "ref_high":          r_high,
            "num_visits":        len(q_points),
            "num_abnormal":      sum(1 for p in q_points if p.get("abnormal")),
            "all_points":        [{"date": p["date"], "value": p["value"],
                                   "status": _qual_status(p)} for p in q_points],
        }

    return trends


# ---------------------------------------------------------------------------
# Stage 12: Medication longitudinal
# ---------------------------------------------------------------------------

def analyze_medication_longitudinal(timeline: dict) -> dict:
    """
    Tracks each medication across visits.  Absence in a later visit is
    "not documented in later visit(s)" — never asserted as discontinued.
    """
    by_med: dict[str, list] = defaultdict(list)

    for visit in timeline["visits"]:
        for med in visit["medications"]:
            by_med[med["name"]].append({
                "visit_id":          visit["visit_id"],
                "visit_date":        visit["visit_date"],
                "dose":              med["dose"],
                "unit":              med["unit"],
                "frequency":         med["frequency"],
                "needs_verification": med["needs_verification"],
            })

    result = {}
    for name, records in by_med.items():
        records.sort(key=lambda r: (r["visit_date"] is None, r["visit_date"] or ""))
        changes = []
        for prev, curr in zip(records, records[1:]):
            if prev["dose"] != curr["dose"] and None not in (prev["dose"], curr["dose"]):
                changes.append({
                    "type":       "dose_change",
                    "from":       f"{prev['dose']}{prev['unit'] or ''}",
                    "to":         f"{curr['dose']}{curr['unit'] or ''}",
                    "at_visit":   curr["visit_id"],
                    "visit_date": curr["visit_date"],
                })
            if prev["frequency"] != curr["frequency"] and \
               None not in (prev["frequency"], curr["frequency"]):
                changes.append({
                    "type":       "frequency_change",
                    "from":       prev["frequency"],
                    "to":         curr["frequency"],
                    "at_visit":   curr["visit_id"],
                    "visit_date": curr["visit_date"],
                })

        result[name] = {
            "first_visit_date":       records[0]["visit_date"],
            "latest_visit_date":      records[-1]["visit_date"],
            "num_visits_documented":  len(records),
            "status":                 "continued" if len(records) > 1 else "single_visit_only",
            "changes":                changes,
            "any_needs_verification": any(r["needs_verification"] for r in records),
        }

    return result


# ---------------------------------------------------------------------------
# Stage 13: Diagnosis longitudinal
# ---------------------------------------------------------------------------

def analyze_diagnosis_longitudinal(timeline: dict) -> dict:
    """
    Tracks each diagnosis across visits.

    Rule 13 — status vocabulary:
      persistent      — present in > 1 visit
      documented_once — only 1 visit
    Resolution is never inferred from absence of documentation.
    """
    all_visit_dates = [v["visit_date"] for v in timeline["visits"]]
    by_dx: dict[str, list] = defaultdict(list)

    for visit in timeline["visits"]:
        present_here: set[str] = set()
        for dx in visit["diagnoses"]:
            if dx["assertion"] in ("present", "historical"):
                present_here.add(dx["name"])
        for name in present_here:
            by_dx[name].append(visit["visit_date"])

    result = {}
    for name, visit_dates in by_dx.items():
        dated = [d for d in visit_dates if d]
        result[name] = {
            "first_documented":      min(dated, default=None),
            "last_documented":       max(dated, default=None),
            "num_visits_documented": len(visit_dates),
            "total_visits_in_record": len(all_visit_dates),
            # Rule 13: only mark persistent when patient-specific evidence
            # spans more than one visit; otherwise documented_once.
            "status":               "persistent" if len(visit_dates) > 1 else "documented_once",
        }

    return result
