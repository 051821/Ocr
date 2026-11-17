"""Consistency check: the same pattern must give the same flag and severity in every patient.

Each patient is re-run under transformations that keep the clinical content equal:
  shift_dates   every date moved by the same number of days;
  reorder       prescription lines listed in reverse order within each visit;
  rename_case   medicine names in another letter case, extra spaces;
  swap_brand    a brand replaced by another brand with the same ingredients (medicine_ingredients.json);
  new_id        another patient id.
A rule that fires, disappears or changes level under any of them is reported as inconsistent. Across patients, the
levels each rule fired at are tabulated so a rule whose severity differs between patients can be looked at.
"""
import copy
import json
import re
from collections import defaultdict
from datetime import date, timedelta

import medicine_profile
from clinical_snapshot import build_snapshot


def signature(flag):
    """A flag's rule identity: its title with numbers masked ('BMI 26.0 (obese ...)' -> 'BMI N (obese ...)')."""
    return re.sub(r"\d+(?:\.\d+)?", "N", str(flag.get("title") or ""))


def _flags(patient):
    snap = build_snapshot(copy.deepcopy(patient)) or {}
    return {(signature(f), f["level"]) for f in snap.get("flags") or []}


def _shift(text, days):
    def move(m):
        try:
            return (date.fromisoformat(m.group(0)) + timedelta(days=days)).isoformat()
        except ValueError:
            return m.group(0)
    return re.sub(r"\d{4}-\d{2}-\d{2}", move, text)


def shift_dates(patient, days=91):
    return json.loads(_shift(json.dumps(patient, default=str), days))


def reorder(patient):
    out = copy.deepcopy(patient)
    for v in out.get("visits") or []:
        v["db_medications"] = list(reversed(v.get("db_medications") or []))
    return out


def rename_case(patient):
    out = copy.deepcopy(patient)
    for v in out.get("visits") or []:
        for m in v.get("db_medications") or []:
            m["name"] = "  " + str(m.get("name") or "").swapcase() + " "
    return out


def _brand_twins():
    by_ingredients = defaultdict(list)
    for brand, spec in medicine_profile.REF["brands"].items():
        by_ingredients[(tuple(sorted(spec["ingredients"])), spec.get("form"))].append(brand)
    return {b: next(x for x in group if x != b) for group in by_ingredients.values() if len(group) > 1 for b in group}


def swap_brand(patient):
    twins, out = _brand_twins(), copy.deepcopy(patient)
    for v in out.get("visits") or []:
        for m in v.get("db_medications") or []:
            key = medicine_profile.base_key(m.get("name"))
            if key in twins:
                m["name"] = re.sub(re.escape(key), twins[key].title(), str(m["name"]), flags=re.I)
    return out


def new_id(patient):
    return {**copy.deepcopy(patient), "patient_id": "consistency-check"}


TRANSFORMS = {"shift_dates": shift_dates, "reorder": reorder, "rename_case": rename_case, "swap_brand": swap_brand,
              "new_id": new_id}


def check_patient(patient):
    """[{transform, only_original, only_variant}] for every transform that changes the flags."""
    base, problems = _flags(patient), []
    for name, transform in TRANSFORMS.items():
        variant = _flags(transform(patient))
        if variant != base:
            problems.append({"transform": name, "only_original": sorted(base - variant),
                             "only_variant": sorted(variant - base)})
    return base, problems


def check_all(patients):
    """{"inconsistent": [...], "levels_by_rule": {rule: {level: n}}} over a list of (label, patient)."""
    inconsistent, levels = [], defaultdict(lambda: defaultdict(int))
    for label, patient in patients:
        base, problems = check_patient(patient)
        for rule, level in base:
            levels[rule][level] += 1
        inconsistent += [{"patient": label, **p} for p in problems]
    return {"inconsistent": inconsistent,
            "levels_by_rule": {r: dict(v) for r, v in sorted(levels.items())},
            "rules_with_mixed_levels": sorted(r for r, v in levels.items() if len(v) > 1)}
