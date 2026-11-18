"""Painkiller safety rules from published guidance (condition_registry.json 'medicine_safety_rules').

Each rule names its source and threshold; nothing here is per drug or per disease: medicines match by registry class,
route, ingredient or name pattern, and every number comes from the rule. A rule is draft until a clinician verifies it
(verified / verified_by / verified_on); the concern always says which, with the threshold it was raised at.

check(...) returns concerns as dicts {level, title, evidence, significance, source, threshold, rule}; the snapshot
turns each into a flag. Evidence cites encounters (#n) and dates so the current / past split works as for any flag.
"""
import re

import lab_validation

_LOCAL = ("topical", "local", "eye", "ear", "nasal", "inhaled", "vaginal", "rectal")


def threshold_text(rule, **values):
    """'<threshold> — <source> (draft: not yet clinically verified)' or '... (verified by X on D)'."""
    text = rule["threshold"].format(**values) if values else rule["threshold"]
    state = (f"verified by {rule['verified_by']} on {rule['verified_on']}" if rule.get("verified") and rule.get("verified_by")
             else "draft: not yet clinically verified")
    return f"{text}. Source: {rule['source']} ({state})."


def _matches(line, spec):
    """A prescription line matches a rule's medicine spec: class, ingredient or name pattern, and the route if set."""
    if spec.get("route") == "oral" and str(line.get("route") or "").casefold() in _LOCAL:
        return False
    if spec.get("route") == "oral" and str(line.get("route") or "") == "unclear":
        return False  # only lines written as oral count; unclear routes are left to the route-unclear checks
    ings = {str(i).casefold() for i in line.get("ingredients") or []}
    if ings & {i.casefold() for i in spec.get("exclude_ingredients") or []}:
        return False  # e.g. aspirin is an antiplatelet here, not the NSAID side of an NSAID + antiplatelet check
    if spec.get("classes") and set(spec["classes"]) & set(line.get("classes") or []):
        return True
    if spec.get("ingredients") and set(i.casefold() for i in spec["ingredients"]) & {str(i).casefold() for i in line.get("ingredients") or []}:
        return True
    # Name patterns also check the mapped ingredients, so a brand ('Alprax') counts as its generic (alprazolam).
    if spec.get("name_pattern") and re.search(spec["name_pattern"], " ".join(
            [str(line.get("name") or ""), str(line.get("raw") or "")] + sorted(ings)), re.I):
        return True
    return False


def _window(line):
    start = line.get("start") or line.get("date")
    return start, line.get("end") or start


def _overlap(a, b):
    (s1, e1), (s2, e2) = _window(a), _window(b)
    return bool(s1 and s2) and s1 <= e2 and s2 <= e1


def _cite(line):
    return f"{line['name']} (#{line['i']}, {line.get('date') or 'date unknown'})"


def _daily_mg(line, ingredient):
    """Daily mg of a single-ingredient line (a combination product's strength is not split between its parts)."""
    dose = line.get("dose") or {}
    ings = [str(i).casefold() for i in line.get("ingredients") or []]
    if ings == [ingredient.casefold()] and dose.get("daily_mg"):
        return float(dose["daily_mg"])
    return None


def _visit_days(lines, visits):
    return sorted({str(visits[l["i"] - 1].get("visit_date") or "")[:10] for l in lines if 0 < l["i"] <= len(visits)})


def _latest_lab(labs, key):
    rows = [(d or "", lab_validation.parse_value(lab.get("value"))[1]) for lab, _, d in labs
            if lab_validation.identify(lab.get("test_name")) == key and not str(lab.get("status") or "").startswith("Unverified")]
    rows = [r for r in rows if r[1] is not None]
    return max(rows) if rows else None


def _latest_weight(visits):
    for v in reversed(visits or []):
        for r in v.get("db_vitals_rows") or []:
            if str(r.get("label")) == "Weight":
                n = lab_validation.parse_value(r.get("value"))[1]
                if n:
                    return n
    return None


def check(rules, lines, visits, labs, present, age=None, health_history=None, repeat_days=2):
    """Concerns from the guideline rules for one patient. lines: prescribed lines (medicine_profile.record);
    labs: [(lab, source, date)]; present(names) -> the listed conditions the patient has; repeat_days: the local
    'repeated' threshold (condition_registry.json nsaid_repeat.min_visit_days)."""
    out = []
    history = " ".join(k for k, v in (health_history or {}).items() if isinstance(v, dict) and v.get("present"))
    has_bp = any(str(r.get("label")) == "BP" for v in visits for r in v.get("db_vitals_rows") or [])
    lab_keys = {lab_validation.identify(l.get("test_name")) for l, _, _ in labs}

    def emit(rule, evidence, level=None, **values):
        out.append({"level": level or rule["level"], "title": rule["title"], "evidence": evidence,
                    "significance": rule.get("significance", ""), "source": rule["source"],
                    "threshold": threshold_text(rule, **values), "rule": rule["id"], "problem": rule.get("problem"),
                    "verified": bool(rule.get("verified") and rule.get("verified_by"))})

    for rule in rules or []:
        kind = rule.get("type")
        med = [l for l in lines if _matches(l, rule.get("med") or rule.get("a") or {})]
        if kind == "combo":
            other = [l for l in lines if _matches(l, rule["b"])]
            pairs = [(a, b) for a in med for b in other if a is not b and a["key"] != b.get("key") and _overlap(a, b)]
            if pairs:
                classes = {c for _, b in pairs for c in b.get("classes") or []}
                level = next((lv for cls, lv in (rule.get("level_if_b") or {}).items() if cls in classes), None)
                shown = list(dict.fromkeys(f"{_cite(a)} with {_cite(b)}" for a, b in pairs))
                emit(rule, "; ".join(shown[:3]) + (f" (+{len(shown) - 3} more)" if len(shown) > 3 else ""), level)
        elif kind == "with_conditions":
            conds = present(rule["conditions"])
            if med and conds:
                emit(rule, f"{', '.join(dict.fromkeys(_cite(l) for l in med[-3:]))} · {', '.join(conds)}")
        elif kind == "daily_dose_with_conditions":
            conds = present(rule["conditions"])
            high = [(l, _daily_mg(l, rule["med"]["ingredients"][0])) for l in med]
            high = [(l, mg) for l, mg in high if mg and mg > rule["daily_mg_above"]]
            if high and conds:
                emit(rule, "; ".join(f"{_cite(l)} = {mg:g} mg/day" for l, mg in high[-3:]) + f" · {', '.join(conds)}")
        elif kind in ("repeat_without_tests", "age_repeat"):
            days = _visit_days(med, visits)
            if len(days) < repeat_days:
                continue
            if kind == "age_repeat" and not (age is not None and age >= rule["min_age"]):
                continue
            missing = []
            if kind == "repeat_without_tests":
                if not lab_keys & set(rule["tests"]):
                    missing.append("no creatinine / eGFR result")
                if rule.get("needs_bp") and not has_bp:
                    missing.append("no BP reading")
                if not missing:
                    continue
            names = ", ".join(dict.fromkeys(l["name"] for l in med))
            encs = ", ".join(f"#{n}" for n in sorted({l["i"] for l in med}))
            evidence = f"{names} on {len(days)} visit days ({encs}; {days[0]} to {days[-1]})"
            evidence += f" · {'; '.join(missing)} in the record" if missing else f" · age {age}"
            emit(rule, evidence, min_visit_days=repeat_days)
        elif kind == "lab_below":
            latest = _latest_lab(labs, rule["lab"])
            if med and latest and latest[1] < rule["below"]:
                emit(rule, f"{', '.join(dict.fromkeys(_cite(l) for l in med[-3:]))} · latest eGFR {latest[1]:g} "
                           f"({latest[0] or 'undated'})")
        elif kind == "age_dose":
            if age is None or age <= rule["age_above"]:
                continue
            high = [(l, _daily_mg(l, rule["med"]["ingredients"][0])) for l in med]
            high = [(l, mg) for l, mg in high if mg and mg > rule["daily_mg_above"]]
            if high:
                emit(rule, "; ".join(f"{_cite(l)} = {mg:g} mg/day" for l, mg in high[-3:]) + f" · age {age}")
        elif kind == "age_without_lab":
            if med and age is not None and age >= rule["min_age"] and not lab_keys & set(rule["labs"]):
                emit(rule, f"{', '.join(dict.fromkeys(_cite(l) for l in med[-3:]))} · age {age} · no "
                           f"{' / '.join(rule['labs'])} result in the record")
        elif kind == "dose_with_risk":
            high = [(l, _daily_mg(l, rule["med"]["ingredients"][0])) for l in med]
            high = [(l, mg) for l, mg in high if mg and mg > rule["daily_mg_above"]]
            if not high:
                continue
            weight = _latest_weight(visits)
            risks = ([f"weight {weight:g} kg"] if weight and weight < rule["weight_below_kg"] else []) + \
                present(rule.get("conditions") or []) + \
                (["alcohol use reported"] if rule.get("history_pattern") and re.search(rule["history_pattern"], history, re.I) else [])
            if risks:
                emit(rule, "; ".join(f"{_cite(l)} = {mg:g} mg/day" for l, mg in high[-3:]) + f" · {', '.join(risks)}")
    return out
