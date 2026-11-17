"""Two tiers of the AI Analysis & Summary, built per patient from the pipeline's own outputs.

summary = {headline, tiles, quick, detailed, coverage}

quick     One bedside screen, relative to the LATEST visit:
            now             current concerns (high > verify > medium), one line each, with as_of, area, flag id, #n
            earlier         past high / verify concerns, and past medium ones that touch the latest prescription or
                            an active condition
            treatment       the latest prescription grouped by the condition each medicine treats; courses running
                            but not renewed; medicines stopped recently
            tests_to_check  missing / undated / overdue monitoring tests, one per test family, verified or draft
            hidden_counts   what only the detailed tier shows, and every capped list's overflow
detailed  Everything, each fact once:
            brief           Key findings, Conditions, Treatment, Gaps (a point that states a concern links to it)
            concerns        every concern grouped by topic, plus info notes
            course          consistent / diagnosis shifts / medicine shifts, condition grid, medicine grid, timeline
            reference       lab values with ranges, guideline rows, record-quality notes, unlinked documents,
                            AI-cited visits, AI checks, EHR markdown
coverage  Every flag id and abnormal lab is in detailed, and is either shown in quick or counted in
          quick.hidden_counts. build_summary raises CoverageError otherwise: nothing is dropped silently.

Nothing here is per disease or per drug. Levels, areas, problems and the current / past split come from the snapshot
flags (condition_registry.json, lab_significance.json, clinical_thresholds.json); monitoring tests from the guideline
section of condition_registry.json; caps and word limits from clinical_thresholds.json 'summary_tiers'.
"""
import datetime
import json
import os
import re

import lab_validation

_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_DIR, "data", "reference", "clinical_thresholds.json"), encoding="utf-8") as _f:
    CONFIG = json.load(_f)["summary_tiers"]
with open(os.path.join(_DIR, "data", "reference", "condition_registry.json"), encoding="utf-8") as _f:
    _GUIDELINES = json.load(_f).get("guidelines") or {}

VIEWS = ("quick", "detailed")
_CONCERN_LEVELS = ("high", "verify", "medium")
_LEVEL_RANK = {"high": 0, "verify": 1, "medium": 2, "info": 3, "data_quality": 4}
_TEST_STATE_RANK = {"missing": 0, "advised, no result": 1, "overdue": 2, "undated": 3, "no repeat": 4, "due": 5}
_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# A flag that is about a test that is not in the record ("No urine test in the record", "X without thyroid function
# in the record", "X: monitoring due"): it belongs in tests_to_check, not in the concern lines.
_TEST_FLAG = re.compile(r"(?i)^(?:no\s+(?P<a>.+?)|.+?\bwithout\s+(?P<b>.+?)|(?P<c>[^:]+):\s+no\s+(?P<d>.+?))"
                        r"\s+in the record\b|^(?P<e>[^:]+):\s+monitoring due$")
_GAP_ADVISED = re.compile(r"^Tests advised at #(\d+) \((\d{4}-\d{2}-\d{2})\) with no result in the record: (.+?)\.?$")
_GAP_REPEAT = re.compile(r"^No repeat result after the out-of-range value for: (.+?)\.?$")


class CoverageError(AssertionError):
    """A flag or abnormal lab is missing from the detailed tier, or is neither shown nor counted in quick."""


# ── Small helpers ───────────────────────────────────────────────────────────
def words(text):
    return len(str(text or "").split())


def clip(text, limit):
    """At most `limit` words; a cut line ends with '…'."""
    parts = str(text or "").split()
    return " ".join(parts) if len(parts) <= limit else " ".join(parts[:limit]).rstrip(",;:·") + " …"


def cites(text, encounter_dates=None):
    """Encounter numbers a text cites: '#3', '#2–4', '#2-#4'; else the encounters on the dates it names."""
    text = str(text or "")
    found = {int(n) for n in re.findall(r"#(\d+)", text)}
    for a, b in re.findall(r"#(\d+)\s*[–-]\s*#?(\d+)", text):
        if int(b) - int(a) < 200:
            found.update(range(int(a), int(b) + 1))
    if not found and encounter_dates:
        days = set(_ISO.findall(text))
        found = {int(n) for n, d in encounter_dates.items() if d in days}
    return sorted(found)


def _days(a, b):
    try:
        return (datetime.date.fromisoformat(b[:10]) - datetime.date.fromisoformat(a[:10])).days
    except (TypeError, ValueError):
        return None


def _norm(text):
    return " ".join(re.sub(r"[^a-z0-9/ ]+", " ", str(text or "").casefold()).split())


def _first_word(name):
    m = re.match(r"[A-Za-z][A-Za-z-]{2,}", str(name or "").strip())
    return m.group(0).casefold() if m else ""


def _mentions(text, name):
    name = str(name or "").strip()
    return bool(name) and re.search(rf"(?<![a-z]){re.escape(name.casefold())}(?![a-z])", str(text or "").casefold())


def lab_id(lab, seen):
    base = f"lab-{lab.get('key') or _norm(lab.get('test')).replace(' ', '-')}-{lab.get('date') or 'undated'}"
    lid, n = base, 1
    while lid in seen:
        n += 1
        lid = f"{base}-{n}"
    seen.add(lid)
    return lid


def is_record_quality(flag):
    return bool(flag.get("record_error")) or flag.get("kind") == "data" or flag.get("level") == "data_quality"


def test_labels(flag):
    """The test names a missing-test flag is about ([] when it is not one): 'No blood glucose / HbA1c or vitamin B12
    in the record (tingling)' -> ['blood glucose / HbA1c', 'vitamin B12']."""
    m = _TEST_FLAG.match(re.sub(r"\s*\([^)]*\)\s*$", "", str(flag.get("title") or "")).strip())
    if not m or flag.get("kind") not in ("followup", "lab"):
        return []
    if m.group("e"):
        return [f"{m.group('e')} monitoring"]
    label = m.group("a") or m.group("b") or m.group("d")
    return [p.strip() for p in re.split(r"\s+(?:or|and)\s+|,\s*", label) if p.strip()]


# ── Test families (monitoring map from the guideline section; else the lab catalog key) ──
def _monitoring_families():
    fams = {}
    for section in ("monitoring",):
        for items in ((_GUIDELINES.get(section) or {}).get("conditions") or {}).values():
            for item in items:
                fams[_norm(item["label"])] = frozenset(item["keys"])
    dm = _GUIDELINES.get("diabetes_control") or {}
    if dm.get("analyte"):
        fams.setdefault(_norm(dm.get("label") or dm["analyte"]), frozenset([dm["analyte"]]))
    return fams


_FAMILIES = _monitoring_families()


def _every(days):
    days = int(days)
    return ("yearly" if days >= 360 else f"every {round(days / 30)} months" if days >= 60 and days % 30 in (0, 1, 2, 3, 4, 5)
            else f"every {days} days")


def _intervals_and_controls():
    """{test label: recheck interval text} from the guideline sections, and the labels of a condition's own control
    test (a section with both 'condition' and 'analyte', e.g. HbA1c for diabetes). Read from condition_registry.json."""
    intervals, controls = {}, set()
    for items in ((_GUIDELINES.get("monitoring") or {}).get("conditions") or {}).values():
        for item in items:
            intervals.setdefault(_norm(item["label"]), _every(item["every_days"]))
    for sec in _GUIDELINES.values():
        if isinstance(sec, dict) and sec.get("condition") and sec.get("analyte"):
            label = _norm(sec.get("label") or sec["analyte"])
            controls.add(label)
            above, at = sec.get("recheck_days_above_target"), sec.get("recheck_days_at_target")
            if above and at:
                intervals[label] = f"{_every(above)} above target, {_every(at).replace('every ', '')} at target"
    return intervals, controls


_INTERVALS, _CONTROLS = _intervals_and_controls()


def test_family(label):
    """Family key of a test: the guideline monitoring keys it belongs to; else its lab-catalog key; else its words."""
    n = _norm(label)
    if n in _FAMILIES:
        return "+".join(sorted(_FAMILIES[n]))
    key = lab_validation.identify(label)
    if key:
        for keys in _FAMILIES.values():
            if key in keys:
                return "+".join(sorted(keys))
        return key
    return n


# ── Quick tier ──────────────────────────────────────────────────────────────
def _concern_item(f, encounter_dates, limit):
    text = f"{f.get('title', '')} {f.get('evidence', '')}"
    return {"id": f["id"], "level": f["level"], "area": f.get("area") or "general", "time": f.get("time", "current"),
            "as_of": f.get("as_of") or "", "undated": bool(f.get("undated")), "title": f.get("title", ""),
            "text": clip(f.get("summary") or f.get("title", ""), limit), "cites": cites(text, encounter_dates),
            "src": "rule"}


def _interacts(f, current_names, active):
    """A past concern that still matters now: it names a medicine on the latest prescription (or one of its
    ingredients), or belongs to an active condition."""
    text = f"{f.get('title', '')} {f.get('evidence', '')}"
    if any(_mentions(text, n) for n in current_names):
        return True
    if any(n in current_names for n in (_first_word(m) for m in f.get("related_medicines") or [])):
        return True
    problem = str(f.get("problem") or "").casefold()
    return any(problem and problem == a.casefold() or _mentions(text, a) for a in active)


def _parse_monitoring(row):
    """Tests of a guideline monitoring row: 'creatinine: none in record; lipid profile: last 2024-01-02, overdue'."""
    out = []
    for part in re.split(r";\s+", str(row.get("finding") or "")):
        label, _, state = part.rpartition(": ")
        if not label:
            label, state = row.get("topic", part), "due"
        kind = ("missing" if "none in record" in state
                else "undated" if "undated" in state or "no dated result" in state
                else "overdue" if "overdue" in state else "due")
        out.append({"test": label.strip(), "state": kind, "detail": state.strip()})
    return out


def tests_to_check(snapshot, evidence, flags):
    """Monitoring / missing tests, one entry per test family. Sources: guideline monitoring rows (verified or
    draft), missing-test flags, tests advised with no result, and out-of-range values never repeated."""
    by_family = {}
    latest = (evidence.get("patient") or {}).get("latest_visit", "")

    def add(test, state, detail, reason, basis, verified, source="", flag_ids=(), as_of=""):
        fam = test_family(test)
        control = _norm(test) in _CONTROLS
        e = by_family.get(fam)
        if not e:
            # A condition's control test (HbA1c for diabetes) and a wider test that includes it ('blood glucose /
            # HbA1c' for hypertension) are one row, named after the control test.
            keys = set(fam.split("+"))
            e = next((x for x in by_family.values() if (control or x["control"]) and keys & set(x["family"].split("+"))), None)
            if e and control:
                e.update(test=test, control=True, interval=_INTERVALS.get(_norm(test), e["interval"]))
        if not e:
            e = by_family[fam] = {"test": test, "family": fam, "state": state, "detail": detail, "reasons": [],
                                  "basis": basis, "verified": verified, "source": source, "flag_ids": [],
                                  "as_of": as_of or latest, "control": control,
                                  "interval": _INTERVALS.get(_norm(test), "")}
        elif _TEST_STATE_RANK.get(state, 9) < _TEST_STATE_RANK.get(e["state"], 9):
            e.update(state=state, detail=detail)
        if basis == "guideline" and e["basis"] != "guideline":
            e.update(basis="guideline", verified=verified, source=source)
        elif basis == "guideline":
            e["verified"] = bool(e["verified"] or verified)
        if reason and reason not in e["reasons"]:
            e["reasons"].append(reason)
        e["flag_ids"] += [i for i in flag_ids if i not in e["flag_ids"]]

    rows = [r for r in (snapshot.get("guideline") or {}).get("rows") or [] if r.get("kind") == "followup"]
    for r in rows:
        cond = r.get("condition") or ""
        ids = [f["id"] for f in flags if f.get("title") == f"{cond}: monitoring due"]
        for t in _parse_monitoring(r):
            add(t["test"], t["state"], t["detail"], cond, "guideline", bool(r.get("verified")), r.get("source", ""), ids)
    monitored = {r.get("condition") for r in rows}
    for f in flags:
        labels = test_labels(f)
        if not labels:
            continue
        reason = re.search(r"\(([^)]+)\)\s*$", f.get("title", ""))
        reason = reason.group(1) if reason else (f.get("problem") or "")
        for label in labels:
            generic = label.endswith(" monitoring") or "related" in label
            if generic:
                cond = label.replace(" monitoring", "").strip() or f.get("problem") or ""
                cond = re.sub(r"(?i):?\s*no related.*$", "", f.get("title", "")).strip() if "related" in label else cond
                if cond in monitored:  # the guideline row lists the tests one by one
                    for e in by_family.values():
                        if cond in e["reasons"] and f["id"] not in e["flag_ids"]:
                            e["flag_ids"].append(f["id"])
                    continue
            add(label, "missing", "none in record", reason, "rule", None, flag_ids=[f["id"]], as_of=f.get("as_of", ""))
    for gap in evidence.get("followup_gaps") or []:
        m = _GAP_ADVISED.match(gap)
        if m:
            for t in re.split(r",\s*", m.group(3)):
                add(t, "advised, no result", f"advised at #{m.group(1)} ({m.group(2)})", "advised", "rule", None,
                    as_of=m.group(2))
            continue
        m = _GAP_REPEAT.match(gap)
        if m:
            for t in re.split(r",\s*(?![^()]*\))", m.group(1)):
                name, _, date = t.partition(" (")
                add(name.strip(), "no repeat", f"out of range on {date.rstrip(')')}, not repeated", "out of range",
                    "rule", None, as_of=date.rstrip(")"))
    # A condition's own control test first (HbA1c for diabetes), then by how urgent the state is.
    items = sorted(by_family.values(), key=lambda e: (not e["control"], _TEST_STATE_RANK.get(e["state"], 9),
                                                      e["verified"] is not True, e["test"].casefold()))
    return items


def treatment(snapshot, evidence, limit):
    """The latest prescription grouped by the condition each medicine treats ('Not linked to a recorded condition'
    when the record links it to none), then medicines stopped within recent_stop_days of the latest visit."""
    latest = (evidence.get("patient") or {}).get("latest_visit", "")
    info = {m["name"]: m for m in evidence.get("medicines") or []}
    groups, items = {}, []
    for c in snapshot.get("current_medicines") or []:
        m = info.get(c["name"]) or {}
        item = {"name": c["name"], "regimen": m.get("regimen", ""), "class": m.get("class", ""),
                "status": c.get("status", "current"), "since": c.get("since", ""), "until": c.get("until") or "",
                "note": c.get("note", ""), "specialization": c.get("specialization") or ""}
        items.append(item)
        for cond in (m.get("for") or ["Not linked to a recorded condition"]):
            groups.setdefault(cond, []).append(item)
    shown = {id(i) for i in items[:limit]}
    out = [{"condition": cond, "medicines": [i for i in meds if id(i) in shown]} for cond, meds in groups.items()]
    out = [g for g in out if g["medicines"]]
    out.sort(key=lambda g: g["condition"].startswith("Not linked"))
    stopped = []
    for m in evidence.get("medicines") or []:
        gap = _days(m.get("last") or "", latest)
        if m.get("status") == "not on latest prescription" and gap is not None and gap <= CONFIG["recent_stop_days"]:
            stopped.append({"name": m["name"], "last": m.get("last", ""), "encounters": m.get("encounters", ""),
                            "for": m.get("for") or []})
    running = [i["name"] for i in items if i["status"] != "current"]
    return {"by_condition": out, "running_not_renewed": running, "recently_stopped": stopped,
            "overflow": max(0, len(items) - limit)}


def _headline(snapshot, evidence, n_now, n_high, today):
    """Rule headline (<= headline_words): who, how long followed, active conditions, concerns now; a stale-record
    note first when the latest visit is older than stale_latest_days."""
    pt = evidence.get("patient") or {}
    latest = pt.get("latest_visit", "")
    n = pt.get("clinical_days") or pt.get("encounters") or 0
    who = f"{pt.get('age_sex') or 'Age/sex not recorded'} · {n} visit{'s' if n != 1 else ''}"
    span = pt.get("span") if n > 1 and not str(pt.get("span") or "").startswith("0 ") else ""
    who += f" over {span}" if span else (f" ({latest})" if latest else "")
    concerns = (f"{n_now} current concern{'s' if n_now != 1 else ''}" + (f" ({n_high} high)" if n_high else "")
                if n_now else "No current concerns")
    stale = ""
    age = _days(latest, today) if latest and today else None
    if age is not None and age > CONFIG["stale_latest_days"]:
        stale = f"Latest visit {latest}, {age // 30} months ago; 'now' means as of then."
    active = list(snapshot.get("active") or [])
    for k in range(len(active), -1, -1):
        shown = active[:k]
        more = f" +{len(active) - k}" if len(active) > k else ""
        act = f"Active: {', '.join(shown)}{more}." if shown else (f"{len(active)} active conditions." if active else "")
        text = " ".join(x for x in (stale, f"{who}.", act, f"{concerns}.") if x)
        if words(text) <= CONFIG["headline_words"]:
            return text
    return clip(" ".join(x for x in (stale, f"{who}.", f"{concerns}.") if x), CONFIG["headline_words"])


def build_quick(snapshot, evidence, today=None):
    flags = snapshot.get("flags") or []
    enc_dates = evidence.get("encounter_dates") or {}
    limit = CONFIG["line_words"]
    order = {i: k for k, i in enumerate(snapshot.get("concerns") or [])}
    ranked = sorted(range(len(flags)), key=lambda i: (_LEVEL_RANK.get(flags[i]["level"], 9), order.get(i, 999), i))
    tests = tests_to_check(snapshot, evidence, flags)
    test_ids = {i for t in tests for i in t["flag_ids"]}

    current_names = {_first_word(c["name"]) for c in snapshot.get("current_medicines") or []} - {""}
    active = list(snapshot.get("active") or [])
    hidden = {k: [] for k in ("past_concerns", "info", "record_quality", "now_overflow", "earlier_overflow",
                              "tests_overflow")}
    now, earlier = [], []
    for i in ranked:
        f = flags[i]
        if f["id"] in test_ids:
            continue
        if is_record_quality(f):
            hidden["record_quality"].append(f["id"])
        elif f["level"] not in _CONCERN_LEVELS:
            hidden["info"].append(f["id"])
        elif f.get("time") != "past":
            now.append(_concern_item(f, enc_dates, limit))
        elif f["level"] in ("high", "verify") or _interacts(f, current_names, active):
            earlier.append(_concern_item(f, enc_dates, limit))
        else:
            hidden["past_concerns"].append(f["id"])
    n_now, n_high = len(now), sum(1 for c in now if c["level"] == "high")
    hidden["now_overflow"] = [c["id"] for c in now[CONFIG["max_now"]:]]
    hidden["earlier_overflow"] = [c["id"] for c in earlier[CONFIG["max_earlier"]:]]
    shown_test_ids = {i for t in tests[:CONFIG["max_tests"]] for i in t["flag_ids"]}
    hidden["tests_overflow"] = list(dict.fromkeys(i for t in tests[CONFIG["max_tests"]:] for i in t["flag_ids"]
                                                  if i not in shown_test_ids))
    now, earlier = now[:CONFIG["max_now"]], earlier[:CONFIG["max_earlier"]]
    rx = treatment(snapshot, evidence, CONFIG["max_treatment_items"])
    quick = {
        "headline": {"text": _headline(snapshot, evidence, n_now, n_high, today),
                     "src": "rule"},
        "as_of": (evidence.get("patient") or {}).get("latest_visit", ""),
        "now": now,
        "earlier": earlier,
        "treatment": {k: v for k, v in rx.items() if k != "overflow"},
        "tests_to_check": tests[:CONFIG["max_tests"]],
        "_hidden_ids": hidden,
        "_tests_overflow": max(0, len(tests) - CONFIG["max_tests"]),
        "_treatment_overflow": rx["overflow"],
    }
    return quick


# ── Detailed tier ───────────────────────────────────────────────────────────
_THEME_NAMES = {"key findings": "Key findings", "conditions": "Conditions", "treatment": "Treatment",
                "gaps to check": "Gaps", "gaps": "Gaps"}


def _brief(merged, flags):
    """Brief themes without Course (it is part of the course section). A Key-findings or Gaps point that states a
    concern becomes a link to it ('see'), so the concern is written once."""
    from clinical_insights import _point_flag
    themes, course_points = [], []
    for b in merged or []:
        name = b["theme"].casefold()
        srcs = b.get("src") or ["rule"] * len(b["points"])
        if name == "course":
            course_points = [{"text": t, "src": s} for t, s in zip(b["points"], srcs)]
            continue
        links = b.get("links") or []
        pts = []
        for k, text in enumerate(b["points"]):
            ids = [flags[i]["id"] for i in (links[k] if k < len(links) else []) if i < len(flags)]
            if not ids and not text.startswith("+"):
                f = _point_flag(text, flags)
                # Gaps: any point that states a concern. Elsewhere only a point written as the concern itself
                # ('<flag title> — <evidence>'), so 'Hypertension: Amlodipine' stays a treatment line.
                lead = _norm(re.sub(r"^\W+", "", text))
                if f and (name.startswith("gaps") or lead.startswith(_norm(f["title"])[:24])):
                    ids = [f["id"]]
            area = (b.get("areas") or [])[k] if k < len(b.get("areas") or []) else []
            pts.append({"see": ids, "src": srcs[k] if k < len(srcs) else "rule", "areas": area} if ids
                       else {"text": text, "src": srcs[k] if k < len(srcs) else "rule", "areas": area})
        themes.append({"theme": _THEME_NAMES.get(name, b["theme"]), "points": pts})
    return themes, course_points


def _concern_full(f, enc_dates):
    return {"id": f["id"], "level": f["level"], "area": f.get("area") or "general", "time": f.get("time", "current"),
            "as_of": f.get("as_of") or "", "undated": bool(f.get("undated")), "kind": f.get("kind", ""),
            "title": f.get("title", ""), "evidence": f.get("evidence", ""), "significance": f.get("significance", ""),
            "line": f.get("summary", ""), "cites": cites(f"{f.get('title', '')} {f.get('evidence', '')}", enc_dates),
            **({"source": f["source"]} if f.get("source") else {})}


def build_detailed(snapshot, evidence, course, view):
    flags = snapshot.get("flags") or []
    enc_dates = evidence.get("encounter_dates") or {}
    brief, course_points = _brief(view.get("merged") or [], flags)
    groups = []
    for g in snapshot.get("concern_groups") or []:
        if g["group"] == "error":
            continue
        groups.append({"group": g["group"], "label": g["label"],
                       "concerns": [_concern_full(flags[i], enc_dates) for i in g["concerns"] if not is_record_quality(flags[i])]})
    groups = [g for g in groups if g["concerns"]]
    grouped = {c["id"] for g in groups for c in g["concerns"]}
    info = [_concern_full(f, enc_dates) for f in flags
            if f["id"] not in grouped and not is_record_quality(f) and f["level"] not in _CONCERN_LEVELS]
    # A concern-level flag order_concerns did not place (should not happen) still goes under 'Other'.
    stray = [_concern_full(f, enc_dates) for f in flags if f["id"] not in grouped and not is_record_quality(f)
             and f["level"] in _CONCERN_LEVELS]
    if stray:
        groups.append({"group": "other", "label": "Other", "concerns": stray})
    seen = set()
    labs = [{"id": lab_id(g, seen), "test": g.get("test"), "as_written": g.get("test_as_written"),
             "value": g.get("value"), "unit": g.get("unit"), "reference": g.get("reference"),
             "reference_source": g.get("reference_source"), "status": g.get("status"), "critical": g.get("critical"),
             "date": g.get("date") or "", "encounters": g.get("encounters") or "", "significance": g.get("significance", "")}
            for g in evidence.get("abnormal_labs") or []]
    reports = view.get("reports") or []
    return {
        "brief": brief,
        "concerns": {"groups": groups, "info": info},
        "course": {
            "brief": course_points,
            "consistent": course.get("consistent") or [],
            "diagnosis_shifts": course.get("diagnosis_shifts") or [],
            "medicine_shifts": course.get("medicine_shifts") or [],
            "condition_grid": view.get("snap_timeline") or [],
            "medicine_grid": snapshot.get("medicine_timeline") or [],
            "dates": snapshot.get("dates") or [],
            "timeline": course.get("timeline") or [],
        },
        "reference": {
            "labs": labs,
            "lab_counts": evidence.get("lab_counts") or {},
            "guideline": {"version": (snapshot.get("guideline") or {}).get("version", ""),
                          "rows": (snapshot.get("guideline") or {}).get("rows") or []},
            "record_quality": {"flags": [_concern_full(f, enc_dates) for f in flags if is_record_quality(f)],
                               "checks": evidence.get("record_checks") or [],
                               "discrepancies": view.get("discrepancies") or []},
            "unlinked_documents": [{"label": r.get("label"), "likely_visit": r.get("likely_visit", ""),
                                    "impression": r.get("impression", ""), "labs": len(r.get("labs") or [])}
                                   for r in reports],
            "ai_cited_visits": view.get("ai_evidence") or [],
            "ai_checks": {"corrections": view.get("corrections"), "citation_log": view.get("citation_log") or []},
            "ehr_markdown": view.get("ai_markdown") or "",
        },
    }


# ── Coverage ────────────────────────────────────────────────────────────────
def _detailed_flag_ids(detailed):
    ids = [c["id"] for g in detailed["concerns"]["groups"] for c in g["concerns"]]
    ids += [c["id"] for c in detailed["concerns"]["info"]]
    ids += [c["id"] for c in detailed["reference"]["record_quality"]["flags"]]
    return ids


def _quick_shown_ids(quick):
    return ([c["id"] for c in quick["now"]] + [c["id"] for c in quick["earlier"]]
            + [i for t in quick["tests_to_check"] for i in t["flag_ids"]])


def _lab_in_quick(lab, quick):
    text = " ".join([quick["headline"]["text"]] + [f"{c['title']} {c['text']}" for c in quick["now"] + quick["earlier"]]
                    + [t["test"] for t in quick["tests_to_check"]])
    names = {n for n in (lab.get("test"), lab.get("as_written")) if n}
    return any(_mentions(text, n) for n in names)


def check_coverage(flags, quick, detailed):
    """{'ok', 'flags', 'labs', 'problems'}; every flag id and abnormal lab is in detailed, and either shown in quick
    or counted once in hidden_counts."""
    problems = []
    all_ids = [f["id"] for f in flags]
    det = _detailed_flag_ids(detailed)
    missing_det = sorted(set(all_ids) - set(det))
    if missing_det:
        problems.append(f"flags missing from detailed: {missing_det}")
    if len(det) != len(set(det)):
        problems.append("a flag appears twice in detailed")
    shown = _quick_shown_ids(quick)
    hidden = [i for ids in quick["_hidden_ids"].values() for i in ids]
    if len(set(c["id"] for c in quick["now"] + quick["earlier"])) != len(quick["now"]) + len(quick["earlier"]):
        problems.append("a concern appears twice in quick")
    accounted = set(shown) | set(hidden)
    lost = sorted(set(all_ids) - accounted)
    if lost:
        problems.append(f"flags neither shown nor counted in quick: {lost}")
    both = sorted(set(c["id"] for c in quick["now"] + quick["earlier"]) & set(hidden))
    if both:
        problems.append(f"flags both shown and counted hidden: {both}")
    for bucket, ids in quick["_hidden_ids"].items():
        if bucket != "tests_overflow" and quick["hidden_counts"].get(bucket) != len(ids):
            problems.append(f"hidden_counts.{bucket} = {quick['hidden_counts'].get(bucket)}, expected {len(ids)}")
    labs = detailed["reference"]["labs"]
    lab_shown = [l["id"] for l in labs if _lab_in_quick(l, quick)]
    lab_hidden = quick["_hidden_labs"]
    if sorted(set(lab_shown) | set(lab_hidden)) != sorted(l["id"] for l in labs) or set(lab_shown) & set(lab_hidden):
        problems.append("abnormal labs not accounted for in quick")
    if quick["hidden_counts"]["labs"] != len(lab_hidden):
        problems.append("hidden_counts.labs does not match")
    return {"ok": not problems, "flags": len(all_ids), "flags_shown_in_quick": len(set(shown)),
            "flags_counted_hidden": len(set(hidden)), "labs": len(labs), "labs_shown_in_quick": len(lab_shown),
            "labs_counted_hidden": len(lab_hidden), "problems": problems,
            "hidden_ids": quick["_hidden_ids"], "hidden_labs": lab_hidden}


# ── Entry point ─────────────────────────────────────────────────────────────
def build_summary(snapshot, evidence, course, view=None, today=None, ai_quick=None, strict=True):
    """summary = {headline, tiles, quick, detailed, coverage} for one patient. `view` is insight_view.build_patient's
    output (merged brief, reports, AI parts); `today` the date the page is built (stale-record note);
    `ai_quick` the AI quick tier (quick_review.generate), validated here line by line. Raises CoverageError when
    strict and any flag or abnormal lab is unaccounted for."""
    view = view or {}
    today = today or datetime.date.today().isoformat()
    snapshot = snapshot or {}
    flags = snapshot.get("flags") or []
    quick = build_quick(snapshot, evidence, today)
    detailed = build_detailed(snapshot, evidence, course or {}, view)
    quick["ai"] = None  # rule-built unless a checked AI wording is applied
    if ai_quick:
        import quick_review
        quick_review._attach_evidence(quick, snapshot)
        quick_review.apply(quick, ai_quick, evidence)
    labs = detailed["reference"]["labs"]
    quick["_hidden_labs"] = [l["id"] for l in labs if not _lab_in_quick(l, quick)]
    h = quick["_hidden_ids"]
    quick["hidden_counts"] = {
        **{k: len(v) for k, v in h.items()},
        "labs": len(quick["_hidden_labs"]),
        "course": sum(len(detailed["course"][k]) for k in ("consistent", "diagnosis_shifts", "medicine_shifts")),
        "tests_overflow": quick.pop("_tests_overflow"),  # tests, not flags
        "treatment_overflow": quick.pop("_treatment_overflow"),
    }
    coverage = check_coverage(flags, quick, detailed)
    quick.pop("_hidden_ids")
    quick.pop("_hidden_labs")
    headline = quick.pop("headline")  # once, at the top of the summary
    if strict and not coverage["ok"]:
        raise CoverageError("; ".join(coverage["problems"]))
    return {"headline": headline, "tiles": snapshot.get("tiles") or [], "quick": quick,
            "detailed": detailed, "coverage": coverage}


def resolve_view(requested=None, preference=None):
    """The tier that opens: ?view= override, else the doctor's saved default, else the configured default."""
    for v in (requested, preference, CONFIG.get("default_view"), "quick"):
        if str(v or "").casefold() in VIEWS:
            return str(v).casefold()
    return "quick"
