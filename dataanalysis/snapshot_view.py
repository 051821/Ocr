"""Clinical Snapshot screen: status tiles per area, and each concern as one short line.

Built only from the snapshot's own flags and the patient's record (labs, BP, weight, follow-ups); nothing is per patient.
The full evidence stays on the flag; the short line is for reading at a glance.
"""
import re

import lab_validation
from difflib import SequenceMatcher

from medicine_profile import GROUP_CLASSES, LOCAL_ROUTES

AREAS = [("glucose", "Glucose"), ("bp", "BP"), ("kidney", "Kidney"), ("liver", "Liver"),
         ("medicine", "Medicine safety"), ("weight", "Weight / BMI"), ("followup", "Follow-up"),
         ("labs", "Labs missing")]
_STATUS = {"high": "red", "verify": "amber", "medium": "amber"}
_MAX_WORDS = 20
_MAX_DATES = 3
_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_LAB_KEYS = {"glucose": ("hba1c", "glucose_fasting", "glucose_pp", "glucose_random"),
             "kidney": ("creatinine", "egfr", "urea", "bun", "urine_acr"),
             "liver": ("alt", "ast", "bilirubin_total", "alp", "ggt")}


def flag_area(flag):
    """The tile a flag belongs to, or None (record errors and patterns with no tile)."""
    title, kind = str(flag.get("title") or ""), flag.get("kind")
    if kind == "data":
        return None
    if re.search(r"(?i)missed follow-up", title):
        return "followup"
    if re.search(r"(?i)liver|\b(?:SGPT|SGOT|ALT|AST)\b|bilirubin", title):
        return "liver"
    if re.match(r"No .+ in the record|.+: no related lab tests", title):
        return "labs"
    if re.search(r"(?i)creatinine|egfr|kidney|renal|\burea\b|albumin|proteinuria", title):
        return "kidney"
    if re.search(r"(?i)^BMI|weight", title):
        return "weight"
    if re.search(r"(?i)^BP\b|blood pressure|Hypertension medicine", title):
        return "bp"
    if kind == "medication":
        return "medicine"
    if re.search(r"(?i)HbA1c|glucose|hyperglyc|with diabetes", title):
        return "glucose"
    return None


# ── One short line per concern ──────────────────────────────────────────────

def _day(iso):
    y, m, d = iso.split("-")
    return f"{int(d)} {_MONTHS[int(m) - 1]}"


def _bmi_class(m):
    india, who = m.group(1), m.group(2)
    return f"({india})" if india == who else f"({india} by Indian cut-off, {who} by WHO)"


# Titles in plain words for a busy doctor: no shorthand such as Rx or CCB (HTN, T2DM and ≥ kept), same or fewer words.
_TITLE_WORDS = (
    (r"Type 2 diabetes mellitus", "T2DM"), (r"\bHypertension\b", "HTN"), (r"\boral\s+", ""),
    (r"\bOral\s+(\w)", lambda m: m.group(1).upper()),
    (r"^Same-class medicines on the same day", "Same type of medicine twice in a day"),
    (r"\bon the same day\b", "same day"), (r"\(route unclear\)", "(tablet or gel not written)"),
    (r"\bat/above\b", "≥"),
    (r"\b1 missed follow-up\(s\)", "1 missed follow-up"), (r"follow-up\(s\)", "follow-ups"),
    (r"\bincl\. the latest\b", "including latest"), (r"\bin the (?:structured )?record\b", "in record"),
    (r"^Daily dose above the usual maximum", "Dose above usual maximum"),
    (r"^Weight change of 5% or more within 30 days", "Weight change 5% or more in 30 days"),
    (r"^Impossible vital reading in record", "Impossible vital reading"),
    (r"^Symptoms consistent with hyperglycaemia", "Symptoms of high blood sugar"),
    (r"^Possible neuropathic symptoms", "Possible nerve symptoms (neuropathy)"),
    (r"\brenal/fluid\b", "kidney/fluid"),
    (r"for the same complaint", "for same complaint"), (r" with a calcium-channel blocker", " with calcium-channel blocker"),
    (r" with ARB / ACE inhibitor", " with ARB/ACE inhibitor"),
    (r"^(\w+) pattern result with no matching diagnosis", r"\1 result, not diagnosed"),
    (r"\((\w+), Asian Indian cut-off; WHO: (\w+)\)", _bmi_class),
    (r"\(s\)", "s"), (r"≥ ", "≥"))
# Wording that says the same in fewer words; values are never touched.
_COMPACT = (
    (r"^Last \d+ readings:\s*", ""), (r"\bstill recorded at the latest\b", "ongoing"), (r"\babout every\b", "every"),
    (r"^At (\d+ of \d+ visits)", r"\1"), (r"\bdifferent treatments\b", "treatments"),
    (r"\bregimen changed at\b", "changed"), (r"\breturn visits\b", "returns"), (r"\(usual max\b", "(max"),
    (r"\s*\(≈[^)]*\)", ""), (r"\s*—\s*likely °C entered as °F", ""), (r"\bvisit\(s\)", "visits"),
    (r"(\d+(?:\.\d+)?) kg → (\d+(?:\.\d+)?) kg", r"\1→\2 kg"), (r"^Serum\s+", ""), (r"\s+last$", ""),
    (r",?\s*up to$", ""), (r"\s+for '[^']*'$", ""), (r"\bdue\s*(?=[,;]|$)", ""), (r"^[,;\s]+", ""), (r"\bno visit recorded since\b", "no visit since"),
    (r"\b(\d+) days overdue\b", r"seen \1 days late"), (r"\s+and \d+ more day(?:s|\(s\))?\b", ""),
    (r"\s+Tab(?:let)?\b", ""), (r"\babout (\d+) days stated\b", r"\1 days total"),
    (r"~(\d+) days stated\b", r"\1 days total"),
    (r"^(\d+ of \d+ visits)", r"at \1"), (r"\bchanged (\d+) of \d+ returns\b", r"treatment changed \1 times"),
    (r"^[…\s]*\(([^()]*)\)?$", r"\1"), (r"^(.+?): both contain (.+)$", r"\1 (both \2)"),
    (r"\s*\([\d.]+\s*°C calculated\)", ""), (r"\s*\([^()\d]*not counted\)", ""),
    (r"^systolic ≥\d+ or diastolic ≥\d+[\s…]*$", ""))  # the raised-reading encounters it listed are gone


def _short_title(title, encounter_dates=()):
    for old, new in _TITLE_WORDS:
        title = re.sub(old, new, title)
    # 'since #13' -> 'since 6 Aug'
    title = re.sub(r"#(\d+)", lambda m: _day(encounter_dates[int(m.group(1)) - 1])
                   if 0 < int(m.group(1)) <= len(encounter_dates) and encounter_dates[int(m.group(1)) - 1]
                   else m.group(0), title)
    return title[:1].upper() + title[1:]


def _rx(segment):
    """'not on the latest prescription (last #16, …)' -> 'not in latest prescription'; None when not a prescription
    status."""
    if "latest prescription" not in segment:
        return None
    some = re.match(r"not all on the latest prescription \((.+?) on it", segment)
    if some:
        return f"only {some.group(1)} still prescribed"
    if segment.startswith("not on the latest prescription"):
        return "not in latest prescription"
    stopped = re.match(r"(.+?) last prescribed at .*not on the latest prescription", segment)
    if stopped:
        return f"{stopped.group(1)} not in latest prescription"
    also = re.match(r"also on the latest prescription:\s*(.+?)(?:\s*\([^()]*\))?$", segment)
    if also:  # another medicine of the finding, named, so the line never implies the listed ones continue
        return f"{also.group(1)} still prescribed"
    named = re.match(r"(.+?) on the latest prescription", segment)
    return f"{named.group(1)} still prescribed" if named else "still prescribed"


def _strip_refs(text):
    """Encounter numbers, ISO dates and source notes out of a piece of evidence; values stay."""
    text = re.sub(r"\((?:Abnormal )?\((High|Low)\)\)", r"(\1)", text)
    text = re.sub(r"\s*\([^()]*(?:#\d|\d{4}-\d{2}-\d{2}|undated|Document|Encounter|Visit #|same day)[^()]*\)", "", text)
    text = re.sub(r"\s*\bat (?:#\d+(?:\s*Document \d+)?(?:,\s*)?)+…?", "", text)
    text = re.sub(r"\s*#\d+(?:\s*Document \d+)?", "", text)
    text = re.sub(r"\s*\b(?:on|since)\s+\d{4}-\d{2}-\d{2}", "", text)
    text = _ISO.sub("", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    text = re.sub(r"(?:\s*,)+\s*…?\s*$", "", text)
    for old, new in _COMPACT:
        text = re.sub(old, new, text.strip())
    text = re.sub(r"(?:\s*,){2,}", ",", text)  # an emptied slot ('late, , no visit') leaves one comma
    return re.sub(r"\s{2,}", " ", text).strip(" ,;:·…")


def _merge(items):
    """'A with X', 'B with X' -> 'A, B with X'; "'a' → 'b'", "'b' → 'c'" -> "'a' → 'b' → 'c'";
    'Temperature 36.5 °F', 'Temperature 36 °F' -> 'Temperature 36.5, 36 °F'."""
    out = []
    for item in items:
        prev = out[-1] if out else ""
        w, pw = re.match(r"^(.+?) with (.+)$", item), re.match(r"^(.+?) with (.+)$", prev)
        if w and pw and w.group(2).casefold() == pw.group(2).casefold():
            if w.group(1).casefold() not in pw.group(1).casefold():
                out[-1] = f"{pw.group(1)}, {w.group(1)} with {pw.group(2)}"
            continue
        a, b = re.match(r"^('[^']+') → ('[^']+')$", item), re.search(r"→ ('[^']+')$", prev)
        if a and b and a.group(1).casefold() == b.group(1).casefold():
            out[-1] = f"{prev} → {a.group(2)}"
            continue
        v, pv = re.match(r"^(\D+?) (\d+(?:\.\d+)?) (\S+)$", item), re.match(r"^(\D+?) ([\d., ]+) (\S+)$", prev)
        if v and pv and v.group(1) == pv.group(1) and v.group(3) == pv.group(3):
            if v.group(2) not in pv.group(2).split(", "):
                out[-1] = f"{pv.group(1)} {pv.group(2)}, {v.group(2)} {pv.group(3)}"
            continue
        out.append(item)
    return out


def _words(text):
    """Word count; a date ('14 Aug') and a '+N more' are one value each, so each counts as one word."""
    text = re.sub(r"\b\d{1,2} (?:%s)\b" % "|".join(_MONTHS), "D", text)
    text = re.sub(r"\+\d+ more\b", "M", text)
    return len([w for w in text.split() if w not in ("→", "=", "+")])


def concern_line(flag, encounter_dates):
    """One line, ≤20 words: title + key values + dates, e.g.
    'Two NSAIDs same day: Diclofenac + Nucoxia; 2 days (10 Jul, 27 Aug); not on latest Rx.'
    Items holding a number (dose, lab value, BP, %) are the last to go when the line is too long."""
    from clinical_insights import _expand_ranges
    title = _short_title(str(flag.get("title") or ""), encounter_dates)
    segments = [s.strip() for s in str(flag.get("evidence") or "").split(" · ") if s.strip()]
    rx = next((r for r in map(_rx, segments) if r), None)
    segments = [s for s in segments if not _rx(s)]
    main, extras = (segments[0] if segments else ""), segments[1:]

    days, more = None, 0
    for pattern in (r"^On (\d+) days?(?:\(s\))?:\s*", r"^Stated durations added over (\d+) prescription days?(?:\(s\))?:?[\s,]*",
                    r"\s*\bat \d+ visits on (\d+) days\b", r"\s*\bon (\d+) day\(s\)"):
        m = re.search(pattern, main)
        if m:
            days, main = int(m.group(1)), main[:m.start()] + main[m.end():]
            break
    m = re.search(r"\s*\(\+(\d+) more(?: days)?\)", main)
    if m:
        more, main = int(m.group(1)), main[:m.start()] + main[m.end():]

    def dates_in(text):
        found = ["-".join(d) for d in _ISO.findall(text)]
        return found or [encounter_dates[int(n) - 1] for n in re.findall(r"#(\d+)", _expand_ranges(text))
                         if 0 < int(n) <= len(encounter_dates) and encounter_dates[int(n) - 1]]

    entries = [e for e in re.split(r";\s*", main) if e.strip()]
    if len(entries) == 1 and len(_ISO.findall(main)) > 1 and not re.search(r"\([^)]*\d{4}-\d{2}-\d{2}", main):
        entries = re.split(r",\s*(?=\d{4}-\d{2}-\d{2})", main)
    dates, contents = [], []
    for entry in entries:
        dates += dates_in(entry)
        text = _strip_refs(entry)
        if text and text.casefold() not in title.casefold() and text.casefold() not in {c.casefold() for c in contents}:
            contents.append(text)
    if not dates:  # nothing dated in the main finding: the encounters named next to it
        dates = [d for e in extras for d in dates_in(e)]
    dates = sorted(set(dates))
    total = days if days else len(dates) + more
    if flag.get("kind") == "medication":  # one drug class for all items: said once, after the title
        classes = {c for i in contents for c in re.findall(r"\(([A-Z][a-z][^()\d]*)\)", i)}
        if len(classes) == 1:
            title += f" ({next(iter(classes)).lower()})"
        contents = [re.sub(r"\s*\([A-Z][a-z][^()\d]*\)", "", i) for i in contents]
    contents = _merge(list(dict.fromkeys(contents)))
    extras = [_strip_refs(e) for e in extras]
    extras = [e for e in extras if e and not re.match(r"(?i)no .*\bin the record$", e)
              and e.casefold() not in title.casefold()]
    if rx and rx != "still prescribed" and not rx.startswith(("only", "not")) and rx.endswith(" still prescribed"):
        drug = rx[: -len(" still prescribed")]
        rx = "still prescribed" if drug.casefold() in " ".join(contents).casefold() else rx

    def when_text(n):
        shown = dates[:n]
        hidden = max(total - len(shown), len(dates) - len(shown), 0)
        # When every date is known, the latest always shows: '29 May, 25 Jun, +1 more, 20 Oct'.
        if hidden and n >= 2 and len(dates) > n and len(dates) >= total:
            text = ", ".join([*map(_day, dates[:n - 1]), f"+{hidden} more", _day(dates[-1])])
        else:
            text = ", ".join(_day(d) for d in shown) + (f" +{hidden} more" if hidden else "")
        return (f"{days} day{'s' if days != 1 else ''}" + (f" ({text})" if text else "")) if days else text

    def build(items, extra, dropped=0, n_dates=_MAX_DATES):
        when = when_text(n_dates)
        body = ", ".join(items) + (f" +{dropped} more" if dropped and items else "")
        if when and not days and body:
            body = f"{body} ({when})"
        tail = [when] if when and (days or not body) else []
        parts = [p for p in [body, *(tail if days else []), *extra, *([] if days else tail), rx or ""] if p]
        return (f"{title}: " + "; ".join(parts) if parts else title) + "."

    # 2 = a key value (BP, dose, lab value, weight, %), 1 = another number, 0 = no number.
    def weight(t):
        if re.search(r"\d{2,3}/\d{2,3}|\d\s*(?:%|mg|kg|mmHg|°F|/cumm|g/dL|U/L|IU)", t):
            return 2
        return 1 if re.search(r"\d", t) else 0

    def names_only(item):
        words = re.findall(r"[a-z]{4,}", item.casefold())
        return flag.get("kind") == "medication" and words and not any(w in title.casefold() for w in words)

    items, extra, n_dates, dropped_extras = list(contents), list(extras), _MAX_DATES, []
    line = lambda: build(items, extra, len(contents) - len(items), n_dates)
    # Drop, in order: extras / items with no number, the 3rd date, extras / items with a plain number, the 2nd date,
    # then key values (extras before items). One item always stays.
    for pool, level in [(extra, 0), (items, 0), ("dates", 2), (extra, 1), (items, 1), ("dates", 1), (extra, 2), (items, 2)]:
        if pool == "dates":
            if _words(line()) > _MAX_WORDS and len(dates) > level:
                n_dates = level
            continue
        while _words(line()) > _MAX_WORDS:
            idx = next((i for i in range(len(pool) - 1, -1, -1) if weight(pool[i]) == level
                        and not (pool is items and len(items) == 1
                                 and (level == 2 or not extra or (level == 0 and names_only(items[0]))))), None)
            # (the last item stays when it holds a key value, has nothing beside it, or is a list of names - e.g. the
            # medicines - that the title does not already say)
            if idx is None:
                break
            gone = pool.pop(idx)
            if pool is extra:
                dropped_extras.append(gone)
    # Put back what still fits: dates first, then dropped extras with a key value.
    while n_dates < _MAX_DATES and len(dates) > n_dates:
        n_dates += 1
        if _words(line()) > _MAX_WORDS:
            n_dates -= 1
            break
    for e in sorted(dropped_extras, key=weight, reverse=True):
        extra.append(e)
        if _words(line()) > _MAX_WORDS:
            extra.pop()
    if _words(line()) > _MAX_WORDS and items:  # one long item left: shorten it, never the dates or the Rx status
        keep = _MAX_WORDS - (_words(line()) - _words(items[0]))
        items[0] = " ".join(items[0].split()[:max(keep, 3)]) + "…"
    return line()


# ── Status tiles ────────────────────────────────────────────────────────────

def _latest_lab(labs, keys):
    """'HbA1c 7.6 %' for the most recent result of the first test in `keys` that has one (undated results last)."""
    for key in keys:
        rows = [(lab, date) for lab, _, date in labs if lab_validation.identify(lab.get("test_name")) == key
                and lab_validation.parse_value(lab.get("value"))[1] is not None]
        if rows:
            lab, date = max(rows, key=lambda r: r[1] or "")
            name = _catalog_name(key)
            return f"{name} {str(lab.get('value')).strip()}" + ("" if date else " (undated)")
    return None


def _catalog_name(key):
    return {"hba1c": "HbA1c", "glucose_fasting": "FBS", "glucose_pp": "PPBS",
                                 "glucose_random": "RBS", "creatinine": "Creatinine", "egfr": "eGFR",
                                 "urea": "Urea", "bun": "BUN", "urine_acr": "Urine ACR", "alt": "SGPT (ALT)",
                                 "ast": "SGOT (AST)", "bilirubin_total": "Bilirubin", "alp": "ALP",
                                 "ggt": "GGT"}.get(key, key)


def _tile_value(area, flags, ctx):
    """The one key value shown on a tile."""
    top = flags[0] if flags else None
    if area in _LAB_KEYS:
        value = _latest_lab(ctx["labs"], _LAB_KEYS[area])
        if value:
            return value
        return "Not tested"
    if area == "bp":
        return f"BP {ctx['bp'][-1]}" if ctx["bp"] else "Not recorded"
    if area == "weight":
        bmi = next((re.search(r"BMI \d+(?:\.\d+)?", f["title"]).group() for f in ctx["all_flags"]
                    if re.match(r"BMI \d", f["title"])), None)
        bmi = bmi or next((re.search(r"BMI \d+(?:\.\d+)?", r["finding"]).group() for r in ctx["guideline"]
                           if r.get("topic") == "BMI" and re.search(r"BMI \d", r.get("finding") or "")), None)
        change = next((f for f in flags if f["title"].startswith("Weight change")), None)
        pct = re.search(r"[−+-]\d+%", change["evidence"]).group() if change and re.search(r"[−+-]\d+%", change["evidence"]) else ""
        base = bmi or (f"{ctx['weight'][-1]}" if ctx["weight"] else "Not recorded")
        return f"{base} · {pct} weight" if pct else base
    if area == "followup":
        n = len(ctx["missed"])
        return f"{n} missed" if n else "None missed"
    if area == "medicine":
        if not top:
            return f"{ctx['meds']} medicine{'s' if ctx['meds'] != 1 else ''} at latest visit · no flags"
        dose = re.search(r"=\s*([\d.]+\s*mg/day)\s*\(usual max [\d.]+\s*mg/day ([A-Za-z]+)", top["evidence"])
        return f"{dose.group(1)} {dose.group(2)}" if dose else _short_title(top["title"])
    if area == "labs":
        names = [re.sub(r"^No (.+?) in the record.*", r"\1", f["title"]) for f in flags]
        names = [re.sub(r"^(.+?): no related lab tests.*", r"\1 labs", n) for n in names]
        names = [_short_title(n) for n in names]
        return names[0] + (f" +{len(names) - 1} more" if len(names) > 1 else "") if names else "None"
    return _short_title(top["title"]) if top else ""


def status_tiles(snapshot, patient_data, labs):
    """One tile per area that applies to this patient: label, status (red / amber / green / grey), key value."""
    visits = patient_data.get("visits") or []
    flags = snapshot.get("flags") or []
    active = " ".join(snapshot.get("active") or []).casefold()
    by_area = {}
    for f in flags:
        by_area.setdefault(f.get("area"), []).append(f)

    def series(label):
        out = []
        for v in visits:
            row = next((r for r in v.get("db_vitals_rows") or [] if r.get("label") == label), None)
            value = str((row or {}).get("value") or "").strip()
            if value and re.search(r"\d", value):
                out.append(re.sub(r"\s*mmHg$", "", value))
        return out

    bp = [v for v in series("BP") if re.match(r"^\d{2,3}\s*/\s*\d{2,3}", v)]
    weight = series("Weight")
    missed = [f for v in visits for f in v.get("followups") or [] if "MISSED" in str(f.get("status", "")).upper()]
    has_lab = {area: any(lab_validation.identify(lab.get("test_name")) in keys for lab, _, _ in labs)
               for area, keys in _LAB_KEYS.items()}
    diabetes, htn = "diabet" in active, "hypertension" in active
    nsaid = any(re.search(r"NSAID", f["title"]) and f["level"] in ("high", "verify", "medium") for f in flags)
    applies = {
        "glucose": diabetes or has_lab["glucose"] or bool(by_area.get("glucose")),
        "bp": bool(bp) or htn or bool(by_area.get("bp")),
        "kidney": has_lab["kidney"] or bool(by_area.get("kidney")) or diabetes or htn or nsaid,
        "liver": has_lab["liver"] or bool(by_area.get("liver")),
        "medicine": any(v.get("db_medications") for v in visits) or bool(by_area.get("medicine")),
        "weight": bool(weight) or bool(by_area.get("weight")),
        "followup": any(v.get("followups") for v in visits) or bool(by_area.get("followup")),
        "labs": bool(by_area.get("labs")),
    }
    ctx = {"labs": labs, "bp": bp, "weight": weight, "missed": missed, "all_flags": flags,
           "guideline": (snapshot.get("guideline") or {}).get("rows") or [], "meds": snapshot.get("active_meds") or 0}
    tiles = []
    for area, label in AREAS:
        if not applies[area]:
            continue
        area_flags = by_area.get(area) or []
        level = next((lvl for lvl in ("high", "verify", "medium") if any(f["level"] == lvl for f in area_flags)), None)
        value = _tile_value(area, area_flags, ctx)
        no_data = value in ("Not tested", "Not recorded")
        status = _STATUS.get(level) or ("grey" if no_data else "green")
        tiles.append({"area": area, "label": label, "status": status, "value": value,
                      "concerns": sum(1 for f in area_flags if f["level"] in ("high", "verify", "medium"))})
    return tiles


_LEVEL_RANK = {"high": 0, "verify": 1, "medium": 2, "info": 3}
_GROUP_LABELS = {"medicine": "Medicines", "bp": "Blood pressure", "glucose": "Diabetes / blood sugar",
                 "kidney": "Kidney", "liver": "Liver", "weight": "Weight", "labs": "Tests (missing or abnormal)",
                 "followup": "Follow-up", "symptoms": "Symptoms to check", "course": "How the illness is going",
                 "other": "Other", "error": "Record errors (check the entry)"}
_KIND_GROUP = {"lab": "labs", "risk": "symptoms", "pattern": "course"}


def concern_group(flag):
    """The topic a concern is shown under: its tile area, else by kind (an abnormal result with no tile is a test,
    a symptom is a symptom, a course pattern is the illness course); record errors apart."""
    if flag.get("record_error") or flag.get("kind") == "data":
        return "error"
    return flag.get("area") or _KIND_GROUP.get(flag.get("kind"), "other")


def order_concerns(flags):
    """Concerns (high / verify / medium flags) grouped by topic: [{'group', 'label', 'concerns': [flag indexes]}].
    Within a group: level (high > verify > medium > info), then current before past, record errors last. Groups: by
    their highest level, then by how many concerns they hold; record errors last."""
    rank = lambda i: (_LEVEL_RANK.get(flags[i]["level"], 3), flags[i].get("time") == "past",
                      bool(flags[i].get("record_error")), i)
    groups = {}
    for i in sorted((i for i, f in enumerate(flags) if f["level"] in ("high", "verify", "medium")), key=rank):
        groups.setdefault(concern_group(flags[i]), []).append(i)
    ordered = sorted(groups.items(), key=lambda g: (g[0] == "error", min(rank(i)[0] for i in g[1]), -len(g[1]),
                                                    min(rank(i) for i in g[1])))
    return [{"group": g, "label": _GROUP_LABELS[g], "concerns": idx} for g, idx in ordered]


def flag_id(flag, seen=None):
    """Stable id of a flag within one patient's snapshot ('f-' + 8 hex): from its kind and title, which do not change
    when the evidence text is reworded. A second flag with the same kind and title gets a numeric suffix."""
    import hashlib
    base = "f-" + hashlib.sha1(f"{flag.get('kind', '')}|{flag.get('title', '')}".encode("utf-8")).hexdigest()[:8]
    fid, n = base, 1
    while seen is not None and fid in seen:
        n += 1
        fid = f"{base}-{n}"
    if seen is not None:
        seen.add(fid)
    return fid


def decorate(snapshot, patient_data, labs):
    """Adds to the snapshot: each flag's area, short line and record-error mark; concerns in reading order; tiles."""
    dates = snapshot.get("dates") or []
    seen = set()
    for f in snapshot.get("flags") or []:
        f["id"] = flag_id(f, seen)
        f["area"] = flag_area(f)
        f["summary"] = concern_line(f, dates)
        f["record_error"] = f.get("kind") == "data"
    snapshot["concern_groups"] = order_concerns(snapshot["flags"])
    snapshot["concerns"] = [i for g in snapshot["concern_groups"] for i in g["concerns"]]
    snapshot["tiles"] = status_tiles(snapshot, patient_data, labs)
    return snapshot


# ── Area tags for brief points and lab details (same keys as the tiles) ─────
_AREA_WORDS = {
    "glucose": r"hba1c|glucose|sugar|diabet|t2dm|\bdm\b|hyperglyc|hypoglyc|\b(?:fbs|ppbs|rbs)\b|metformin|glic|glim|"
               r"gliptin|insulin|gliflozin|pioglit|voglib|acarbose",
    "bp": r"\bbp\b|blood pressure|hypertens|\bhtn\b|systolic|diastolic|\b\d{2,3}/\d{2,3}\b|sartan\b|amlodip|cilnid|"
          r"nifedip|olol\b|pril\b|chlorthal|hydrochlorothiazide",
    "kidney": r"creatinine|egfr|kidney|renal|\burea\b|\bbun\b|albumin|proteinuria|\bacr\b|nephro",
    "liver": r"liver|\b(?:sgpt|sgot|alt|ast|alp|ggt)\b|bilirubin|hepat",
    "medicine": r"medicine|medication|\bdrugs?\b|\bdoses?\b|dosing|prescri|tablet|\btab\b|nsaid|steroid|interaction|"
                r"duplicate|side effect|0-0-0|not continued|stopped",
    "weight": r"\bbmi\b|weight|obes",
    "followup": r"follow-?up|missed (?:visit|review)|revisit|recheck|re-check",
    "labs": r"\b(?:no|not|never)\b[^.;]{0,50}\b(?:tests?|tested|labs?|checked|results?|report)\b|not (?:tested|checked)",
}
_THEME_AREA = {"treatment": "medicine"}
_AREA_ORDER = [a for a, _ in AREAS]


def point_areas(text, flags=(), theme=""):
    """The tile areas a brief point or lab detail is about (AREAS keys, in tile order); ['general'] when none fits.
    From the point's wording, then the rule flag it states, then its theme (Treatment points are medicine)."""
    low = str(text or "").casefold()
    found = {a for a, pattern in _AREA_WORDS.items() if re.search(pattern, low)}
    if flags and not found - {"labs"}:
        from clinical_insights import _point_flag
        f = _point_flag(str(text), list(flags))
        if f and (f.get("area") or flag_area(f)):
            found.add(f.get("area") or flag_area(f))
    if _THEME_AREA.get(str(theme).casefold()):
        found.add(_THEME_AREA[str(theme).casefold()])
    return [a for a in _AREA_ORDER if a in found] or ["general"]


def lab_areas(text):
    """Tile areas of a lab detail line: the catalog test it names, else its wording; ['labs'] when none fits."""
    head = re.split(r"\d", re.sub(r"\*\*", "", str(text or "")), maxsplit=1)[0].strip(" :-")
    key = lab_validation.identify(head) if head else None
    found = [a for a, keys in _LAB_KEYS.items() if key in keys]
    return found or [a for a in point_areas(text) if a != "general"] or ["labs"]


def split_brief_points(body):
    """A theme's points: split at '; ' outside brackets (the brief's own point separator)."""
    return [p.strip().rstrip(".").strip() for p in re.split(r";\s+(?![^()]*\))", str(body or "")) if p.strip(" .")]


def tag_brief(lines, flags=()):
    """Brief bullets ('- **Theme:** point; point') as [{'theme', 'points': [{'text', 'areas'}]}]."""
    out = []
    for line in lines or []:
        m = re.match(r"^\s*(?:[-*•]\s*)?\*\*(.+?):?\*\*:?\s*(.*)$", str(line).strip())
        if not m:
            continue
        theme = m.group(1).strip(" :")
        out.append({"theme": theme, "points": [{"text": p, "areas": point_areas(p, flags, theme)}
                                               for p in split_brief_points(m.group(2))]})
    return out


def tag_review(markdown, flags=()):
    """Area tags for a review's markdown: the Clinical Insights points and the Abnormal Results (lab details) lines."""
    insights, labs, section = [], [], None
    for line in str(markdown or "").splitlines():
        s = line.strip()
        if s.startswith("## "):
            section = s[3:].strip().casefold()
            continue
        if not s or not s.startswith(("- ", "* ", "• ")):
            continue
        if section == "clinical insights":
            insights.append(s)
        elif section == "abnormal results":
            text = re.sub(r"^[-*•]\s*", "", s)
            labs.append({"text": text, "areas": lab_areas(text)})
    return {"insights": tag_brief(insights, flags), "labs": labs}


# ── Medicine-class timeline (rows under the visit timeline) ─────────────────
_NSAID = "NSAID (pain / anti-inflammatory)"
_MAX_MED_ROWS = 8


def _same_product(a, b):
    """Two lines are one product: same ingredients when both are known, else near-identical names (spelling)."""
    ga, gb = set(a.get("ingredients") or []), set(b.get("ingredients") or [])
    if ga and gb:
        return ga == gb
    wa, wb = str(a["name"]).casefold().split()[:1], str(b["name"]).casefold().split()[:1]
    return bool(wa and wb and SequenceMatcher(None, wa[0], wb[0]).ratio() >= 0.75)


def medicine_timeline(rec, visits):
    """One row per medicine class prescribed at 2+ visits (or twice on one day): for each visit, the medicines of that
    class, and whether two different ones were given on the same date (a same-class duplicate). NSAIDs come first,
    split by route (oral / topical); a line with no written form is shown under oral, marked 'route unclear'."""
    dates = [str(v.get("visit_date") or "")[:10] for v in visits]
    rows = {}
    for line in rec.prescribed():
        i = line.get("i")
        if not i or i > len(visits):
            continue
        for cls in line.get("classes") or []:
            if cls == _NSAID:
                key = "NSAID — topical" if line.get("route") in LOCAL_ROUTES else "NSAID — oral"
            else:
                key = cls.split(" (")[0]
            name = line["name"] + (" (route unclear)" if line.get("route") == "unclear" else "")
            row = rows.setdefault(key, {"label": key, "cls": cls, "cells": {}, "lines": {}})
            if name not in row["cells"].setdefault(i, []):
                row["cells"][i].append(name)
                row["lines"].setdefault(i, []).append(line)
    out = []
    for key, row in rows.items():
        # Same-class duplicate: two different products (or one product at two visits) of a duplicate-prone class (the classes the same-class rule
        # checks; combination therapy such as two antidiabetics is not one) on one date, by any route but topical.
        by_date = {}
        for i, lines in row["lines"].items():  # systemic lines only: a cream with a tablet is not a duplicate
            by_date.setdefault(dates[i - 1], []).extend(l for l in lines if l.get("route") not in LOCAL_ROUTES)
        dup_dates = {d for d, lines in by_date.items() if d and row["cls"] in GROUP_CLASSES and "topical" not in key
                     and any(not _same_product(a, b) or a["i"] != b["i"] for n, a in enumerate(lines) for b in lines[n + 1:])}
        if len(row["cells"]) < 2 and not dup_dates:
            continue
        cells = [{"encounter": i, "date": dates[i - 1], "medicines": names, "duplicate": dates[i - 1] in dup_dates,
                  # Prescriber specialization per medicine of the cell (same order as 'medicines'); None when unknown.
                  "specializations": [l.get("specialization") for l in row["lines"].get(i, [])][:len(names)]}
                 for i, names in sorted(row["cells"].items())]
        out.append({"label": key, "class": row["cls"], "cells": cells,
                    "visits": len(cells), "duplicate_days": len(dup_dates)})
    out.sort(key=lambda r: (not r["label"].startswith("NSAID"), r["label"] != "NSAID — oral",
                            -r["duplicate_days"], -r["visits"]))
    return out[:_MAX_MED_ROWS]
