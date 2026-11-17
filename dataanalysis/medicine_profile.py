"""Each prescription line as the rules see it: active ingredients, route, rule classes and course window.

Ingredients, not brand names, drive the class rules. A line is expanded (registry data/reference/medicine_ingredients.json):
  1. the brand table;
  2. a '+' combination, part by part ('Miconazole cream + Eberfine M cream');
  3. ingredient names written in the line ('Calcium d3', 'Diclofenac sodium tab'), misspellings by similarity;
  4. the cached term resolution (clinical_terms, model-resolved once per term);
  otherwise it is unmapped. Unmapped products, and products only the model resolved, go to the unmapped-brands report
  (config UNMAPPED_BRANDS_REPORT) instead of being skipped silently.
The NSAID / corticosteroid / azole / PPI / calcium-vitamin D classes come from the ingredients' groups; other classes
(antihistamine, antihypertensive ...) keep coming from the term resolution or the registry patterns.

Route of a line: the form written on it -> the brand's registry form -> an ingredient made for one route only -> the
resolved route of a product with no multi-route ingredient -> the same product with a written form in a near-identical
prescription elsewhere in the record (template) -> 'unclear'. A rule that needs an oral / systemic route treats 'unclear'
as unknown: it reports the finding for checking instead of assuming the oral form.

Course window: from the prescription date to date + stated duration (core.encounters.course_end).
"""
import json
import logging
import os
import re
import threading
from datetime import date as _date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import clinical_terms
from core.encounters import course_end, dose_not_specified, duration_days, unclear_dose
from patient_chat import REGISTRY, _med_display

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent
with (_DIR / "data" / "reference" / "medicine_ingredients.json").open(encoding="utf-8") as _f:
    REF = json.load(_f)
_BRANDS = REF["brands"]
_INGREDIENTS = REF["ingredients"]
_GROUPS = REF["groups"]
_FORM = {route: re.compile(rf"(?<![a-z])(?:{p})(?![a-z])", re.I) for route, p in REF["forms"].items()}
_ANY_FORM = re.compile(r"(?<![a-z])(?:" + "|".join(REF["forms"].values()) + r")(?![a-z])", re.I)
_ORAL_PREFIX = re.compile(r"^\s*(?:" + "|".join(map(re.escape, REF["oral_prefixes"])) + r")\.?\s+", re.I)
_STRENGTH = re.compile(r"(?<![a-z\d.])\d+(?:\.\d+)?\s*(?:mg|mcg|gm|g|ml|iu|%)?(?![a-z\d])", re.I)
_SPLIT = re.compile(r"\s*(?:\+|&|\bplus\b)\s*", re.I)
_NOT_A_MEDICINE = re.compile(r"(?i)\b(?:diet|salt|exercise|walk(?:ing)?|advice|advised|counsel\w*|review|report|"
                             r"monitor\w*|follow[\s-]*up|plenty|fluids?\s+intake|rest|steam)\b")
# Words that do not change what a product is ('Diclofenac Sodium', 'Omeprazole IP').
_FILLER_WORDS = {"sodium", "potassium", "hcl", "hydrochloride", "ip", "bp", "usp", "forte", "sr", "er", "xr", "mr",
                 "ds", "dt", "od", "oral", "plain", "vit", "vitamin", "and", "with"}
_CLASS_PATTERNS = [(re.compile(e["pattern"], re.I), e["class"]) for e in REGISTRY["medicine_classes"]]
GROUP_CLASSES = {c for g in _GROUPS.values() for c in (g.get("class"), g.get("systemic_class"), g.get("topical_class")) if c}
SYSTEMIC_ROUTES = {"oral", "injection"}
LOCAL_ROUTES = {"topical", "drops", "inhaled"}
_FORM_ROUTE = {"tablet": "oral", "capsule": "oral", "syrup": "oral", "cream": "topical", "gel": "topical",
               "ointment": "topical", "lotion": "topical", "injection": "injection", "drops": "drops", "inhaler": "inhaled"}
# Ingredient spellings, longest first, so 'calcium carbonate' is read before 'calcium' and 'vitamin d3' before 'd3'.
_NAMES = sorted({(n.casefold(), canon) for canon, e in _INGREDIENTS.items() for n in [canon, *e["names"]]},
                key=lambda x: -len(x[0]))
_FUZZY = [(n, canon) for n, canon in _NAMES if len(n) >= 6 and " " not in n]


# ── Names, forms and ingredients ────────────────────────────────────────────

def base_key(name):
    """'Tab Nucoxia 90' -> 'nucoxia', 'LOBET GM CREAM' -> 'lobet gm', 'CAL+D3' -> 'cal d3'."""
    text = _ORAL_PREFIX.sub("", str(name or "")).casefold()
    text = _STRENGTH.sub(" ", text)
    text = _ANY_FORM.sub(" ", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def written_route(name):
    """The route a line's written dose form states ('gel' -> topical, 'Tab' -> oral), else None."""
    raw = str(name or "")
    if _ORAL_PREFIX.match(raw):
        return "oral"
    for route in ("topical", "injection", "drops", "inhaled", "oral"):
        if _FORM[route].search(raw):
            return route
    return None


def _from_name(key):
    """(ingredients written in a base key, leftover words that are not an ingredient or a filler word)."""
    text, found = f" {key} ", []
    for n, canon in _NAMES:
        pattern = rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])"
        if re.search(pattern, text):
            found.append(canon)
            text = re.sub(pattern, " ", text)
    leftover = []
    for token in text.split():
        best = max(((SequenceMatcher(None, token, n).ratio(), canon) for n, canon in _FUZZY), default=(0, None)) \
            if len(token) >= 6 else (0, None)
        if best[0] >= 0.85:
            found.append(best[1])
        elif token not in _FILLER_WORDS and not token.isdigit():
            leftover.append(token)
    return list(dict.fromkeys(found)), leftover


def canonical(ingredient):
    """A resolved generic name in the registry's spelling ('colecalciferol' -> 'cholecalciferol'), else as given."""
    found, _ = _from_name(base_key(ingredient))
    return found[0] if len(found) == 1 else str(ingredient).casefold()


def is_medicine(patient_data, name):
    info = clinical_terms.medicine(patient_data, name)
    if info and info.get("is_medicine") is False:
        return False
    return not _NOT_A_MEDICINE.search(str(name or ""))


def ingredients(patient_data, name, _part=False):
    """(ingredients, source, unmapped parts). source: registry | name | model | combination | unmapped."""
    key = base_key(name)
    if not key:
        return [], "unmapped", []
    brand = _BRANDS.get(key)
    if brand:
        return list(brand["ingredients"]), "registry", []
    parts = [p for p in _SPLIT.split(str(name)) if base_key(p)] if not _part else []
    if len(parts) > 1:
        found, missing = [], []
        for part in parts:
            ing, source, unmapped = ingredients(patient_data, part, _part=True)
            found += ing
            missing += unmapped or ([part.strip()] if source == "unmapped" else [])
        return list(dict.fromkeys(found)), ("combination" if found else "unmapped"), missing
    named, leftover = _from_name(key)
    if named and not leftover:
        return named, "name", []
    info = clinical_terms.medicine(patient_data, name)
    if info and info.get("generic"):
        generic = list(dict.fromkeys(canonical(g) for g in info["generic"]))
        # A name that is itself the generic ('Metformin', 'Glimipride') is not a brand the model had to resolve.
        written = any(SequenceMatcher(None, key, base_key(g)).ratio() >= 0.85 or key.startswith(base_key(g))
                      for g in info["generic"])
        return generic, ("name" if written else "model"), []
    if named:
        return named, "name", []
    return [], "unmapped", [str(name).strip()]


def _ingredient_routes(ings):
    return [set(_INGREDIENTS[g]["routes"]) for g in ings if g in _INGREDIENTS]


def classes(patient_data, name, ings, route):
    """Registry medicine classes of a line. NSAID / corticosteroid / azole / PPI / calcium-vitamin D come from the
    ingredients when every ingredient is in the registry; other classes from the term resolution or the patterns."""
    out = set()
    for g in (grp for ing in ings for grp in (_INGREDIENTS.get(ing) or {}).get("groups", [])):
        spec = _GROUPS[g]
        if spec.get("class"):
            out.add(spec["class"])
        elif route in LOCAL_ROUTES:
            out.add(spec["topical_class"])
        elif route in SYSTEMIC_ROUTES or route == "other":
            out.add(spec["systemic_class"])
        else:  # route unclear: either, so both rule families see it (systemic rules then ask for checking)
            out |= {spec["topical_class"], spec["systemic_class"]}
    info = clinical_terms.medicine(patient_data, name) or {}
    other = set(info.get("classes") or []) or {c for p, c in _CLASS_PATTERNS if p.search(str(name or ""))}
    if ings and all(i in _INGREDIENTS for i in ings):
        other -= GROUP_CLASSES
    return out | other


def groups(ings):
    return {g for ing in ings for g in (_INGREDIENTS.get(ing) or {}).get("groups", [])}


# ── The record's lines ──────────────────────────────────────────────────────

def _route(patient_data, line):
    """(route, basis) without the template step; ('unclear', None) when only a template can tell."""
    written = written_route(line["raw"])
    if written:
        return written, "written"
    brand = _BRANDS.get(line["key"]) or {}
    if brand.get("form") in _FORM_ROUTE:
        return _FORM_ROUTE[brand["form"]], "brand registry"
    routes = _ingredient_routes(line["ingredients"])
    common = set.intersection(*routes) if routes else set()
    if len(common) == 1:  # one ingredient made for one route fixes the product's route (calcium + D3 is oral)
        return next(iter(common)), "ingredient"
    if any(len(r) > 1 for r in routes):
        return "unclear", None
    info = clinical_terms.medicine(patient_data, line["name"]) or {}
    if info.get("route") and info["route"] != "other":
        return info["route"], "resolved"
    return "oral", "assumed (no form written, single-route product)"


_COUNT = r"(\d(?:\.\d)?|1/2|½)"
_REGIMEN = re.compile(rf"\s*{_COUNT}\s*-\s*{_COUNT}\s*-\s*{_COUNT}(?:\s*-\s*{_COUNT})?\s*")
_SOS = re.compile(r"(?i)\b(?:sos|prn|as\s+(?:needed|required)|stat|od|bd|bid|tds|tid|qid|hs|once|twice)\b")


def parse_dose(line_med, name):
    """{status, per_day, strength_mg, daily_mg}. status: stated | not specified (0-0-0 + duration) | unclear
    (0-0-0, no duration) | as needed / frequency word | unreadable."""
    dosage = str(line_med.get("dosage") or "").strip()
    if dose_not_specified(line_med):
        return {"status": "not specified"}
    if unclear_dose(line_med):
        return {"status": "unclear"}
    match = _REGIMEN.fullmatch(dosage)
    if not match:
        return {"status": "frequency word" if _SOS.search(dosage) else ("unreadable" if dosage else "not written")}
    per_day = sum(0.5 if x in ("1/2", "½") else float(x) for x in match.groups() if x)
    strength = re.search(r"(?<![a-z\d.])(\d+(?:\.\d+)?)\s*(?:mg)?(?![a-z\d%])", str(name or ""), re.I)
    out = {"status": "stated", "per_day": per_day}
    if strength and not _SPLIT.search(str(name)):
        out["strength_mg"] = float(strength.group(1))
        out["daily_mg"] = per_day * out["strength_mg"]
    return out


def _default_route(line):
    """The registry default route when every multi-route ingredient of the line has the same one, else unclear."""
    defaults = {(_INGREDIENTS.get(g) or {}).get("default_route") for g in line["ingredients"]
                if len((_INGREDIENTS.get(g) or {}).get("routes", [])) > 1}
    if len(defaults) == 1 and None not in defaults:
        return defaults.pop(), "registry default route"
    return line["route"], line["route_basis"]


def _same(a, b):
    ia, ib = set(a["ingredients"]), set(b["ingredients"])
    if ia and ib:
        return ia == ib
    return SequenceMatcher(None, a["key"], b["key"]).ratio() >= 0.8


class MedRecord:
    """All prescription lines of a record, per encounter (1-based)."""

    def __init__(self, patient_data):
        visits = patient_data.get("visits") or []
        self.visits = visits
        self.lines = []
        for i, v in enumerate(visits, 1):
            date = str(v.get("visit_date") or "")[:10]
            row = []
            for m in v.get("db_medications") or []:
                if not m.get("name"):
                    continue
                name = _med_display(m.get("name"))
                medicine = is_medicine(patient_data, name)
                ings, source, unmapped = ingredients(patient_data, name) if medicine else ([], "not a medicine", [])
                row.append({"i": i, "date": date, "raw": str(m.get("name")), "name": name, "key": base_key(name),
                            "ingredients": ings, "source": source, "unmapped": unmapped, "medicine": medicine,
                            "prescribed": not unclear_dose(m), "dose_not_specified": dose_not_specified(m),
                            "duration": duration_days(m), "start": date or None,
                            "end": course_end(date, m) if date else None, "dosage": str(m.get("dosage") or ""),
                            "dose": parse_dose(m, name),
                            # Specialization of the doctor who wrote this line (never a name); None when unknown.
                            "specialization": (str(m.get("prescriber_specialization") or "").strip() or None)})
            self.lines.append(row)
        for row in self.lines:
            for line in row:
                line["route"], line["route_basis"] = _route(patient_data, line)
        for row in self.lines:
            for line in row:
                if line["route"] == "unclear":
                    line["route"], line["route_basis"] = self._template_route(line)
                if line["route"] == "unclear":
                    line["route"], line["route_basis"] = _default_route(line)
        for row in self.lines:
            for line in row:
                line["classes"] = classes(patient_data, line["name"], line["ingredients"], line["route"])
                line["groups"] = groups(line["ingredients"])

    def _template_route(self, line):
        """The route written for the same product in a near-identical prescription elsewhere in the record."""
        spec = REF["template_inference"]
        mine = {l["key"] for l in self.lines[line["i"] - 1] if l is not line and l["prescribed"]}
        best = None
        for row in self.lines:
            for other in row:
                if other["i"] == line["i"] or other["route_basis"] not in ("written", "brand registry") \
                        or not _same(other, line):
                    continue
                theirs = {l["key"] for l in row if l is not other and l["prescribed"]}
                shared = len(mine & theirs)
                score = shared / len(mine | theirs) if mine | theirs else 0
                if shared >= spec["min_shared"] and score >= spec["min_similarity"] and (best is None or score > best[0]):
                    best = (score, other["route"], other["i"])
        if best:
            return best[1], f"template #{best[2]}"
        return "unclear", "route unclear (no form written)"

    # Lookups by encounter and display name (the names clinical_snapshot works with).
    def line(self, i, name):
        return next((l for l in self.lines[i - 1] if l["name"] == name), None) if 1 <= i <= len(self.lines) else None

    def route(self, i, name):
        line = self.line(i, name)
        return line["route"] if line else None

    def systemic(self, i, name):
        """'yes' / 'no' / 'unclear': is this line a systemic (oral / injected) product? A name with no line here
        (not expected) counts as systemic, as the rules always assumed."""
        route = self.route(i, name)
        if route == "unclear":
            return "unclear"
        return "no" if route in LOCAL_ROUTES else "yes"

    def classes(self, i, name):
        line = self.line(i, name)
        return line["classes"] if line else set()

    def prescribed(self):
        return [l for row in self.lines for l in row if l["prescribed"] and l["medicine"]]

    def running_on(self, line, on_date):
        """True when the latest course of this line's medicine started on or before on_date still runs on it."""
        courses = [l for l in self.prescribed() if _same(l, line) and l["start"] and l["start"] <= on_date]
        if not courses:
            return False
        last = max(courses, key=lambda l: (l["start"], l["end"] or ""))
        return bool(last["end"] and last["end"] >= on_date)

    def course_end(self, line):
        """End of the latest course of this line's medicine (None when that course states no duration)."""
        courses = [l for l in self.prescribed() if _same(l, line) and l["start"]]
        return max(courses, key=lambda l: l["start"])["end"] if courses else None

    def same(self, a, b):
        return _same(a, b)

    def unmapped(self):
        return [(l["name"], part) for l in self.prescribed() for part in l["unmapped"]]

    def model_only(self):
        return [l["name"] for l in self.prescribed() if l["source"] == "model"]


def record(patient_data):
    return MedRecord(patient_data)


# ── Medicines taken before a lab result (cause attribution) ─────────────────

def _days_before(a, b):
    try:
        return (_date.fromisoformat(b) - _date.fromisoformat(a)).days
    except (TypeError, ValueError):
        return None


def effect_window_medicines(rec, lab_key, lab_date, direction):
    """[{name, effect, window_days, last_taken}] for medicines whose course overlaps the plausible window before a
    lab test (registry drug_lab_effects). Empty when the test is undated: an undated result cannot be placed."""
    if not lab_key or not lab_date:
        return []
    found = {}
    for rule in REF["drug_lab_effects"]:
        if lab_key not in rule["labs"] or (rule.get("direction") and direction and rule["direction"] != direction):
            continue
        earliest = (_date.fromisoformat(lab_date) - timedelta(days=rule["window_days"])).isoformat()
        for line in rec.prescribed():
            if rule["group"] not in line["groups"] or not line["start"]:
                continue
            route_ok = (line["route"] in SYSTEMIC_ROUTES if rule["route"] == "systemic"
                        else line["route"] == rule["route"] if rule["route"] else True)
            last = line["end"] or line["start"]
            if route_ok and line["start"] <= lab_date and last >= earliest:
                found.setdefault(line["name"].casefold(), {"name": line["name"], "effect": rule["effect"],
                                                           "window_days": rule["window_days"],
                                                           "last_taken": min(last, lab_date)})
    return list(found.values())


# ── Unmapped-brands report ──────────────────────────────────────────────────

_REPORT_LOCK = threading.Lock()


def report_path():
    """Where unmapped products are reported (UNMAPPED_BRANDS_REPORT); None when switched off ('off' / '')."""
    value = os.getenv("UNMAPPED_BRANDS_REPORT")
    if value is None:
        return _DIR / "logs" / "unmapped_brands.json"
    return None if value.strip().casefold() in ("", "off", "false", "0", "none") else Path(value)


def report_unmapped(rec):
    """Add this record's unmapped and model-only products to the report file. Product names only, no patient data.
    Never raises."""
    path = report_path()
    unmapped, model_only = rec.unmapped(), rec.model_only()
    if not path or not (unmapped or model_only):
        return
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with _REPORT_LOCK:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            data.setdefault("_doc", "Products the ingredient registry (data/reference/medicine_ingredients.json) "
                                    "does not map. 'unmapped': no ingredients found, so class rules cannot see them. "
                                    "'model_only': ingredients from the model's term resolution only; check and add "
                                    "them to the registry.")
            for section, items in (("unmapped", [part for _, part in unmapped]), ("model_only", model_only)):
                bucket = data.setdefault(section, {})
                for name in items:
                    entry = bucket.setdefault(base_key(name) or name.casefold(), {"examples": [], "times_seen": 0})
                    entry["times_seen"] += 1
                    entry["last_seen"] = now
                    if name not in entry["examples"] and len(entry["examples"]) < 5:
                        entry["examples"].append(name)
            data["updated"] = now
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
    except Exception as exc:  # reporting must never break the snapshot
        logger.warning("unmapped-brands report not written: %s", exc)
