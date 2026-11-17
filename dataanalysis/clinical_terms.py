"""Dynamic resolution of the record's own terms: medicines, diagnosis entries and complaint phrases.

The rule layer (clinical_snapshot, clinical_insights) reasons in a fixed vocabulary: registry medicine classes,
registry conditions and symptom tags. Records use free text no list can cover (brands, abbreviations,
misspellings, run-together phrases). This module asks the model once per unseen term to place it in that
vocabulary, validates the answer against the vocabulary, and caches it per term, so each new brand or
abbreviation is resolved once for every patient. Doses, limits and clinical rules stay in the registry: the
model only says what a term *is*, never what to do.

Resolved terms are attached to patient_data["_terms"]; every consumer falls back to the registry patterns when
a term is unresolved (offline, disabled, or the model declined).
"""
import hashlib
import json
import logging
import re

from core.config import CHAT_MODEL, OPENROUTER_API_KEY, flag
from patient_chat import REGISTRY, _med_display, long_term_labels

logger = logging.getLogger(__name__)

TERMS_VERSION = "terms_v3"
_BATCH = 30

CLASS_VOCAB = [e["class"] for e in REGISTRY["medicine_classes"]]
CONDITION_VOCAB = [c["name"] for c in REGISTRY["condition_map"]]
SYMPTOM_VOCAB = {
    "chest_pain": "chest pain, tightness or discomfort",
    "breathlessness": "breathlessness or breathing difficulty",
    "palpitations": "palpitations",
    "syncope": "fainting or blackout",
    "dizziness": "dizziness, giddiness or light-headedness",
    "weakness_fatigue": "weakness or tiredness",
    "headache": "headache",
    "fever": "fever or chills",
    "cough_cold": "cough, cold or runny nose",
    "sore_throat": "sore throat",
    "foot_or_leg_wound": "wound, ulcer, injury, cut, blister or infection on a foot, toe or leg",
    "neuropathic_sensory": "numbness, tingling, pins and needles or burning in hands, feet or limbs",
    "visual_change": "blurred or changed vision",
    "leg_swelling": "swelling of legs, ankles or feet",
    "itching_rash": "itching, rash or skin lesion",
    "musculoskeletal_pain": "pain in back, neck, joints, muscles or limbs, body ache",
    "abdominal_pain": "abdominal pain",
    "acidity_heartburn": "acidity, heartburn or burning in chest after food",
    "vomiting_diarrhoea": "vomiting, nausea or loose motions",
    "urinary_symptoms": "burning or frequent urination",
    "thirst_polyuria": "excessive thirst or passing much urine",
    "weight_loss": "weight loss",
    "bleeding": "bleeding",
    "mood_sleep": "low mood, anxiety, mood swings or poor sleep",
    "ear_hearing": "ear pain or hearing loss",
}
ROUTES = ("oral", "topical", "inhaled", "injection", "drops", "other")

_SYSTEM = """You map clinical record terms to a fixed vocabulary. Answer only with JSON. Use general medical knowledge
of what a term is; never guess: when you are not sure, give the empty / null value.

MEDICINE CLASSES: {classes}
CONDITIONS: {conditions}
SYMPTOM TAGS: {symptoms}

For each item in "medicines" (as written on a prescription, often an Indian brand, abbreviation or misspelling):
  {{"confidence": "high" only when you know this exact product's ingredients, else "low",
    "is_medicine": false when it is advice, a test or an instruction ("Salt restricted diet", "Review with report"),
    "generic": [active ingredients, lower case],
    "classes": [one or more MEDICINE CLASSES, exactly as written; "Other" when none fits - never the nearest
                wrong class],
    "route": one of {routes},
    "combination": true when it has more than one active ingredient,
    "strength_mg": strength in mg of a single-ingredient product written in the name, else null,
    "treats": [CONDITIONS this medicine is a specific treatment for; [] for symptomatic medicines such as
               painkillers, antacids, vitamins, antihistamines]}}
For each item in "diagnoses" (a diagnosis field, possibly several diagnoses, abbreviations or typos):
  {{"is_diagnosis": false when it names no condition ("No trauma", "Follow up", "Review"),
    "parts": [{{"text": the part as written, "condition": the CONDITION this part is (or a specific form of),
               else null - an acute infection or any condition not in the list is null, never the nearest one,
               "meaning": the plain full name when the text is an abbreviation or misspelling and its meaning is
                          unambiguous, else null}}]}}
For each item in "complaints" (a symptom phrase):
  {{"symptoms": [SYMPTOM TAGS the words of the phrase name; a reading ("BP 150/100"), a diagnosis, a test or a
                visit reason is not a symptom; [] when unsure],
    "duration_days": how long the phrase says it has lasted, in days, else null}}

Return {{"medicines": {{item: ...}}, "diagnoses": {{item: ...}}, "complaints": {{item: ...}}}} with every item as
its key, exactly as given."""


def enabled():
    """Dynamic resolution runs when switched on (default) and the model is reachable."""
    return flag("ENABLE_DYNAMIC_ANALYSIS", True) and bool(OPENROUTER_API_KEY)


def term_key(text):
    return " ".join(str(text or "").casefold().split())


def _cache_key(kind, term):
    digest = hashlib.blake2b(f"{TERMS_VERSION}|{kind}|{term_key(term)}".encode("utf-8"), digest_size=16).hexdigest()
    return f"terms:{digest}"


# ── Validation: only vocabulary values survive ─────────────────────────────

def _strings(value):
    return [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


_DOSE_FORM = re.compile(r"(?i)\d\s*(?:mg|ml|mcg|g|iu|%)\b|\b(?:tab|tablet|cap|capsule|syp|syr|syrup|inj|cream|gel|"
                        r"lotion|oint|ointment|drops?|susp|suspension|sachet|spray|neb)\b")
_NOT_A_DIAGNOSIS = re.compile(r"(?i)^\s*(?:no|nil|not|none|follow|review|report|check|refer|repeat|routine|ok|"
                              r"normal|nad|wnl|na\b|n/a|under\s+(?:evaluation|observation|investigation)|unknown|-)")
_LIQUID = re.compile(r"(?i)\d\s*ml\b|\b(?:syp|syr|syrup|susp|suspension|drops?)\b")
_CONDITION_MEDS = REGISTRY["condition_meds"]
_CLASS_TREATS = REGISTRY["class_treats"]
# Words a condition name shares with unrelated ones; they do not show that a meaning names the condition.
_GENERIC_WORDS = {"tract", "infection", "disease", "disorder", "condition", "pain", "chronic", "acute", "syndrome"}


def _treats(condition, generic, classes, term):
    """A model 'treats' answer is kept only when the registry agrees: the condition's own medicine list names an
    ingredient, or the medicine's class is that condition's treatment class. Conditions the registry has no
    medicine list for take the model's answer."""
    pattern = _CONDITION_MEDS.get(condition)
    if _CLASS_TREATS.get(condition) in classes:
        return True
    if not pattern:
        return True
    return any(re.search(rf"\b(?:{pattern})", text, re.I) for text in generic + [term])


def _registry_classes(term):
    return list(dict.fromkeys(e["class"] for e in REGISTRY["medicine_classes"]
                              if re.search(e["pattern"], term, re.I)))[:3]


def _clean_medicine(raw, term=""):
    if not isinstance(raw, dict):
        return None
    route = str(raw.get("route") or "").casefold()
    known = str(raw.get("confidence") or "").casefold() == "high"
    # An unsure answer about ingredients is no answer: unknown is safer than a wrong class.
    generic = [g.casefold() for g in _strings(raw.get("generic"))][:4] if known else []
    classes = [c for c in _strings(raw.get("classes")) if c in CLASS_VOCAB and c != "Other"][:3] if known else []
    # A name the registry recognises keeps the registry's (curated) class.
    classes = _registry_classes(term) or classes
    return {
        # Something written with a dose or a dose form is a medicine whatever the model says ("MVT 5ml").
        "is_medicine": raw.get("is_medicine") is not False or bool(_DOSE_FORM.search(term)),
        "generic": generic,
        "classes": classes,
        "route": route if route in ROUTES else "other",
        "combination": bool(raw.get("combination")) or len(generic) > 1,
        # A liquid's number is a volume or a per-5-ml amount, never a tablet strength.
        "strength_mg": None if _LIQUID.search(term) or not known else _number(raw.get("strength_mg")),
        "treats": [c for c in _strings(raw.get("treats"))
                   if c in CONDITION_VOCAB and _treats(c, generic, classes, term)][:3] if known else [],
    }


def _registry_condition(text):
    from patient_chat import conditions_matching
    found = conditions_matching(text)
    return found[0] if len(found) == 1 else None


def _checked_condition(text, condition, meaning):
    """The registry's own mapping wins; otherwise the model's, unless its meaning names something else
    ("URTI" meaning upper respiratory tract infection is not "Urinary tract infection")."""
    registry = _registry_condition(text) or (_registry_condition(meaning) if meaning else None)
    if registry:
        return registry
    if condition not in CONDITION_VOCAB:
        return None
    if meaning:
        words = lambda s: {w for w in re.findall(r"[a-z]{4,}", s.casefold())} - _GENERIC_WORDS
        if not words(condition) & words(meaning):
            return None
    return condition


def _clean_diagnosis(raw, term=""):
    if not isinstance(raw, dict):
        return None
    parts = []
    for part in raw.get("parts") or []:
        if not isinstance(part, dict) or not str(part.get("text") or "").strip():
            continue
        text = str(part["text"]).strip()
        meaning = str(part.get("meaning") or "").strip()[:60] or None
        parts.append({"text": text, "condition": _checked_condition(text, str(part.get("condition") or "").strip(), meaning),
                      "meaning": meaning})
    # "Not a diagnosis" is accepted only for negations and visit reasons ("No trauma", "Review"); a recorded
    # symptom-level diagnosis ("Fever") stays.
    if raw.get("is_diagnosis") is False and not _NOT_A_DIAGNOSIS.match(term):
        parts = parts or [{"text": term.strip(), "condition": _registry_condition(term), "meaning": None}]
        return {"is_diagnosis": True, "parts": parts[:6]}
    return {"is_diagnosis": raw.get("is_diagnosis") is not False and bool(parts), "parts": parts[:6]}


def _clean_complaint(raw, term=""):
    if not isinstance(raw, dict):
        return None
    days = _number(raw.get("duration_days"))
    return {"symptoms": [s for s in _strings(raw.get("symptoms")) if s in SYMPTOM_VOCAB][:5],
            "duration_days": int(days) if days else None}


_CLEAN = {"medicines": _clean_medicine, "diagnoses": _clean_diagnosis, "complaints": _clean_complaint}


# ── Cache and model ─────────────────────────────────────────────────────────

def _cached(kind, term):
    from storage.ai_analysis_csv import get_llm_cache
    try:
        value = get_llm_cache(_cache_key(kind, term))
        return json.loads(value) if value else None
    except Exception as exc:
        logger.warning("terms cache read failed: %s", exc)
        return None


def _store(kind, term, value, patient_id):
    from storage.ai_analysis_csv import set_llm_cache
    try:
        set_llm_cache(_cache_key(kind, term), json.dumps(value, ensure_ascii=False), patient_id=patient_id)
    except Exception as exc:
        logger.warning("terms cache write failed: %s", exc)


def _ask_model(batch):
    from core.llm_client import complete
    system = _SYSTEM.format(classes=json.dumps(CLASS_VOCAB), conditions=json.dumps(CONDITION_VOCAB),
                            symptoms=json.dumps(SYMPTOM_VOCAB), routes=json.dumps(ROUTES))
    # The chat model: broader drug knowledge (Indian brands) than the small review models.
    reply = complete([{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(batch, ensure_ascii=False)}],
                     "terms", model=CHAT_MODEL, temperature=0, cache=False)
    text = re.sub(r"^```(?:json)?|```$", "", str(reply.get("content") or "").strip(), flags=re.M)
    match = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(match.group(0)) if match else {}


def complaint_parts(visits):
    """[(encounter, part)]: each chief complaint split at commas / semicolons / line breaks, as written."""
    parts = []
    for idx, v in enumerate(visits, 1):
        for part in re.split(r"[,;\n]", str(v.get("db_chief_complaint") or v.get("chief_complaint") or "")):
            part = " ".join(part.strip(" .-:").split())
            if len(part) >= 3:
                parts.append((idx, part[:120]))
    return parts


def _terms_of(patient_data):
    visits = patient_data.get("visits") or []
    medicines = {_med_display(m.get("name")) for v in visits for m in v.get("db_medications") or [] if m.get("name")}
    diagnoses = {str(v.get(f) or "").strip() for v in visits
                 for f in ("db_provisional_dx", "db_confirmed_dx", "provisional_diagnosis", "confirmed_diagnosis")}
    diagnoses |= {label for label, _ in long_term_labels(patient_data)}
    complaints = {term_key(part): part for _, part in complaint_parts(visits)}
    return {"medicines": sorted(m for m in medicines if m), "diagnoses": sorted(d for d in diagnoses if d),
            "complaints": sorted(complaints.values())}


def resolve_terms(patient_data):
    """Attach patient_data["_terms"] = {kind: {term_key: resolved}}; cached terms cost nothing, unseen ones
    are resolved in a few batched calls. Never raises: unresolved terms fall back to the registry patterns."""
    resolved = {"medicines": {}, "diagnoses": {}, "complaints": {}}
    patient_data["_terms"] = resolved
    if not enabled():
        return resolved
    missing = {"medicines": [], "diagnoses": [], "complaints": []}
    for kind, terms in _terms_of(patient_data).items():
        for term in terms:
            hit = _cached(kind, term)
            if hit is not None:
                resolved[kind][term_key(term)] = hit
            else:
                missing[kind].append(term)
    queue = [(kind, term) for kind, terms in missing.items() for term in terms]
    for start in range(0, len(queue), _BATCH):
        chunk = queue[start:start + _BATCH]
        batch = {kind: [t for k, t in chunk if k == kind] for kind in resolved}
        try:
            answer = _ask_model({k: v for k, v in batch.items() if v})
        except Exception as exc:
            logger.warning("term resolution failed; registry patterns will be used: %s", exc)
            continue
        for kind, term in chunk:
            raw = (answer.get(kind) or {}).get(term)
            value = _CLEAN[kind](raw, term) if raw is not None else None
            if value is not None:
                resolved[kind][term_key(term)] = value
                _store(kind, term, value, patient_data.get("patient_id"))
    return resolved


# ── Lookups used by the rules ───────────────────────────────────────────────

def medicine(patient_data, name):
    return ((patient_data or {}).get("_terms") or {}).get("medicines", {}).get(term_key(_med_display(name)))


def diagnosis(patient_data, text):
    return ((patient_data or {}).get("_terms") or {}).get("diagnoses", {}).get(term_key(text))


def complaint(patient_data, phrase):
    return ((patient_data or {}).get("_terms") or {}).get("complaints", {}).get(term_key(phrase))
