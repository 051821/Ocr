"""HTTP client for the Patient Intelligence API (used by the Streamlit app)."""
import os

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")

FAST_TIMEOUT = 60
LLM_TIMEOUT = 300

_session = requests.Session()
_session.headers.update({"X-API-Key": API_KEY})


class ApiError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _request(method, path, timeout=FAST_TIMEOUT, **kwargs):
    try:
        resp = _session.request(method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(0, f"API unreachable at {API_BASE_URL}: {exc}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(resp.status_code, detail)
    return resp.json()


def list_patients():
    return _request("GET", "/patients")["patient_ids"]


def get_patient(patient_id):
    return _request("GET", f"/patients/{patient_id}")


def refresh_patient(patient_id):
    return _request("POST", f"/patients/{patient_id}/refresh")


def get_snapshot(patient_id):
    """Returns {"snapshot": ..., "clinical_course": ..., "brief": rule-based review markdown}."""
    return _request("GET", f"/patients/{patient_id}/snapshot")


def get_insight_view(patient_id, view=None, doctor_id=None):
    """{"patient": ..., "ai_checks": [...], "view": "quick" | "detailed"}: data for the AI Analysis & Summary and Lab
    Results tabs (no LLM). view: ?view= override; doctor_id: whose saved default view applies."""
    params = {k: v for k, v in (("view", view), ("doctor_id", doctor_id)) if v}
    return _request("GET", f"/patients/{patient_id}/insight-view", params=params)


def list_doctors():
    return _request("GET", "/doctors")["doctors"]


def get_doctor_settings(doctor_id):
    """{"doctor_id", "default_view", "saved"}; default_view is 'quick' when the doctor saved none."""
    return _request("GET", f"/doctors/{doctor_id}/settings")


def set_doctor_default_view(doctor_id, default_view):
    return _request("PUT", f"/doctors/{doctor_id}/settings", json={"default_view": default_view})


def get_medication_journey(patient_id):
    return _request("GET", f"/patients/{patient_id}/medications")


def get_side_effects(patient_id, use_ai=False):
    timeout = LLM_TIMEOUT if use_ai else FAST_TIMEOUT
    return _request("GET", f"/patients/{patient_id}/side-effects", timeout=timeout, params={"use_ai": use_ai})


def ai_assist(patient_id):
    return _request("POST", f"/patients/{patient_id}/ai-assist", timeout=LLM_TIMEOUT)


def get_review(patient_id):
    """Cached current review, or None when none is current."""
    try:
        return _request("GET", f"/patients/{patient_id}/review")
    except ApiError as exc:
        if exc.status_code == 404:
            return None
        raise


def generate_review(patient_id, force_refresh=False):
    return _request("POST", f"/patients/{patient_id}/review", timeout=LLM_TIMEOUT,
                    params={"force_refresh": force_refresh})


def chat(patient_id, question, history=None):
    """Returns (answer, evidence, meta)."""
    history = [{"role": m["role"], "content": str(m.get("content") or "")}
               for m in (history or []) if m.get("role") in {"user", "assistant"}]
    data = _request("POST", f"/patients/{patient_id}/chat", timeout=LLM_TIMEOUT,
                    json={"question": question, "history": history})
    return data["answer"], data.get("evidence", []), data.get("meta", {})


def get_faq(patient_id, limit=5):
    return _request("GET", f"/patients/{patient_id}/faq", params={"limit": limit})["questions"]


def graph_sync(patient_id):
    """True when the graph synced; False on failure or when Neo4j is not configured."""
    try:
        _request("POST", f"/patients/{patient_id}/graph-sync", timeout=LLM_TIMEOUT)
        return True
    except ApiError:
        return False


def get_settings():
    return _request("GET", "/settings")
