"""Shared FastAPI dependencies: API-key auth and patient context loading."""
import hmac
import os

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from services.patient_context import load_patient_context

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
# The only routes read-only keys may call: IDs and analysis, never documents, the full
# record (raw OCR text) or free-text chat questions. New routes stay closed until added here.
READONLY_ROUTES = {
    ("GET", "/patients"),
    ("GET", "/patients/{patient_id}/summary"),
    ("GET", "/patients/{patient_id}/demographics"),
    ("GET", "/patients/{patient_id}/snapshot"),
    ("GET", "/patients/{patient_id}/patient-type"),
    ("GET", "/patients/{patient_id}/long-term-conditions"),
    ("GET", "/patients/{patient_id}/medications"),
    ("GET", "/patients/{patient_id}/timeline"),
    ("GET", "/doctors"),
    ("GET", "/doctors/{doctor_id}/patients"),
}


def _keys(var):
    return [k.strip() for k in os.getenv(var, "").split(",") if k.strip()]


def _matches(api_key, keys):
    return any(hmac.compare_digest(api_key, k) for k in keys)


def require_api_key(request: Request, api_key: str = Security(_api_key_header)):
    """Check X-API-Key against API_KEY (full access) or READONLY_API_KEYS (READONLY_ROUTES only).

    Both are comma-separated lists in .env, so each consumer can get its own key
    and be revoked without touching the others.
    """
    full_keys, readonly_keys = _keys("API_KEY"), _keys("READONLY_API_KEYS")
    if not full_keys and not readonly_keys:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API_KEY is not configured on the server.")
    if not api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key.")
    if _matches(api_key, full_keys):
        return
    if _matches(api_key, readonly_keys):
        route = getattr(request.scope.get("route"), "path", "")
        if (request.method, route) not in READONLY_ROUTES:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "This API key cannot access this endpoint.")
        return
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key.")


def get_patient_context(patient_id: str, _: None = Depends(require_api_key)):
    """Return (patient_data, chat_index); 404 when the patient has no encounters."""
    try:
        patient_data, chat_index = load_patient_context(patient_id)
    except Exception as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Could not load patient record: {exc}")
    if not patient_data or patient_data.get("total_visits", 0) == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No clinical encounters found for this patient.")
    return patient_data, chat_index


def public_view(patient_data):
    """Copy of patient_data without internal underscore-prefixed keys."""
    return {k: v for k, v in patient_data.items() if not str(k).startswith("_")}
