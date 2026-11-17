"""FastAPI entry point. Run from the repo root: uvicorn api.main:app --reload"""
import api  # noqa: F401  (sets sys.path before service imports)

from fastapi import Depends, FastAPI

import core.config
from api.deps import require_api_key
from api.routers import doctors, patients
from services.graph_service import is_neo4j_configured

FLAG_NAMES = ["ENABLE_HARDENED_CONFIG", "ENABLE_EXTERNAL_PROMPTS", "ENABLE_PROMPT_INJECTION_SANDBOX",
              "ENABLE_ENCRYPTED_CACHE", "ENABLE_CHAT_OUTPUT_SANITIZER", "ENABLE_ANONYMIZED_AUDIT_LOGS",
              "ENABLE_STRICT_HTML_SANITIZATION"]

app = FastAPI(title="Patient Intelligence API", version="1.0.0")
app.include_router(patients.router)
app.include_router(doctors.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/settings", tags=["settings"], dependencies=[Depends(require_api_key)])
def settings():
    """Read-only feature flags and model configuration (for the Settings page)."""
    return {
        "flags": {name: core.config.flag(name) for name in FLAG_NAMES},
        "chat_model": core.config.CHAT_MODEL,
        "review_models": list(core.config.DEFAULT_MODELS),
        "neo4j_configured": is_neo4j_configured(),
    }
