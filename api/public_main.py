"""Public entry point: only the routes shared with read-only keys (api.deps.READONLY_ROUTES), plus /health.

Hosted copy for teammates: uvicorn api.public_main:app
Rule-based only: no AI model calls, no saved-analysis file. Every response is computed fresh from the
patient database. Every other route (review, chat, FAQ, side effects, ai-assist, graph-sync, settings...)
does not exist on this app.
"""
import os

# Rules only, set before any project import: an empty key (load_dotenv never overrides an existing
# variable, even an empty one) and dynamic analysis off mean nothing on this app can call the AI model.
os.environ["RULES_ONLY"] = "true"
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["ENABLE_DYNAMIC_ANALYSIS"] = "false"
# No outside services either: no LangSmith traces, no Neo4j graph. Only the patient database is contacted.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["NEO4J_URI"] = ""

import api  # noqa: E402,F401  (sets sys.path before service imports)

from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute

from api.deps import READONLY_ROUTES, get_patient_context
from api.routers import doctors, patients

app = FastAPI(title="Patient Intelligence API", version="1.0.0")

_OWN = {("GET", "/patients/{patient_id}/summary")}  # defined below without the AI field

for _router in (patients.router, doctors.router):
    for _route in _router.routes:
        if isinstance(_route, APIRoute) and any(
                (m, _route.path) in READONLY_ROUTES - _OWN for m in _route.methods):
            app.router.routes.append(_route)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/patients/{patient_id}/summary", tags=["patients"])
def get_summary(patient_id: str, ctx=Depends(get_patient_context)):
    """Everything for one patient in one call, rule-based (no ai_review)."""
    return patients.build_summary(patient_id, ctx[0], include_ai=False)
