"""Public entry point: only the routes shared with read-only keys (api.deps.READONLY_ROUTES), plus /health.

Hosted copy for teammates: uvicorn api.public_main:app
Saved AI reviews live in your own cache database (CACHE_DATABASE_URL), shared with your laptop.
Every other route (chat, FAQ, side effects, ai-assist, graph-sync, settings...) does not exist on this app.
"""
import os

# No outside services besides the patient database, the cache database and the AI model:
# no LangSmith traces, no Neo4j graph. Set before any project import.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["NEO4J_URI"] = ""

import api  # noqa: E402,F401  (sets sys.path before service imports)

from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402

from api.deps import READONLY_ROUTES  # noqa: E402
from api.routers import doctors, patients  # noqa: E402

app = FastAPI(title="Patient Intelligence API", version="1.0.0")

for _router in (patients.router, doctors.router):
    for _route in _router.routes:
        if isinstance(_route, APIRoute) and any((m, _route.path) in READONLY_ROUTES for m in _route.methods):
            app.router.routes.append(_route)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
