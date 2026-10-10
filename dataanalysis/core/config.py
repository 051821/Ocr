"""Central configuration constants for the analytics application."""
import os

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = (
    "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-001",
    "inclusionai/ling-3.0-flash-sante:free",
)
MAX_TOKENS = {"review": 600, "chat": 400, "extraction": 300, "side_effects": 300}
try:
    from config import OPENROUTER_API_KEY as _CONFIG_API_KEY
except Exception:
    _CONFIG_API_KEY = ""

OPENROUTER_API_KEY = _CONFIG_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
