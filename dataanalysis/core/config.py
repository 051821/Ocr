"""Central configuration constants for the analytics application."""
import os


def flag(name: str, default: bool = False) -> bool:
    """Read a boolean configuration flag from environment variables."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Canonical chat model
CHAT_MODEL = "google/gemini-2.5-flash"

_RAW_DEFAULT_MODELS = (
    "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-001",
    "inclusionai/ling-3.0-flash-sante:free",
)

if flag("ENABLE_HARDENED_CONFIG"):
    DEFAULT_MODELS = tuple(m for m in _RAW_DEFAULT_MODELS if not m.endswith(":free"))
else:
    DEFAULT_MODELS = _RAW_DEFAULT_MODELS

# Approved models allowlist (remaining defaults + canonical chat model)
APPROVED_MODELS = tuple(dict.fromkeys(
    tuple(m for m in _RAW_DEFAULT_MODELS if not m.endswith(":free")) + (CHAT_MODEL,)
))

MAX_TOKENS = {"review": 600, "chat": 400, "extraction": 300, "side_effects": 300}

if flag("ENABLE_HARDENED_CONFIG"):
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
else:
    try:
        from config import OPENROUTER_API_KEY as _CONFIG_API_KEY
    except Exception:
        _CONFIG_API_KEY = ""
    OPENROUTER_API_KEY = _CONFIG_API_KEY or os.getenv("OPENROUTER_API_KEY", "")


def get_openrouter_api_key(explicit_key: str = None) -> str:
    """Resolve the active OpenRouter API key, enforcing strict checks if hardened."""
    key = explicit_key if explicit_key is not None else (
        os.getenv("OPENROUTER_API_KEY", "") if flag("ENABLE_HARDENED_CONFIG") else OPENROUTER_API_KEY
    )
    if flag("ENABLE_HARDENED_CONFIG") and not key:
        raise RuntimeError("OpenRouter API key is not configured in environment (OPENROUTER_API_KEY).")
    return key
