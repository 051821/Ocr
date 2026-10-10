"""Shared, cached OpenRouter client for every application LLM request."""
import hashlib
import json
import logging
import time

import requests
from core.config import DEFAULT_MODELS, MAX_TOKENS, OPENROUTER_API_KEY, OPENROUTER_URL

try:
    from storage.ai_analysis_csv import get_llm_cache, set_llm_cache
except ImportError:
    from ..storage.ai_analysis_csv import get_llm_cache, set_llm_cache


logger = logging.getLogger(__name__)


def _cache_key(task, messages, models, temperature, max_tokens):
    payload = json.dumps(
        {"task": task, "messages": messages, "models": models,
         "temperature": temperature, "max_tokens": max_tokens},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return "llm:" + hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def complete(messages, task, api_key=None, model=None, temperature=0.1, max_tokens=None, cache=True):
    """Return a cached or generated response with model and usage metadata."""
    key = api_key if api_key is not None else OPENROUTER_API_KEY
    if not key:
        raise RuntimeError("OpenRouter API key is not configured")
    temperature = min(float(temperature), 0.2)
    max_tokens = int(max_tokens or MAX_TOKENS[task])
    models = (model,) if model else DEFAULT_MODELS
    cache_key = _cache_key(task, messages, models, temperature, max_tokens)
    if cache:
        try:
            cached = get_llm_cache(cache_key)
            if cached:
                result = json.loads(cached)
                result["cache_hit"] = True
                return result
        except Exception as exc:
            logger.warning("LLM cache read failed: %s", exc)

    last_error = None
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for candidate in models:
        for attempt, backoff in enumerate((0, 1, 2)):
            if backoff:
                time.sleep(backoff)
            try:
                response = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json={"model": candidate, "messages": messages,
                          "temperature": temperature, "max_tokens": max_tokens},
                    timeout=60,
                )
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices") or []
                content = choices[0].get("message", {}).get("content") if choices else None
                if not content:
                    raise ValueError("LLM response did not contain message content")
                usage = body.get("usage") or {}
                logger.info(
                    "llm task=%s model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    task, candidate, usage.get("prompt_tokens"),
                    usage.get("completion_tokens"), usage.get("total_tokens"),
                )
                result = {"content": content, "model": candidate, "usage": usage, "cache_hit": False}
                if cache:
                    try:
                        set_llm_cache(cache_key, json.dumps(result, ensure_ascii=False))
                    except Exception as exc:
                        logger.warning("LLM cache write failed: %s", exc)
                return result
            except Exception as exc:
                logger.warning("LLM task=%s model=%s attempt=%d failed: %s", task, candidate, attempt + 1, exc)
                last_error = exc
    raise RuntimeError(f"LLM request failed for task {task}: {last_error}")
