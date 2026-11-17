"""Optional LangSmith tracing for the RAG chat and AI review pipelines.

Tracing is active only when the ``langsmith`` package is installed and
``LANGSMITH_TRACING=true``; otherwise ``traced`` returns the function
unchanged, so nothing here can break a request. Large patient records are
reduced to a short summary before they are sent to LangSmith.
"""
import functools
import logging

logger = logging.getLogger(__name__)

try:
    from langsmith import traceable as _traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:  # langsmith is optional
    _traceable = None
    get_current_run_tree = None

_PATIENT_KEYS = ("patient_data", "data")


def _summarize_patient(value):
    if not isinstance(value, dict):
        return value
    visits = value.get("visits") or []
    return {
        "patient_id": value.get("patient_id"),
        "visit_count": len(visits) if isinstance(visits, list) else None,
        "keys": sorted(k for k in value if not str(k).startswith("_"))[:40],
    }


def _process_inputs(inputs):
    try:
        out = dict(inputs)
        for key in _PATIENT_KEYS:
            if key in out:
                out[key] = _summarize_patient(out[key])
        for key in ("api_key", "key"):
            if key in out:
                out[key] = "***"
        return out
    except Exception:
        return {}


def traced(name=None, run_type="chain", **kwargs):
    """Decorate a function as a LangSmith run; a no-op when tracing is unavailable."""
    def decorator(fn):
        if _traceable is None:
            return fn
        try:
            return _traceable(name=name or fn.__name__, run_type=run_type,
                              process_inputs=kwargs.pop("process_inputs", _process_inputs), **kwargs)(fn)
        except Exception as exc:
            logger.warning("LangSmith tracing disabled for %s: %s", fn.__name__, exc)
            return fn
    return decorator


def annotate_run(metadata=None, tags=None):
    """Attach metadata/tags (patient id, model, cache hit ...) to the current run."""
    if get_current_run_tree is None:
        return
    try:
        run = get_current_run_tree()
        if run is None:
            return
        if metadata:
            run.metadata.update({k: v for k, v in metadata.items() if v is not None})
        if tags:
            run.tags = list(dict.fromkeys((run.tags or []) + list(tags)))
    except Exception as exc:
        logger.debug("LangSmith annotate failed: %s", exc)


def patient_context_outputs(result):
    """Summarize a ``(patient_data, chat_index)`` pair instead of uploading the whole record."""
    try:
        patient_data, chat_index = result
        return {"patient": _summarize_patient(patient_data),
                "index_entries": len(chat_index) if isinstance(chat_index, list) else None}
    except Exception:
        return {}


def llm_outputs(result):
    """Shape a ``core.llm_client.complete`` result so LangSmith shows tokens and cost."""
    try:
        # A cache hit costs nothing, so it is reported with zero usage.
        usage = {} if result.get("cache_hit") else (result.get("usage") or {})
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        usage_metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        }
        # OpenRouter reports the real billed USD cost when usage accounting is requested.
        cost = usage.get("cost")
        if cost is not None:
            usage_metadata["total_cost"] = float(cost)
        return {
            "content": result.get("content"),
            "model": result.get("model"),
            "cache_hit": result.get("cache_hit"),
            "truncated": result.get("truncated"),
            "usage_metadata": usage_metadata,
        }
    except Exception:
        return {"output": result}
