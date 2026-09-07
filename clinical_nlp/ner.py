"""
Optional generic biomedical NER layer.

This is deliberately NOT relied on for labs/medications (see labs.py /
medications.py docstrings — dedicated extraction logic is used there
instead). It's provided as a supplementary signal, e.g. for catching
anatomy/procedure mentions the rule-based extractors don't target, and as
the extension point for a stronger model later.

Loading transformers + a HF model is optional and lazy: if the package or
model isn't available (e.g. no internet access, or you haven't installed
`transformers`/`torch`), calls to run_ner() simply return an empty list
instead of crashing the pipeline.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

_ner_pipeline = None
_load_attempted = False


def _get_pipeline():
    global _ner_pipeline, _load_attempted
    if _ner_pipeline is not None or _load_attempted:
        return _ner_pipeline

    _load_attempted = True
    try:
        from transformers import pipeline
        _ner_pipeline = pipeline(
            "token-classification",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
        )
        logger.info("Loaded biomedical NER model d4data/biomedical-ner-all")
    except Exception as e:  # noqa: BLE001 - intentionally broad, this is optional
        logger.warning("Biomedical NER model unavailable (%s). "
                        "Falling back to rule-based extraction only.", e)
        _ner_pipeline = None

    return _ner_pipeline


def run_ner(text: str):
    """
    Returns a list of {entity_group, word, start, end, score} dicts, or
    [] if the model isn't available. Never raises.
    """
    if not text:
        return []
    nlp = _get_pipeline()
    if nlp is None:
        return []
    try:
        return nlp(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("NER inference failed: %s", e)
        return []
