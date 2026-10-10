"""Cache-first review service facade."""
from ai_review_service import get_or_generate_review, is_review_current

__all__ = ["get_or_generate_review", "is_review_current"]
