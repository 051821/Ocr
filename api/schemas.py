"""Request and response models for the API."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class PatientList(BaseModel):
    patient_ids: list[str]
    total: int


class ReviewResponse(BaseModel):
    patient_id: str
    raw_markdown: str
    model_name: str = ""
    model_version: str = ""
    generated_at: str = ""
    cache_hit: bool = False
    status: str = ""
    # {"insights": [{"theme", "points": [{"text", "areas"}]}], "labs": [{"text", "areas"}]}
    area_tags: dict = {}
    # AI wording of the quick tier {"headline", "lines": {flag id: text}, "model"}; None = rule-built quick tier.
    quick_ai: Optional[dict] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)


class Evidence(BaseModel):
    date: Optional[str] = None
    visit_id: Optional[str] = None
    visit_index: Optional[Any] = None
    category: Optional[str] = None
    source_id: Optional[str] = None
    content: str = ""


class ChatResponse(BaseModel):
    patient_id: str
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class FaqResponse(BaseModel):
    patient_id: str
    questions: list[Any]


class OcrJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    summary: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class DoctorSettings(BaseModel):
    doctor_id: str = ""
    # Which tier of the AI Analysis & Summary opens by default.
    default_view: Literal["quick", "detailed"] = "quick"
    saved: bool = False


class DoctorSettingsUpdate(BaseModel):
    default_view: Literal["quick", "detailed"]
