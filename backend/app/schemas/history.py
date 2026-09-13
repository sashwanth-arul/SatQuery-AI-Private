from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict


class AnalysisHistoryItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str
    created_at: str
    query: str
    intent: str | None = None
    mode: str | None = None
    status: str
    summary_answer: str | None = None
    confidence: float | None = None
    metrics_summary: dict[str, Any] | None = None
    input_summary: dict[str, Any] | None = None


class AnalysisHistoryResponse(BaseModel):
    items: list[AnalysisHistoryItem]
    total: int
    limit: int
    offset: int
