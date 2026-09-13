"""Pydantic contracts for surface area changes (built-up, water, vegetation)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import EvidenceRegion, Metric


class SurfaceDomainKind(str, Enum):
    BUILT_UP = "built_up"
    WATER = "water"
    VEGETATION = "vegetation"


class SurfaceAreaChangeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: SurfaceDomainKind
    earlier_image_id: str
    later_image_id: str
    before_area_m2: float = Field(ge=0.0)
    after_area_m2: float = Field(ge=0.0)
    difference_m2: float
    percentage_change: float
    primary_index: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_regions: list[EvidenceRegion] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
