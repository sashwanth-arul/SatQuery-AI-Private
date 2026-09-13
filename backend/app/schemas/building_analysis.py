"""Pydantic contracts for building detection and temporal footprint matching."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import GeoJSONGeometry, Metric


class BuildingStatus(str, Enum):
    DETECTED = "detected"
    NEW = "new"
    REMOVED = "removed"
    UNCHANGED = "unchanged"
    SIGNIFICANTLY_CHANGED = "significantly_changed"


class BuildingFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(description="[minx, miny, maxx, maxy] in EPSG:4326")
    geometry: GeoJSONGeometry
    area_m2: float = Field(ge=0.0)
    status: BuildingStatus = BuildingStatus.DETECTED
    iou_with_match: float | None = None
    matched_id: str | None = None


class BuildingDetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    count: int = Field(ge=0)
    detections: list[BuildingFootprint]
    detector: str
    detector_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    total_area_m2: float = 0.0
    metrics: list[Metric] = Field(default_factory=list)
    detector_metadata: dict[str, Any] = Field(default_factory=dict)


class BuildingTemporalMatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    earlier_image_id: str
    later_image_id: str
    before_count: int = Field(ge=0)
    after_count: int = Field(ge=0)
    new_count: int = Field(ge=0)
    removed_count: int = Field(ge=0)
    unchanged_count: int = Field(ge=0)
    changed_count: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    matcher_name: str
    matched_footprints: list[BuildingFootprint]
    metrics: list[Metric] = Field(default_factory=list)
