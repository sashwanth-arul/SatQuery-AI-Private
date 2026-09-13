"""Common geometric and metric domain schemas (leaf module to avoid circular dependencies)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GeoJSONGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["Polygon", "MultiPolygon", "Point", "LineString"]
    coordinates: list[Any]

    @field_validator("coordinates")
    @classmethod
    def coordinates_not_empty(cls, v: list[Any]) -> list[Any]:
        if not v:
            raise ValueError("geometry coordinates must not be empty")
        return v


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | int | str
    unit: str | None = None
    source: str


class EvidenceRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    geometry: GeoJSONGeometry
    type: str = "change"
    confidence: float = Field(ge=0, le=1)
    metrics: list[Metric] = Field(default_factory=list)
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
