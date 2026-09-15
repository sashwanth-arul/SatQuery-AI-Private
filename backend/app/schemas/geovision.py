"""Domain contracts for GeoVision workspace (Phase 1).

Completely isolated contracts for aerial/drone/remote-sensing object detection,
grounded visual question answering, and multi-turn image reasoning.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GeoVisionIntent(str, Enum):
    AUTOMATIC = "automatic"
    SINGLE_IMAGE_DESCRIPTION = "single_image_description"
    VISUAL_QA = "visual_qa"
    OBJECT_COUNT = "object_count"
    OBJECT_DETECTION = "object_detection"
    SCENE_CLASSIFICATION = "scene_classification"
    REGION_CAPTION = "region_caption"
    REFERRING_EXPRESSION = "referring_expression"
    GROUNDED_DESCRIPTION = "grounded_description"
    DETAILED_DESCRIPTION = "detailed_description"
    COMPLEX_REASONING = "complex_reasoning"
    MULTI_TURN_CONVERSATION = "multi_turn_conversation"
    OBJECT_ATTRIBUTE = "object_attribute"
    OBJECT_RELATIONSHIP = "object_relationship"


class DetectedObject(BaseModel):
    """Grounded detection contract."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Unique object identifier within the image")
    class_name: str = Field(description="Categorical label (e.g. building, car, tree, road)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from trained detector")
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] in pixel coordinates",
    )
    segmentation_mask: list[list[float]] | None = Field(
        default=None,
        description="Optional polygon contour coordinates [[x, y], ...]",
    )
    center: list[float] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Center point [x, y]",
    )
    area_px: float | None = Field(default=None, ge=0.0, description="Area in pixel units")
    area_m2: float | None = Field(default=None, ge=0.0, description="Area in square meters if georeferenced")
    evidence_type: str = Field(default="bounding_box", description="Evidence type: bounding_box, polygon_mask, or linear_feature")
    size_class: str | None = Field(default=None, description="Size classification: Large, Medium, or Small")
    geographic_coordinates: list[list[float]] | None = Field(default=None, description="Geographic boundary coordinates if georeferenced")
    score_type: str = Field(default="confidence", description="Score category: 'confidence' for ML model, 'heuristic_score' for deterministic CV")
    details: str | None = Field(default=None, description="Descriptive detail or constraint note")
    properties: dict[str, Any] = Field(default_factory=dict, description="Feature attributes such as coverage_percent, density, length_px")
    source_model: str = Field(description="Name of the model/provider that produced this detection")
    model_version: str = Field(default="v1.2.0-aerial-finetuned", description="Trained model version identifier")
    tile_id: str | None = Field(default=None, description="Tile ID where detection originated during sliding-window inference")
    coordinate_space: str = Field(default="image_pixel_original", description="Target coordinate reference frame")


class GeoVisionTraceStep(BaseModel):
    """Step in the 12-phase execution audit trace."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    name: str
    status: str  # "completed" | "running" | "skipped" | "unavailable" | "failed"
    duration_ms: int = 0
    model_or_provider: str = "system"
    details: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class GeoVisionUploadResponse(BaseModel):
    """Response returned upon uploading an aerial/drone image."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    filename: str
    format: str
    width: int
    height: int
    file_size_bytes: int
    georeferenced: bool = False
    crs: str | None = None
    bounds: list[float] | None = None
    preview_url: str


class GeoVisionAnalyzeRequest(BaseModel):
    """Analysis query for an uploaded GeoVision image."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    query: str = Field(min_length=1, max_length=2000)
    intent: GeoVisionIntent = GeoVisionIntent.AUTOMATIC
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)


class GeoVisionAnalyzeResponse(BaseModel):
    """Structured response from GeoVision analysis pipeline."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    image_id: str
    detected_intent: GeoVisionIntent
    answer: str
    is_one_word: bool = False
    confidence: float | None = None
    confidence_available: bool = False
    score_type: str = Field(default="confidence", description="'confidence' for ML models, 'heuristic_score' for deterministic CV")
    heuristic_score: float | None = Field(default=None, description="Explicit heuristic score for deterministic CV")
    coverage_percentage: float | None = Field(default=None, description="Calculated area feature coverage percentage")
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    object_summary: dict[str, int] = Field(default_factory=dict)
    trace: list[GeoVisionTraceStep] = Field(default_factory=list)
    models_used: dict[str, str] = Field(default_factory=dict)
    provider_status: dict[str, str] = Field(
        default_factory=lambda: {
            "geochat": "not_connected",
            "yolo_detector": "not_connected",
            "sam2_segmenter": "not_connected",
            "controlled_demo_cv": "idle",
        }
    )
    georeferenced: bool = False
    crs: str | None = None
    model_metadata: dict[str, Any] | None = None
    evaluation_metrics: dict[str, Any] | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
