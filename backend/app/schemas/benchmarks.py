"""Standardized evaluation contracts for Remote Sensing benchmarks.

Supports:
- VRSBench (VQA, captioning, grounding)
- RSVQA (High-resolution and low-resolution land-cover VQA)
- CDVQA (Change Detection VQA)
- BigEarthNet / BigEarthNet.txt (Multilabel classification and VQA formatting)
- ISRO/SAC evaluation format (Disaster, built-up, and water-body change monitoring)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkDataset(str, Enum):
    VRSBENCH = "vrsbench"
    RSVQA = "rsvqa"
    CDVQA = "cdvqa"
    BIGEARTHNET = "bigearthnet"
    ISRO_SAC = "isro_sac"


class BenchmarkTask(str, Enum):
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    OBJECT_COUNTING = "object_counting"
    CHANGE_DETECTION = "change_detection"
    MULTIMODAL_FUSION = "multimodal_fusion"


class BenchmarkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    dataset: BenchmarkDataset
    task: BenchmarkTask
    question: str
    image_paths: list[str]
    reference_answer: str | None = None
    reference_count: int | None = None
    reference_bboxes: list[list[float]] | None = None  # [minx, miny, maxx, maxy]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    dataset: BenchmarkDataset
    task: BenchmarkTask
    model_answer: str
    predicted_count: int | None = None
    predicted_bboxes: list[list[float]] | None = None
    confidence: float | None = None
    tool_or_model: str
    runtime_ms: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: BenchmarkDataset
    task: BenchmarkTask
    metric_name: str
    metric_value: float
    num_samples: int
    evaluation_metadata: dict[str, Any] = Field(default_factory=dict)
