"""Specialist tool for object detection and exact building counting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.building.base import BuildingDetector
from app.adapters.building.development import DevelopmentBuildingDetector
from app.schemas.building_analysis import BuildingDetectionResult
from app.schemas.domain import EvidenceRegion, Metric
from app.schemas.input import ImageInput


class BuildingCountTool:
    name = "detect_buildings"
    description = "Detect structural footprints and return exact building count and georeferenced polygons."

    def __init__(self, detector: BuildingDetector | None = None) -> None:
        self._detector = detector or DevelopmentBuildingDetector()

    async def execute(
        self,
        image: ImageInput,
        raster_path: Path,
        *,
        min_area_m2: float = 25.0,
    ) -> tuple[BuildingDetectionResult, list[EvidenceRegion]]:
        result = await self._detector.detect(image, raster_path, min_area_m2=min_area_m2)

        evidence_regions: list[EvidenceRegion] = []
        for idx, det in enumerate(result.detections, 1):
            region = EvidenceRegion(
                id=f"building-evidence-{idx:04d}",
                geometry=det.geometry,
                type="building_footprint",
                confidence=det.confidence,
                metrics=[
                    Metric(name="area_m2", value=det.area_m2, unit="m²", source="building_detector"),
                    Metric(name="building_index", value=idx, unit=None, source="building_detector"),
                ],
                source="building_detector",
                metadata={
                    "building_id": det.id,
                    "bbox": det.bbox,
                    "status": det.status.value,
                    "confidence": det.confidence,
                },
            )
            evidence_regions.append(region)

        return result, evidence_regions
