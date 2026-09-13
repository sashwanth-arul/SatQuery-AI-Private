"""Abstract protocol for building detection adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.schemas.building_analysis import BuildingDetectionResult
from app.schemas.input import ImageInput


class BuildingDetector(Protocol):
    @property
    def name(self) -> str: ...

    async def detect(
        self,
        image: ImageInput,
        raster_path: Path,
        *,
        min_area_m2: float = 25.0,
        max_area_m2: float = 50_000.0,
    ) -> BuildingDetectionResult: ...
