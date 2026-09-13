"""Specialist tool for water surface area change analysis (NDWI)."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom

from app.adapters.change.bi_temporal.raster_io import (
    load_raster,
    match_raster,
    needs_coregistration,
)
from app.core.errors import SatQueryError
from app.evidence.geometry import ensure_ring_closed
from app.schemas.domain import EvidenceRegion, GeoJSONGeometry, Metric
from app.schemas.input import ImageInput
from app.schemas.surface_change import SurfaceAreaChangeResult, SurfaceDomainKind


class WaterChangeTool:
    name = "analyze_water"
    description = "Compute before/after water surface area in m², water body area difference, and percentage change."

    async def execute(
        self,
        earlier_image: ImageInput,
        later_image: ImageInput,
        earlier_path: Path,
        later_path: Path,
    ) -> tuple[SurfaceAreaChangeResult, list[EvidenceRegion]]:
        return await asyncio.to_thread(
            self._execute_sync,
            earlier_image,
            later_image,
            earlier_path,
            later_path,
        )

    def _execute_sync(
        self,
        earlier_image: ImageInput,
        later_image: ImageInput,
        earlier_path: Path,
        later_path: Path,
    ) -> tuple[SurfaceAreaChangeResult, list[EvidenceRegion]]:
        r_t1 = load_raster(earlier_path)
        r_t2 = load_raster(later_path)

        if needs_coregistration(r_t1, r_t2):
            r_t2 = match_raster(r_t2, r_t1)

        # Pixel area in m²
        mean_lat = (r_t1.bounds.bottom + r_t1.bounds.top) / 2.0
        if r_t1.crs and r_t1.crs.is_geographic:
            lat_scale = 111_320.0
            lon_scale = 111_320.0 * math.cos(math.radians(mean_lat))
            pixel_area_m2 = abs(r_t1.transform.a) * lon_scale * abs(r_t1.transform.e) * lat_scale
        else:
            pixel_area_m2 = abs(r_t1.transform.a * r_t1.transform.e)

        water_t1 = self._segment_water(r_t1.data)
        water_t2 = self._segment_water(r_t2.data)

        before_px = int(np.sum(water_t1))
        after_px = int(np.sum(water_t2))

        before_area_m2 = round(before_px * pixel_area_m2, 1)
        after_area_m2 = round(after_px * pixel_area_m2, 1)
        diff_m2 = round(after_area_m2 - before_area_m2, 1)
        pct_change = round((diff_m2 / max(before_area_m2, 1e-6)) * 100.0, 2)

        # Vectorize water change regions (expansion or shrinkage)
        change_mask = (water_t2 ^ water_t1).astype(np.uint8)
        evidence_regions: list[EvidenceRegion] = []
        region_idx = 1

        for geom, val in shapes(change_mask, mask=change_mask > 0, transform=r_t1.transform):
            if val == 0 or geom["type"] != "Polygon":
                continue

            if r_t1.crs and r_t1.crs.to_string() != "EPSG:4326":
                try:
                    reprojected = transform_geom(r_t1.crs.to_string(), "EPSG:4326", geom)
                except Exception:
                    continue
            else:
                reprojected = geom

            coords = reprojected.get("coordinates")
            if not coords or not coords[0]:
                continue
            ring = ensure_ring_closed(coords[0])
            if len(ring) < 4:
                continue

            evidence_regions.append(
                EvidenceRegion(
                    id=f"water-change-{region_idx:03d}",
                    geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                    type="water_surface_change",
                    confidence=0.91,
                    metrics=[
                        Metric(name="region_index", value=region_idx, unit=None, source="water_tool"),
                    ],
                    source="water_tool",
                    metadata={"domain": "water", "type": "surface_change"},
                )
            )
            region_idx += 1
            if region_idx > 50:
                break

        metrics = [
            Metric(name="before_water_area_m2", value=before_area_m2, unit="m²", source="water_tool"),
            Metric(name="after_water_area_m2", value=after_area_m2, unit="m²", source="water_tool"),
            Metric(name="water_difference_m2", value=diff_m2, unit="m²", source="water_tool"),
            Metric(name="water_percentage_change", value=pct_change, unit="%", source="water_tool"),
        ]

        result = SurfaceAreaChangeResult(
            domain=SurfaceDomainKind.WATER,
            earlier_image_id=earlier_image.id,
            later_image_id=later_image.id,
            before_area_m2=before_area_m2,
            after_area_m2=after_area_m2,
            difference_m2=diff_m2,
            percentage_change=pct_change,
            primary_index="NDWI",
            confidence=0.92,
            evidence_regions=evidence_regions,
            metrics=metrics,
            metadata={
                "pixel_area_m2": pixel_area_m2,
                "before_pixels": before_px,
                "after_pixels": after_px,
            },
        )

        return result, evidence_regions

    def _segment_water(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] >= 4:
            # Green = band 1 (0-indexed), NIR = band 3
            green = data[1].astype(np.float32)
            nir = data[3].astype(np.float32)
            ndwi = (green - nir) / np.maximum(green + nir, 1e-6)
            return ndwi > 0.0
        # For RGB: Blue band typically has higher response over water, NIR/Red is low
        if data.shape[0] >= 3:
            red = data[0].astype(np.float32)
            green = data[1].astype(np.float32)
            blue = data[2].astype(np.float32)
            # Water index proxy in RGB: (Green - Red) / (Green + Red + 1e-6) and blue > red
            ratio = (green - red) / np.maximum(green + red, 1e-6)
            return (blue > red) & (ratio > 0.05)
        # 1-band (e.g. SAR or grayscale)
        band = data[0].astype(np.float32)
        p25 = np.percentile(band, 25)
        return band < p25
