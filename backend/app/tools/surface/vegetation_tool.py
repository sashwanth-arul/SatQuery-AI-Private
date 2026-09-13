"""Specialist tool for vegetation canopy change analysis (NDVI)."""

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


class VegetationChangeTool:
    name = "analyze_vegetation"
    description = "Compute before/after vegetation cover area in m², canopy area difference, and percentage change."

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

        veg_t1 = self._segment_vegetation(r_t1.data)
        veg_t2 = self._segment_vegetation(r_t2.data)

        before_px = int(np.sum(veg_t1))
        after_px = int(np.sum(veg_t2))

        before_area_m2 = round(before_px * pixel_area_m2, 1)
        after_area_m2 = round(after_px * pixel_area_m2, 1)
        diff_m2 = round(after_area_m2 - before_area_m2, 1)
        pct_change = round((diff_m2 / max(before_area_m2, 1e-6)) * 100.0, 2)

        # Vectorize vegetation change regions (loss or gain)
        change_mask = (veg_t2 ^ veg_t1).astype(np.uint8)
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
                    id=f"veg-change-{region_idx:03d}",
                    geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                    type="vegetation_change",
                    confidence=0.90,
                    metrics=[
                        Metric(name="region_index", value=region_idx, unit=None, source="vegetation_tool"),
                    ],
                    source="vegetation_tool",
                    metadata={"domain": "vegetation", "type": "surface_change"},
                )
            )
            region_idx += 1
            if region_idx > 50:
                break

        metrics = [
            Metric(name="before_vegetation_area_m2", value=before_area_m2, unit="m²", source="vegetation_tool"),
            Metric(name="after_vegetation_area_m2", value=after_area_m2, unit="m²", source="vegetation_tool"),
            Metric(name="vegetation_difference_m2", value=diff_m2, unit="m²", source="vegetation_tool"),
            Metric(name="vegetation_percentage_change", value=pct_change, unit="%", source="vegetation_tool"),
        ]

        result = SurfaceAreaChangeResult(
            domain=SurfaceDomainKind.VEGETATION,
            earlier_image_id=earlier_image.id,
            later_image_id=later_image.id,
            before_area_m2=before_area_m2,
            after_area_m2=after_area_m2,
            difference_m2=diff_m2,
            percentage_change=pct_change,
            primary_index="NDVI",
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

    def _segment_vegetation(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] >= 4:
            # Red = band 0, NIR = band 3
            red = data[0].astype(np.float32)
            nir = data[3].astype(np.float32)
            ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
            return ndvi > 0.25
        # RGB: Visible Atmospherically Resistant Index (VARI) or excess green index
        if data.shape[0] >= 3:
            red = data[0].astype(np.float32)
            green = data[1].astype(np.float32)
            blue = data[2].astype(np.float32)
            # Excess green: (2*G - R - B) / (2*G + R + B + 1e-6)
            exg = (2.0 * green - red - blue) / np.maximum(2.0 * green + red + blue, 1e-6)
            return exg > 0.08
        band = data[0].astype(np.float32)
        p75 = np.percentile(band, 75)
        return band > p75
