"""Deterministic building detection adapter using geospatial raster analysis."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import shape as shapely_shape

from app.adapters.building.base import BuildingDetector
from app.core.errors import SatQueryError
from app.evidence.geometry import ensure_ring_closed
from app.schemas.building_analysis import (
    BuildingDetectionResult,
    BuildingFootprint,
    BuildingStatus,
)
from app.schemas.domain import GeoJSONGeometry
from app.schemas.input import ImageInput


def _calc_geodesic_polygon_area_m2(ring: list[list[float]], mean_lat: float) -> float:
    if len(ring) < 3:
        return 0.0
    closed = ring[:-1] if ring[0] == ring[-1] else ring
    lat_scale = 111_320.0
    lon_scale = 111_320.0 * math.cos(math.radians(mean_lat))
    area = 0.0
    for i in range(len(closed)):
        x1, y1 = closed[i][0] * lon_scale, closed[i][1] * lat_scale
        x2, y2 = closed[(i + 1) % len(closed)][0] * lon_scale, closed[(i + 1) % len(closed)][1] * lat_scale
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


class DevelopmentBuildingDetector(BuildingDetector):
    """
    Deterministic building detection adapter.
    Segments discrete structural footprints using high-contrast spectral/spatial gradients
    and morphological boundary analysis on actual raster pixels.
    """

    @property
    def name(self) -> str:
        return "development_morphological_building_detector"

    async def detect(
        self,
        image: ImageInput,
        raster_path: Path,
        *,
        min_area_m2: float = 25.0,
        max_area_m2: float = 2_000_000.0,
    ) -> BuildingDetectionResult:
        return await asyncio.to_thread(
            self._detect_sync,
            image,
            raster_path,
            min_area_m2=min_area_m2,
            max_area_m2=max_area_m2,
        )

    def _detect_sync(
        self,
        image: ImageInput,
        raster_path: Path,
        *,
        min_area_m2: float = 25.0,
        max_area_m2: float = 2_000_000.0,
    ) -> BuildingDetectionResult:
        if not raster_path.exists():
            raise SatQueryError("image_not_found", f"Raster not found: {raster_path}", status_code=404)

        try:
            with rasterio.open(raster_path) as src:
                crs = src.crs
                transform = src.transform
                width, height = src.width, src.height
                count = src.count

                # Read representative band or luminance
                if count >= 3:
                    # Normalized red and green bands
                    r = src.read(1).astype(np.float32)
                    g = src.read(2).astype(np.float32)
                    b = src.read(3).astype(np.float32)
                    img = 0.299 * r + 0.587 * g + 0.114 * b
                else:
                    img = src.read(1).astype(np.float32)

        except Exception as exc:
            raise SatQueryError("invalid_raster", f"Failed to read raster for building detection: {exc}", status_code=400) from exc

        # Normalize pixel values
        vmin, vmax = np.percentile(img, 2), np.percentile(img, 98)
        norm = np.clip((img - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)

        # High-frequency structural feature response via morphological gradient
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(norm, k3)
        eroded = cv2.erode(norm, k3)
        gradient = dilated - eroded

        # Also detect bright compact rooftop structures
        local_mean = cv2.boxFilter(norm, -1, (15, 15))
        bright_structures = (norm - local_mean) > 0.08
        structure_mask = ((gradient > 0.12) | bright_structures).astype(np.uint8)

        # Morphological opening and closing to separate adjacent buildings
        cross_k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        structure_mask = cv2.morphologyEx(structure_mask, cv2.MORPH_OPEN, cross_k)
        structure_mask = cv2.morphologyEx(structure_mask, cv2.MORPH_CLOSE, rect_k)

        # Vectorize mask to polygons using rasterio.features.shapes
        polygon_generator = shapes(structure_mask, mask=structure_mask > 0, transform=transform)

        detections: list[BuildingFootprint] = []
        detection_idx = 1

        for geom, val in polygon_generator:
            if val == 0:
                continue
            if geom["type"] != "Polygon":
                continue

            # Reproject to EPSG:4326 if needed
            if crs and crs.to_string() != "EPSG:4326":
                try:
                    reprojected_geom = transform_geom(crs.to_string(), "EPSG:4326", geom)
                except Exception:
                    continue
            else:
                reprojected_geom = geom

            coords = reprojected_geom.get("coordinates")
            if not coords or not coords[0]:
                continue

            raw_ring = coords[0]
            clean_ring = ensure_ring_closed(raw_ring)
            if len(clean_ring) < 4:
                continue

            # Calculate geodesic area
            mean_lat = sum(p[1] for p in clean_ring) / len(clean_ring)
            area_m2 = _calc_geodesic_polygon_area_m2(clean_ring, mean_lat)

            if area_m2 < min_area_m2 or area_m2 > max_area_m2:
                continue

            lons = [p[0] for p in clean_ring]
            lats = [p[1] for p in clean_ring]
            bbox = [round(min(lons), 6), round(min(lats), 6), round(max(lons), 6), round(max(lats), 6)]

            # Confidence based on compactness and size plausibility
            perimeter = sum(
                math.hypot(
                    (clean_ring[i][0] - clean_ring[i - 1][0]) * 111_320.0 * math.cos(math.radians(mean_lat)),
                    (clean_ring[i][1] - clean_ring[i - 1][1]) * 111_320.0,
                )
                for i in range(len(clean_ring))
            )
            isoperimetric_ratio = (4.0 * math.pi * area_m2) / max(perimeter**2, 1e-6)
            confidence = float(np.clip(0.65 + 0.30 * isoperimetric_ratio, 0.60, 0.95))

            detections.append(
                BuildingFootprint(
                    id=f"building-{detection_idx:04d}",
                    confidence=round(confidence, 3),
                    bbox=bbox,
                    geometry=GeoJSONGeometry(type="Polygon", coordinates=[clean_ring]),
                    area_m2=round(area_m2, 1),
                    status=BuildingStatus.DETECTED,
                )
            )
            detection_idx += 1

        total_area = round(sum(d.area_m2 for d in detections), 1)
        from app.schemas.common import Metric
        metrics = [
            Metric(name="building_count", value=len(detections), unit="buildings", source=self.name),
            Metric(name="total_footprint_area_m2", value=total_area, unit="m²", source=self.name),
        ]
        return BuildingDetectionResult(
            image_id=image.id,
            count=len(detections),
            detections=detections,
            detector=self.name,
            detector_name=self.name,
            confidence=0.88,
            total_area_m2=total_area,
            metrics=metrics,
            detector_metadata={
                "min_area_m2": min_area_m2,
                "max_area_m2": max_area_m2,
                "raster_width": width,
                "raster_height": height,
                "raster_crs": crs.to_string() if crs else "unknown",
            },
        )
