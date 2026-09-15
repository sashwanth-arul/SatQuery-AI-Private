"""Specialist tools for domain-specific remote sensing analysis:
- Water detection & water resource assessment
- Agriculture & vegetation condition monitoring
- Forest canopy & deforestation assessment
- Flood & disaster inundation mapping
- Comprehensive multi-class land cover analysis
- Evidence overlay raster generation (actual pixel masks composite)
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import shapes
from rasterio.warp import transform_geom

from app.adapters.change.bi_temporal.raster_io import (
    load_raster,
    match_raster,
    needs_coregistration,
)
from app.core.errors import SatQueryError
from app.evidence.geometry import ensure_ring_closed
from app.schemas.common import GeoJSONGeometry, Metric
from app.schemas.domain import EvidenceOverlay, EvidenceRegion
from app.schemas.input import ImageInput


def _compute_pixel_area_m2(transform: Any, crs: Any, bounds: Any) -> float:
    """Compute true ground pixel area in square meters."""
    mean_lat = (bounds.bottom + bounds.top) / 2.0
    if crs and crs.is_geographic:
        lat_scale = 111_320.0
        lon_scale = 111_320.0 * math.cos(math.radians(mean_lat))
        return abs(transform.a) * lon_scale * abs(transform.e) * lat_scale
    return abs(transform.a * transform.e)


def _reproject_polygon(geom: dict[str, Any], src_crs: Any) -> list[list[float]] | None:
    if geom.get("type") != "Polygon":
        return None
    if src_crs and src_crs.to_string() != "EPSG:4326":
        try:
            reprojected = transform_geom(src_crs.to_string(), "EPSG:4326", geom)
        except Exception:
            return None
    else:
        reprojected = geom

    coords = reprojected.get("coordinates")
    if not coords or not coords[0]:
        return None
    ring = ensure_ring_closed(coords[0])
    if len(ring) < 4:
        return None
    return ring


def render_evidence_overlay(
    raw_data: np.ndarray,
    layers: dict[str, tuple[np.ndarray, tuple[int, int, int]]],
    *,
    alpha: float = 0.55,
    max_dim: int = 640,
) -> tuple[str, int, int]:
    """
    Render a real composite RGBA PNG evidence overlay from actual raster analysis masks.
    Returns: (data_uri, width, height)
    """
    # 1. Base RGB background from raster bands
    c, h, w = raw_data.shape
    if c >= 3:
        r = raw_data[0].astype(np.float32)
        g = raw_data[1].astype(np.float32)
        b = raw_data[2].astype(np.float32)
    else:
        r = g = b = raw_data[0].astype(np.float32)

    # Normalize base image contrast
    vmin, vmax = np.percentile(r, 2), np.percentile(r, 98)
    r_norm = np.clip((r - vmin) / max(vmax - vmin, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    vmin, vmax = np.percentile(g, 2), np.percentile(g, 98)
    g_norm = np.clip((g - vmin) / max(vmax - vmin, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    vmin, vmax = np.percentile(b, 2), np.percentile(b, 98)
    b_norm = np.clip((b - vmin) / max(vmax - vmin, 1e-6) * 255.0, 0, 255).astype(np.uint8)

    base_rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)

    # 2. Blend colored mask layers on top of base image
    overlay_rgb = base_rgb.copy().astype(np.float32)
    for _layer_name, (mask, color_rgb) in layers.items():
        if mask is None or not np.any(mask):
            continue
        # Resize mask if dimensions mismatch
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        color_arr = np.array(color_rgb, dtype=np.float32)
        mask_idx = mask > 0
        overlay_rgb[mask_idx] = (1.0 - alpha) * overlay_rgb[mask_idx] + alpha * color_arr

    final_rgb = np.clip(overlay_rgb, 0, 255).astype(np.uint8)

    # 3. Downscale if image is very large to maintain fast UI delivery
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        out_w, out_h = int(w * scale), int(h * scale)
        final_rgb = cv2.resize(final_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        out_w, out_h = w, h

    pil_img = Image.fromarray(final_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"
    return data_uri, out_w, out_h


# ─────────────────────────────────────────────────────────────
# 1. Single-Image Water Resource Assessment Tool
# ─────────────────────────────────────────────────────────────

class SingleImageWaterTool:
    name = "water_detection_specialist"
    description = "Detect water bodies, compute surface water area in m² and percentage of total area."

    async def execute(
        self,
        image: ImageInput,
        raster_path: Path,
    ) -> tuple[dict[str, Any], list[EvidenceRegion], EvidenceOverlay]:
        return await asyncio.to_thread(self._execute_sync, image, raster_path)

    def _execute_sync(
        self,
        image: ImageInput,
        raster_path: Path,
    ) -> tuple[dict[str, Any], list[EvidenceRegion], EvidenceOverlay]:
        r = load_raster(raster_path)
        pixel_area = _compute_pixel_area_m2(r.transform, r.crs, r.bounds)
        total_pixels = r.data.shape[1] * r.data.shape[2]
        total_area_m2 = total_pixels * pixel_area

        # Compute NDWI or proxy
        water_mask = self._segment_water(r.data)
        water_px = int(np.sum(water_mask))
        water_area_m2 = round(water_px * pixel_area, 1)
        water_pct = round((water_area_m2 / max(total_area_m2, 1e-6)) * 100.0, 2)

        # Vectorize polygons
        evidence_regions: list[EvidenceRegion] = []
        region_idx = 1
        for geom, val in shapes(water_mask.astype(np.uint8), mask=water_mask > 0, transform=r.transform):
            if val == 0:
                continue
            ring = _reproject_polygon(geom, r.crs)
            if not ring:
                continue

            evidence_regions.append(
                EvidenceRegion(
                    id=f"water-body-{region_idx:03d}",
                    geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                    type="water_body",
                    confidence=0.92,
                    metrics=[
                        Metric(name="water_body_index", value=region_idx, unit=None, source=self.name),
                    ],
                    source=self.name,
                    metadata={"claim_type": "water_body", "domain": "water"},
                )
            )
            region_idx += 1
            if region_idx > 50:
                break

        # Generate evidence overlay
        data_uri, out_w, out_h = render_evidence_overlay(
            r.data,
            {"water": (water_mask, (2, 132, 199))},  # Cyan/Blue #0284c7
        )

        overlay = EvidenceOverlay(
            overlay_id=f"overlay-water-{image.id}",
            image_id=image.id,
            supported_layers=["original", "water"],
            data_uri=data_uri,
            width=out_w,
            height=out_h,
            crs=r.crs.to_string() if r.crs else "EPSG:4326",
            bounds=[r.bounds.left, r.bounds.bottom, r.bounds.right, r.bounds.top],
            statistics={
                "water_area_m2": water_area_m2,
                "water_percentage": water_pct,
                "total_area_m2": total_area_m2,
                "primary_index": "NDWI",
            },
        )

        stats = {
            "domain": "water",
            "water_area_m2": water_area_m2,
            "water_percentage": water_pct,
            "total_area_m2": total_area_m2,
            "water_body_count": len(evidence_regions),
            "primary_index": "NDWI",
            "confidence": 0.92,
        }

        return stats, evidence_regions, overlay

    def _segment_water(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] >= 4:
            # Sentinel-2: Green=B3 (index 1), NIR=B8 (index 3)
            green = data[1].astype(np.float32)
            nir = data[3].astype(np.float32)
            ndwi = (green - nir) / np.maximum(green + nir, 1e-6)
            return ndwi > 0.0
        if data.shape[0] >= 3:
            red = data[0].astype(np.float32)
            green = data[1].astype(np.float32)
            blue = data[2].astype(np.float32)
            ratio = (green - red) / np.maximum(green + red, 1e-6)
            return (blue > red) & (ratio > 0.05)
        # 1-band (e.g. SAR backscatter specular reflection)
        band = data[0].astype(np.float32)
        p25 = np.percentile(band, 25)
        return band < p25


# ─────────────────────────────────────────────────────────────
# 2. Single-Image Vegetation & Forest Monitoring Tool
# ─────────────────────────────────────────────────────────────

class SingleImageVegetationTool:
    name = "vegetation_forest_specialist"
    description = "Analyze agricultural and forest vegetation, compute canopy cover in m² and percentage."

    async def execute(
        self,
        image: ImageInput,
        raster_path: Path,
        *,
        mode: str = "general",  # 'general', 'agriculture', 'forest'
    ) -> tuple[dict[str, Any], list[EvidenceRegion], EvidenceOverlay]:
        return await asyncio.to_thread(self._execute_sync, image, raster_path, mode)

    def _execute_sync(
        self,
        image: ImageInput,
        raster_path: Path,
        mode: str,
    ) -> tuple[dict[str, Any], list[EvidenceRegion], EvidenceOverlay]:
        r = load_raster(raster_path)
        pixel_area = _compute_pixel_area_m2(r.transform, r.crs, r.bounds)
        total_pixels = r.data.shape[1] * r.data.shape[2]
        total_area_m2 = total_pixels * pixel_area

        ndvi_map = self._compute_ndvi(r.data)

        # Configurable thresholds:
        # Dense forest canopy: NDVI > 0.55
        # Agriculture / moderate vegetation: 0.22 <= NDVI <= 0.55
        forest_mask = ndvi_map > 0.55
        agri_mask = (ndvi_map >= 0.22) & (ndvi_map <= 0.55)
        total_veg_mask = ndvi_map >= 0.22

        forest_area_m2 = round(float(np.sum(forest_mask)) * pixel_area, 1)
        agri_area_m2 = round(float(np.sum(agri_mask)) * pixel_area, 1)
        total_veg_area_m2 = round(float(np.sum(total_veg_mask)) * pixel_area, 1)

        forest_pct = round((forest_area_m2 / max(total_area_m2, 1e-6)) * 100.0, 2)
        agri_pct = round((agri_area_m2 / max(total_area_m2, 1e-6)) * 100.0, 2)
        total_veg_pct = round((total_veg_area_m2 / max(total_area_m2, 1e-6)) * 100.0, 2)

        # Determine target mask for polygon extraction based on query mode
        if mode == "forest":
            target_mask = forest_mask
            claim = "forest_cover"
            region_type = "forest_canopy"
        elif mode == "agriculture":
            target_mask = agri_mask
            claim = "agricultural_area"
            region_type = "agricultural_parcel"
        else:
            target_mask = total_veg_mask
            claim = "vegetation_cover"
            region_type = "vegetation_tract"

        evidence_regions: list[EvidenceRegion] = []
        region_idx = 1
        for geom, val in shapes(target_mask.astype(np.uint8), mask=target_mask > 0, transform=r.transform):
            if val == 0:
                continue
            ring = _reproject_polygon(geom, r.crs)
            if not ring:
                continue

            evidence_regions.append(
                EvidenceRegion(
                    id=f"veg-region-{region_idx:03d}",
                    geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                    type=region_type,
                    confidence=0.90,
                    metrics=[
                        Metric(name="region_index", value=region_idx, unit=None, source=self.name),
                    ],
                    source=self.name,
                    metadata={"claim_type": claim, "domain": "vegetation"},
                )
            )
            region_idx += 1
            if region_idx > 50:
                break

        # Evidence overlay with both agriculture (lime) and dense forest (dark green)
        data_uri, out_w, out_h = render_evidence_overlay(
            r.data,
            {
                "agriculture": (agri_mask, (132, 204, 22)),  # Lime green #84cc16
                "forest": (forest_mask, (21, 128, 61)),      # Forest green #15803d
            },
        )

        overlay = EvidenceOverlay(
            overlay_id=f"overlay-veg-{image.id}",
            image_id=image.id,
            supported_layers=["original", "vegetation", "forest", "agriculture"],
            data_uri=data_uri,
            width=out_w,
            height=out_h,
            crs=r.crs.to_string() if r.crs else "EPSG:4326",
            bounds=[r.bounds.left, r.bounds.bottom, r.bounds.right, r.bounds.top],
            statistics={
                "forest_area_m2": forest_area_m2,
                "forest_percentage": forest_pct,
                "agricultural_area_m2": agri_area_m2,
                "agricultural_percentage": agri_pct,
                "total_vegetation_area_m2": total_veg_area_m2,
                "total_vegetation_percentage": total_veg_pct,
                "primary_index": "NDVI",
            },
        )

        stats = {
            "domain": "vegetation" if mode == "general" else mode,
            "mode": mode,
            "forest_area_m2": forest_area_m2,
            "forest_percentage": forest_pct,
            "agricultural_area_m2": agri_area_m2,
            "agricultural_percentage": agri_pct,
            "total_vegetation_area_m2": total_veg_area_m2,
            "total_vegetation_percentage": total_veg_pct,
            "total_area_m2": total_area_m2,
            "region_count": len(evidence_regions),
            "primary_index": "NDVI",
            "confidence": 0.90,
        }

        return stats, evidence_regions, overlay

    def _compute_ndvi(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] >= 4:
            red = data[0].astype(np.float32)
            nir = data[3].astype(np.float32)
            return (nir - red) / np.maximum(nir + red, 1e-6)
        if data.shape[0] >= 3:
            red = data[0].astype(np.float32)
            green = data[1].astype(np.float32)
            blue = data[2].astype(np.float32)
            # Visible Atmospherically Resistant Index (VARI)
            return (green - red) / np.maximum(green + red - blue, 1e-6)
        band = data[0].astype(np.float32)
        lo, hi = np.percentile(band, 5), np.percentile(band, 95)
        return (band - lo) / max(hi - lo, 1e-6)


# ─────────────────────────────────────────────────────────────
# 3. Disaster Management & Flood Inundation Tool
# ─────────────────────────────────────────────────────────────

class FloodAnalysisTool:
    name = "flood_disaster_specialist"
    description = "Detect flooded and inundated areas from optical (NDWI) or SAR imagery; assess flood extent and change."

    async def execute_single(
        self,
        image: ImageInput,
        raster_path: Path,
    ) -> tuple[dict[str, Any], list[EvidenceRegion], EvidenceOverlay]:
        return await asyncio.to_thread(self._execute_single_sync, image, raster_path)

    async def execute_temporal(
        self,
        earlier_image: ImageInput,
        later_image: ImageInput,
        earlier_path: Path,
        later_path: Path,
    ) -> tuple[dict[str, Any], list[EvidenceRegion], EvidenceOverlay]:
        return await asyncio.to_thread(
            self._execute_temporal_sync,
            earlier_image,
            later_image,
            earlier_path,
            later_path,
        )

    def _execute_single_sync(
        self,
        image: ImageInput,
        raster_path: Path,
    ) -> tuple[dict[str, Any], list[EvidenceRegion], EvidenceOverlay]:
        r = load_raster(raster_path)
        pixel_area = _compute_pixel_area_m2(r.transform, r.crs, r.bounds)
        total_pixels = r.data.shape[1] * r.data.shape[2]
        total_area_m2 = total_pixels * pixel_area

        flood_mask = self._segment_flood(r.data)
        flood_px = int(np.sum(flood_mask))
        flood_area_m2 = round(flood_px * pixel_area, 1)
        flood_pct = round((flood_area_m2 / max(total_area_m2, 1e-6)) * 100.0, 2)

        evidence_regions: list[EvidenceRegion] = []
        region_idx = 1
        for geom, val in shapes(flood_mask.astype(np.uint8), mask=flood_mask > 0, transform=r.transform):
            if val == 0:
                continue
            ring = _reproject_polygon(geom, r.crs)
            if not ring:
                continue

            evidence_regions.append(
                EvidenceRegion(
                    id=f"flood-zone-{region_idx:03d}",
                    geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                    type="flood_inundation",
                    confidence=0.91,
                    metrics=[
                        Metric(name="zone_index", value=region_idx, unit=None, source=self.name),
                    ],
                    source=self.name,
                    metadata={"claim_type": "flooded_area", "domain": "disaster"},
                )
            )
            region_idx += 1
            if region_idx > 50:
                break

        data_uri, out_w, out_h = render_evidence_overlay(
            r.data,
            {"flood": (flood_mask, (56, 189, 248))},  # Flood sky blue #38bdf8
        )

        overlay = EvidenceOverlay(
            overlay_id=f"overlay-flood-{image.id}",
            image_id=image.id,
            supported_layers=["original", "flood"],
            data_uri=data_uri,
            width=out_w,
            height=out_h,
            crs=r.crs.to_string() if r.crs else "EPSG:4326",
            bounds=[r.bounds.left, r.bounds.bottom, r.bounds.right, r.bounds.top],
            statistics={
                "flood_inundation_area_m2": flood_area_m2,
                "flood_percentage": flood_pct,
                "total_area_m2": total_area_m2,
                "primary_index": "NDWI_or_SAR_Drop",
            },
        )

        stats = {
            "domain": "disaster",
            "task": "flood_analysis",
            "flood_area_m2": flood_area_m2,
            "flood_percentage": flood_pct,
            "total_area_m2": total_area_m2,
            "zone_count": len(evidence_regions),
            "primary_index": "NDWI / SAR Inundation",
            "confidence": 0.91,
        }

        return stats, evidence_regions, overlay

    def _execute_temporal_sync(
        self,
        earlier_image: ImageInput,
        later_image: ImageInput,
        earlier_path: Path,
        later_path: Path,
    ) -> tuple[dict[str, Any], list[EvidenceRegion], EvidenceOverlay]:
        r_t1 = load_raster(earlier_path)
        r_t2 = load_raster(later_path)
        if needs_coregistration(r_t1, r_t2):
            r_t2 = match_raster(r_t2, r_t1)

        pixel_area = _compute_pixel_area_m2(r_t1.transform, r_t1.crs, r_t1.bounds)
        total_pixels = r_t1.data.shape[1] * r_t1.data.shape[2]
        total_area_m2 = total_pixels * pixel_area

        water_t1 = self._segment_flood(r_t1.data)
        water_t2 = self._segment_flood(r_t2.data)

        # Inundated land = water in later date that was not water before
        new_flood_mask = water_t2 & (~water_t1)
        receded_mask = water_t1 & (~water_t2)

        before_area_m2 = round(float(np.sum(water_t1)) * pixel_area, 1)
        after_area_m2 = round(float(np.sum(water_t2)) * pixel_area, 1)
        new_flood_area_m2 = round(float(np.sum(new_flood_mask)) * pixel_area, 1)
        receded_area_m2 = round(float(np.sum(receded_mask)) * pixel_area, 1)
        net_change_m2 = round(after_area_m2 - before_area_m2, 1)

        evidence_regions: list[EvidenceRegion] = []
        region_idx = 1
        for geom, val in shapes(new_flood_mask.astype(np.uint8), mask=new_flood_mask > 0, transform=r_t1.transform):
            if val == 0:
                continue
            ring = _reproject_polygon(geom, r_t1.crs)
            if not ring:
                continue

            evidence_regions.append(
                EvidenceRegion(
                    id=f"inundation-zone-{region_idx:03d}",
                    geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                    type="flood_inundation",
                    confidence=0.93,
                    metrics=[
                        Metric(name="zone_index", value=region_idx, unit=None, source=self.name),
                    ],
                    source=self.name,
                    metadata={"claim_type": "flooded_area", "domain": "disaster"},
                )
            )
            region_idx += 1
            if region_idx > 50:
                break

        # Evidence overlay showing baseline water (cyan) + new flood inundation (bright sky blue)
        data_uri, out_w, out_h = render_evidence_overlay(
            r_t2.data,
            {
                "baseline_water": (water_t1, (2, 132, 199)),
                "new_flood": (new_flood_mask, (56, 189, 248)),
            },
        )

        overlay = EvidenceOverlay(
            overlay_id=f"overlay-flood-temporal-{later_image.id}",
            image_id=later_image.id,
            supported_layers=["original", "flood", "change"],
            data_uri=data_uri,
            width=out_w,
            height=out_h,
            crs=r_t1.crs.to_string() if r_t1.crs else "EPSG:4326",
            bounds=[r_t1.bounds.left, r_t1.bounds.bottom, r_t1.bounds.right, r_t1.bounds.top],
            statistics={
                "newly_flooded_area_m2": new_flood_area_m2,
                "receded_area_m2": receded_area_m2,
                "before_water_area_m2": before_area_m2,
                "after_water_area_m2": after_area_m2,
                "net_flood_change_m2": net_change_m2,
            },
        )

        stats = {
            "domain": "disaster",
            "task": "temporal_flood_change",
            "newly_flooded_area_m2": new_flood_area_m2,
            "receded_area_m2": receded_area_m2,
            "before_water_area_m2": before_area_m2,
            "after_water_area_m2": after_area_m2,
            "net_flood_change_m2": net_change_m2,
            "total_area_m2": total_area_m2,
            "inundation_zone_count": len(evidence_regions),
            "primary_index": "NDWI / SAR Inundation Delta",
            "confidence": 0.93,
        }

        return stats, evidence_regions, overlay

    def _segment_flood(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] >= 4:
            green = data[1].astype(np.float32)
            nir = data[3].astype(np.float32)
            ndwi = (green - nir) / np.maximum(green + nir, 1e-6)
            return ndwi > 0.05
        if data.shape[0] >= 3:
            red = data[0].astype(np.float32)
            green = data[1].astype(np.float32)
            blue = data[2].astype(np.float32)
            return (blue > red * 1.1) & (green > red * 1.05)
        # SAR specular drop: low backscatter
        band = data[0].astype(np.float32)
        p20 = np.percentile(band, 20)
        return band < p20


# ─────────────────────────────────────────────────────────────
# 4. Multi-Class Land Cover & Environmental Analysis Tool
# ─────────────────────────────────────────────────────────────

class LandCoverAnalysisTool:
    name = "land_cover_environmental_specialist"
    description = "Classify multi-class land cover (Water, Forest, Agriculture, Built-up) and compute distribution statistics."

    async def execute(
        self,
        image: ImageInput,
        raster_path: Path,
    ) -> tuple[dict[str, Any], list[EvidenceRegion], EvidenceOverlay]:
        return await asyncio.to_thread(self._execute_sync, image, raster_path)

    def _execute_sync(
        self,
        image: ImageInput,
        raster_path: Path,
    ) -> tuple[dict[str, Any], list[EvidenceRegion], EvidenceOverlay]:
        r = load_raster(raster_path)
        pixel_area = _compute_pixel_area_m2(r.transform, r.crs, r.bounds)
        total_pixels = r.data.shape[1] * r.data.shape[2]
        total_area_m2 = total_pixels * pixel_area

        data = r.data
        h, w = data.shape[1], data.shape[2]

        # 1. Water mask
        if data.shape[0] >= 5:
            green = data[1].astype(np.float32)
            nir = data[3].astype(np.float32)
            swir = data[4].astype(np.float32)
            ndwi = (green - nir) / np.maximum(green + nir, 1e-6)
            water_mask = (ndwi > 0.0) & (swir < green)
        elif data.shape[0] >= 4:
            green = data[1].astype(np.float32)
            nir = data[3].astype(np.float32)
            ndwi = (green - nir) / np.maximum(green + nir, 1e-6)
            water_mask = ndwi > 0.0
        elif data.shape[0] >= 3:
            water_mask = (data[2] > data[0] * 1.15) & (data[1] > data[0] * 1.05)
        else:
            water_mask = data[0] < np.percentile(data[0], 20)

        # 2. Vegetation & Forest masks (excluding water)
        if data.shape[0] >= 4:
            red = data[0].astype(np.float32)
            nir = data[3].astype(np.float32)
            ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        elif data.shape[0] >= 3:
            red = data[0].astype(np.float32)
            green = data[1].astype(np.float32)
            blue = data[2].astype(np.float32)
            ndvi = (green - red) / np.maximum(green + red - blue, 1e-6)
        else:
            lo, hi = np.percentile(data[0], 5), np.percentile(data[0], 95)
            ndvi = (data[0].astype(np.float32) - lo) / max(hi - lo, 1e-6)

        forest_mask = (ndvi > 0.55) & (~water_mask)
        agri_mask = (ndvi >= 0.22) & (ndvi <= 0.55) & (~water_mask)

        # 3. Built-up / Urban mask (excluding water & vegetation)
        if data.shape[0] >= 5:
            swir = data[4].astype(np.float32)
            nir = data[3].astype(np.float32)
            ndbi = (swir - nir) / np.maximum(swir + nir, 1e-6)
            built_mask = (ndbi > 0.05) & (~water_mask) & (~forest_mask) & (~agri_mask)
        elif data.shape[0] >= 3:
            lum = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
            built_mask = (lum > np.percentile(lum, 70)) & (~water_mask) & (~forest_mask) & (~agri_mask)
        else:
            built_mask = (data[0] > np.percentile(data[0], 70)) & (~water_mask) & (~forest_mask) & (~agri_mask)

        # Other / Bare ground
        bare_mask = (~water_mask) & (~forest_mask) & (~agri_mask) & (~built_mask)

        # Compute areas and percentages
        water_m2 = round(float(np.sum(water_mask)) * pixel_area, 1)
        forest_m2 = round(float(np.sum(forest_mask)) * pixel_area, 1)
        agri_m2 = round(float(np.sum(agri_mask)) * pixel_area, 1)
        built_m2 = round(float(np.sum(built_mask)) * pixel_area, 1)
        bare_m2 = round(float(np.sum(bare_mask)) * pixel_area, 1)

        water_pct = round((water_m2 / max(total_area_m2, 1e-6)) * 100.0, 1)
        forest_pct = round((forest_m2 / max(total_area_m2, 1e-6)) * 100.0, 1)
        agri_pct = round((agri_m2 / max(total_area_m2, 1e-6)) * 100.0, 1)
        built_pct = round((built_m2 / max(total_area_m2, 1e-6)) * 100.0, 1)
        bare_pct = round((bare_m2 / max(total_area_m2, 1e-6)) * 100.0, 1)

        # Extract representative polygons
        evidence_regions: list[EvidenceRegion] = []
        region_idx = 1
        class_masks = [
            ("water", water_mask, "water_body", "water_body"),
            ("forest", forest_mask, "forest_canopy", "forest_cover"),
            ("agriculture", agri_mask, "agricultural_parcel", "agricultural_area"),
            ("built_up", built_mask, "built_up_area", "built_up_area"),
        ]

        for cls_name, mask, reg_type, claim in class_masks:
            count_for_class = 0
            for geom, val in shapes(mask.astype(np.uint8), mask=mask > 0, transform=r.transform):
                if val == 0:
                    continue
                ring = _reproject_polygon(geom, r.crs)
                if not ring:
                    continue

                evidence_regions.append(
                    EvidenceRegion(
                        id=f"landcover-{cls_name}-{region_idx:03d}",
                        geometry=GeoJSONGeometry(type="Polygon", coordinates=[ring]),
                        type=reg_type,
                        confidence=0.88,
                        metrics=[
                            Metric(name="land_cover_class", value=cls_name, unit=None, source=self.name),
                        ],
                        source=self.name,
                        metadata={"claim_type": claim, "class": cls_name, "domain": "environmental"},
                    )
                )
                region_idx += 1
                count_for_class += 1
                if count_for_class >= 10:
                    break

        # Evidence overlay with multi-class coloring
        data_uri, out_w, out_h = render_evidence_overlay(
            r.data,
            {
                "water": (water_mask, (2, 132, 199)),       # Blue
                "forest": (forest_mask, (21, 128, 61)),     # Dark green
                "agriculture": (agri_mask, (132, 204, 22)), # Light green
                "built_up": (built_mask, (249, 115, 22)),   # Orange
            },
        )

        overlay = EvidenceOverlay(
            overlay_id=f"overlay-landcover-{image.id}",
            image_id=image.id,
            supported_layers=["original", "water", "vegetation", "forest", "built_up"],
            data_uri=data_uri,
            width=out_w,
            height=out_h,
            crs=r.crs.to_string() if r.crs else "EPSG:4326",
            bounds=[r.bounds.left, r.bounds.bottom, r.bounds.right, r.bounds.top],
            statistics={
                "water_m2": water_m2,
                "water_pct": water_pct,
                "forest_m2": forest_m2,
                "forest_pct": forest_pct,
                "agri_m2": agri_m2,
                "agri_pct": agri_pct,
                "built_m2": built_m2,
                "built_pct": built_pct,
                "bare_m2": bare_m2,
                "bare_pct": bare_pct,
                "total_area_m2": total_area_m2,
            },
        )

        stats = {
            "domain": "environmental",
            "task": "land_cover_analysis",
            "water_m2": water_m2,
            "water_percentage": water_pct,
            "forest_m2": forest_m2,
            "forest_percentage": forest_pct,
            "agricultural_m2": agri_m2,
            "agricultural_percentage": agri_pct,
            "built_up_m2": built_m2,
            "built_up_percentage": built_pct,
            "bare_m2": bare_m2,
            "bare_percentage": bare_pct,
            "total_area_m2": total_area_m2,
            "region_count": len(evidence_regions),
            "primary_index": "Multi-Spectral Classification",
            "confidence": 0.88,
        }

        return stats, evidence_regions, overlay
