"""Uploaded optical/multispectral analysis for cross-modal workflows."""

from __future__ import annotations

import hashlib

from app.schemas.cross_modal import CrossModalProviderKind, ModalityAnalysisSummary, OpticalAnalysisInput
from app.schemas.domain import EvidenceRegion, GeoJSONGeometry, Metric
from app.adapters.imagery.uploaded.bi_temporal_bridge import aoi_from_bounds


class UploadedOpticalAnalysisTool:
    name = "optical_analysis"
    description = "Extract optical/multispectral surface cues from an uploaded image."

    async def execute(self, payload: OpticalAnalysisInput) -> ModalityAnalysisSummary:
        bounds = payload.bounds
        digest = hashlib.sha256(f"optical:{payload.optical_image_id}:{payload.query}".encode()).hexdigest()[:8]
        west = min(bounds[0], bounds[2])
        south = min(bounds[1], bounds[3])
        east = max(bounds[0], bounds[2])
        north = max(bounds[1], bounds[3])
        cx = (west + east) / 2
        cy = (south + north) / 2
        w = max((east - west) * 0.25, 0.0001)
        h = max((north - south) * 0.25, 0.0001)
        built_poly = GeoJSONGeometry(
            type="Polygon",
            coordinates=[
                [
                    [round(cx - w / 2, 6), round(cy - h / 2, 6)],
                    [round(cx + w / 2, 6), round(cy - h / 2, 6)],
                    [round(cx + w / 2, 6), round(cy + h / 2, 6)],
                    [round(cx - w / 2, 6), round(cy + h / 2, 6)],
                    [round(cx - w / 2, 6), round(cy - h / 2, 6)],
                ]
            ],
        )
        water_cy = cy - h
        water_poly = GeoJSONGeometry(
            type="Polygon",
            coordinates=[
                [
                    [round(cx - w / 2, 6), round(water_cy - h / 4, 6)],
                    [round(cx + w / 2, 6), round(water_cy - h / 4, 6)],
                    [round(cx + w / 2, 6), round(water_cy + h / 4, 6)],
                    [round(cx - w / 2, 6), round(water_cy + h / 4, 6)],
                    [round(cx - w / 2, 6), round(water_cy - h / 4, 6)],
                ]
            ],
        )
        regions = [
            EvidenceRegion(
                id=f"optical-built-{digest}",
                geometry=built_poly,
                type="built_up_candidate",
                confidence=0.62,
                metrics=[
                    Metric(name="surface_class", value="built_up", source="development_optical_analysis"),
                ],
                source="development_optical_analysis",
                metadata={"evidence_modality": "optical", "surface_class": "built_up"},
            ),
            EvidenceRegion(
                id=f"optical-water-{digest}",
                geometry=water_poly,
                type="water_candidate",
                confidence=0.58,
                metrics=[
                    Metric(name="surface_class", value="water", source="development_optical_analysis"),
                ],
                source="development_optical_analysis",
                metadata={"evidence_modality": "optical", "surface_class": "water"},
            ),
        ]
        q = payload.query.lower()
        focus = []
        if "built" in q:
            focus.append("built-up")
        if "water" in q:
            focus.append("water")
        focus_clause = f" focusing on {', '.join(focus)}" if focus else ""
        return ModalityAnalysisSummary(
            modality="optical",
            summary=(
                f"[development mock optical analysis — not Earth Engine/Dynamic World] "
                f"Optical cues suggest built-up and water-covered candidates{focus_clause} "
                f"in image {payload.optical_image_id} (ref {digest})."
            ),
            analyzer="development_uploaded_optical_analysis",
            provider=CrossModalProviderKind.DEVELOPMENT,
            regions=regions,
            confidence_available=False,
            metadata={"mock": True, "image_id": payload.optical_image_id},
        )
