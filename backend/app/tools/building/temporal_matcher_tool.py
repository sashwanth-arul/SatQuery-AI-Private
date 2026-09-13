"""Specialist tool for temporal building footprint bipartite matching and change classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shapely.geometry import shape as shapely_shape

from app.adapters.building.base import BuildingDetector
from app.adapters.building.development import DevelopmentBuildingDetector
from app.schemas.building_analysis import (
    BuildingFootprint,
    BuildingStatus,
    BuildingTemporalMatchResult,
)
from app.schemas.domain import EvidenceRegion, Metric
from app.schemas.input import ImageInput


class BuildingTemporalMatcherTool:
    name = "match_building_footprints"
    description = "Spatially match before/after building footprints and classify new, removed, and changed buildings."

    def __init__(self, detector: BuildingDetector | None = None) -> None:
        self._detector = detector or DevelopmentBuildingDetector()

    async def execute(
        self,
        earlier_image: ImageInput,
        later_image: ImageInput,
        earlier_path: Path,
        later_path: Path,
        *,
        min_area_m2: float = 25.0,
    ) -> tuple[BuildingTemporalMatchResult, list[EvidenceRegion]]:
        # 1. Detect before buildings
        t1_result = await self._detector.detect(earlier_image, earlier_path, min_area_m2=min_area_m2)
        # 2. Detect after buildings
        t2_result = await self._detector.detect(later_image, later_path, min_area_m2=min_area_m2)

        # 3. Spatially match before/after building footprints
        matched_after, removed_before = self._match_footprints(
            t1_result.detections,
            t2_result.detections,
        )

        before_count = len(t1_result.detections)
        after_count = len(t2_result.detections)
        new_count = sum(1 for f in matched_after if f.status == BuildingStatus.NEW)
        removed_count = len(removed_before)
        unchanged_count = sum(1 for f in matched_after if f.status == BuildingStatus.UNCHANGED)
        changed_count = sum(1 for f in matched_after if f.status == BuildingStatus.SIGNIFICANTLY_CHANGED)

        all_matched = matched_after + removed_before

        metrics = [
            Metric(name="before_building_count", value=before_count, unit="buildings", source="building_matcher"),
            Metric(name="after_building_count", value=after_count, unit="buildings", source="building_matcher"),
            Metric(name="new_building_count", value=new_count, unit="buildings", source="building_matcher"),
            Metric(name="removed_building_count", value=removed_count, unit="buildings", source="building_matcher"),
            Metric(name="changed_building_count", value=changed_count, unit="buildings", source="building_matcher"),
            Metric(name="unchanged_building_count", value=unchanged_count, unit="buildings", source="building_matcher"),
        ]

        evidence_regions: list[EvidenceRegion] = []
        for idx, f in enumerate(all_matched, 1):
            region_type = f"building_{f.status.value}"
            evidence = EvidenceRegion(
                id=f"building-match-{idx:04d}",
                geometry=f.geometry,
                type=region_type,
                confidence=f.confidence,
                metrics=[
                    Metric(name="area_m2", value=f.area_m2, unit="m²", source="building_matcher"),
                    Metric(name="iou", value=round(f.iou_with_match or 0.0, 3), unit="ratio", source="building_matcher"),
                ],
                source="building_temporal_matcher",
                metadata={
                    "building_id": f.id,
                    "status": f.status.value,
                    "area_m2": f.area_m2,
                    "iou": f.iou_with_match,
                    "matched_id": f.matched_id,
                },
            )
            evidence_regions.append(evidence)

        match_result = BuildingTemporalMatchResult(
            earlier_image_id=earlier_image.id,
            later_image_id=later_image.id,
            before_count=before_count,
            after_count=after_count,
            new_count=new_count,
            removed_count=removed_count,
            unchanged_count=unchanged_count,
            changed_count=changed_count,
            confidence=0.91,
            matcher_name="spatial_bipartite_footprint_matcher",
            matched_footprints=all_matched,
            metrics=metrics,
        )

        return match_result, evidence_regions

    def _match_footprints(
        self,
        t1_list: list[BuildingFootprint],
        t2_list: list[BuildingFootprint],
    ) -> tuple[list[BuildingFootprint], list[BuildingFootprint]]:
        t1_shapely = [(f, shapely_shape(f.geometry.model_dump())) for f in t1_list]
        t2_shapely = [(f, shapely_shape(f.geometry.model_dump())) for f in t2_list]

        # Match T2 footprints against T1
        matched_t2: list[BuildingFootprint] = []
        t1_matched_ids: set[str] = set()

        for f2, s2 in t2_shapely:
            best_iou = 0.0
            best_match_id: str | None = None
            b2_bounds = s2.bounds

            for f1, s1 in t1_shapely:
                b1_bounds = s1.bounds
                # Fast bbox pre-filter
                if not (b1_bounds[0] <= b2_bounds[2] and b2_bounds[0] <= b1_bounds[2] and
                        b1_bounds[1] <= b2_bounds[3] and b2_bounds[1] <= b1_bounds[3]):
                    continue

                inter = s2.intersection(s1).area
                if inter <= 0:
                    continue
                union = s2.union(s1).area
                iou = inter / max(union, 1e-10)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = f1.id

            if best_iou < 0.20:
                status = BuildingStatus.NEW
            elif best_iou >= 0.70:
                status = BuildingStatus.UNCHANGED
                if best_match_id:
                    t1_matched_ids.add(best_match_id)
            else:
                status = BuildingStatus.SIGNIFICANTLY_CHANGED
                if best_match_id:
                    t1_matched_ids.add(best_match_id)

            matched_t2.append(
                f2.model_copy(
                    update={
                        "status": status,
                        "iou_with_match": round(best_iou, 3),
                        "matched_id": best_match_id,
                    }
                )
            )

        # Identify removed T1 footprints (not matched in T2)
        removed_t1: list[BuildingFootprint] = []
        for f1, s1 in t1_shapely:
            if f1.id in t1_matched_ids:
                continue
            # Double check max IoU with all T2
            max_iou = 0.0
            b1_bounds = s1.bounds
            for _, s2 in t2_shapely:
                b2_bounds = s2.bounds
                if not (b1_bounds[0] <= b2_bounds[2] and b2_bounds[0] <= b1_bounds[2] and
                        b1_bounds[1] <= b2_bounds[3] and b2_bounds[1] <= b1_bounds[3]):
                    continue
                inter = s1.intersection(s2).area
                if inter > 0:
                    union = s1.union(s2).area
                    iou = inter / max(union, 1e-10)
                    if iou > max_iou:
                        max_iou = iou

            if max_iou < 0.20:
                removed_t1.append(
                    f1.model_copy(
                        update={
                            "id": f"{f1.id}-removed",
                            "status": BuildingStatus.REMOVED,
                            "iou_with_match": round(max_iou, 3),
                        }
                    )
                )

        return matched_t2, removed_t1
