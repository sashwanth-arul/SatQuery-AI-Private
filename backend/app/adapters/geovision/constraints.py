"""GeoVision Demonstration Constraints & Post-Processing Filtering.

Applies configurable size, area, and scientific validity rules.
Zero fabrication: never invents square meters or tree height when uncalibrated.
"""
from __future__ import annotations

from typing import Any
from app.schemas.geovision import DetectedObject


MIN_BUILDING_AREA_M2 = 100.0  # Required minimum area when georeferenced
MIN_BUILDING_PIXEL_AREA = 8000.0  # Minimum pixel area for demo image-space candidates
MIN_TREE_PIXEL_AREA = 1800.0  # Minimum pixel area for canopy cluster


def apply_building_constraints(
    raw_buildings: list[DetectedObject],
    *,
    georeferenced: bool,
    pixel_area_m2: float | None = None,
) -> list[DetectedObject]:
    """Filters building detections according to strict physical scale constraints.

    - Georeferenced: enforces area >= 100 m²; calculates exact area_m2.
    - Non-georeferenced (JPG/PNG): enforces minimum pixel area; labels as Large building candidate;
      strictly leaves area_m2 as None.
    """
    valid: list[DetectedObject] = []
    for bldg in raw_buildings:
        px_area = bldg.area_px or (bldg.bbox[2] - bldg.bbox[0]) * (bldg.bbox[3] - bldg.bbox[1])

        if georeferenced and pixel_area_m2 is not None and pixel_area_m2 > 0:
            area_m2 = round(px_area * pixel_area_m2, 1)
            if area_m2 < MIN_BUILDING_AREA_M2:
                continue  # Filter out structures under 100 m²
            bldg.area_m2 = area_m2
            bldg.size_class = "Large" if area_m2 >= 250 else "Medium"
            bldg.details = f"Verified structure ({area_m2} m²)"
        else:
            if px_area < MIN_BUILDING_PIXEL_AREA:
                continue
            bldg.area_m2 = None
            bldg.size_class = "Large" if px_area >= 25000 else "Medium"
            bldg.details = "Large building candidate detected in image coordinates; geographic area unavailable."

        valid.append(bldg)

    return valid


def apply_tree_constraints(
    raw_trees: list[DetectedObject],
) -> list[DetectedObject]:
    """Filters tree detections and prevents uncalibrated height claims.

    - Retains only sufficiently large tree / canopy clusters.
    - Labels explicitly as 'Large tree/canopy detected' without fabricating heights.
    """
    valid: list[DetectedObject] = []
    for tree in raw_trees:
        px_area = tree.area_px or (tree.bbox[2] - tree.bbox[0]) * (tree.bbox[3] - tree.bbox[1])
        if px_area < MIN_TREE_PIXEL_AREA:
            continue

        tree.size_class = "Large" if px_area >= 6000 else "Medium"
        tree.details = "Large tree/canopy detected"
        tree.area_m2 = None
        valid.append(tree)

    return valid


def format_water_body_feature(
    *,
    object_id: str,
    bbox: list[float],
    polygon_points: list[list[float]],
    pixel_area: float,
    coverage_percent: float,
    georeferenced: bool,
    pixel_area_m2: float | None = None,
    source_model: str = "Controlled-Demo-CV",
    score_type: str = "heuristic_score",
    score: float = 0.96,
) -> DetectedObject:
    """Formats a continuous water body as an area feature with polygon mask."""
    area_m2 = round(pixel_area * pixel_area_m2, 1) if (georeferenced and pixel_area_m2) else None
    details = (
        f"Water Body: {coverage_percent:.1f}% of image. Large continuous water region."
        if not area_m2
        else f"Water Body: {coverage_percent:.1f}% of image ({area_m2} m²). Large continuous water region."
    )

    return DetectedObject(
        id=object_id,
        class_name="water_body",
        confidence=score,
        score_type=score_type,
        bbox=bbox,
        segmentation_mask=polygon_points,
        area_px=float(pixel_area),
        area_m2=area_m2,
        evidence_type="polygon_mask",
        size_class="Large",
        details=details,
        properties={
            "coverage_percent": coverage_percent,
            "feature_type": "Large continuous water region",
            "evidence": "polygon mask",
        },
        source_model=source_model,
        model_version="v1.0-deterministic",
    )


def format_forest_feature(
    *,
    object_id: str,
    bbox: list[float],
    polygon_points: list[list[float]],
    pixel_area: float,
    coverage_percent: float,
    density: str,
    georeferenced: bool,
    pixel_area_m2: float | None = None,
    source_model: str = "Controlled-Demo-CV",
    score_type: str = "heuristic_score",
    score: float = 0.95,
) -> DetectedObject:
    """Formats forest canopy as a segmented area feature."""
    area_m2 = round(pixel_area * pixel_area_m2, 1) if (georeferenced and pixel_area_m2) else None
    details = f"Forest: {coverage_percent:.1f}% coverage. Density: {density}."

    return DetectedObject(
        id=object_id,
        class_name="forest",
        confidence=score,
        score_type=score_type,
        bbox=bbox,
        segmentation_mask=polygon_points,
        area_px=float(pixel_area),
        area_m2=area_m2,
        evidence_type="polygon_mask",
        size_class="Large",
        details=details,
        properties={
            "coverage_percent": coverage_percent,
            "density": density,
            "evidence": "segmentation mask",
        },
        source_model=source_model,
        model_version="v1.0-deterministic",
    )


def format_road_feature(
    *,
    object_id: str,
    bbox: list[float],
    pixel_length: float,
    polygon_points: list[list[float]] | None = None,
    georeferenced: bool = False,
    meters_per_pixel: float | None = None,
    source_model: str = "Controlled-Demo-CV",
    score_type: str = "heuristic_score",
    score: float = 0.90,
) -> DetectedObject:
    """Formats connected road segments."""
    length_m = round(pixel_length * meters_per_pixel, 1) if (georeferenced and meters_per_pixel) else None
    details = f"Connected road corridor (~{int(pixel_length)} px)" if not length_m else f"Connected road corridor (~{length_m} m)"

    return DetectedObject(
        id=object_id,
        class_name="road",
        confidence=score,
        score_type=score_type,
        bbox=bbox,
        segmentation_mask=polygon_points,
        area_px=float(pixel_length * 15.0),
        area_m2=None,
        evidence_type="linear_feature",
        size_class="Large",
        details=details,
        properties={
            "pixel_length": pixel_length,
            "length_m": length_m,
            "evidence": "connected linear network",
        },
        source_model=source_model,
        model_version="v1.0-deterministic",
    )
