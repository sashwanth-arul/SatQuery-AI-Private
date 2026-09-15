"""Test suite for SIH Agentic Remote Sensing Domain Specialists:
- Agriculture monitoring
- Disaster management (Flood inundation & change)
- Urban planning & built-up area
- Forest monitoring (Canopy cover)
- Water-resource assessment
- Infrastructure mapping
- Environmental land cover analysis
- Real raster-derived evidence overlay generation
- 9-facet execution trace emission
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.schemas.domain import QueryRequest
from app.schemas.input import ImageFormat, ImageInput, ImageModality
from app.schemas.planning import QueryIntent
from app.services.planner.deterministic import build_deterministic_plan
from app.services.query_controller import QueryController
from app.storage.factory import get_image_storage, get_metadata_registry
from app.tools.surface.domain_specialists import (
    FloodAnalysisTool,
    LandCoverAnalysisTool,
    SingleImageVegetationTool,
    SingleImageWaterTool,
    render_evidence_overlay,
)


def _create_synthetic_multispectral_geotiff(
    path: Path,
    *,
    width: int = 100,
    height: int = 100,
    bounds: tuple[float, float, float, float] = (55.10, 25.10, 55.15, 25.15),
    water_box: tuple[int, int, int, int] | None = (10, 10, 30, 40),
    agri_box: tuple[int, int, int, int] | None = (40, 10, 60, 40),
    forest_box: tuple[int, int, int, int] | None = (10, 60, 40, 90),
    built_box: tuple[int, int, int, int] | None = (60, 60, 90, 90),
) -> None:
    """
    Create a 5-band synthetic Sentinel-2 raster:
    B1: Blue, B2: Green, B3: Red, B4: NIR, B5: SWIR
    """
    # Background bare ground
    data = np.full((5, height, width), 60, dtype=np.uint16)
    data[0] = 50   # Blue
    data[1] = 60   # Green
    data[2] = 70   # Red
    data[3] = 80   # NIR
    data[4] = 90   # SWIR

    # Water: High Green/Blue, very low NIR/SWIR (NDWI > 0)
    if water_box:
        r1, c1, r2, c2 = water_box
        data[0, r1:r2, c1:c2] = 120  # Blue
        data[1, r1:r2, c1:c2] = 150  # Green
        data[2, r1:r2, c1:c2] = 40   # Red
        data[3, r1:r2, c1:c2] = 10   # NIR (low)
        data[4, r1:r2, c1:c2] = 5    # SWIR (low)

    # Agriculture: Moderate Green, high NIR, low Red (0.25 <= NDVI <= 0.55)
    if agri_box:
        r1, c1, r2, c2 = agri_box
        data[0, r1:r2, c1:c2] = 30
        data[1, r1:r2, c1:c2] = 90
        data[2, r1:r2, c1:c2] = 30
        data[3, r1:r2, c1:c2] = 90   # NIR
        data[4, r1:r2, c1:c2] = 40

    # Forest: Low Red, very high NIR (NDVI > 0.60)
    if forest_box:
        r1, c1, r2, c2 = forest_box
        data[0, r1:r2, c1:c2] = 20
        data[1, r1:r2, c1:c2] = 80
        data[2, r1:r2, c1:c2] = 20   # Low Red
        data[3, r1:r2, c1:c2] = 200  # High NIR
        data[4, r1:r2, c1:c2] = 30

    # Built-up: High SWIR, moderate NIR, bright visible (NDBI > 0.05)
    if built_box:
        r1, c1, r2, c2 = built_box
        data[0, r1:r2, c1:c2] = 140
        data[1, r1:r2, c1:c2] = 110
        data[2, r1:r2, c1:c2] = 140
        data[3, r1:r2, c1:c2] = 120
        data[4, r1:r2, c1:c2] = 190  # High SWIR (NDBI > 0.05)

    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=5,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for b in range(5):
            dst.write(data[b], b + 1)


@pytest.fixture
def multispectral_raster(tmp_path: Path):
    tif_path = tmp_path / "scene_5band.tif"
    _create_synthetic_multispectral_geotiff(tif_path)
    image = ImageInput(
        id="a1" * 16,
        filename="scene_5band.tif",
        file_size_bytes=tif_path.stat().st_size,
        format=ImageFormat.GEOTIFF,
        modality=ImageModality.OPTICAL,
        width=100,
        height=100,
        crs="EPSG:4326",
        bounds=[55.10, 25.10, 55.15, 25.15],
        georeferenced=True,
        acquisition_datetime=datetime(2024, 3, 15, 10, 0, 0),
    )
    return tif_path, image


@pytest.mark.asyncio
async def test_single_image_water_tool(multispectral_raster):
    tif_path, image = multispectral_raster
    tool = SingleImageWaterTool()
    stats, evidence_regions, overlay = await tool.execute(image, tif_path)

    assert stats["domain"] == "water"
    assert stats["water_area_m2"] > 0
    assert stats["water_percentage"] > 0
    assert len(evidence_regions) >= 1
    assert evidence_regions[0].type == "water_body"
    assert overlay.data_uri.startswith("data:image/png;base64,")
    assert "water" in overlay.supported_layers


@pytest.mark.asyncio
async def test_single_image_vegetation_tool(multispectral_raster):
    tif_path, image = multispectral_raster
    tool = SingleImageVegetationTool()

    # Agriculture mode
    stats_agri, regions_agri, overlay_agri = await tool.execute(image, tif_path, mode="agriculture")
    assert stats_agri["agricultural_area_m2"] > 0
    assert len(regions_agri) >= 1
    assert regions_agri[0].type == "agricultural_parcel"

    # Forest mode
    stats_forest, regions_forest, overlay_forest = await tool.execute(image, tif_path, mode="forest")
    assert stats_forest["forest_area_m2"] > 0
    assert len(regions_forest) >= 1
    assert regions_forest[0].type == "forest_canopy"


@pytest.mark.asyncio
async def test_land_cover_tool(multispectral_raster):
    tif_path, image = multispectral_raster
    tool = LandCoverAnalysisTool()
    stats, evidence_regions, overlay = await tool.execute(image, tif_path)

    assert stats["domain"] == "environmental"
    assert stats["water_percentage"] > 0
    assert stats["forest_percentage"] > 0
    assert stats["agricultural_percentage"] > 0
    assert stats["built_up_percentage"] > 0
    assert len(evidence_regions) >= 4
    assert overlay.data_uri.startswith("data:image/png;base64,")


def test_planner_domain_intent_routing():
    """Verify deterministic routing for all 7 application domains."""
    queries_and_expected = [
        ("Show water bodies in this scene", QueryIntent.WATER_DETECTION),
        ("What is the water extent?", QueryIntent.WATER_DETECTION),
        ("Show agricultural areas in this region", QueryIntent.AGRICULTURE_MONITORING),
        ("What is the vegetation condition?", QueryIntent.AGRICULTURE_MONITORING),
        ("Show forested areas", QueryIntent.FOREST_MONITORING),
        ("Show flooded areas after the cyclone", QueryIntent.FLOOD_ANALYSIS),
        ("How many buildings are there?", QueryIntent.BUILDING_COUNT),
        ("Show buildings and infrastructure", QueryIntent.INFRASTRUCTURE_MAPPING),
        ("Describe the land cover", QueryIntent.LAND_COVER_ANALYSIS),
        ("Classify the land cover", QueryIntent.LAND_COVER_ANALYSIS),
    ]

    for q, expected_intent in queries_and_expected:
        plan = build_deterministic_plan(QueryRequest(query=q, image_id="img-1"))
        assert plan.user_intent == expected_intent, f"Query '{q}' routed to {plan.user_intent}, expected {expected_intent}"


@pytest.mark.asyncio
async def test_end_to_end_domain_analysis_with_trace(multispectral_raster, monkeypatch):
    tif_path, image = multispectral_raster

    registry = get_metadata_registry()
    storage = get_image_storage()
    registry.save(image)
    with open(tif_path, "rb") as f:
        storage.save(image.id, ".tif", f)

    controller = QueryController()

    try:
        # Query 1: Water resource assessment
        req_water = QueryRequest(query="Show water bodies", image_id=image.id)
        res_water = await controller.submit(req_water)
        assert res_water.status.value == "completed"
        assert res_water.domain == "water_resources"
        assert res_water.evidence_overlay is not None
        assert len(res_water.evidence) >= 1
        assert "water resource assessment" in res_water.answer.lower()

        # Verify 9-facet trace steps
        tool_names = [step.tool_name for step in res_water.trace]
        assert "input_validation" in tool_names
        assert "plan_query" in tool_names
        assert "select_specialist" in tool_names
        assert "geospatial_processing" in tool_names
        assert "calculate_statistics" in tool_names
        assert "generate_evidence" in tool_names
        assert "grounded_answer" in tool_names

        # Query 2: Agriculture monitoring
        req_agri = QueryRequest(query="Show agricultural areas in this region", image_id=image.id)
        res_agri = await controller.submit(req_agri)
        assert res_agri.status.value == "completed"
        assert res_agri.domain == "agriculture"
        assert "agricultural" in res_agri.answer.lower()

        # Query 3: Multi-class land cover analysis
        req_lc = QueryRequest(query="Describe the land cover", image_id=image.id)
        res_lc = await controller.submit(req_lc)
        assert res_lc.status.value == "completed"
        assert res_lc.domain == "environmental_analysis"
        assert "water:" in res_lc.answer.lower()
        assert "forest:" in res_lc.answer.lower()
    finally:
        if registry.exists(image.id):
            registry.delete(image.id)
        if storage.exists(image.id, ".tif"):
            storage.delete(image.id, ".tif")
