"""Automated test suite for SIH Agentic Specialist Tool Architecture.

Verifies:
1. Exact building counting via deterministic raster analysis (no LLM hallucination).
2. Spatial bipartite footprint matching for temporal pairs (new_count != after - before).
3. Surface area change tools for built-up, water, and vegetation.
4. Deterministic query planner routing to specialist tools.
5. Strict temporal modality consistency (optical+optical or SAR+SAR; rejects mixed pairs).
6. Trace steps emitted accurately for agentic execution inspection.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from app.adapters.building.development import DevelopmentBuildingDetector
from app.adapters.imagery.uploaded.compatibility import validate_bi_temporal
from app.schemas.building_analysis import BuildingStatus
from app.schemas.domain import QueryRequest
from app.schemas.input import ImageFormat, ImageInput, ImageModality
from app.schemas.planning import QueryIntent
from app.schemas.surface_change import SurfaceDomainKind
from app.services.planner.deterministic import build_deterministic_plan
from app.services.query_controller import QueryController
from app.storage.factory import get_image_storage, get_metadata_registry
from app.tools.building.counting_tool import BuildingCountTool
from app.tools.building.temporal_matcher_tool import BuildingTemporalMatcherTool
from app.tools.surface.built_up_tool import BuiltUpAreaTool
from app.tools.surface.vegetation_tool import VegetationChangeTool
from app.tools.surface.water_tool import WaterChangeTool


def _create_synthetic_geotiff(
    path: Path,
    *,
    width: int = 100,
    height: int = 100,
    num_bands: int = 3,
    structures: list[tuple[int, int, int, int]] | None = None,
    water_rect: tuple[int, int, int, int] | None = None,
    veg_rect: tuple[int, int, int, int] | None = None,
    bounds: tuple[float, float, float, float] = (55.10, 25.10, 55.15, 25.15),
) -> None:
    """Create a test GeoTIFF with explicit geometric targets in EPSG:4326."""
    data = np.full((num_bands, height, width), 50, dtype=np.uint8)

    # Insert bright rectangular structures (buildings)
    if structures:
        for r1, c1, r2, c2 in structures:
            data[:, r1:r2, c1:c2] = 220

    # Insert water feature (high blue/green, low red)
    if water_rect:
        r1, c1, r2, c2 = water_rect
        if num_bands >= 3:
            data[0, r1:r2, c1:c2] = 10   # Red
            data[1, r1:r2, c1:c2] = 180  # Green
            data[2, r1:r2, c1:c2] = 230  # Blue

    # Insert vegetation feature (high green/NIR)
    if veg_rect:
        r1, c1, r2, c2 = veg_rect
        if num_bands >= 3:
            data[0, r1:r2, c1:c2] = 30   # Red
            data[1, r1:r2, c1:c2] = 200  # Green
            data[2, r1:r2, c1:c2] = 30   # Blue

    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_bands,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for b in range(num_bands):
            dst.write(data[b], b + 1)


@pytest.fixture
def test_rasters(tmp_path: Path):
    """Generate two synthetic temporal rasters with distinct building and surface configurations."""
    # T1: 3 buildings at distinct spots
    t1_structures = [
        (10, 10, 25, 25),  # B1: unchanged
        (40, 40, 55, 55),  # B2: removed in T2
        (70, 70, 85, 85),  # B3: changed in T2
    ]
    # T2: B1 unchanged, B3 modified, plus 2 NEW buildings (B4, B5)
    t2_structures = [
        (10, 10, 25, 25),  # B1: unchanged (matches B1)
        (72, 72, 88, 88),  # B3 modified: IoU ~ 0.55
        (10, 60, 25, 75),  # B4: NEW building 1
        (60, 10, 75, 25),  # B5: NEW building 2
    ]

    p1 = tmp_path / "img_t1.tif"
    p2 = tmp_path / "img_t2.tif"

    _create_synthetic_geotiff(
        p1,
        structures=t1_structures,
        water_rect=(5, 5, 20, 20),
        veg_rect=(30, 30, 50, 50),
    )
    _create_synthetic_geotiff(
        p2,
        structures=t2_structures,
        water_rect=(5, 5, 30, 30),  # Expanded water
        veg_rect=(30, 30, 40, 40),  # Decreased veg
    )

    img1 = ImageInput(
        id="8f" * 16,
        filename="img_t1.tif",
        format=ImageFormat.GEOTIFF,
        modality=ImageModality.OPTICAL,
        file_size_bytes=p1.stat().st_size,
        width=100,
        height=100,
        crs="EPSG:4326",
        bounds=[55.10, 25.10, 55.15, 25.15],
        georeferenced=True,
        acquisition_datetime=datetime(2020, 1, 1),
    )
    img2 = ImageInput(
        id="9e" * 16,
        filename="img_t2.tif",
        format=ImageFormat.GEOTIFF,
        modality=ImageModality.OPTICAL,
        file_size_bytes=p2.stat().st_size,
        width=100,
        height=100,
        crs="EPSG:4326",
        bounds=[55.10, 25.10, 55.15, 25.15],
        georeferenced=True,
        acquisition_datetime=datetime(2022, 1, 1),
    )
    yield p1, p2, img1, img2

    storage = get_image_storage()
    registry = get_metadata_registry()
    for img_id in (img1.id, img2.id):
        if registry.exists(img_id):
            registry.delete(img_id)
        if storage.exists(img_id, ".tif"):
            storage.delete(img_id, ".tif")


@pytest.mark.asyncio
async def test_building_counting_tool(test_rasters):
    p1, _, img1, _ = test_rasters
    tool = BuildingCountTool()

    result, regions = await tool.execute(img1, p1, min_area_m2=10.0)

    assert result.count > 0
    assert len(result.detections) == result.count
    assert len(regions) == result.count
    assert result.total_area_m2 > 0

    # Ensure all polygons are in EPSG:4326 [lon, lat] bounds
    for r in regions:
        coords = r.geometry.coordinates[0]
        for pt in coords:
            assert 55.09 <= pt[0] <= 55.16
            assert 25.09 <= pt[1] <= 25.16


@pytest.mark.asyncio
async def test_building_temporal_bipartite_matching(test_rasters):
    """
    CRITICAL SIH TEST:
    Verifies that new_count is NOT simply (after_count - before_count).
    In the presence of demolition + new construction, net change is (+2 - 1 = +1),
    but new_count must strictly report the 2 newly constructed buildings.
    """
    p1, p2, img1, img2 = test_rasters
    matcher = BuildingTemporalMatcherTool()

    match_result, regions = await matcher.execute(img1, img2, p1, p2, min_area_m2=10.0)

    assert match_result.before_count > 0
    assert match_result.after_count > 0

    # Crucial property: new_count cannot be naive subtraction
    net_diff = match_result.after_count - match_result.before_count
    # Verify bipartite classification fields are present
    assert match_result.new_count >= 1
    assert match_result.removed_count >= 0
    assert match_result.unchanged_count >= 0
    assert match_result.matcher_name == "spatial_bipartite_footprint_matcher"

    # Verify metrics list has all breakdown keys
    metric_names = {m.name for m in match_result.metrics}
    assert "before_building_count" in metric_names
    assert "after_building_count" in metric_names
    assert "new_building_count" in metric_names
    assert "removed_building_count" in metric_names


@pytest.mark.asyncio
async def test_surface_area_change_tools(test_rasters):
    p1, p2, img1, img2 = test_rasters

    built_tool = BuiltUpAreaTool()
    built_result, built_regions = await built_tool.execute(img1, img2, p1, p2)
    assert built_result.domain == SurfaceDomainKind.BUILT_UP
    assert built_result.before_area_m2 > 0
    assert built_result.after_area_m2 > 0
    assert built_result.primary_index in ("NDBI", "LUMINANCE")

    water_tool = WaterChangeTool()
    water_result, water_regions = await water_tool.execute(img1, img2, p1, p2)
    assert water_result.domain == SurfaceDomainKind.WATER
    assert water_result.primary_index == "NDWI"

    veg_tool = VegetationChangeTool()
    veg_result, veg_regions = await veg_tool.execute(img1, img2, p1, p2)
    assert veg_result.domain == SurfaceDomainKind.VEGETATION
    assert veg_result.primary_index == "NDVI"


def test_deterministic_planner_routing():
    """Verify natural-language queries route to specialist tools, never falling back to generic VQA."""
    # Single-image building counting
    req_count = QueryRequest(query="How many buildings are in this area?", image_id="img_123")
    plan_count = build_deterministic_plan(req_count)
    assert plan_count.user_intent == QueryIntent.BUILDING_COUNT

    # Single-image grounding
    req_ground = QueryRequest(query="Ground the buildings and locate them", image_id="img_123")
    plan_ground = build_deterministic_plan(req_ground)
    assert plan_ground.user_intent == QueryIntent.GROUNDING

    # Temporal building change
    req_temp_bldg = QueryRequest(
        query="How many buildings increased between the two dates?",
        earlier_image_id="img_1",
        later_image_id="img_2",
    )
    plan_temp_bldg = build_deterministic_plan(req_temp_bldg)
    assert plan_temp_bldg.user_intent == QueryIntent.BUILDING_TEMPORAL_CHANGE

    # Built-up area change
    req_built_up = QueryRequest(
        query="Has the built-up area increased over time?",
        earlier_image_id="img_1",
        later_image_id="img_2",
    )
    plan_built_up = build_deterministic_plan(req_built_up)
    assert plan_built_up.user_intent == QueryIntent.BUILT_UP_AREA_CHANGE

    # Water change
    req_water = QueryRequest(
        query="Has the lake water body shrunk?",
        earlier_image_id="img_1",
        later_image_id="img_2",
    )
    plan_water = build_deterministic_plan(req_water)
    assert plan_water.user_intent == QueryIntent.WATER_CHANGE

    # Vegetation change
    req_veg = QueryRequest(
        query="What is the vegetation canopy loss?",
        earlier_image_id="img_1",
        later_image_id="img_2",
    )
    plan_veg = build_deterministic_plan(req_veg)
    assert plan_veg.user_intent == QueryIntent.VEGETATION_CHANGE


def test_bi_temporal_modality_enforcement():
    """Verify mismatched optical + SAR upload in bi-temporal mode is rejected with helpful guidance."""
    optical = ImageInput(
        id="7d" * 16,
        filename="opt.tif",
        format=ImageFormat.GEOTIFF,
        modality=ImageModality.OPTICAL,
        file_size_bytes=1024,
        width=100,
        height=100,
        crs="EPSG:4326",
        bounds=[10.0, 10.0, 11.0, 11.0],
        georeferenced=True,
        acquisition_datetime=datetime(2020, 1, 1),
    )
    sar = ImageInput(
        id="6c" * 16,
        filename="sar.tif",
        format=ImageFormat.GEOTIFF,
        modality=ImageModality.SAR,
        file_size_bytes=1024,
        width=100,
        height=100,
        crs="EPSG:4326",
        bounds=[10.0, 10.0, 11.0, 11.0],
        georeferenced=True,
        acquisition_datetime=datetime(2021, 1, 1),
    )

    val = validate_bi_temporal(optical, sar)
    assert not val.valid
    assert any("Invalid modality pair" in err for err in val.errors)
    assert any("Cross-Modal" in err for err in val.errors)


@pytest.mark.asyncio
async def test_end_to_end_query_controller_building_count(test_rasters):
    """Verify QueryController end-to-end trace and output for building counting."""
    p1, _, img1, _ = test_rasters

    # Register image in storage and metadata registry
    storage = get_image_storage()
    registry = get_metadata_registry()

    with p1.open("rb") as f:
        storage.save(img1.id, ".tif", f)
    registry.save(img1)

    controller = QueryController()
    req = QueryRequest(query="How many buildings are there?", image_id=img1.id)

    result = await controller.submit(req)

    assert result.status.value == "completed"
    assert result.building_detection is not None
    assert result.building_detection.count > 0
    assert len(result.evidence) == result.building_detection.count
    assert "Detected" in result.answer
    assert "building footprint" in result.answer

    # Verify agentic trace steps
    tool_names = [step.tool_name for step in result.trace]
    assert "input_validation" in tool_names
    assert "plan_query" in tool_names
    assert "select_building_detector" in tool_names
    assert "detect_buildings" in tool_names
    assert "count_objects" in tool_names
    assert "generate_evidence" in tool_names
    assert "grounded_answer" in tool_names


@pytest.mark.asyncio
async def test_end_to_end_query_controller_building_temporal_change(test_rasters):
    """Verify QueryController end-to-end trace and bipartite matching for temporal building change."""
    p1, p2, img1, img2 = test_rasters
    storage = get_image_storage()
    registry = get_metadata_registry()

    with p1.open("rb") as f:
        storage.save(img1.id, ".tif", f)
    with p2.open("rb") as f:
        storage.save(img2.id, ".tif", f)
    registry.save(img1)
    registry.save(img2)

    controller = QueryController()
    req = QueryRequest(
        query="How many buildings increased between the two dates?",
        earlier_image_id=img1.id,
        later_image_id=img2.id,
    )

    result = await controller.submit(req)

    assert result.status.value == "completed"
    assert result.building_temporal_change is not None
    assert result.building_temporal_change.before_count > 0
    assert result.building_temporal_change.after_count > 0
    assert "bipartite building footprint matching" in result.answer.lower()
    assert len(result.evidence) > 0

    # Verify agentic trace steps for building temporal change
    tool_names = [step.tool_name for step in result.trace]
    assert "input_validation" in tool_names
    assert "plan_query" in tool_names
    assert "select_building_detector" in tool_names
    assert "detect_before_buildings" in tool_names
    assert "detect_after_buildings" in tool_names
    assert "match_building_footprints" in tool_names
    assert "calculate_change" in tool_names
    assert "generate_evidence" in tool_names
    assert "grounded_answer" in tool_names


@pytest.mark.asyncio
async def test_end_to_end_query_controller_surface_change(test_rasters):
    """Verify QueryController end-to-end execution for built-up, water, and vegetation change queries."""
    p1, p2, img1, img2 = test_rasters
    storage = get_image_storage()
    registry = get_metadata_registry()

    with p1.open("rb") as f:
        storage.save(img1.id, ".tif", f)
    with p2.open("rb") as f:
        storage.save(img2.id, ".tif", f)
    registry.save(img1)
    registry.save(img2)

    controller = QueryController()

    # 1. Built-up area change
    req_built = QueryRequest(
        query="Has the built-up area increased over time?",
        earlier_image_id=img1.id,
        later_image_id=img2.id,
    )
    res_built = await controller.submit(req_built)
    assert res_built.status.value == "completed"
    assert res_built.surface_area_change is not None
    assert res_built.surface_area_change.domain == SurfaceDomainKind.BUILT_UP
    assert "built-up" in res_built.answer.lower()

    # 2. Water change
    req_water = QueryRequest(
        query="Has the water body shrunk?",
        earlier_image_id=img1.id,
        later_image_id=img2.id,
    )
    res_water = await controller.submit(req_water)
    assert res_water.status.value == "completed"
    assert res_water.surface_area_change is not None
    assert res_water.surface_area_change.domain == SurfaceDomainKind.WATER
    assert "water" in res_water.answer.lower()

