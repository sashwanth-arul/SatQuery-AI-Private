"""Automated tests for geospatial coordinate reading, CRS transformation, and cross-modal evidence rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.imagery.uploaded.metadata import probe_tiff
from app.evidence.geometry import pixel_polygon_to_wgs84, pixel_to_wgs84
from app.main import app
from app.storage.factory import get_image_storage, get_metadata_registry
from app.adapters.imagery.uploaded.factory import get_uploaded_imagery_provider
from app.core.config import get_settings
from tests.fixtures.rasters import write_geotiff, write_utm_geotiff


@pytest.fixture
def clean_storage(tmp_path: Path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("UPLOAD_DIR", str(root))
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "8")
    monkeypatch.setenv("GEOCHAT_VQA_PROVIDER", "development")
    get_settings.cache_clear()

    get_image_storage.cache_clear()
    get_metadata_registry.cache_clear()
    get_uploaded_imagery_provider.cache_clear()
    yield root
    get_image_storage.cache_clear()
    get_metadata_registry.cache_clear()
    get_uploaded_imagery_provider.cache_clear()
    get_settings.cache_clear()


@pytest.fixture
async def client(clean_storage):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def test_probe_tiff_epsg4326_metadata(tmp_path: Path):
    path = tmp_path / "epsg4326.tif"
    write_geotiff(path, width=64, height=48, origin_lon=80.28, origin_lat=13.08, pixel_size=0.001)

    probe = probe_tiff(path)
    assert probe.width == 64
    assert probe.height == 48
    assert probe.band_count == 1
    assert probe.crs == "EPSG:4326"
    assert probe.georeferenced is True
    assert probe.transform is not None
    assert len(probe.transform) == 6
    assert probe.bounds is not None
    min_lon, min_lat, max_lon, max_lat = probe.bounds
    assert min_lon < max_lon
    assert min_lat < max_lat
    assert 80.0 <= min_lon <= 81.0
    assert 13.0 <= min_lat <= 14.0


def test_probe_tiff_utm_transformation_to_epsg4326(tmp_path: Path):
    path = tmp_path / "chennai_utm.tif"
    # St. George Fort, Chennai: UTM Zone 44N (EPSG:32644)
    # Approx easting 422000 m, northing 1446000 m corresponds to lon ~80.28 deg, lat ~13.08 deg
    write_utm_geotiff(
        path,
        width=128,
        height=128,
        origin_x=422000.0,
        origin_y=1446000.0,
        pixel_size=10.0,
        epsg=32644,
        bands=3,
    )

    probe = probe_tiff(path)
    assert probe.width == 128
    assert probe.height == 128
    assert probe.band_count == 3
    assert probe.crs == "EPSG:32644"
    assert probe.georeferenced is True
    assert probe.transform is not None
    assert probe.native_bounds is not None
    assert probe.native_bounds[0] == 422000.0

    # Bounds must be converted to EPSG:4326
    assert probe.bounds is not None
    min_lon, min_lat, max_lon, max_lat = probe.bounds
    # Must be valid geographic coordinates, NOT UTM meter values!
    assert -180.0 <= min_lon <= 180.0
    assert -90.0 <= min_lat <= 90.0
    assert -180.0 <= max_lon <= 180.0
    assert -90.0 <= max_lat <= 90.0
    assert min_lon < max_lon
    assert min_lat < max_lat
    # Chennai longitude ~80.28, latitude ~13.08
    assert 80.2 <= min_lon <= 80.4
    assert 13.0 <= min_lat <= 13.2


def test_pixel_to_wgs84_helpers(tmp_path: Path):
    path = tmp_path / "utm_grid.tif"
    write_utm_geotiff(path, origin_x=422000.0, origin_y=1446000.0, pixel_size=10.0, epsg=32644)
    probe = probe_tiff(path)

    # Pixel center
    lon, lat = pixel_to_wgs84(32.0, 32.0, transform=probe.transform, crs=probe.crs)
    assert 80.25 <= lon <= 80.35
    assert 13.05 <= lat <= 13.15

    # Pixel polygon ring
    px_coords = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0)]
    wgs_ring = pixel_polygon_to_wgs84(px_coords, transform=probe.transform, crs=probe.crs)
    assert len(wgs_ring) == 5
    assert wgs_ring[0] == wgs_ring[-1]
    for pt in wgs_ring:
        assert len(pt) == 2
        # [longitude, latitude]
        assert 80.25 <= pt[0] <= 80.35
        assert 13.05 <= pt[1] <= 13.15


@pytest.mark.asyncio
async def test_cross_modal_utm_and_wgs84_evidence_rendering(client, clean_storage: Path):
    # Create optical in UTM 44N (Chennai Fort St. George)
    optical_path = clean_storage / "chennai_optical_utm.tif"
    write_utm_geotiff(
        optical_path,
        width=128,
        height=128,
        origin_x=422000.0,
        origin_y=1446000.0,
        pixel_size=10.0,
        epsg=32644,
        bands=3,
    )

    probe_opt = probe_tiff(optical_path)
    opt_bounds = probe_opt.bounds
    assert opt_bounds is not None

    # Create SAR in EPSG:4326 covering overlapping extent
    sar_path = clean_storage / "chennai_sar_4326.tif"
    write_geotiff(
        sar_path,
        width=128,
        height=128,
        origin_lon=opt_bounds[0],
        origin_lat=opt_bounds[3],
        pixel_size=(opt_bounds[2] - opt_bounds[0]) / 128.0,
        epsg=4326,
        bands=1,
    )

    with optical_path.open("rb") as f:
        res_opt = await client.post(
            "/api/v1/imagery/upload",
            files={"file": ("chennai_optical_utm.tif", f, "image/tiff")},
            data={"modality": "optical"},
        )
    assert res_opt.status_code == 200
    opt_id = res_opt.json()["data"]["image"]["id"]
    opt_data = res_opt.json()["data"]["image"]
    assert opt_data["crs"] == "EPSG:32644"
    assert opt_data["bounds"] is not None
    assert opt_data["bounds"][0] < opt_data["bounds"][2]
    # Ensure not a thin line (width should be proper, not 3)
    assert opt_data["width"] == 128
    assert opt_data["height"] == 128

    with sar_path.open("rb") as f:
        res_sar = await client.post(
            "/api/v1/imagery/upload",
            files={"file": ("chennai_sar_4326.tif", f, "image/tiff")},
            data={"modality": "sar"},
        )
    assert res_sar.status_code == 200
    sar_id = res_sar.json()["data"]["image"]["id"]

    # Submit cross-modal query
    submit_res = await client.post(
        "/api/v1/query/submit",
        json={
            "query": "Identify built-up and water regions using optical and SAR.",
            "optical_image_id": opt_id,
            "sar_image_id": sar_id,
        },
    )
    assert submit_res.status_code == 200
    result = submit_res.json()["data"]["result"]
    assert result["status"] == "completed"

    # Verify debug information in inference_metadata
    cross_modal = result["cross_modal"]
    debug_meta = cross_modal.get("inference_metadata", {}).get("debug_geospatial")
    assert debug_meta is not None
    assert debug_meta["optical_crs"] == "EPSG:32644"
    assert debug_meta["sar_crs"] == "EPSG:4326"
    assert debug_meta["overlap_bounds"] is not None
    assert len(debug_meta["overlap_bounds"]) == 4

    # Verify evidence regions and GeoJSON coordinates
    evidence_regions = result["evidence"]
    assert len(evidence_regions) > 0

    for region in evidence_regions:
        geom = region["geometry"]
        assert geom["type"] == "Polygon"
        coords = geom["coordinates"][0]
        assert len(coords) >= 4
        # Closed polygon
        assert coords[0] == coords[-1]
        for pt in coords:
            assert len(pt) == 2
            lon, lat = pt[0], pt[1]
            # Must be valid longitude/latitude in Chennai area
            assert 80.0 <= lon <= 81.0, f"Expected longitude in [80, 81], got {lon}"
            assert 13.0 <= lat <= 14.0, f"Expected latitude in [13, 14], got {lat}"

        # Check that evidence is NOT a thin line
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        poly_width = max(lons) - min(lons)
        poly_height = max(lats) - min(lats)
        assert poly_width > 0.00005, f"Polygon width too small (thin line bug): {poly_width}"
        assert poly_height > 0.00005, f"Polygon height too small: {poly_height}"
