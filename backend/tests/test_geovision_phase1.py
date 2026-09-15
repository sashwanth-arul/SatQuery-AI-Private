"""Tests for GeoVision Phase 1 endpoints and baseline contracts."""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

from app.main import app


def _create_png_bytes(width: int = 120, height: int = 80) -> bytes:
    img = Image.new("RGB", (width, height), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_jpg_bytes(width: int = 100, height: int = 100) -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 150, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _create_geotiff_bytes(width: int = 64, height: int = 64) -> bytes:
    buf = io.BytesIO()
    transform = from_bounds(77.5, 12.9, 77.6, 13.0, width, height)
    data = np.full((3, height, width), 100, dtype=np.uint8)
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(data[i], i + 1)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_geovision_upload_png():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        png_bytes = _create_png_bytes(150, 100)
        res = await client.post(
            "/api/v1/geovision/upload",
            files={"file": ("aerial_drone.png", png_bytes, "image/png")},
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["filename"] == "aerial_drone.png"
        assert data["format"] == "PNG"
        assert data["width"] == 150
        assert data["height"] == 100
        assert data["georeferenced"] is False
        assert "bytes" in data["preview_url"]


@pytest.mark.asyncio
async def test_geovision_upload_geotiff():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        tif_bytes = _create_geotiff_bytes(80, 80)
        res = await client.post(
            "/api/v1/geovision/upload",
            files={"file": ("ortho_scene.tif", tif_bytes, "image/tiff")},
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["width"] == 80
        assert data["height"] == 80
        assert data["georeferenced"] is True
        assert data["crs"] == "EPSG:4326"
        assert len(data["bounds"]) == 4


@pytest.mark.asyncio
async def test_geovision_upload_invalid_extension():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/v1/geovision/upload",
            files={"file": ("document.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "unsupported_format"


@pytest.mark.asyncio
async def test_geovision_get_image_bytes():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        png_bytes = _create_png_bytes(60, 40)
        up = await client.post(
            "/api/v1/geovision/upload",
            files={"file": ("small_drone.png", png_bytes, "image/png")},
        )
        image_id = up.json()["data"]["image_id"]

        bytes_res = await client.get(f"/api/v1/geovision/{image_id}/bytes")
        assert bytes_res.status_code == 200
        assert bytes_res.headers["content-type"].startswith("image/")
        assert len(bytes_res.content) > 0


@pytest.mark.asyncio
async def test_geovision_analyze_one_word():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        jpg_bytes = _create_jpg_bytes(100, 100)
        up = await client.post(
            "/api/v1/geovision/upload",
            files={"file": ("city.jpg", jpg_bytes, "image/jpeg")},
        )
        image_id = up.json()["data"]["image_id"]

        res = await client.post(
            "/api/v1/geovision/analyze",
            json={
                "image_id": image_id,
                "query": "Is this rural or urban? Answer in one word.",
            },
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["is_one_word"] is True
        assert data["detected_intent"] == "scene_classification"
        assert data["answer"] in ("Urban", "Rural")
        assert len(data["trace"]) == 12
        assert data["trace"][0]["name"] == "Image uploaded"
        assert data["trace"][11]["name"] in ("Analysis saved", "Result persisted")


@pytest.mark.asyncio
async def test_geovision_analyze_honest_provider_status():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        png_bytes = _create_png_bytes(100, 100)
        up = await client.post(
            "/api/v1/geovision/upload",
            files={"file": ("survey.png", png_bytes, "image/png")},
        )
        image_id = up.json()["data"]["image_id"]

        res = await client.post(
            "/api/v1/geovision/analyze",
            json={
                "image_id": image_id,
                "query": "How many cars are there?",
            },
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["detected_intent"] == "object_count"
        # Verify no fabricated fake detections
        assert len(data["detected_objects"]) == 0
        assert data["confidence_available"] is False
        assert "geochat" in data["provider_status"]
        assert "yolo_detector" in data["provider_status"]
