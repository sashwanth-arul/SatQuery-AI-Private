"""Unit and integration tests for GeoVision Professional Aerial Detection Upgrade."""

from __future__ import annotations

import io
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.adapters.geovision.detector import (
    AerialObjectDetectionProvider,
    get_aerial_detection_provider,
)
from app.adapters.geovision.evaluation import (
    get_model_metadata,
    load_evaluation_report,
)
from app.adapters.geovision.taxonomy import (
    is_aerial_area,
    is_aerial_object,
    normalize_class_name,
    validate_detection_class,
)
from app.adapters.geovision.tiling import (
    compute_iou,
    generate_tile_windows,
    map_tile_box_to_original,
    merge_tile_detections_nms,
    TileWindow,
)
from app.main import app


def _create_sample_png(width: int = 800, height: int = 600) -> bytes:
    img = Image.new("RGB", (width, height), color=(60, 80, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_taxonomy_discrete_objects_vs_area_classes():
    # Discrete objects must be recognized
    assert is_aerial_object("building") is True
    assert is_aerial_object("car") is True
    assert is_aerial_object("truck") is True
    assert is_aerial_object("aircraft") is True
    assert is_aerial_object("plane") is True  # Aliased

    # Area classes must be recognized
    assert is_aerial_area("water") is True
    assert is_aerial_area("road") is True
    assert is_aerial_area("vegetation") is True
    assert is_aerial_area("forest") is True

    # Rejection of area classes as object detections
    with pytest.raises(ValueError, match="cannot be treated as a discrete object"):
        validate_detection_class("water")

    with pytest.raises(ValueError, match="cannot be treated as a discrete object"):
        validate_detection_class("road")

    assert validate_detection_class("pedestrian") == "person"
    assert validate_detection_class("van") == "truck"


def test_tiling_geometry_and_coordinate_mapping():
    # 1200 x 800 image with 640 tile size and 0.2 overlap
    windows = generate_tile_windows(
        image_width=1200,
        image_height=800,
        tile_size=640,
        overlap_ratio=0.20,
    )
    assert len(windows) >= 4

    # Test coordinate mapping
    tile = TileWindow(tile_id="tile_r1_c1", x_offset=512, y_offset=200, width=640, height=600)
    # Box [10, 20, 100, 150] in tile
    mapped = map_tile_box_to_original(
        tile_box=[10, 20, 100, 150],
        tile=tile,
        image_width=1200,
        image_height=800,
    )
    assert mapped == [522.0, 220.0, 612.0, 350.0]

    # Test clamping on image border
    border_box = map_tile_box_to_original(
        tile_box=[600, 550, 750, 700],
        tile=tile,
        image_width=1200,
        image_height=800,
    )
    assert border_box[2] <= 1200.0
    assert border_box[3] <= 800.0

    # Inverted box should be rejected
    inv = map_tile_box_to_original(
        tile_box=[100, 100, 50, 50],
        tile=tile,
        image_width=1200,
        image_height=800,
    )
    assert inv is None


def test_tiling_iou_and_nms_deduplication():
    box1 = [100.0, 100.0, 200.0, 200.0]
    box2 = [110.0, 110.0, 210.0, 210.0]
    iou = compute_iou(box1, box2)
    assert 0.5 < iou < 1.0

    dets = [
        {"class_name": "building", "confidence": 0.92, "bbox": box1, "tile_id": "tile_1"},
        {"class_name": "building", "confidence": 0.81, "bbox": box2, "tile_id": "tile_2"},
        {"class_name": "car", "confidence": 0.88, "bbox": [300.0, 300.0, 350.0, 330.0], "tile_id": "tile_1"},
    ]

    merged = merge_tile_detections_nms(dets, iou_threshold=0.45)
    # The duplicate building should be suppressed, keeping the higher confidence one
    assert len(merged) == 2
    classes = [d["class_name"] for d in merged]
    assert "building" in classes
    assert "car" in classes
    building_det = next(d for d in merged if d["class_name"] == "building")
    assert building_det["confidence"] == 0.92


def test_provider_honest_unconfigured_status():
    provider = AerialObjectDetectionProvider(weights_path="/nonexistent/best.pt")
    assert provider.is_available is False

    img = Image.new("RGB", (300, 300), color="white")
    objects, summary, meta = provider.detect(img)
    assert objects == []
    assert summary == {}
    assert meta["status"] == "unavailable"
    assert "Trained detection model unavailable" in meta["reason"]


def test_evaluation_report_loader():
    report = load_evaluation_report()
    assert "global_metrics" in report
    assert "per_class_metrics" in report
    assert report["global_metrics"]["map50"] > 0.5
    assert "building" in report["per_class_metrics"]
    assert "car" in report["per_class_metrics"]

    metadata = get_model_metadata()
    assert metadata["model_name"] == "SatQuery-Aerial-YOLOv8"
    assert metadata["confidence_threshold"] == 0.25


@pytest.mark.asyncio
async def test_geovision_evaluation_metrics_api():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/geovision/evaluation-metrics")
        assert res.status_code == 200
        data = res.json()["data"]
        assert "global_metrics" in data
        assert "precision" in data["global_metrics"]
        assert "recall" in data["global_metrics"]
        assert "map50" in data["global_metrics"]
        assert "per_class_metrics" in data


@pytest.mark.asyncio
async def test_geovision_12_step_trace_and_honest_reporting():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        png_bytes = _create_sample_png(800, 600)
        up = await client.post(
            "/api/v1/geovision/upload",
            files={"file": ("unseen_aerial_test.png", png_bytes, "image/png")},
        )
        assert up.status_code == 200
        image_id = up.json()["data"]["image_id"]

        # 1. Test detailed description query
        res = await client.post(
            "/api/v1/geovision/analyze",
            json={
                "image_id": image_id,
                "query": "Describe this image in detail.",
                "intent": "detailed_description",
            },
        )
        assert res.status_code == 200
        data = res.json()["data"]

        # Check 12-step trace
        trace = data["trace"]
        assert len(trace) == 12
        step_names = [s["name"] for s in trace]
        assert step_names == [
            "Image uploaded",
            "File validated",
            "Image dimensions detected",
            "Image type identified",
            "Query intent classified",
            "Specialist selected",
            "Detection/segmentation executed",
            "Constraints applied",
            "Evidence generated",
            "Confidence calculated",
            "Grounded response generated",
            "Analysis saved",
        ]

        # In unconfigured environment for unseen image, detection step must be unavailable
        det_step = next(s for s in trace if s["name"] == "Detection/segmentation executed")
        assert det_step["status"] in ("skipped", "unavailable")

        # Check response structure
        assert "Scene:" in data["answer"] or "Scene" in data["answer"]

        # Check model metadata and evaluation metrics in response
        assert "model_metadata" in data
        assert "evaluation_metrics" in data
        assert data["provider_status"]["sam2_segmenter"] == "unavailable"
