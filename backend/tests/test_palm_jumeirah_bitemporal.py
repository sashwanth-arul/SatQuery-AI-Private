"""End-to-end test for Palm Jumeirah Sentinel-2 bi-temporal change detection."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app

DOWNLOADS = Path(r"C:\Users\it212\Downloads")
FILE_EARLIER = DOWNLOADS / "2016-09-06-00_00_2016-09-06-23_59_Sentinel-2_L2A_True_color.tiff"
FILE_LATER = DOWNLOADS / "2026-09-09-00_00_2026-09-09-23_59_Sentinel-2_L2A_True_color.tiff"


@pytest.mark.skipif(
    not (FILE_EARLIER.exists() and FILE_LATER.exists()),
    reason="Palm Jumeirah Sentinel-2 files not present in Downloads",
)
def test_palm_jumeirah_temporal_pair_end_to_end():
    client = TestClient(app)

    # 1. Upload earlier image
    with FILE_EARLIER.open("rb") as fp:
        res1 = client.post(
            "/api/v1/imagery/upload",
            files={"file": (FILE_EARLIER.name, fp, "image/tiff")},
            data={"modality": "optical", "acquisition_datetime": "2023-01-01T00:00:00+00:00"},
        )
    assert res1.status_code == 200, res1.text
    img1 = res1.json()["data"]["image"]
    assert img1["id"]
    # Filename date extraction overrides UI default
    assert img1["acquisition_datetime"].startswith("2016-09-06")

    # 2. Upload later image
    with FILE_LATER.open("rb") as fp:
        res2 = client.post(
            "/api/v1/imagery/upload",
            files={"file": (FILE_LATER.name, fp, "image/tiff")},
            data={"modality": "optical", "acquisition_datetime": "2024-01-01T00:00:00+00:00"},
        )
    assert res2.status_code == 200, res2.text
    img2 = res2.json()["data"]["image"]
    assert img2["id"]
    assert img2["acquisition_datetime"].startswith("2026-09-09")

    # 3. Submit temporal pair query
    qres = client.post(
        "/api/v1/query/submit",
        json={
            "query": "Detect land-cover change in Palm Jumeirah Dubai",
            "earlier_image_id": img1["id"],
            "later_image_id": img2["id"],
        },
    )
    assert qres.status_code == 200, qres.text
    payload = qres.json()["data"]
    result = payload["result"]
    assert result["status"] == "completed"
    assert "2016-09-06" in result["answer"]
    assert "2026-09-09" in result["answer"]

    # 4. Verify actual change regions and EPSG:4326 [lon, lat] coordinates
    evidence = result["evidence"]
    assert len(evidence) > 0, "No change regions detected"
    first_region = evidence[0]
    assert first_region["geometry"]["type"] == "Polygon"
    ring = first_region["geometry"]["coordinates"][0]
    assert len(ring) >= 4
    for coord in ring:
        lon, lat = coord[0], coord[1]
        assert 54.8 <= lon <= 55.2, f"Longitude out of bounds for Palm Jumeirah: {lon}"
        assert 24.8 <= lat <= 25.2, f"Latitude out of bounds for Palm Jumeirah: {lat}"
