from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from starlette.testclient import TestClient

from app.main import app
from app.schemas.domain import (
    AnalysisResult,
    AnalysisStatus,
    DataMode,
    EvidenceRegion,
    GeoJSONGeometry,
    Metric,
    TraceStatus,
    TraceStep,
)
from app.schemas.building_analysis import BuildingDetectionResult, BuildingFootprint, BuildingTemporalMatchResult
from app.storage.database import DatabaseStorage
from app.services.session_store import SessionStore
from app.services.report_service import report_service


def _sample_analysis_result(session_id: str, query: str = "How many buildings are there?") -> AnalysisResult:
    footprint = BuildingFootprint(
        id="b1",
        geometry=GeoJSONGeometry(
            type="Polygon",
            coordinates=[[[77.59, 12.97], [77.60, 12.97], [77.60, 12.98], [77.59, 12.98], [77.59, 12.97]]],
        ),
        bbox=[77.59, 12.97, 77.60, 12.98],
        area_m2=250.0,
        confidence=0.92,
        status="new",
    )
    region = EvidenceRegion(
        id="reg-001",
        geometry=footprint.geometry,
        type="building_footprint",
        confidence=0.92,
        source="DevelopmentBuildingDetector",
        metrics=[Metric(name="area_m2", value=250.0, unit="m²", source="computer_vision")],
        metadata={"claim_type": "new_building", "evidence_modality": "optical"},
    )
    b_detection = BuildingDetectionResult(
        image_id="img-001",
        count=5,
        detections=[footprint],
        confidence=0.91,
        detector="DevelopmentBuildingDetector",
        detector_name="DevelopmentBuildingDetector",
        total_area_m2=1250.0,
        metrics=[Metric(name="building_count", value=5, unit=None, source="computer_vision")],
    )
    trace = [
        TraceStep(
            id="t1",
            tool_name="select_building_detector",
            status=TraceStatus.COMPLETED,
            duration_ms=4,
            summary="Selected DevelopmentBuildingDetector",
        ),
        TraceStep(
            id="t2",
            tool_name="detect_buildings",
            status=TraceStatus.COMPLETED,
            duration_ms=35,
            summary="Extracted 5 building footprints",
        ),
    ]
    return AnalysisResult(
        status=AnalysisStatus.COMPLETED,
        session_id=session_id,
        answer="Building footprint analysis identified 5 buildings with a total area of 1,250 m².",
        confidence=0.91,
        confidence_available=True,
        metrics=[Metric(name="building_count", value=5, unit=None, source="computer_vision")],
        evidence=[region],
        trace=trace,
        mode=DataMode.DEVELOPMENT,
        building_detection=b_detection,
    )


def test_sqlite_database_creation_and_session_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_satquery.db"
        storage = DatabaseStorage(db_path)
        assert db_path.exists()

        session_id = "test-session-123456"
        res = _sample_analysis_result(session_id)
        storage.save_session(
            session_id=session_id,
            created_at=datetime.now(UTC),
            query="Count buildings in this area",
            intent="building_count",
            mode="upload",
            status=AnalysisStatus.COMPLETED,
            summary_answer=res.answer,
            result=res,
            trace=res.trace,
        )

        row = storage.get_session(session_id)
        assert row is not None
        assert row["session_id"] == session_id
        assert row["query"] == "Count buildings in this area"
        assert row["mode"] == "upload"
        assert row["status"] == "completed"
        assert "1,250 m²" in row["summary_answer"]


def test_session_store_persistence_and_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_satquery.db"
        storage = DatabaseStorage(db_path)

        store1 = SessionStore()
        # Monkeypatch store1 to use isolated storage
        import app.services.session_store as sm
        original_db = sm.db_storage
        sm.db_storage = storage

        try:
            session_id = store1.create(query="Has construction occurred?", mode="temporal_pair")
            res = _sample_analysis_result(session_id, "Has construction occurred?")
            store1.complete(session_id, res, query="Has construction occurred?", mode="temporal_pair")

            # Simulate complete server restart with a brand new SessionStore
            store2 = SessionStore()
            assert session_id not in store2._sessions

            # Reloading from SQLite
            reloaded = store2.get(session_id)
            assert reloaded is not None
            assert reloaded.session_id == session_id
            assert reloaded.query == "Has construction occurred?"
            assert reloaded.mode == "temporal_pair"
            assert reloaded.result is not None
            assert reloaded.result.answer == res.answer
            assert len(reloaded.trace) == 2

            # Test history listing from persistent store
            items, total = store2.list_history(limit=10, offset=0)
            assert total >= 1
            assert any(item["session_id"] == session_id for item in items)
        finally:
            sm.db_storage = original_db


def test_history_api_endpoint():
    client = TestClient(app)
    # 1. Check history endpoint returns 200
    resp = client.get("/api/v1/query/history?limit=10&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "items" in data["data"]
    assert "total" in data["data"]
    assert isinstance(data["data"]["items"], list)


def test_report_generation():
    session_id = "report-test-session-789"
    res = _sample_analysis_result(session_id)
    html_report = report_service.generate_html_report(res)

    assert "<!DOCTYPE html>" in html_report
    assert "SatQuery AI" in html_report
    assert "Building Detection" in html_report
    assert "1,250 m²" in html_report
    assert "reg-001" in html_report
    assert "select_building_detector" in html_report


def test_report_api_endpoint():
    client = TestClient(app)
    # Create and complete a sample session in session_store
    from app.services.session_store import session_store
    session_id = session_store.create(query="Report test query", mode="upload")
    res = _sample_analysis_result(session_id, "Report test query")
    session_store.complete(session_id, res, query="Report test query", mode="upload")

    resp = client.get(f"/api/v1/query/{session_id}/report")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert f"satquery-report-{session_id[:8]}.html" in resp.headers["content-disposition"]
    assert "SatQuery AI" in resp.text
    assert "Building footprint analysis" in resp.text


def test_history_mode_filtering():
    client = TestClient(app)
    from app.services.session_store import session_store
    s1 = session_store.create(query="Filter test upload", mode="upload")
    r1 = _sample_analysis_result(s1, "Filter test upload")
    session_store.complete(s1, r1, query="Filter test upload", mode="upload")

    resp_all = client.get("/api/v1/query/history?limit=50")
    assert resp_all.status_code == 200
    all_data = resp_all.json()["data"]["items"]
    assert any(item["session_id"] == s1 for item in all_data)

    resp_filter = client.get("/api/v1/query/history?mode=upload&limit=50")
    assert resp_filter.status_code == 200
    filter_data = resp_filter.json()["data"]["items"]
    assert all(item["mode"] == "upload" for item in filter_data)
    assert any(item["session_id"] == s1 for item in filter_data)

    resp_empty = client.get("/api/v1/query/history?mode=non_existent_mode")
    assert resp_empty.status_code == 200
    assert len(resp_empty.json()["data"]["items"]) == 0


def test_cross_modal_report_generation():
    from app.schemas.cross_modal import (
        CrossModalOpticalSARResult,
        CrossModalFusionSummary,
        ModalityAnalysisSummary,
        CoRegistrationStatus,
        CrossModalProviderKind,
    )
    cm_result = CrossModalOpticalSARResult(
        optical_image_id="opt-1",
        sar_image_id="sar-1",
        provenance="Test provenance",
        question="What can optical and SAR tell us about this area?",
        optical_analysis=ModalityAnalysisSummary(
            modality="optical",
            analyzer="TestOptical",
            provider=CrossModalProviderKind.DEVELOPMENT,
            summary="Optical reflectance shows urban density.",
        ),
        sar_analysis=ModalityAnalysisSummary(
            modality="sar",
            analyzer="TestSAR",
            provider=CrossModalProviderKind.DEVELOPMENT,
            summary="SAR backscatter shows rough texture.",
        ),
        fused_analysis=CrossModalFusionSummary(
            summary="Joint analysis confirms dense urban structures with high radar return.",
            fused_region_count=4,
            fusion_policy="cross_modal_v1",
        ),
        co_registration_status=CoRegistrationStatus.VERIFIED_BENCHMARK,
        co_registration_provenance="Test benchmark verification",
        provider="development",
        answer="Joint optical-SAR assessment verified 4 regions.",
    )
    res = AnalysisResult(
        status=AnalysisStatus.COMPLETED,
        session_id="cm-session-456",
        answer="Joint optical-SAR assessment verified 4 regions.",
        confidence=0.88,
        confidence_available=True,
        metrics=[],
        evidence=[],
        trace=[],
        mode=DataMode.DEVELOPMENT,
        cross_modal=cm_result,
    )
    report = report_service.generate_html_report(res)
    assert "Cross-Modal Optical + SAR Analysis" in report
    assert "Optical reflectance shows urban density" in report
    assert "SAR backscatter shows rough texture" in report
    assert "Joint analysis confirms dense urban structures" in report
    assert "verified benchmark" in report
