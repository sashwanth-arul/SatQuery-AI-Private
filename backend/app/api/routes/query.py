from fastapi import APIRouter
from fastapi.responses import Response

from app.core.errors import SatQueryError
from app.core.responses import ApiResponse, SubmitQueryData, success
from app.schemas.domain import AnalysisResult, QueryRequest, TraceStep
from app.schemas.history import AnalysisHistoryItem, AnalysisHistoryResponse
from app.schemas.region_chat import RegionChatData, RegionChatRequest, SessionChatRequest
from app.services.session_chat import session_chat_service
from app.schemas.ground_context import GroundContextResult
from app.schemas.region_interpretation import InterpretRegionData, RegionInterpretationRequest
from app.schemas.region_ranking import BiTemporalRegionRankingResult
from app.services.mock_ground_context import mock_ground_context_service
from app.services.query_controller import query_controller
from app.services.region_chat import region_chat_service
from app.services.region_evidence_export import region_evidence_export_service
from app.services.region_interpretation import region_interpretation_service
from app.services.region_ranking import region_ranking_service
from app.services.report_service import report_service
from app.services.session_store import session_store

router = APIRouter(prefix="/query", tags=["query"])


@router.get("/history", response_model=ApiResponse[AnalysisHistoryResponse])
async def get_history(
    limit: int = 50,
    offset: int = 0,
    mode: str | None = None,
    status: str | None = None,
) -> ApiResponse[AnalysisHistoryResponse]:
    items_raw, total = session_store.list_history(
        limit=limit,
        offset=offset,
        mode=mode,
        status=status,
    )
    items = [AnalysisHistoryItem.model_validate(item) for item in items_raw]
    return success(
        AnalysisHistoryResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )
    )


@router.post("/submit", response_model=ApiResponse[SubmitQueryData])
async def submit_query(request: QueryRequest) -> ApiResponse[SubmitQueryData]:
    result = await query_controller.submit(request)
    return success(SubmitQueryData(session_id=result.session_id, result=result))


@router.get("/{session_id}/report")
async def get_analysis_report(session_id: str) -> Response:
    session = session_store.get(session_id)
    if not session or not session.result:
        raise SatQueryError(
            "session_not_found",
            f"No result found for session: {session_id}",
            status_code=404,
        )
    html_content = report_service.generate_html_report(session.result)
    filename = f"satquery-report-{session_id[:8]}.html"
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{session_id}/trace", response_model=ApiResponse[list[TraceStep]])
async def get_trace(session_id: str) -> ApiResponse[list[TraceStep]]:
    trace = query_controller.get_trace(session_id)
    return success(trace)


@router.get("/{session_id}/result", response_model=ApiResponse[AnalysisResult])
async def get_result(session_id: str) -> ApiResponse[AnalysisResult]:
    result = query_controller.get_result(session_id)
    return success(result)


@router.get(
    "/{session_id}/regions/ranked",
    response_model=ApiResponse[BiTemporalRegionRankingResult],
)
async def get_ranked_regions(session_id: str) -> ApiResponse[BiTemporalRegionRankingResult]:
    ranking = region_ranking_service.rank_session_regions(session_id)
    return success(ranking)


@router.post(
    "/{session_id}/regions/{region_id}/interpret",
    response_model=ApiResponse[InterpretRegionData],
)
async def interpret_region(
    session_id: str,
    region_id: str,
    request: RegionInterpretationRequest,
) -> ApiResponse[InterpretRegionData]:
    interpretation, trace_step = await region_interpretation_service.interpret_region(
        session_id,
        region_id,
        request.question,
    )
    return success(
        InterpretRegionData(
            interpretation=interpretation,
            trace_step=trace_step.model_dump(mode="json"),
        )
    )


@router.get(
    "/{session_id}/regions/{region_id}/ground-context",
    response_model=ApiResponse[GroundContextResult],
)
async def get_region_ground_context(
    session_id: str,
    region_id: str,
) -> ApiResponse[GroundContextResult]:
    context = mock_ground_context_service.get_ground_context(session_id, region_id)
    return success(context)


@router.get("/{session_id}/regions/{region_id}/evidence")
async def export_region_evidence(session_id: str, region_id: str) -> Response:
    zip_bytes, filename = region_evidence_export_service.export_region_evidence(
        session_id,
        region_id,
    )
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/{session_id}/regions/{region_id}/chat",
    response_model=ApiResponse[RegionChatData],
)
async def chat_region(
    session_id: str,
    region_id: str,
    request: RegionChatRequest,
) -> ApiResponse[RegionChatData]:
    chat, trace_step = await region_chat_service.chat(
        session_id,
        region_id,
        request.message,
    )
    return success(
        RegionChatData(
            chat=chat,
            trace_step=trace_step.model_dump(mode="json"),
        )
    )


@router.post(
    "/{session_id}/chat",
    response_model=ApiResponse[RegionChatData],
)
async def chat_session(
    session_id: str,
    request: SessionChatRequest,
) -> ApiResponse[RegionChatData]:
    chat, trace_step = await session_chat_service.chat(
        session_id,
        request.message,
        region_id=request.region_id,
    )
    return success(
        RegionChatData(
            chat=chat,
            trace_step=trace_step.model_dump(mode="json"),
        )
    )
