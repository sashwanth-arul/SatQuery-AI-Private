from __future__ import annotations

from datetime import date

from app.schemas.domain import QueryRequest
from app.schemas.input import ImageModality
from app.schemas.planning import (
    AnalysisProfileName,
    PlannerToolName,
    QueryAnalysisPlan,
    QueryIntent,
    RequestedModality,
    SensorRequirement,
)
from app.services.building_temporal_intent import is_building_temporal_query
from app.services.change_domain import domain_uses_construction_pipeline, resolve_change_domain
from app.services.single_image_intent import single_image_intent_from_query

BUILDING_CONSTRUCTION_PROFILE = "building_construction"
CONSTRUCTION_KEYWORDS = frozenset({"construction", "built", "development"})
SAR_KEYWORDS = frozenset({"sar", "radar", "sentinel-1", "sentinel1", "backscatter"})
MULTIMODAL_KEYWORDS = frozenset({"optical", "radar", "compare", "multimodal"})


def _extract_tokens(query: str) -> set[str]:
    tokens: set[str] = set()
    for word in query.split():
        cleaned = word.strip(".,!?\"'").lower()
        tokens.add(cleaned)
        tokens.update(part for part in cleaned.split("-") if part)
    return tokens

BASE_TOOLS = [
    PlannerToolName.FETCH_IMAGERY,
    PlannerToolName.DETECT_CHANGE,
    PlannerToolName.FUSE_EVIDENCE,
    PlannerToolName.GENERATE_EVIDENCE,
]


def _intent_from_query(query: str) -> QueryIntent:
    if is_building_temporal_query(query):
        return QueryIntent.BUILDING_TEMPORAL_CHANGE

    tokens = _extract_tokens(query)
    has_construction = bool(tokens & CONSTRUCTION_KEYWORDS) or "building" in tokens
    has_sar = bool(tokens & SAR_KEYWORDS)
    has_multimodal = bool(tokens & MULTIMODAL_KEYWORDS) or (
        "optical" in tokens and ("radar" in tokens or "sar" in tokens)
    )

    if has_construction and (has_sar or has_multimodal):
        return QueryIntent.MULTIMODAL_COMPARISON
    if has_construction:
        return QueryIntent.CONSTRUCTION
    if has_sar and not has_construction:
        return QueryIntent.RADAR_CHANGE
    return QueryIntent.SPECTRAL_CHANGE


def build_deterministic_plan(
    request: QueryRequest,
    *,
    planner: str = "deterministic",
    image_modality: ImageModality | None = None,
) -> QueryAnalysisPlan:
    """Keyword-based constrained planner (safe fallback)."""
    if request.is_cross_modal_upload:
        return QueryAnalysisPlan(
            user_intent=QueryIntent.CROSS_MODAL_OPTICAL_SAR,
            requested_modalities=[RequestedModality.OPTICAL, RequestedModality.SAR],
            analysis_profile=AnalysisProfileName.NONE,
            required_tools=[
                PlannerToolName.OPTICAL_ANALYSIS,
                PlannerToolName.SAR_ANALYSIS,
                PlannerToolName.CROSS_MODAL_FUSION,
                PlannerToolName.GENERATE_EVIDENCE,
            ],
            earlier_date=None,
            later_date=None,
            sensor_requirement=SensorRequirement.NOT_APPLICABLE,
            aoi_required=False,
            user_intent_summary="Route to uploaded cross-modal optical+SAR joint analysis.",
            planner=planner,  # type: ignore[arg-type]
        )

    if request.is_bi_temporal_upload:
        q_lower = request.query.strip().lower()
        tokens = {w.strip(".,!?\"'") for w in q_lower.split()}

        # 1. Building footprint matching
        if bool({"building", "buildings", "houses", "structures"} & tokens) and bool(
            {"increase", "increased", "new", "added", "decrease", "decreased", "removed", "change", "changed", "how many", "count"} & tokens
            or "how many" in q_lower
        ):
            return QueryAnalysisPlan(
                user_intent=QueryIntent.BUILDING_TEMPORAL_CHANGE,
                requested_modalities=[RequestedModality.OPTICAL],
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.DETECT_BUILDINGS,
                    PlannerToolName.MATCH_BUILDING_FOOTPRINTS,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to uploaded bi-temporal building footprint detection and spatial matching specialist.",
                planner=planner,  # type: ignore[arg-type]
            )

        # 2. Built-up area change
        if "built-up area" in q_lower or "built up area" in q_lower or "urban area" in q_lower or ("built-up" in q_lower and "area" in q_lower):
            return QueryAnalysisPlan(
                user_intent=QueryIntent.BUILT_UP_AREA_CHANGE,
                requested_modalities=[RequestedModality.OPTICAL],
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.ANALYZE_BUILT_UP,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to uploaded bi-temporal built-up surface area change specialist.",
                planner=planner,  # type: ignore[arg-type]
            )

        # 3. Water change
        if bool({"water", "lake", "reservoir", "river"} & tokens) and bool(
            {"shrinkage", "shrinking", "drying", "receding", "increase", "expansion", "expanded", "change", "lost", "gain"} & tokens
            or "water body" in q_lower
            or "water-body" in q_lower
        ):
            return QueryAnalysisPlan(
                user_intent=QueryIntent.WATER_CHANGE,
                requested_modalities=[RequestedModality.OPTICAL],
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.ANALYZE_WATER_CHANGE,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to uploaded bi-temporal water surface change specialist.",
                planner=planner,  # type: ignore[arg-type]
            )

        # 4. Vegetation change
        if bool({"vegetation", "forest", "canopy", "tree", "trees"} & tokens) and bool(
            {"loss", "lost", "decrease", "deforestation", "deforest", "growth", "greening", "change"} & tokens
        ):
            return QueryAnalysisPlan(
                user_intent=QueryIntent.VEGETATION_CHANGE,
                requested_modalities=[RequestedModality.OPTICAL],
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.ANALYZE_VEGETATION_CHANGE,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to uploaded bi-temporal vegetation canopy change specialist.",
                planner=planner,  # type: ignore[arg-type]
            )

        # Flood / inundation change
        if bool({"flood", "flooded", "flooding", "inundation", "inundated"} & tokens):
            return QueryAnalysisPlan(
                user_intent=QueryIntent.FLOOD_ANALYSIS,
                requested_modalities=[RequestedModality.OPTICAL],
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.ANALYZE_FLOOD,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to uploaded bi-temporal flood inundation and disaster assessment specialist.",
                planner=planner,  # type: ignore[arg-type]
            )

        change_domain = resolve_change_domain(request.query)
        return QueryAnalysisPlan(
            user_intent=QueryIntent.BI_TEMPORAL_CHANGE_VQA,
            requested_modalities=[RequestedModality.OPTICAL],
            analysis_profile=AnalysisProfileName.NONE,
            required_tools=[
                PlannerToolName.DETECT_CHANGE,
                PlannerToolName.CHANGE_UNDERSTANDING,
                PlannerToolName.GENERATE_EVIDENCE,
            ],
            earlier_date=None,
            later_date=None,
            sensor_requirement=SensorRequirement.NOT_APPLICABLE,
            aoi_required=False,
            change_domain=change_domain,
            user_intent_summary=(
                f"Route to uploaded bi-temporal change detection and interpretation"
                + (f" (domain: {change_domain.value})." if change_domain else ".")
            ),
            planner=planner,  # type: ignore[arg-type]
        )

    if request.is_single_image_vqa:
        modality = image_modality or ImageModality.OPTICAL
        requested = (
            [RequestedModality.SAR]
            if modality == ImageModality.SAR
            else [RequestedModality.OPTICAL]
        )
        intent = single_image_intent_from_query(request.query)
        if intent == QueryIntent.BUILDING_COUNT:
            return QueryAnalysisPlan(
                user_intent=QueryIntent.BUILDING_COUNT,
                requested_modalities=requested,
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.DETECT_BUILDINGS,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to building detection and exact object counting specialist.",
                planner=planner,  # type: ignore[arg-type]
            )
        if intent == QueryIntent.WATER_DETECTION:
            return QueryAnalysisPlan(
                user_intent=QueryIntent.WATER_DETECTION,
                requested_modalities=requested,
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.ANALYZE_WATER,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to water resource detection and surface area segmentation specialist.",
                planner=planner,  # type: ignore[arg-type]
            )
        if intent == QueryIntent.FLOOD_ANALYSIS:
            return QueryAnalysisPlan(
                user_intent=QueryIntent.FLOOD_ANALYSIS,
                requested_modalities=requested,
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.ANALYZE_FLOOD,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to disaster management and flood inundation mapping specialist.",
                planner=planner,  # type: ignore[arg-type]
            )
        if intent == QueryIntent.AGRICULTURE_MONITORING:
            return QueryAnalysisPlan(
                user_intent=QueryIntent.AGRICULTURE_MONITORING,
                requested_modalities=requested,
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.ANALYZE_VEGETATION,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to agricultural crop monitoring and vegetation condition specialist.",
                planner=planner,  # type: ignore[arg-type]
            )
        if intent == QueryIntent.FOREST_MONITORING:
            return QueryAnalysisPlan(
                user_intent=QueryIntent.FOREST_MONITORING,
                requested_modalities=requested,
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.ANALYZE_VEGETATION,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to forest canopy monitoring and woodland assessment specialist.",
                planner=planner,  # type: ignore[arg-type]
            )
        if intent == QueryIntent.INFRASTRUCTURE_MAPPING:
            return QueryAnalysisPlan(
                user_intent=QueryIntent.INFRASTRUCTURE_MAPPING,
                requested_modalities=requested,
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.MAP_INFRASTRUCTURE,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to infrastructure and built structural footprint specialist.",
                planner=planner,  # type: ignore[arg-type]
            )
        if intent == QueryIntent.LAND_COVER_ANALYSIS:
            return QueryAnalysisPlan(
                user_intent=QueryIntent.LAND_COVER_ANALYSIS,
                requested_modalities=requested,
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.ANALYZE_LAND_COVER,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to multi-class land cover classification and environmental analysis specialist.",
                planner=planner,  # type: ignore[arg-type]
            )
        if intent == QueryIntent.GROUNDING:
            return QueryAnalysisPlan(
                user_intent=QueryIntent.GROUNDING,
                requested_modalities=requested,
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.GROUND_OBJECTS,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to spatial object grounding and localization specialist.",
                planner=planner,  # type: ignore[arg-type]
            )
        if intent == QueryIntent.SINGLE_IMAGE_CAPTION:
            return QueryAnalysisPlan(
                user_intent=QueryIntent.SINGLE_IMAGE_CAPTION,
                requested_modalities=requested,
                analysis_profile=AnalysisProfileName.NONE,
                required_tools=[
                    PlannerToolName.GEOCHAT_CAPTION,
                    PlannerToolName.GENERATE_EVIDENCE,
                ],
                earlier_date=None,
                later_date=None,
                sensor_requirement=SensorRequirement.NOT_APPLICABLE,
                aoi_required=False,
                user_intent_summary="Route to GeoChat single-image scene description specialist.",
                planner=planner,  # type: ignore[arg-type]
            )
        return QueryAnalysisPlan(
            user_intent=QueryIntent.SINGLE_IMAGE_VQA,
            requested_modalities=requested,
            analysis_profile=AnalysisProfileName.NONE,
            required_tools=[
                PlannerToolName.GEOCHAT_VQA,
                PlannerToolName.GENERATE_EVIDENCE,
            ],
            earlier_date=None,
            later_date=None,
            sensor_requirement=SensorRequirement.NOT_APPLICABLE,
            aoi_required=False,
            user_intent_summary="Route to GeoChat single-image VQA specialist.",
            planner=planner,  # type: ignore[arg-type]
        )

    intent = _intent_from_query(request.query)
    change_domain = resolve_change_domain(request.query)

    if intent == QueryIntent.BUILDING_TEMPORAL_CHANGE:
        return QueryAnalysisPlan(
            user_intent=intent,
            requested_modalities=[RequestedModality.OPTICAL],
            analysis_profile=AnalysisProfileName.NONE,
            required_tools=[PlannerToolName.IMAGERY_POLICY],
            earlier_date=request.earlier_date,
            later_date=request.later_date,
            sensor_requirement=SensorRequirement.SENTINEL_2,
            user_intent_summary=(
                "Route to building-instance temporal imagery policy "
                "(segmentation not yet implemented)."
            ),
            planner=planner,  # type: ignore[arg-type]
        )

    if intent == QueryIntent.RADAR_CHANGE:
        return QueryAnalysisPlan(
            user_intent=intent,
            requested_modalities=[RequestedModality.SAR],
            analysis_profile=AnalysisProfileName.NONE,
            required_tools=BASE_TOOLS,
            earlier_date=request.earlier_date,
            later_date=request.later_date,
            sensor_requirement=SensorRequirement.SENTINEL_1,
            user_intent_summary="Route to Sentinel-1 SAR change detection.",
            planner=planner,  # type: ignore[arg-type]
        )

    if intent == QueryIntent.CONSTRUCTION or (
        change_domain and domain_uses_construction_pipeline(change_domain)
    ):
        domain_note = f" (domain: {change_domain.value})" if change_domain else ""
        return QueryAnalysisPlan(
            user_intent=QueryIntent.CONSTRUCTION,
            requested_modalities=[RequestedModality.OPTICAL, RequestedModality.SEMANTIC],
            analysis_profile=AnalysisProfileName.BUILDING_CONSTRUCTION,
            required_tools=[
                *BASE_TOOLS[:2],
                PlannerToolName.ANALYZE_SEMANTICS,
                *BASE_TOOLS[2:],
            ],
            earlier_date=request.earlier_date,
            later_date=request.later_date,
            sensor_requirement=SensorRequirement.SENTINEL_2,
            change_domain=change_domain,
            user_intent_summary=(
                f"Route to optical CVA and Dynamic World built semantic analysis{domain_note}."
            ),
            planner=planner,  # type: ignore[arg-type]
        )

    if intent == QueryIntent.MULTIMODAL_COMPARISON:
        return QueryAnalysisPlan(
            user_intent=intent,
            requested_modalities=[
                RequestedModality.OPTICAL,
                RequestedModality.SEMANTIC,
                RequestedModality.SAR,
            ],
            analysis_profile=AnalysisProfileName.BUILDING_CONSTRUCTION,
            required_tools=[
                PlannerToolName.FETCH_IMAGERY,
                PlannerToolName.DETECT_CHANGE,
                PlannerToolName.ANALYZE_SEMANTICS,
                PlannerToolName.DETECT_SAR_CHANGE,
                PlannerToolName.FUSE_EVIDENCE,
                PlannerToolName.GENERATE_EVIDENCE,
            ],
            earlier_date=request.earlier_date,
            later_date=request.later_date,
            sensor_requirement=SensorRequirement.SENTINEL_2_AND_1,
            user_intent_summary="Route to optical CVA, Dynamic World semantics, and Sentinel-1 SAR with fusion.",
            planner=planner,  # type: ignore[arg-type]
        )

    domain_note = f" (domain: {change_domain.value})" if change_domain else ""
    return QueryAnalysisPlan(
        user_intent=QueryIntent.SPECTRAL_CHANGE,
        requested_modalities=[RequestedModality.OPTICAL],
        analysis_profile=AnalysisProfileName.NONE,
        required_tools=BASE_TOOLS,
        earlier_date=request.earlier_date,
        later_date=request.later_date,
        sensor_requirement=SensorRequirement.SENTINEL_2,
        change_domain=change_domain,
        user_intent_summary=f"Route to Sentinel-2 CVA spectral change detection{domain_note}.",
        planner=planner,  # type: ignore[arg-type]
    )


def resolve_analysis_profile(query: str) -> str | None:
    plan = build_deterministic_plan(
        QueryRequest(
            query=query,
            aoi=_minimal_aoi(),
            earlier_date=date(2024, 1, 1),
            later_date=date(2024, 6, 1),
        )
    )
    return plan.profile


def _minimal_aoi():
    from app.schemas.domain import AOI, GeoJSONGeometry

    return AOI(
        geometry=GeoJSONGeometry(
            type="Polygon",
            coordinates=[[[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01], [0.0, 0.0]]],
        )
    )
