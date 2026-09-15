"""Single-image upload query intent routing (VQA vs scene caption)."""

from __future__ import annotations

from app.schemas.planning import QueryIntent

_CAPTION_PHRASES = (
    "describe this scene",
    "describe this satellite scene",
    "describe the satellite scene",
    "give a caption",
    "caption for this",
    "caption for the",
    "what does this image show",
    "what does this scene show",
    "what does this satellite image show",
    "what does this satellite scene show",
)


def is_building_count_query(query: str) -> bool:
    q = query.strip().lower()
    tokens = {w.strip(".,!?\"'") for w in q.split()}
    has_count_word = bool(
        {"count", "number", "quantity", "many"} & tokens
        or "how many" in q
        or "total number" in q
    )
    has_building_word = bool(
        {"building", "buildings", "structures", "houses", "rooftops"} & tokens
        or "built-up" in q
    )
    return has_count_word and has_building_word


def is_grounding_query(query: str) -> bool:
    q = query.strip().lower()
    return any(
        p in q
        for p in (
            "locate",
            "bounding box",
            "where is",
            "where are",
            "ground ",
            "grounding",
            "detect and locate",
        )
    )


def is_water_detection_query(query: str) -> bool:
    q = query.strip().lower()
    tokens = {w.strip(".,!?\"'") for w in q.split()}
    if any(vqa_kw in q for vqa_kw in ("what color", "why is", "who", "when was", "is there", "are there", "visible?")):
        return False
    if bool({"water", "waterbody", "waterbodies", "lake", "lakes", "reservoir", "reservoirs", "river", "rivers"} & tokens):
        if any(w in q for w in ("show water", "detect water", "water bodies", "water-body", "water body", "water extent", "water area", "water surface", "water resources", "find water")):
            return True
        if not ("?" in q and any(q.startswith(p) for p in ("what ", "which ", "how ", "where "))):
            return True
        if "water extent" in q or "water area" in q:
            return True
    return False


def is_flood_query(query: str) -> bool:
    q = query.strip().lower()
    tokens = {w.strip(".,!?\"'") for w in q.split()}
    return bool({"flood", "flooded", "flooding", "inundation", "inundated"} & tokens)


def is_agriculture_query(query: str) -> bool:
    q = query.strip().lower()
    tokens = {w.strip(".,!?\"'") for w in q.split()}
    if bool({"agriculture", "agricultural", "crop", "crops", "farmland", "farming"} & tokens):
        return True
    if "vegetation condition" in q or "vegetation health" in q or "crop health" in q:
        return True
    return False


def is_forest_query(query: str) -> bool:
    q = query.strip().lower()
    tokens = {w.strip(".,!?\"'") for w in q.split()}
    return bool({"forest", "forested", "trees", "canopy", "woodland"} & tokens)


def is_infrastructure_query(query: str) -> bool:
    q = query.strip().lower()
    return "infrastructure" in q or "buildings and infrastructure" in q or "structural network" in q


def is_land_cover_query(query: str) -> bool:
    q = query.strip().lower()
    # Keep GeoChat VQA for exploratory / qualitative / visual questions
    if "in this image" in q or "visible" in q or "major objects" in q:
        return False
    if any(marker in q for marker in ("what land-cover", "what land cover", "what types", "visible?")):
        return False
    if "?" in q and any(q.startswith(prefix) for prefix in ("what ", "which ", "how ", "where ", "when ", "who ")):
        return False
    return (
        "classify land" in q
        or "segment land" in q
        or "land cover classification" in q
        or "land cover analysis" in q
        or "land-cover analysis" in q
        or "environmental analysis" in q
        or "land use classification" in q
        or "land cover distribution" in q
        or "show land cover" in q
        or "map land cover" in q
        or "describe the land cover" in q
        or "describe land cover" in q
        or (("land cover" in q or "land-cover" in q) and "?" not in q and not q.startswith("what ") and not q.startswith("describe "))
    )


def single_image_intent_from_query(query: str) -> QueryIntent:
    """
    Distinguish building count, domain specialists (water, flood, agriculture, forest,
    infrastructure, land cover), grounding, scene-description (caption), and targeted VQA on uploaded images.
    """
    q = query.strip().lower()
    if not q:
        return QueryIntent.SINGLE_IMAGE_VQA

    if is_building_count_query(query):
        return QueryIntent.BUILDING_COUNT

    if is_flood_query(query):
        return QueryIntent.FLOOD_ANALYSIS

    if is_water_detection_query(query):
        return QueryIntent.WATER_DETECTION

    if is_agriculture_query(query):
        return QueryIntent.AGRICULTURE_MONITORING

    if is_forest_query(query):
        return QueryIntent.FOREST_MONITORING

    if is_infrastructure_query(query):
        return QueryIntent.INFRASTRUCTURE_MAPPING

    if is_land_cover_query(query):
        return QueryIntent.LAND_COVER_ANALYSIS

    if is_grounding_query(query):
        return QueryIntent.GROUNDING

    if any(phrase in q for phrase in _CAPTION_PHRASES):
        return QueryIntent.SINGLE_IMAGE_CAPTION

    if "surface pattern" in q and q.startswith("describe"):
        return QueryIntent.SINGLE_IMAGE_CAPTION

    if q.startswith("describe ") and (" scene" in q or "satellite scene" in q):
        return QueryIntent.SINGLE_IMAGE_CAPTION

    if "?" in q:
        if any(
            q.startswith(prefix)
            for prefix in ("what ", "which ", "how ", "where ", "when ", "who ")
        ):
            return QueryIntent.SINGLE_IMAGE_VQA

    if any(
        marker in q
        for marker in (
            "what land-cover",
            "what land cover",
            "what types",
            "what structures",
            "how many",
            "which ",
        )
    ):
        return QueryIntent.SINGLE_IMAGE_VQA

    # Imperative describe focused on image content (Phase 10 VQA compat).
    if q.startswith("describe ") and "in this image" in q:
        return QueryIntent.SINGLE_IMAGE_VQA

    if q.startswith("describe "):
        return QueryIntent.SINGLE_IMAGE_CAPTION

    return QueryIntent.SINGLE_IMAGE_VQA
