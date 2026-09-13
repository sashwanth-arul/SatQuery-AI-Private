"""Bridge uploaded optical+SAR pairs to shared AOI / imagery contracts."""

from __future__ import annotations

from datetime import date

from app.adapters.imagery.uploaded.bi_temporal_bridge import _intersection_bounds, aoi_from_bounds
from app.schemas.domain import (
    AOI,
    DataMode,
    ImageryResult,
    ImageryScene,
    SensorType,
    SpatialMetadata,
)
from app.schemas.input import ImageInput


def _ensure_epsg4326_bounds(bounds: list[float] | None, crs: str | None) -> list[float] | None:
    if not bounds:
        return None
    west, south, east, north = bounds
    if (
        -180.0 <= min(west, east)
        and max(west, east) <= 180.0
        and -90.0 <= min(south, north)
        and max(south, north) <= 90.0
    ):
        return [min(west, east), min(south, north), max(west, east), max(south, north)]
    if crs and crs != "EPSG:4326":
        try:
            from rasterio.warp import transform_bounds

            w_left, w_bottom, w_right, w_top = transform_bounds(
                crs, "EPSG:4326", west, south, east, north
            )
            return [
                min(float(w_left), float(w_right)),
                min(float(w_bottom), float(w_top)),
                max(float(w_left), float(w_right)),
                max(float(w_bottom), float(w_top)),
            ]
        except Exception:
            pass
    return bounds


def pair_bounds(optical: ImageInput, sar: ImageInput) -> list[float]:
    opt_b = _ensure_epsg4326_bounds(optical.bounds, optical.crs)
    sar_b = _ensure_epsg4326_bounds(sar.bounds, sar.crs)
    if opt_b and sar_b:
        return _intersection_bounds(opt_b, sar_b)
    if opt_b:
        return opt_b
    if sar_b:
        return sar_b
    if optical.benchmark_dataset and sar.benchmark_dataset:
        # Benchmark JPEG/PNG without georeferencing — development-only neutral extent.
        return [0.0, 0.0, 0.01, 0.01]
    raise ValueError("Both images must include geographic bounds for cross-modal analysis.")


def build_cross_modal_imagery_result(optical: ImageInput, sar: ImageInput) -> ImageryResult:
    bounds = pair_bounds(optical, sar)
    acq = optical.acquisition_datetime or sar.acquisition_datetime
    acq_date = acq.date() if acq else date(1970, 1, 1)
    return ImageryResult(
        source="uploaded_cross_modal",
        mode=DataMode.DEVELOPMENT,
        sensor=SensorType.SENTINEL_2,
        scenes=[
            ImageryScene(
                scene_id=optical.id,
                acquisition_date=acq_date,
                platform_id=f"upload://{optical.id}",
                metadata={"filename": optical.filename, "role": "optical"},
            ),
            ImageryScene(
                scene_id=sar.id,
                acquisition_date=acq_date,
                platform_id=f"upload://{sar.id}",
                metadata={"filename": sar.filename, "role": "sar"},
            ),
        ],
        spatial=SpatialMetadata(crs=optical.crs or sar.crs or "EPSG:4326", bbox=bounds),
        collection_id="uploaded_optical_sar_pair",
        message="Uploaded cross-modal optical+SAR pair — development adapters.",
    )


def pair_aoi(optical: ImageInput, sar: ImageInput) -> AOI:
    return aoi_from_bounds(pair_bounds(optical, sar))
