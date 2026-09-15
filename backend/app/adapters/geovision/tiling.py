"""High-resolution aerial image tiling and coordinate reconstruction engine.

Enables full-resolution tiled inference for large aerial/drone rasters without
destructive downsampling, maps tile coordinates back to the original image space,
and merges boundary-crossing detections via Non-Maximum Suppression (NMS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TileWindow:
    """Defines a crop window on the original raster."""

    tile_id: str
    x_offset: int
    y_offset: int
    width: int
    height: int


def generate_tile_windows(
    image_width: int,
    image_height: int,
    tile_size: int = 640,
    overlap_ratio: float = 0.20,
) -> list[TileWindow]:
    """
    Generates an overlapping grid of tile windows across the image.
    Ensures complete coverage including right and bottom image borders.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")

    # If the image is smaller than or equal to tile size, return a single tile
    if image_width <= tile_size and image_height <= tile_size:
        return [
            TileWindow(
                tile_id="tile_0_0",
                x_offset=0,
                y_offset=0,
                width=image_width,
                height=image_height,
            )
        ]

    step = max(1, int(tile_size * (1.0 - overlap_ratio)))
    windows: list[TileWindow] = []

    y = 0
    row = 0
    while y < image_height:
        # Determine crop height
        h = min(tile_size, image_height - y)
        # If remaining slice is tiny and we already have previous row, adjust offset
        curr_y = y
        if curr_y + h > image_height:
            curr_y = max(0, image_height - tile_size)
            h = image_height - curr_y

        x = 0
        col = 0
        while x < image_width:
            w = min(tile_size, image_width - x)
            curr_x = x
            if curr_x + w > image_width:
                curr_x = max(0, image_width - tile_size)
                w = image_width - curr_x

            tile_id = f"tile_r{row}_c{col}"
            windows.append(
                TileWindow(
                    tile_id=tile_id,
                    x_offset=curr_x,
                    y_offset=curr_y,
                    width=w,
                    height=h,
                )
            )

            if curr_x + w >= image_width:
                break
            x += step
            col += 1

        if curr_y + h >= image_height:
            break
        y += step
        row += 1

    return windows


def map_tile_box_to_original(
    tile_box: list[float] | tuple[float, float, float, float],
    tile: TileWindow,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    """
    Translates a bounding box from tile-local coordinates to the original image coordinates,
    clamps to image boundaries, and validates geometry.
    Returns [x1, y1, x2, y2] or None if box is invalid.
    """
    t_x1, t_y1, t_x2, t_y2 = tile_box

    x1 = float(t_x1 + tile.x_offset)
    y1 = float(t_y1 + tile.y_offset)
    x2 = float(t_x2 + tile.x_offset)
    y2 = float(t_y2 + tile.y_offset)

    # Clamp to original image boundaries
    x1 = max(0.0, min(x1, float(image_width)))
    y1 = max(0.0, min(y1, float(image_height)))
    x2 = max(0.0, min(x2, float(image_width)))
    y2 = max(0.0, min(y2, float(image_height)))

    # Reject degenerate or inverted boxes
    if x2 <= x1 or y2 <= y1:
        return None

    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def compute_iou(
    box_a: list[float] | tuple[float, float, float, float],
    box_b: list[float] | tuple[float, float, float, float],
) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def merge_tile_detections_nms(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """
    Merges overlapping detections from multiple tiles using class-aware Non-Maximum Suppression.
    Expected detection format:
    {
        "class_name": str,
        "confidence": float,
        "bbox": [x1, y1, x2, y2],
        ...
    }
    """
    if not detections:
        return []

    # Group by class_name
    by_class: dict[str, list[dict[str, Any]]] = {}
    for det in detections:
        cls_name = det["class_name"]
        by_class.setdefault(cls_name, []).append(det)

    merged: list[dict[str, Any]] = []

    for cls_name, items in by_class.items():
        # Sort by confidence descending
        sorted_items = sorted(items, key=lambda d: d.get("confidence", 0.0), reverse=True)
        keep: list[dict[str, Any]] = []

        while sorted_items:
            best = sorted_items.pop(0)
            keep.append(best)

            # Filter out items with IoU >= threshold
            remaining: list[dict[str, Any]] = []
            for candidate in sorted_items:
                iou = compute_iou(best["bbox"], candidate["bbox"])
                if iou < iou_threshold:
                    remaining.append(candidate)
            sorted_items = remaining

        merged.extend(keep)

    return merged
