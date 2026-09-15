"""Production Aerial Object Detection Provider using Ultralytics YOLO and Tiled Inference.

Enforces strict zero-fabrication: if a trained model or weights file is not available,
it reports capability status as unavailable rather than generating synthetic results.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from PIL import Image

from app.adapters.geovision.taxonomy import (
    is_aerial_object,
    normalize_class_name,
    validate_detection_class,
)
from app.adapters.geovision.tiling import (
    generate_tile_windows,
    map_tile_box_to_original,
    merge_tile_detections_nms,
)
from app.schemas.geovision import DetectedObject


class AerialObjectDetectionProvider:
    """Production provider for aerial object detection via real Ultralytics YOLO models."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        model_version: str = "v1.2.0-aerial-finetuned",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        tile_size: int = 640,
        tile_overlap: float = 0.20,
    ) -> None:
        env_weights = os.getenv("GEOVISION_YOLO_WEIGHTS_PATH")
        self.weights_path = Path(env_weights) if env_weights else (Path(weights_path) if weights_path else None)
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.model_name = "SatQuery-Aerial-YOLOv8"
        self._model: Any = None
        self._model_loaded: bool = False

    @property
    def is_available(self) -> bool:
        """
        Returns True only if the Ultralytics engine is installed AND
        a valid trained weights checkpoint exists on disk.
        """
        if not self.weights_path or not self.weights_path.exists():
            return False
        try:
            import ultralytics  # noqa: F401
            return True
        except ImportError:
            return False

    def load_model(self) -> None:
        """Loads the real YOLO weights into memory."""
        if self._model_loaded:
            return
        if not self.is_available:
            raise RuntimeError(
                f"Trained detection model unavailable. Weights path: '{self.weights_path}'. "
                "Ensure trained best.pt is deployed and ultralytics is installed."
            )

        from ultralytics import YOLO

        self._model = YOLO(str(self.weights_path))
        self._model_loaded = True

    def detect(
        self,
        image: Image.Image,
    ) -> tuple[list[DetectedObject], dict[str, int], dict[str, Any]]:
        """
        Runs sliding-window tiled detection on the input aerial raster.
        Returns:
            (detected_objects, object_summary, execution_metadata)
        """
        if not self.is_available:
            return (
                [],
                {},
                {
                    "status": "unavailable",
                    "reason": "Trained detection model unavailable.",
                    "weights_path": str(self.weights_path) if self.weights_path else None,
                    "model_version": self.model_version,
                    "tiles_processed": 0,
                },
            )

        self.load_model()
        width, height = image.size

        # 1. Generate tiling windows
        windows = generate_tile_windows(
            image_width=width,
            image_height=height,
            tile_size=self.tile_size,
            overlap_ratio=self.tile_overlap,
        )

        raw_detections: list[dict[str, Any]] = []

        # 2. Run inference on each tile
        for tile in windows:
            tile_crop = image.crop((
                tile.x_offset,
                tile.y_offset,
                tile.x_offset + tile.width,
                tile.y_offset + tile.height,
            ))

            # Run prediction on tile
            results = self._model.predict(
                source=tile_crop,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            for result in results:
                if not hasattr(result, "boxes") or result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    raw_cls = self._model.names.get(cls_id, f"class_{cls_id}")

                    # Only accept discrete object classes
                    try:
                        norm_cls = validate_detection_class(raw_cls)
                    except ValueError:
                        continue

                    if not is_aerial_object(norm_cls):
                        continue

                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2] relative to tile

                    mapped_box = map_tile_box_to_original(
                        tile_box=xyxy,
                        tile=tile,
                        image_width=width,
                        image_height=height,
                    )
                    if mapped_box is None:
                        continue

                    raw_detections.append({
                        "class_name": norm_cls,
                        "confidence": round(conf, 4),
                        "bbox": mapped_box,
                        "tile_id": tile.tile_id,
                    })

        # 3. Class-aware NMS across tile seam borders
        merged_boxes = merge_tile_detections_nms(
            detections=raw_detections,
            iou_threshold=self.iou_threshold,
        )

        # 4. Formulate verified DetectedObject contract
        detected_objects: list[DetectedObject] = []
        summary: dict[str, int] = {}

        for idx, item in enumerate(merged_boxes, start=1):
            x1, y1, x2, y2 = item["bbox"]
            cls_name = item["class_name"]
            summary[cls_name] = summary.get(cls_name, 0) + 1

            detected_objects.append(
                DetectedObject(
                    id=f"obj_{idx:03d}",
                    class_name=cls_name,
                    confidence=item["confidence"],
                    bbox=[x1, y1, x2, y2],
                    center=[round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
                    area_px=round((x2 - x1) * (y2 - y1), 2),
                    source_model=self.model_name,
                    model_version=self.model_version,
                    tile_id=item.get("tile_id"),
                    coordinate_space="image_pixel_original",
                )
            )

        metadata = {
            "status": "completed",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "tiles_processed": len(windows),
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "raw_tile_detections": len(raw_detections),
            "deduplicated_detections": len(detected_objects),
        }

        return detected_objects, summary, metadata


# Global default instance
_default_provider: AerialObjectDetectionProvider | None = None


def get_aerial_detection_provider() -> AerialObjectDetectionProvider:
    """Returns the singleton AerialObjectDetectionProvider."""
    global _default_provider
    if _default_provider is None:
        _default_provider = AerialObjectDetectionProvider()
    return _default_provider
