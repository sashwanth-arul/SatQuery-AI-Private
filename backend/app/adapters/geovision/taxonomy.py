"""Explicit SatQuery Aerial Object and Area Taxonomy.

Strictly separates discrete physical objects (for bounding box detection and counting)
from continuous spatial regions (for segmentation and land-cover analysis).
"""

from __future__ import annotations

from typing import Final

# Discrete physical objects supported by aerial object detection
AERIAL_OBJECT_CLASSES: Final[set[str]] = {
    "building",
    "house",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "aircraft",
    "boat",
    "person",
}

# Continuous landscape and area classes (must be analyzed via segmentation/masks, NOT bbox detections)
AERIAL_AREA_CLASSES: Final[set[str]] = {
    "water",
    "road",
    "vegetation",
    "forest",
    "grass",
    "sports_ground",
}

# Mapping common dataset labels (VisDrone, xView, DOTA) to SatQuery standard classes
DATASET_CLASS_MAPPINGS: Final[dict[str, str]] = {
    # VisDrone
    "pedestrian": "person",
    "people": "person",
    "bicycle": "motorcycle",
    "motor": "motorcycle",
    "van": "truck",
    # DOTA / xView
    "small-vehicle": "car",
    "large-vehicle": "truck",
    "plane": "aircraft",
    "airplane": "aircraft",
    "ship": "boat",
    "vessel": "boat",
    "harbor": "boat",
    "bridge": "road",
    "swimming-pool": "water",
    "ground-track-field": "sports_ground",
    "baseball-diamond": "sports_ground",
    "tennis-court": "sports_ground",
    "basketball-court": "sports_ground",
    "soccer-ball-field": "sports_ground",
    "roundabout": "road",
    "intersection": "road",
    "storage-tank": "building",
}


def is_aerial_object(name: str) -> bool:
    """Returns True if the class is a discrete object valid for bounding-box detection."""
    norm = name.strip().lower().replace(" ", "_").replace("-", "_")
    return norm in AERIAL_OBJECT_CLASSES or DATASET_CLASS_MAPPINGS.get(norm) in AERIAL_OBJECT_CLASSES


def is_aerial_area(name: str) -> bool:
    """Returns True if the class is a continuous surface/area class."""
    norm = name.strip().lower().replace(" ", "_").replace("-", "_")
    return norm in AERIAL_AREA_CLASSES or DATASET_CLASS_MAPPINGS.get(norm) in AERIAL_AREA_CLASSES


def normalize_class_name(raw_name: str) -> str:
    """Maps raw detector or dataset label into standardized SatQuery class name."""
    norm = raw_name.strip().lower().replace(" ", "_").replace("-", "_")
    if norm in AERIAL_OBJECT_CLASSES or norm in AERIAL_AREA_CLASSES:
        return norm
    if norm in DATASET_CLASS_MAPPINGS:
        return DATASET_CLASS_MAPPINGS[norm]
    return norm


def validate_detection_class(label: str) -> str:
    """
    Validates that a detection label belongs to a discrete object class.
    Raises ValueError if an area class is submitted as a bounding box detection.
    """
    norm = normalize_class_name(label)
    if is_aerial_area(norm):
        raise ValueError(
            f"Area class '{norm}' cannot be treated as a discrete object detection box. "
            f"Area surfaces must be analyzed via segmentation or land-cover area analysis."
        )
    return norm
