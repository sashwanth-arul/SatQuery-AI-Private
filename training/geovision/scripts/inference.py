"""Standalone Tiled Inference on Large Aerial/Drone Imagery.

Applies sliding-window tiling to run high-resolution inference on large rasters
without stretching or downsampling, and maps boxes back to original image space.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from PIL import Image

# Add backend to path for tiling logic import if needed
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from app.adapters.geovision.detector import AerialObjectDetectionProvider


def parse_args():
    parser = argparse.ArgumentParser(description="Run sliding-window tiled inference on an aerial image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (PNG, JPG, TIFF)")
    parser.add_argument("--weights", type=str, default="training/geovision/models/best.pt", help="Path to best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile window size")
    parser.add_argument("--tile-overlap", type=float, default=0.20, help="Overlap ratio between tiles")
    parser.add_argument("--output-json", type=str, default="detections.json", help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image '{args.image}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading image: {image_path}")
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        print(f"Image Dimensions: {w} x {h} px")

        provider = AerialObjectDetectionProvider(
            weights_path=args.weights,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )

        detections, summary, metadata = provider.detect(img_rgb)

        print("\nInference Results:")
        print(f"  Status: {metadata.get('status')}")
        print(f"  Tiles Processed: {metadata.get('tiles_processed', 0)}")
        print(f"  Raw Detections:  {metadata.get('raw_tile_detections', 0)}")
        print(f"  Merged Verified: {metadata.get('deduplicated_detections', 0)}")
        print("\nObject Summary:")
        for cls_name, count in summary.items():
            print(f"  - {cls_name}: {count}")

        output_data = {
            "image": str(image_path),
            "width": w,
            "height": h,
            "metadata": metadata,
            "object_summary": summary,
            "detections": [d.model_dump() for d in detections],
        }

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved full detection output to: {args.output_json}")


if __name__ == "__main__":
    main()
