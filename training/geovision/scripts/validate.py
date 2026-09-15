"""SatQuery Aerial YOLO Validation Script.

Validates trained model weights against the independent validation/test splits
to calculate Precision, Recall, F1, mAP50, and mAP50-95.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate trained aerial YOLO weights")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--data", type=str, default=str(Path(__file__).parent.parent / "configs" / "dataset.yaml"), help="Path to dataset.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Tile resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    print("==================================================")
    print("SatQuery Aerial Detection Validation")
    print(f"Weights: {args.weights}")
    print(f"Split:   {args.split}")
    print("==================================================")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' is required. Install with: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    print("\nValidation Results:")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
