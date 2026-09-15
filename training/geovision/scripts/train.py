"""SatQuery Aerial YOLO Fine-Tuning Script.

Designed to execute in Google Colab (with GPU) or local CUDA environments
for transfer learning from pretrained YOLO weights onto the SatQuery aerial dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on SatQuery Aerial Object Dataset")
    parser.add_argument("--weights", type=str, default="yolov8m.pt", help="Pretrained weights checkpoint (e.g. yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--data", type=str, default=str(Path(__file__).parent.parent / "configs" / "dataset.yaml"), help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input tile size in pixels")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--project", type=str, default=str(Path(__file__).parent.parent / "models"), help="Output directory")
    parser.add_argument("--name", type=str, default="satquery_aerial", help="Run experiment name")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"==================================================")
    print(f"SatQuery Aerial YOLO Fine-Tuning Pipeline")
    print(f"==================================================")
    print(f"Base Weights: {args.weights}")
    print(f"Dataset:      {args.data}")
    print(f"Epochs:       {args.epochs} (Patience: {args.patience})")
    print(f"Tile Size:    {args.imgsz}x{args.imgsz} px")
    print(f"Batch Size:   {args.batch}")
    print(f"Output Dir:   {args.project}/{args.name}")
    print(f"==================================================")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] 'ultralytics' is required for training. Install with: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.weights)

    # Launch transfer learning
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        patience=args.patience,
        project=args.project,
        name=args.name,
        # Aerial domain augmentations
        mosaic=1.0,
        mixup=0.15,
        degrees=10.0,
        fliplr=0.5,
        flipud=0.5,
        scale=0.5,
        # Checkpoint saving
        save=True,
        save_period=10,
        exist_ok=True,
    )

    print("\nTraining completed successfully.")
    print(f"Best checkpoint saved at: {args.project}/{args.name}/weights/best.pt")
    print(f"Last checkpoint saved at: {args.project}/{args.name}/weights/last.pt")
    print("\nTo deploy this model into SatQuery AI:")
    print(f"1. Copy 'best.pt' to backend/app/models/best.pt or training/geovision/models/best.pt")
    print(f"2. Set environment variable: GEOVISION_YOLO_WEIGHTS_PATH=training/geovision/models/best.pt")


if __name__ == "__main__":
    main()
