# SatQuery GeoVision — Standalone Aerial YOLO Training Pipeline

This directory contains the training, validation, and evaluation framework for fine-tuning Ultralytics YOLO models on high-resolution aerial and drone imagery.

---

## 1. Directory Structure

```
training/geovision/
├── configs/
│   ├── dataset.yaml            # YOLOv8 dataset configuration (train/val/test splits)
│   └── class_mapping.yaml      # Taxonomy mapping for VisDrone, xView, and DOTA
├── scripts/
│   ├── train.py                # Transfer learning training script (epochs, batch, augmentations)
│   ├── validate.py             # Validation script calculating precision, recall, mAP
│   ├── inference.py            # Standalone sliding-window tiled inference
│   └── evaluate.py             # Full evaluation report and confusion matrix generation
├── notebooks/
│   └── geovision_yolo_training.ipynb  # End-to-end Google Colab GPU training notebook
├── evaluation/
│   ├── evaluation_report.json  # Audited metrics (Precision, Recall, F1, mAP50, mAP50-95)
│   └── error_analysis.md       # Comprehensive edge-case and hard-example analysis
└── README.md
```

---

## 2. SatQuery Class Taxonomy

SatQuery enforces an explicit distinction between **discrete physical objects** and **continuous spatial areas**:

### 2.1 Discrete Object Classes (Bounding Box Detection & Counting)
- `building`
- `house`
- `car`
- `truck`
- `bus`
- `motorcycle`
- `aircraft`
- `boat`
- `person`

### 2.2 Continuous Area Classes (Strictly Handled via Segmentation Masks)
- `water`
- `road`
- `vegetation`
- `forest`
- `grass`
- `sports_ground`

Area classes are never treated as bounding-box detections.

---

## 3. Training Workflow (Google Colab / Cloud GPU)

Training is completely decoupled from the local SatQuery web server and runs in Google Colab with GPU acceleration:

1. Open `training/geovision/notebooks/geovision_yolo_training.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → T4 GPU** (or A100).
3. Execute the cells to download the unified aerial dataset and start transfer learning:
   ```bash
   python scripts/train.py --weights yolov8m.pt --epochs 100 --batch 16 --imgsz 640
   ```
4. Run validation on the independent validation split:
   ```bash
   python scripts/validate.py --weights models/satquery_aerial/weights/best.pt
   ```
5. Generate the formal evaluation report:
   ```bash
   python scripts/evaluate.py --weights models/satquery_aerial/weights/best.pt
   ```
6. Download `best.pt`.

---

## 4. Deploying to SatQuery AI Backend

Once training is complete, deploy the checkpoint to SatQuery:

1. Place the weights file on the machine, e.g. at:
   `training/geovision/models/best.pt`
2. Set the environment variable in your `.env` or terminal:
   ```bash
   GEOVISION_YOLO_WEIGHTS_PATH=training/geovision/models/best.pt
   ```
3. Restart the backend. GeoVision will automatically detect the weights, activate the `AerialObjectDetectionProvider`, and execute high-resolution sliding-window tiled inference.

---

## 5. Zero-Fabrication Policy

When `GEOVISION_YOLO_WEIGHTS_PATH` is not configured or `ultralytics` is not installed, GeoVision transparently reports:
> **"Trained detection model unavailable."**

It will **never** draw fabricated bounding boxes, invent confidence numbers, or pretend that morphological contours are a trained semantic object detector.
