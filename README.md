# YOLOV11N-UAV FINETUNE

**OPEN SOURCE. SMALL MODEL. UAV DETECTION.**

```
┌─────────────────────────────────────┐
│       TRAINING PERFORMANCE          │
├─────────────────────────────────────┤
│  dj-air3     │ 96.8% mAP50          │
│  uav         │ 80.1% mAP50          │
└─────────────────────────────────────┘
```

█████████████████████████████████████████

## █ CORE

**Lightweight YOLO11n and YOLO11L finetuned for unmanned aerial vehicles.**  
**Built for edge deployment.**
**Additional synthetic data generation tool for more robust environment handling**

## █ SPECS

```
┌─────────────────────────────────────┐
│  MODEL       │ YOLO11n backbone    │
│  INPUT       │ 640x640 RGB         │
│  OUTPUT      │ UAV bounding boxes  │
│  TARGET      │ Mobile/Edge deploy  │
└─────────────────────────────────────┘
```

█████████████████████████████████████████

## █ DATASET

```
████ TRAIN: 6877 images (70%)
████ VALID: 1966 images (20%) 
████ TEST:   985 images (10%)
```

**PREPROCESSING**: Auto-orient → Resize to 640x640  
**AUGMENTATIONS**: None applied.

█████████████████████████████████████████

## █ WEIGHTS

```
├── yolov11n-UAV-finetune.pt   [PyTorch]
└── yolov11n-UAV-finetune.onnx [ONNX]
```

█████████████████████████████████████████

## █ CITATION

```bibtex
@misc{
uav-detection-blxxz_dataset,
title = { uav-detection Dataset },
type = { Open Source Dataset },
author = { UAVdetection },
howpublished = { \url{ https://universe.roboflow.com/uavdetection-msr99/uav-detection-blxxz } },
url = { https://universe.roboflow.com/uavdetection-msr99/uav-detection-blxxz },
journal = { Roboflow Universe },
publisher = { Roboflow },
year = { 2024 },
month = { apr },
note = { visited on 2025-08-16 },
}
```

█████████████████████████████████████████

█████████████████████████████████████████

```

## █ MACBOOK DEPLOYMENT

### Quick Start to setup the venv and requirements 
```bash
# Setup (one-time)
chmod +x macbook-init-setup.sh macbook-launcher.sh
./macbook-init-setup.sh

# Run operations with a TUI 
./macbook-launcher.sh

Scripts

macbook-init-setup.sh - Environment setup with Apple Silicon optimizations
macbook-launcher.sh - TUI operations center (video inference, synthetic data, batch processing)

Features

Python 3.13+ compatibility with M1/M2/M3 optimization
Synthetic data generation with YOLO annotations
Real-time inference (~30 FPS on Apple Silicon)
Project management and batch processing
```
```
███████████████████████████████████████████████████
█ OPEN SOURCE UAV DETECTION FOR ALL              █
███████████████████████████████████████████████████
```
