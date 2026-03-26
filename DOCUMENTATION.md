# FatigueSense Documentation

FatigueSense is a real-time fatigue detection system using YOLO11 object detection. It identifies four body regions — **Head**, **Torso**, **Mouth**, and **Eyes** — from video frames to assess driver/operator fatigue.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Workflow Overview](#workflow-overview)
  - [1. Video Collection](#1-video-collection)
  - [2. Frame Extraction](#2-frame-extraction)
  - [3. Annotation with CVAT](#3-annotation-with-cvat)
  - [4. Label Remapping (If Needed)](#4-label-remapping-if-needed)
  - [5. Merging Datasets](#5-merging-datasets)
  - [6. Training the Model](#6-training-the-model)
  - [7. Inference](#7-inference)
- [Class Labels](#class-labels)
- [Dataset Format](#dataset-format)
- [Scripts Reference](#scripts-reference)
- [Notebooks Reference](#notebooks-reference)
- [Training Infrastructure](#training-infrastructure)
- [Uploading Data](#uploading-data)

---

## Project Overview

The pipeline works as follows:

1. Record videos of subjects
2. Extract frames from videos at regular intervals
3. Manually annotate frames in [CVAT](https://www.cvat.ai/) with bounding boxes
4. Export annotations in YOLO format
5. Merge per-video datasets into one combined dataset with train/val splits
6. Train a YOLO11 model on the merged dataset
7. Run inference (webcam demo, video files, etc.)

---

## Repository Structure

```
FatigueSense/
├── scripts/                    # Data processing and upload utilities
│   ├── frame_extraction.py     # Extract frames from video files
│   ├── merge_datasets.py       # Merge per-video datasets into one
│   ├── remap_labels.py         # Remap class IDs for label consistency
│   ├── remap.py                # Older version of remap_labels.py
│   ├── upload_hf.py            # Upload dataset to Hugging Face
│   ├── upload_dataset.py       # Upload dataset to ClearML (reusable)
│   ├── upload.py               # Upload dataset to ClearML (legacy)
│   └── verify_files.py         # Verify extracted frames against metadata
├── notebooks/                  # Jupyter notebooks
│   ├── YOLOv11_train.ipynb     # Training pipeline + webcam demo
│   ├── data_preprocessing.ipynb# Copy/organize per-video data
│   ├── upload_dataset.ipynb    # Upload to ClearML with versioning
│   └── PERSONAL.ipynb          # Personal utility (download from ClearML)
├── dataset/                    # Per-video annotated datasets
│   ├── vid_001/ ... vid_008/   # One folder per annotated video
│   └── .cache/                 # Hugging Face upload cache
├── data/                       # Merged dataset (train/val split)
├── runs/                       # Training output (weights, metrics, plots)
├── videos/                     # Source video files
├── best.pt                     # Best trained model weights
├── yolo11n.pt                  # YOLO11 Nano pre-trained weights
├── upload_dataset.py           # Upload dataset to Hugging Face (root)
├── SETUP.md                    # Authentication & setup guide
├── README.md                   # Project overview
└── DOCUMENTATION.md            # This file
```

---

## Workflow Overview

### 1. Video Collection

Record videos of subjects and place them in the `videos/` directory following the naming convention:

```
videos/vid_001.mp4
videos/vid_002.mp4
...
```

### 2. Frame Extraction

Extract frames from videos at regular intervals using `scripts/frame_extraction.py`.

```bash
python scripts/frame_extraction.py
```

**Key parameters** (edit in script):

| Parameter    | Default | Description                        |
|--------------|---------|------------------------------------|
| `frame_step` | `15`    | Extract every Nth frame            |
| `extension`  | `.png`  | Output image format                |
| `output_dir` | `./data/images` | Where extracted frames are saved |

Frames are saved with zero-padded names: `frame_000000.png`, `frame_000015.png`, etc.

### 3. Annotation with CVAT

We use [CVAT](https://www.cvat.ai/) to manually annotate bounding boxes on the extracted frames.

**Setting up the label schema in CVAT:**

When creating a new CVAT task, paste this JSON into the label configuration to ensure consistent class ordering:

```json
[
  {
    "name": "Head",
    "id": 9,
    "color": "#ff6037",
    "type": "any",
    "attributes": []
  },
  {
    "name": "Torso",
    "id": 10,
    "color": "#ffcc33",
    "type": "any",
    "attributes": []
  },
  {
    "name": "Mouth",
    "id": 11,
    "color": "#34d1b7",
    "type": "any",
    "attributes": []
  },
  {
    "name": "Eyes",
    "id": 12,
    "color": "#f078f0",
    "type": "any",
    "attributes": []
  }
]
```

**Exporting from CVAT:**

Export the annotations in **YOLO 1.1** format. CVAT will produce a zip containing `images/train/`, `labels/train/`, `data.yaml`, and `train.txt`.

**Organizing the exported data:**

Each exported video annotation should be placed into `dataset/` following the naming convention `vid_xxx`:

```
dataset/
└── vid_xxx/
    ├── images/
    │   └── train/          # Annotated frame images (.png)
    ├── labels/
    │   └── train/          # YOLO label files (.txt)
    ├── data.yaml           # Class definitions and paths
    └── train.txt           # List of training image paths
```

> **Important:** The `images/` and `labels/` folders contain a `train/` subfolder because that's the structure CVAT exports. Keep this structure as-is.

### 4. Label Remapping (If Needed)

The first video (`vid_001`) was annotated with a different class ordering than the rest. The `remap_labels.py` script fixes this by remapping class IDs in-place.

**Original ordering in vid_001:**
```
0: Mouth  →  remapped to 2
1: Torso  →  stays at 1
2: Eyes   →  remapped to 3
3: Head   →  remapped to 0
```

**Run the remapping:**

```bash
python scripts/remap_labels.py
```

This modifies label files in `dataset/vid_001/labels/` in-place.

> **For new videos:** As long as you use the CVAT label JSON above, class IDs will already be correct and no remapping is needed. If a future annotation has a different ordering, update the `REMAP` dictionary in `remap_labels.py` and point it at the appropriate labels directory.

### 5. Merging Datasets

Once all per-video datasets are ready, merge them into a single dataset with a train/val split using `scripts/merge_datasets.py`.

```bash
python scripts/merge_datasets.py
```

**Key parameters** (edit in script):

| Parameter    | Default       | Description                         |
|--------------|---------------|-------------------------------------|
| `DATA_ROOT`  | `data`        | Directory containing per-video datasets |
| `OUTPUT_DIR` | `merged`      | Output directory for merged dataset |
| `VAL_SPLIT`  | `0.15`        | 15% of data goes to validation      |
| `SEED`       | `42`          | Random seed for reproducible splits |

**Output structure:**

```
merged/
├── images/
│   ├── train/        # ~85% of all frames
│   └── val/          # ~15% of all frames
├── labels/
│   ├── train/
│   └── val/
└── data.yaml         # Merged class config
```

The script prefixes filenames with the source dataset name (e.g., `vid_001_frame_000000.png`) to avoid collisions between videos.

### 6. Training the Model

Training is done using the [Ultralytics](https://docs.ultralytics.com/) YOLO11 framework. See `notebooks/YOLOv11_train.ipynb` for the full pipeline.

**Quick start (Python):**

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load pre-trained YOLO11 Nano

model.train(
    data="merged/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs",
    name="fatigue-v1",
    device=0,               # GPU device ID
)
```

**Training outputs** are saved to `runs/fatigue-v1/`:

| File | Description |
|------|-------------|
| `weights/best.pt` | Best model (by validation mAP) |
| `weights/last.pt` | Last epoch checkpoint |
| `results.csv` | Per-epoch metrics (loss, precision, recall, mAP) |
| `results.png` | Training curves visualization |
| `confusion_matrix.png` | Per-class detection accuracy |
| `args.yaml` | Full training configuration |

### 7. Inference

**Webcam demo** (from `notebooks/YOLOv11_train.ipynb`):

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=0.5, max_det=5)
    annotated = results[0].plot()
    cv2.imshow("FatigueSense", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Class Labels

All datasets must use this class ordering:

| ID | Class |
|----|-------|
| 0  | Head  |
| 1  | Torso |
| 2  | Mouth |
| 3  | Eyes  |

The corresponding `data.yaml` should look like:

```yaml
names:
  0: Head
  1: Torso
  2: Mouth
  3: Eyes
path: .
train: train.txt
```

---

## Dataset Format

Each label file (`.txt`) uses the standard YOLO format — one bounding box per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalized** (0–1) relative to image dimensions.

**Example** (`frame_000000.txt`):
```
0 0.463182 0.340278 0.165906 0.443278
3 0.429315 0.327620 0.053995 0.049741
3 0.510547 0.325440 0.043688 0.055843
2 0.470052 0.481199 0.061844 0.061954
1 0.465156 0.773519 0.370104 0.442407
```

This describes one Head, two Eyes, one Mouth, and one Torso detection.

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `frame_extraction.py` | Extract frames from video files at fixed intervals |
| `merge_datasets.py` | Merge per-video datasets into one with train/val split |
| `remap_labels.py` | Remap class IDs to fix ordering inconsistencies |
| `remap.py` | Older version of `remap_labels.py` (legacy) |
| `upload_hf.py` | Upload `dataset/` folder to Hugging Face Hub |
| `upload_dataset.py` | Reusable ClearML dataset upload utility |
| `upload.py` | Legacy ClearML upload script |
| `verify_files.py` | Verify extracted frames match expected metadata |

---

## Notebooks Reference

| Notebook | Purpose |
|----------|---------|
| `YOLOv11_train.ipynb` | Full training pipeline: dataset split → train → webcam inference |
| `data_preprocessing.ipynb` | Consolidate per-video folders into centralized `data/` directory |
| `upload_dataset.ipynb` | Upload images and labels to ClearML with versioning |
| `PERSONAL.ipynb` | Personal utility for downloading frames from ClearML |

---

## Training Infrastructure

**Current setup:**

Training is currently handled by **Dowee**, who has an **RTX 5080** GPU.

**Requirements:**
- NVIDIA GPU with **CUDA** support
- PyTorch installed with the correct **CUDA version** for your GPU
  - Check compatibility: https://pytorch.org/get-started/locally/
  - Verify GPU is available:
    ```python
    import torch
    print(torch.cuda.is_available())   # Should print True
    print(torch.cuda.get_device_name(0))
    ```
- Ultralytics package: `pip install ultralytics`

**Planned:** Migration to cloud GPUs (e.g., NVIDIA A100) for more accessible and scalable training.

---

## Uploading Data

### To Hugging Face

Use `scripts/upload_hf.py` or the root `upload_dataset.py` to push the dataset:

```bash
python scripts/upload_hf.py
# or
python upload_dataset.py
```

Both use `HfApi.upload_large_folder()` to upload the entire `dataset/` directory. You must be logged in first — see [SETUP.md](SETUP.md) for authentication options.

### To ClearML

Legacy upload scripts (`scripts/upload.py`, `scripts/upload_dataset.py`, `notebooks/upload_dataset.ipynb`) use the ClearML `Dataset` API. These require ClearML credentials configured in the `.env` file.
