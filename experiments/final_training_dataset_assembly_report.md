# RoadLens AI — Stage 5D: Final Sign Training Dataset Assembly Report

**Date:** September 2026  
**Environment:** Local macOS (Zero AMD GPU cloud instances used; Zero model training; Zero modifications to production code; Zero modifications to canonical LISA dataset)  
**Target Combined Dataset:** `datasets/roadlens_signs_training/`  
**Test Benchmark Split:** `datasets/roadlens_signs_yolo/images/test` (100% Real LISA, 100% untouched)  

---

## 1. Executive Summary

In Stage 5D, the final 5-class RoadLens AI sign training dataset was assembled at `datasets/roadlens_signs_training/` by fusing:
1. **Real Driving Footage (LISA Traffic Signs):**
   - Training split: 3,735 images (4,291 bounding boxes)
   - Validation split: 894 images (986 bounding boxes)
   - Test split: **781 images (878 bounding boxes) kept strictly isolated and untouched**
2. **Scaled Synthetic Data (`datasets/synthetic_signs/`):**
   - 500 images total (250 `ADVISORY_SPEED` + 250 `WORK_ZONE_SIGN`)
   - Split deterministically with `seed=42`:
     - 400 images (80%) -> `images/train` (200 advisory + 200 work zone)
     - 100 images (20%) -> `images/val` (50 advisory + 50 work zone)
     - 0 images -> test (zero synthetic contamination of benchmark)

The combined dataset contains **5,129 images** and **5,777 annotations** across all 5 classes, with **zero train/val sequence leakage**, **zero duplicate image hashes**, and **100% test split isolation**.

---

## 2. Quantitative Dataset Specification

### A. Image Counts by Split and Origin

| Split | Real LISA Images | Synthetic Images | Total Images | Total Annotations |
| :--- | :---: | :---: | :---: | :---: |
| **Training (`images/train`)** | 3,735 | 400 | **4,135** | **4,691** |
| **Validation (`images/val`)** | 894 | 100 | **994** | **1,086** |
| **Total (Train + Val)** | **4,629** | **500** | **5,129** | **5,777** |
| **Test (Untouched Benchmark)** | 781 | 0 | **781** | **878** |

---

### B. Annotation Distribution Across the 5 Target Classes

| Class ID | Target Class Name | Source Origin | Train Annotations | Val Annotations | Total (Train+Val) | Benchmark Test Annotations |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **0** | `SPEED_LIMIT_SIGN` | Real LISA (speedLimit15..65) | 1,008 | 216 | **1,224** | 152 |
| **1** | `ADVISORY_SPEED` | Synthetic MUTCD W13-1P | 200 | 50 | **250** | 0 |
| **2** | `STOP_SIGN` | Real LISA (stop) | 1,268 | 294 | **1,562** | 259 |
| **3** | `WARNING_SIGN` | Real LISA (pedestrian, signal, merge, etc.) | 2,015 | 476 | **2,491** | 467 |
| **4** | `WORK_ZONE_SIGN` | Synthetic MUTCD W20/W21 series | 200 | 50 | **250** | 0 |
| **TOTAL** | — | — | **4,691** | **1,086** | **5,777** | **878** |

---

## 3. Rigorous 12-Point QA Validation Results

Automated QA validation executed by `run_comprehensive_stage_5d_qa()` in `scratch/assemble_final_training_dataset.py`:

| QA Check | Target Specification | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **1. File Parity (Train)** | `images/train` stems == `labels/train` stems | 4,135 / 4,135 exact match | **PASS** |
| **2. File Parity (Val)** | `images/val` stems == `labels/val` stems | 994 / 994 exact match | **PASS** |
| **3. Valid Decodes** | All images readable by OpenCV/PIL (>50px) | 5,129 / 5,129 valid decodes | **PASS** |
| **4. Valid YOLO Labels** | Exactly 5 fields per line (`class_id xc yc w h`) | 5,129 / 5,129 valid label files | **PASS** |
| **5. Class ID Bounds** | Class IDs strictly in `{0, 1, 2, 3, 4}` | 0 out-of-bounds class IDs | **PASS** |
| **6. Normalized Coords** | All $x_c, y_c, w, h \in (0, 1)$ | All coords strictly in $(0, 1)$ | **PASS** |
| **7. Non-Zero Area** | $w \cdot h > 0$ | 0 zero-area bounding boxes | **PASS** |
| **8. Duplicate Filenames** | Unique filenames across train and val | 0 duplicate filenames | **PASS** |
| **9. Duplicate Image Hashes** | Unique SHA-256 hashes across all images | **0 duplicate hashes** | **PASS** |
| **10. Sequence Leakage** | Zero LISA sequence overlap between train & val | **0 overlapping sequences** (297 train vs 63 val) | **PASS** |
| **11. Test Set Isolation** | No LISA test frame in train/val; no synthetic in test | **0 test leakage / 0 synthetic in test** | **PASS** |
| **12. Manifest Coverage** | Complete metadata provenance for every image | 5,129 / 5,129 samples indexed | **PASS** |

---

## 4. Directory Structure of Assembled Dataset

```
datasets/roadlens_signs_training/
├── images/
│   ├── train/        # 4,135 image files (3,735 PNG + 400 JPG)
│   └── val/          # 994 image files (894 PNG + 100 JPG)
├── labels/
│   ├── train/        # 4,135 YOLO label files
│   └── val/          # 994 YOLO label files
├── dataset.yaml      # Ultralytics YOLOv8 training config pointing to images/train, val, and test
├── manifest.json     # Complete per-image provenance metadata (JSON, 5,129 entries)
└── README.md         # Comprehensive dataset specification & documentation
```

---

## 5. Test Split Isolation Confirmation

The canonical UCSD LISA test split remains **100% untouched and isolated**:
- Path: `datasets/roadlens_signs_yolo/images/test/` (781 images) and `labels/test/` (781 labels).
- Sequence count: 63 distinct driving sequences.
- Zero sequence overlap with train (297 sequences) or validation (63 sequences).
- Referenced directly in `datasets/roadlens_signs_training/dataset.yaml` as the evaluation benchmark.
- Enables unbiased benchmark evaluation of the trained model against physical camera footage without data snooping.

---

## 6. Verification Suite

- **Assembler Unit Tests:** [`scratch/test_assemble_final_training_dataset.py`](file:///Users/samruddhibhagwat/roadlens-ai-local/scratch/test_assemble_final_training_dataset.py): **3/3 tests PASS** in 0.001s.
- **Backend Regression Suite:** [`tests/`](file:///Users/samruddhibhagwat/roadlens-ai-local/tests/): **50/50 tests PASS** in 9.13s.
- **Model Training / GPU Status:** Paused. Zero AMD instances launched.
