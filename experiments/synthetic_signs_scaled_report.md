# RoadLens AI — Stage 5C.2: Scaled Synthetic Sign Dataset Report

**Date:** September 2026  
**Environment:** Local macOS (Zero AMD GPU cloud instances used; Zero model training; Zero modifications to production code; Zero modifications to canonical LISA dataset)  
**Target Directory:** `datasets/synthetic_signs/`  
**Dataset Size:** 500 images (250 `ADVISORY_SPEED` + 250 `WORK_ZONE_SIGN`)  

---

## 1. Executive Summary

In Stage 5C.2, the validated 200-image synthetic pilot was scaled to a complete 500-image training dataset:
- **`1: ADVISORY_SPEED`**: 250 images (100 preserved from pilot + 150 newly generated)
- **`4: WORK_ZONE_SIGN`**: 250 images (100 preserved from pilot + 150 newly generated)
- **Total Images:** 500
- **Total Labels:** 500 (100% 1:1 parity)
- **Total Metadata Files:** 500 (100% audit coverage)
- **Zero Duplicate Scenes:** 222 original road backgrounds + 222 horizontally flipped road backgrounds + 56 spatial crop-shifted road backgrounds.
- **Isolation:** Saved in `datasets/synthetic_signs/` without merging into the LISA dataset or creating train/val/test splits yet.

---

## 2. Dataset Enhancements Applied (Stage 5C.2)

1. **Pilot Preservation:**
   - Samples 0..99 (`syn_advisory_0000..0099` and `syn_workzone_0000..0099`) were copied directly from `datasets/synthetic_sign_pilot/` with 100% byte-for-byte exactness.
2. **Anti-Aliasing on Clean Profiles:**
   - A subtle 1px Gaussian blur ($\sigma=0.6$) is applied to clean-profile signs when composited onto lower-resolution or compressed road backgrounds, eliminating digital edge sharpness discrepancies.
3. **Portable A-Frame Tripod Mounting:**
   - 55 work-zone signs feature dual angled orange powder-coated / galvanized steel legs with horizontal crossbars and ballast sandbag feet, accurately representing temporary traffic control (TTC) roll-up / folding barricade setups.
4. **Realistic Roadside Partial Occlusions (100 samples):**
   - **Traffic Cones:** Safety orange cones with white retroreflective collars encroaching on lower sign corners.
   - **Tree Foliage:** Natural leaf clusters overhanging sign top edges and corners.
   - **Roadside Guardrails:** Horizontal galvanized steel barrier rails occluding lower sign portions.
5. **Scale Ratio Range Expansion:**
   - Small: $0.0312\text{--}0.08$ (distant approaching view)
   - Medium: $0.08\text{--}0.18$ (standard following distance)
   - Large: $0.18\text{--}0.2898$ (close roadside pass)
6. **3D Perspective & Geometric Warping:**
   - Yaw skew ($\pm 9\%$), pitch taper ($\pm 7\%$), and planar roll rotation ($\pm 5^\circ$).

---

## 3. Quantitative Breakdown

### A. Class Distribution
- **Class 1 (`ADVISORY_SPEED`):** 250 (50.0%)
- **Class 4 (`WORK_ZONE_SIGN`):** 250 (50.0%)

### B. Scale Distribution
- **Small ($w \in [0.031, 0.08]$):** 130 images (26.0%)
- **Medium ($w \in [0.08, 0.18]$):** 234 images (46.8%)
- **Large ($w \in [0.18, 0.29]$):** 136 images (27.2%)

### C. Difficulty Profile Distribution
- **Moderate:** 221 images (44.2%)
- **Adverse:** 148 images (29.6%)
- **Clean:** 131 images (26.2%)

### D. Roadside Placement Distribution
- **Right Shoulder:** 345 images (69.0%)
- **Left Median:** 111 images (22.2%)
- **Overhead Gantry:** 44 images (8.8%)

### E. Mounting Styles
- **Galvanized Steel Post:** 220 images (44.0%)
- **Pilot Default Post:** 200 images (40.0%)
- **Portable A-Frame Tripod Stand:** 55 images (11.0%)
- **Overhead Gantry (No ground post):** 25 images (5.0%)

### F. Sign Styles & Legends
- **Advisory Speed Plaques (250):**
  - Standard W13-1P (numeral + `MPH`): 66
  - Portrait W13-1P (numeral + `MPH`/`M.P.H.`): 57
  - Compact W13-1 (numeral only): 59
  - Exit Ramp W13-2 (`EXIT` + numeral + `MPH`): 68
- **Work Zone Signs (250):**
  - Diamond W20 Series (45° rhombus): 206
  - Rectangular G20 Series: 44

### G. Degradations Applied Across Dataset
- Clean / Unmodified: 131
- Directional Motion Blur: 121
- CMOS Sensor ISO Noise: 121
- JPEG Compression Artifacts: 121
- Partial Occlusion (Cones, Foliage, Guardrail): 100
- Gaussian Defocus Blur: 96
- Brightness / Contrast Scaling: 66
- Specular Glare / Lens Flare: 51
- Cast Diagonal Shadows: 46
- Dusk / Night Low Light: 39

---

## 4. Complete QA Validation Results

Validation executed by `run_complete_scaled_qa()` in `scratch/scale_synthetic_signs.py`:

| QA Check | Target Requirement | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Total Images** | Exactly 500 JPEGs | **500** | **PASS** |
| **Total Labels** | Exactly 500 YOLO text files | **500** | **PASS** |
| **Total Metadata** | Exactly 500 JSON audit files | **500** | **PASS** |
| **1:1:1 Parity** | Image stems == Label stems == Metadata stems | **500/500 identical** | **PASS** |
| **Class Counts** | 250 Advisory Speed, 250 Work Zone | **250 / 250** | **PASS** |
| **Corrupted Files** | 0 unreadable images | **0 corrupted (100% valid)** | **PASS** |
| **Coordinate Bounds** | All normalized coords in $(0, 1)$ | **$x_c, y_c, w, h \in (0, 1)$** | **PASS** |
| **Non-Zero Boxes** | $w \cdot h > 0$ | **Min area: $0.0312 \times 0.0141 > 0$** | **PASS** |
| **Frame Confinement** | Bounding box contained within $[0, W] \times [0, H]$ | **$0 \le x_1 < x_2 \le W$, $0 \le y_1 < y_2 \le H$** | **PASS** |
| **Duplicate Scenes** | Zero duplicate background images | **0 duplicate scenes** | **PASS** |

### Coordinate Extremes:
- $x_{\text{center}} \in [0.0535, 0.9383]$
- $y_{\text{center}} \in [0.0825, 0.6075]$
- $\text{width} \in [0.0312, 0.2898]$
- $\text{height} \in [0.0141, 0.6319]$

---

## 5. Verification & Regression Status

- **Unit Tests:** [`scratch/test_scale_synthetic_signs.py`](file:///Users/samruddhibhagwat/roadlens-ai-local/scratch/test_scale_synthetic_signs.py): **5/5 tests PASS** in 0.062s.
- **Backend Regression Suite:** [`tests/`](file:///Users/samruddhibhagwat/roadlens-ai-local/tests/): **50/50 tests PASS** in 8.46s.
- **Untouched Splits:** Canonical LISA dataset (`datasets/roadlens_signs_yolo/`) and test split (`images/test`, `labels/test`) remain completely untouched.
- **No Training Conducted:** Model training and AMD GPU connections remain paused pending user authorization.
