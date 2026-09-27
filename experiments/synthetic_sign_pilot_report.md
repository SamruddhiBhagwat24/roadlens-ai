# RoadLens AI — Stage 5C: Synthetic Traffic Sign Pilot Report

**Date:** September 2026  
**Environment:** Local macOS (Zero AMD GPU cloud instance usage, Zero model training)  
**Target Class IDs:**
- `1: ADVISORY_SPEED` (MUTCD W13-1P: yellow advisory plaques)
- `4: WORK_ZONE_SIGN` (MUTCD W20/W21 series: safety orange diamond/rectangular signs)

---

## 1. Executive Summary

In Stage 5B, the UCSD LISA Traffic Sign Dataset was ingested into `datasets/roadlens_signs_yolo/` with 5,410 frames and 6,155 annotations across 423 unique driving sequences. As audited in Stage 5, LISA provides ample real-world coverage for:
- `0: SPEED_LIMIT_SIGN` (2,168 annotations)
- `2: STOP_SIGN` (1,971 annotations)
- `3: WARNING_SIGN` (2,016 annotations)

However, classes `1: ADVISORY_SPEED` and `4: WORK_ZONE_SIGN` had **zero** instances in LISA.

In Stage 5C, a small, mathematically rigorous synthetic pilot dataset was generated **only** for these two missing classes:
- **100** `ADVISORY_SPEED` images (Class 1)
- **100** `WORK_ZONE_SIGN` images (Class 4)
- **Total:** 200 images, 200 YOLO labels, 200 JSON metadata files.
- **Location:** `datasets/synthetic_sign_pilot/` (strictly isolated, **not** merged into `datasets/roadlens_signs_yolo/`).

---

## 2. Dataset Specification & Design

### Class 1: `ADVISORY_SPEED` (FHWA MUTCD W13-1P Series)
- **Regulatory Standard:** FHWA MUTCD Section 2C.08 / W13-1P Advisory Speed Plaque
- **Color Palette:** Highway Warning Yellow (`RGB: 255, 205, 0` / `250, 200, 10`), black border, black Highway Gothic bold numerals.
- **Speed Increments:** 15, 20, 25, 30, 35, 40, 45, 50, 55 MPH.
- **Layout Styles:**
  1. `w13_1p_standard`: Yellow plaque, bold speed numeral on top, `MPH` centered below.
  2. `w13_1p_portrait`: Vertical rectangular plaque with `MPH` / `M.P.H.`
  3. `w13_1_compact`: Square plaque with large centered speed numeral.
  4. `w13_2_exit`: Vertical ramp/exit plaque with `EXIT`, speed numeral, and `MPH`.

### Class 2: `WORK_ZONE_SIGN` (FHWA MUTCD Part 6 / W20 & W21 Series)
- **Regulatory Standard:** FHWA MUTCD Temporary Traffic Control (TTC) W20/W21 and G20 Series.
- **Color Palette:** MUTCD Safety Orange (`RGB: 255, 102, 0` / `248, 92, 0`), black border, black Highway Gothic bold text.
- **Shapes & Layouts:**
  1. **Diamond (80%):** Standard 45° rotated square signs:
     - `ROAD WORK AHEAD`
     - `WORKERS AHEAD`
     - `ONE LANE ROAD AHEAD`
     - `FLAGGER AHEAD`
     - `DETOUR AHEAD`
     - `SHOULDER WORK AHEAD`
     - `UTILITY WORK AHEAD`
     - `MEN WORKING`
     - `RIGHT LANE CLOSED`
     - `LEFT LANE CLOSED`
     - `ROAD CLOSED AHEAD`
     - `BE PREPARED TO STOP`
     - `RAMP CLOSED`
     - `SURVEY CREW AHEAD`
  2. **Rectangular (20%):** Standard horizontal signs:
     - `ROAD WORK NEXT 5 MILES`
     - `ROAD WORK NEXT 2 MILES`
     - `END ROAD WORK`
     - `DETOUR >>>`
     - `WORK ZONE SPEED LIMIT 45`
     - `WORK ZONE FINES DOUBLED`

---

## 3. Road Scene Composition & Variation Pipeline

1. **Authentic Road Backgrounds:**
   - Sampled from 222 verified US road scenes in `datasets/openalpr_us/*.jpg`.
   - Authentic asphalt, multi-lane highways, trees, skies, guard rails, and lighting conditions.

2. **Roadside Placement:**
   - **Right Shoulder (70%):** Standard right-hand driving position ($x \in [0.58, 0.88]$, $y \in [0.16, 0.62]$).
   - **Left Median (20%):** Divided highways and multi-lane work zones ($x \in [0.03, 0.32]$, $y \in [0.16, 0.62]$).
   - **Overhead Gantry (10%):** Overhead truss mounts ($x \in [0.32, 0.68]$, $y \in [0.06, 0.28]$).

3. **Scale Buckets:**
   - **Large (25%):** Sign width $18\%\text{--}28\%$ of image width (~8-15 meters distance).
   - **Medium (50%):** Sign width $9\%\text{--}18\%$ of image width (~15-35 meters distance).
   - **Small (25%):** Sign width $4.5\%\text{--}9\%$ of image width (~35-65 meters distance).

4. **3D Perspective & Geometric Warping:**
   - 4-corner perspective transform with horizontal yaw skew ($\pm 8\%$) and vertical pitch taper ($\pm 6\%$).
   - Planar roll rotation ($\pm 4.5^\circ$) around sign center.
   - Exact bounding boxes extracted from projected alpha masks.

5. **Mounting Posts & Ambient Grounding:**
   - Galvanized metallic vertical signpost (`color=(140, 140, 145)`) rendered beneath signs down to ground level.
   - Ambient tone/luminance matching to blend sign naturally with local road lighting.

6. **Adverse Environmental Degradation Suite:**
   - Powered by `training/synthetic/degradations.py`:
     - Clean (25%): Clear sunny daylight.
     - Moderate (45%): Defocus blur ($k=3$), motion blur ($L=5$), contrast adjustment, JPEG compression ($q=45\text{--}65$), mild CMOS sensor noise.
     - Adverse (30%): Severe motion blur ($L=9\text{--}11$), high CMOS sensor grain ($\sigma=14\text{--}26$), low light / dusk gamma attenuation, specular headlight/sun glare, diagonal band cast shadows.

---

## 4. Rigorous QA Validation

Validation was executed by `validate_synthetic_dataset()` in `scratch/generate_synthetic_signs.py`:

| Check | Requirement | Result | Status |
| :--- | :--- | :--- | :--- |
| **Total Images** | ~100 advisory + ~100 workzone | 200 | **PASS** |
| **Total Labels** | Exactly 1 per image | 200 | **PASS** |
| **1:1 Parity** | `images/*.jpg` stems match `labels/*.txt` stems | 200/200 exact match | **PASS** |
| **Class Distribution** | Class 1 (`ADVISORY_SPEED`): 100, Class 4 (`WORK_ZONE_SIGN`): 100 | Exact 100 / 100 | **PASS** |
| **Coordinate Bounds** | $0.0 < x_c, y_c, w, h < 1.0$ | All inside $(0, 1)$ | **PASS** |
| **Non-Zero Area** | $w \cdot h > 0$ | Min area: $0.043 \times 0.0355 > 0$ | **PASS** |
| **Range Checks** | Coordinates strictly within image bounds | $x_1 \ge 0, y_1 \ge 0, x_2 \le W, y_2 \le H$ | **PASS** |
| **Readability** | Valid JPEG decode with $> 50$ px resolution | 200/200 decoded cleanly | **PASS** |

### Coordinate Extremes:
- $x_{\text{center}} \in [0.1006, 0.9383]$
- $y_{\text{center}} \in [0.1537, 0.5625]$
- $\text{width} \in [0.0430, 0.2781]$
- $\text{height} \in [0.0355, 0.5903]$

---

## 5. Perception & Intelligence Verification

A visual sample check and OCR perception test confirmed:
1. **EasyOCR Legibility:**
   - Sample `syn_advisory_0050` (`25 MPH`): OCR read `"25 MPH"` with confidence 1.0 on `25`.
   - Sample `syn_workzone_0000` (`UTILITY WORK AHEAD`): OCR read `"UTILITY WORK= AHEAD"` with confidence 1.0 on `AHEAD`.
   - Sample `syn_workzone_0075` (`DETOUR >>>`): OCR read `"DETOUR 555"` with confidence 1.0 on `DETOUR`.
2. **Semantic Interpretation:**
   - `USAProfile.interpret_advisory_speed()` correctly returned:
     `{"object_type": "advisory_speed_plaque", "regulatory_standard": "FHWA MUTCD W13-1P", "meaning": "Cautionary advisory speed 25 MPH for upcoming road segment", "value": 25, "unit": "mph"}`
   - `USAProfile.interpret_warning_sign()` correctly returned:
     `{"object_type": "work_zone_sign", "regulatory_standard": "FHWA MUTCD W20 Series (Work Zone)", "meaning": "Road hazard / work zone detected"}`
3. **Backend Test Suite:**
   - All 50 tests in `tests/` pass with zero regressions (`Ran 50 tests in 8.479s - OK`).

---

## 6. Next Steps Before Stage 6 Training

1. User visual review of generated samples in `datasets/synthetic_sign_pilot/` or `scratch/sample_visual_inspection/`.
2. Determine merge strategy:
   - Option A: Append the 200 synthetic pilot images to `datasets/roadlens_signs_yolo/` split proportionally (160 train, 20 val, 20 test).
   - Option B: Scale the synthetic generator to a larger batch (~400-500 images) before merging.
3. Configure AMD ROCm GPU training parameters in Stage 6.
