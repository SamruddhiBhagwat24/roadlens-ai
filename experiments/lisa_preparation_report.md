# RoadLens AI — LISA Traffic Sign Dataset Preparation Report
**Status:** COMPLETED  
**Dataset Path:** `/Users/samruddhibhagwat/roadlens-ai-local/datasets/roadlens_signs_yolo`  
**Source Path:** `/Users/samruddhibhagwat/roadlens-ai-local/datasets/lisa_signs`  

---

## 1. Frame & Sequence Summary

| Metric | Total | Train | Validation | Test |
|---|---|---|---|---|
| **Sequences** | 423 | 297 (70.2%) | 63 (14.9%) | 63 (14.9%) |
| **Usable Frames** | 5410 | 3735 | 894 | 781 |
| **Total Source Frames** | 54756 | — | — | — |

---

## 2. Annotation Counts by Target Class

| Class ID | Class Name | Total Annotations | Train | Val | Test | Status |
|---|---|---|---|---|---|---|
| **0** | `SPEED_LIMIT_SIGN` | 1376 | 1008 | 216 | 152 | Active |
| **1** | `ADVISORY_SPEED` | 0 | 0 | 0 | 0 | Absent (No LISA source labels) |
| **2** | `STOP_SIGN` | 1821 | 1268 | 294 | 259 | Active |
| **3** | `WARNING_SIGN` | 2958 | 2015 | 476 | 467 | Active |
| **4** | `WORK_ZONE_SIGN` | 0 | 0 | 0 | 0 | Absent (No LISA source labels) |

---

## 3. Quality Assurance (QA) Results

* **Sequence Leakage Detected:** `False` (Zero-leakage partition verified: **True**)
* **Missing Image Files:** `0`
* **Malformed Annotations:** `0`
* **Swapped Coordinates Auto-Corrected:** `0`
* **Out-of-Bounds Boxes Clamped:** `0`
* **Zero-Area Boxes Discarded:** `0`
* **Duplicate Image Assignments:** `0`

---

## 4. Ignored / Excluded Source Labels

The following LISA labels were excluded from the RoadLens target detector:
* `yield`: 236 annotations
* `laneEnds`: 210 annotations
* `stopAhead`: 168 annotations
* `school`: 133 annotations
* `speedLimitUrdbl`: 132 annotations
* `schoolSpeedLimit25`: 105 annotations
* `turnRight`: 92 annotations
* `rightLaneMustTurn`: 77 annotations
* `truckSpeedLimit55`: 60 annotations
* `roundabout`: 53 annotations
* `curveRight`: 50 annotations
* `noLeftTurn`: 47 annotations
* `curveLeft`: 37 annotations
* `dip`: 35 annotations
* `slow`: 34 annotations
* `turnLeft`: 32 annotations
* `rampSpeedAdvisory45`: 29 annotations
* `noRightTurn`: 26 annotations
* `doNotEnter`: 23 annotations
* `zoneAhead25`: 21 annotations
* `zoneAhead45`: 20 annotations
* `thruTrafficMergeLeft`: 19 annotations
* `rampSpeedAdvisory50`: 16 annotations
* `rampSpeedAdvisory20`: 11 annotations
* `doNotPass`: 9 annotations
* `thruMergeRight`: 7 annotations
* `thruMergeLeft`: 5 annotations
* `rampSpeedAdvisory35`: 5 annotations
* `rampSpeedAdvisory40`: 3 annotations
* `rampSpeedAdvisoryUrdbl`: 3 annotations
* `intersection`: 2 annotations

---
## 5. Next Steps
* Dataset is formatted in standard YOLO structure and ready for model training.
* Classes 1 (`ADVISORY_SPEED`) and 4 (`WORK_ZONE_SIGN`) are currently absent and will require targeted synthetic generation or auxiliary datasets before training.
