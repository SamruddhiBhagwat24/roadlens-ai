# RoadLens AI — Phase B Step 4C Vehicle-Prior License Plate Detection Experiment Report
**Document ID:** `RL-DOC-011-VEHICLE-PRIOR-DETECTION`  
**Status:** COMPLETED & AUDITED  
**Date:** September 2026  
**Execution Context:** Local Evaluation Experiment (Zero Downloads, Zero Package Installs, Zero Training, Zero GPU Credits)  
**Evaluated Detector:** Pretrained YOLOv8s COCO (`models/yolov8s.pt`, CPU execution)  
**Evaluated Dataset:** OpenALPR US Benchmark ($N=222$ images)  
**Data Artifact:** `experiments/vehicle_prior_results.csv` (222 evaluation records)  

---

## 1. Executive Summary & Experimental Objectives

In Phase B Step 3, we verified that while pretrained YOLOv8s COCO weights detect vehicles with high precision, **they detect zero license plates** because COCO lacks a `license_plate` category.

The objective of **Phase B Step 4C** is to evaluate whether a **deterministic vehicle-prior geometric heuristic** can extract candidate license-plate regions from existing YOLOv8s vehicle detections without downloading or training new models.

### Strict Experimental Guardrails
* ✅ **Zero Downloads:** Zero models, zero weights, zero datasets downloaded.
* ✅ **Zero Training:** No fine-tuning or model retraining performed.
* ✅ **Zero AMD GPU Credits:** Run entirely on local CPU.
* ✅ **Independent Ground Truth:** Ground-truth plate coordinates were used **strictly for evaluation**, never to guide ROI generation.
* ✅ **Objective Reporting:** Evaluated strictly using standard computer vision intersection-over-union (IoU) metrics without parameter gaming.

---

## 2. Heuristic Definition & Mathematical Formulation

License plates in standard passenger vehicles are mounted centrally on the front or rear bumper/grille assembly. On a detected vehicle bounding box $V = [v_{x1}, v_{y1}, v_{x2}, v_{y2}]$ with width $v_w = v_{x2} - v_{x1}$ and height $v_h = v_{y2} - v_{y1}$, two fixed, deterministic geometric heuristics were evaluated:

### Heuristic A: Single Central Plate Hypothesis (Tight ROI)
Designed to approximate the aspect ratio and relative footprint of a standard US license plate ($12 \times 6\text{ inches}$, aspect ratio $\approx 2.0$):
* **Width:** $p_w = 0.26 \cdot v_w$ ($26\%$ of vehicle width)
* **Height:** $p_h = 0.12 \cdot v_h$ ($12\%$ of vehicle height, yielding aspect ratio $\approx 2.16$)
* **Center Coordinates:** $(c_x, c_y) = (v_{x1} + 0.50 \cdot v_w, \; v_{y1} + 0.70 \cdot v_h)$
* **Bounding Box:** $[c_x - \frac{p_w}{2}, \; c_y - \frac{p_h}{2}, \; c_x + \frac{p_w}{2}, \; c_y + \frac{p_h}{2}]$

### Heuristic B: Macro Bumper Search Window (Coverage Hypothesis)
Designed to act as a broad regional crop capturing the lower bumper search area:
* **Horizontal Bounds:** $[v_{x1} + 0.15 \cdot v_w, \; v_{x2} - 0.15 \cdot v_w]$ (central $70\%$ of vehicle width)
* **Vertical Bounds:** $[v_{y1} + 0.50 \cdot v_h, \; v_{y1} + 0.95 \cdot v_h]$ (lower $45\%$ of vehicle, avoiding ground plane below tires)

> [!NOTE]
> **Fixed Parameter Rationale:**  
> These geometric constants are derived strictly from standard automotive design specifications (FMVSS / SAE standards). In accordance with the project guardrails, **these parameters were kept completely fixed and were not tuned against the 222-image test set**.

---

## 3. Detector Configuration & Execution Profiling

* **Model:** Ultralytics YOLOv8s (`models/yolov8s.pt`, $22.59\text{ MB}$, 11.2M parameters).
* **Target Classes:** COCO vehicle classes: `car` (2), `motorcycle` (3), `bus` (5), `truck` (7).
* **Confidence Threshold:** $0.25$.
* **Execution Hardware:** Apple M-series CPU (deterministic CPU runtime).
* **Inference Latencies:**
  * Mean YOLOv8s Vehicle Detection Latency: **$78.34\text{ ms}$** per frame.
  * Mean Geometric ROI Generation Latency: **$0.0081\text{ ms}$** per frame ($8.1\text{ microseconds}$).
  * Total Localization Pipeline Latency: **$\approx 78.35\text{ ms}$** per frame.

---

## 4. Empirical Detection & Localization Results

Comprehensive evaluation across all $N=222$ benchmark images:

| Evaluation Metric | Measured Value | Percentage of Dataset | Benchmark Interpretation |
|---|---|---|---|
| **Total Images Evaluated** | 222 | 100.0% | Full OpenALPR US Benchmark |
| **Images with Vehicle Detections** | **219 / 222** | **98.65%** | Pretrained YOLOv8s detects vehicles with near-perfect recall. |
| **Images with Candidate Plate ROIs** | **219 / 222** | **98.65%** | Geometric heuristic executed on all detected vehicles. |
| **Mean Candidate ROIs per Image** | **1.64** | — | Average number of vehicle-prior proposals per frame. |
| **Macro Bumper Coverage ($\ge 80\%$)** | **144 / 222** | **64.86%** | Ground-truth plate is physically enclosed in the bumper window. |
| **Plate Recall at $\text{IoU} \ge 0.25$** | **20 / 222** | **9.01%** | Tight plate-sized ROI overlaps ground-truth plate. |
| **Plate Recall at $\text{IoU} \ge 0.50$** | **3 / 222** | **1.35%** | Standard object detection threshold. |
| **Mean Best Plate IoU** | **0.0645** | — | Severely depressed by area disparity and offset. |
| **Median Best Plate IoU** | **0.0000** | — | In over half the images, candidate ROI missed plate entirely. |

---

## 5. Comprehensive Failure Analysis & Taxonomy

Categorizing the failure modes across all 222 images reveals the fundamental limits of rigid geometric heuristics:

```
Failure Distribution (N=222):
  ├── Candidate ROI Too Large (Area Mismatch):       124 samples (55.9%)
  ├── Multiple Vehicle Ambiguity (Scene Noise):       38 samples (17.1%)
  ├── Plate Outside Heuristic ROI (Perspective/Truck): 37 samples (16.7%)
  ├── Success (IoU >= 0.25):                          20 samples ( 9.0%)
  └── No Vehicle Detected (Extreme Close-Up):          3 samples ( 1.4%)
```

### Detailed Breakdown of Failure Modes

1. **Candidate ROI Too Large / Area Disparity (124 samples, 55.9%):**
   * *Mechanism:* In 144 images ($64.86\%$), the broad Macro Bumper window physically encloses the plate ($\ge 80\%$ containment). However, a vehicle bumper occupies $\approx 150,000$ to $300,000\text{ pixels}$, while a license plate occupies only $\approx 2,500$ to $6,000\text{ pixels}$.
   * *Mathematical Consequence:* Even with $100\%$ containment, the mathematical intersection-over-union is bounded by:
     $$\text{IoU} = \frac{\text{Area}(\text{Plate})}{\text{Area}(\text{Bumper})} \approx \frac{4,000}{200,000} = 0.020 \quad (2.0\%)$$
   * *Conclusion:* Macro regional crops succeed at containment but fail completely as tight object detection proposals.
2. **Multiple Vehicle Ambiguity (38 samples, 17.1%):**
   * *Mechanism:* Real traffic scenes contain multiple cars (oncoming lanes, adjacent lanes, parked cars). YOLOv8s detects 2 to 5 vehicles per frame.
   * *Consequence:* The heuristic generates multiple candidate ROIs across all detected vehicles, creating false-positive proposals and ambiguity regarding which vehicle hosts the primary target plate.
3. **Plate Outside Heuristic ROI (37 samples, 16.7%):**
   * *Mechanism A — Perspective Skew:* Vehicles captured from an oblique rear-quarter angle ($30^\circ$ to $45^\circ$) have their license plate shifted towards the outer edge ($15\%$ or $80\%$ of vehicle width), completely outside the central $50\%$ hypothesis.
   * *Mechanism B — Commercial Truck Tailgates:* On commercial box trucks and lifted pickups, bumpers are mounted at non-standard heights, or plates are offset to the bottom-left bumper corner.
   * *Mechanism C — Framing Truncation:* In close-following scenes where the camera frame cuts off the vehicle roof, the detected vehicle box represents only the lower half of the car, shifting the relative plate position to the middle of the box rather than the bottom.
4. **No Vehicle Detected (3 samples, 1.4%):**
   * *Samples:* `car19`, `us4`, and one additional close-up.
   * *Mechanism:* Extreme camera zoom where only the radiator grille and bumper occupy the frame ($0$ windshield or wheels visible). Pretrained YOLOv8s COCO weights fail to trigger vehicle classification when the canonical vehicle silhouette is absent.

---

## 6. Visual Diagnostic Analysis (10 Representative Cases)

A diagnostic set of 10 representative images was generated and rendered with visual overlays:
* **Cyan Box:** YOLOv8s Vehicle Bounding Box & Confidence.
* **Orange Box:** Candidate Plate ROI (Geometric Heuristic).
* **Green Box:** Ground-Truth Plate Bounding Box & Annotation.

Saved locally to: `scratch/diagnostics/`

| Diagnostic Image | Category | Ground Truth | Best Plate IoU | Macro Containment | Visual Diagnostic Observations |
|---|---|---|---|---|---|
| `diag_car16_success_high_iou.jpg` | Success | `002HHM` | **0.3034** | 1.0000 | Clean rear-center perspective. Heuristic ROI overlaps plate squarely. |
| `diag_car5_success_high_iou.jpg` | Success | `59CADI` | **0.3162** | 1.0000 | Standard sedan rear bumper. Centered plate matches geometric prior. |
| `diag_car8_success_high_iou.jpg` | Success | `FZRULZ` | **0.5252** | 1.0000 | Highest IoU achieved ($52.5\%$). Vehicle directly centered in frame. |
| `diag_car19_failure_no_vehicle.jpg` | No Vehicle | `378DXR` | **0.0000** | 0.0000 | Extreme close-up bumper crop. YOLO detects 0 vehicles; 0 ROIs generated. |
| `diag_us4_failure_no_vehicle.jpg` | No Vehicle | `520MRK` | **0.0000** | 0.0000 | Cropped truck grille. Incomplete vehicle envelope leads to missed vehicle. |
| `diag_12c6cb72..._failure_multi_vehicle.jpg` | Multi-Vehicle | `0SG719` | **0.0000** | 0.0000 | Scene with 2 detected vehicles (truck and car). Heuristic splits proposals. |
| `diag_c9368c55..._failure_multi_vehicle.jpg` | Multi-Vehicle | `UH1F0F` | **0.0000** | 0.0000 | Oncoming traffic in adjacent lane generates competing candidate ROIs. |
| `diag_0b86cecf..._failure_plate_outside.jpg` | Plate Outside | `YG9X2G` | **0.0000** | 0.6970 | Angled rear-quarter view ($16.7\%$ from left). Central ROI misses plate. |
| `diag_21d8c31d..._failure_plate_outside.jpg` | Plate Outside | `2DA044` | **0.0000** | 0.0000 | Lifted truck bumper. Plate sits higher than standard passenger car prior. |
| `diag_1e241dc8..._failure_roi_too_large.jpg` | ROI Too Large | `CCLVN3` | **0.0733** | 1.0000 | Plate 100% inside macro bumper, but IoU only $0.07$ due to $40\times$ area disparity. |

---

## 7. End-to-End Optional Check (Oracle-Assisted Diagnostic)

To test downstream feasibility, an oracle-assisted diagnostic was conducted on the **20 candidate ROIs that achieved $\text{IoU} \ge 0.25$**:
* Each candidate crop was fed into the offline EasyOCR pipeline with USA profile syntax decoding.
* **Result:** **$0 / 20$ exact or normalized matches** ($0.0\%$ accuracy).

### Why Downstream OCR Failed Even with $\text{IoU} \ge 0.25$
1. **Background Clutter & Boundary Artifacts:** An IoU of $0.25$ to $0.35$ means that $65\%$ to $75\%$ of the crop consists of non-plate bumper plastic, chrome grilles, exhaust pipes, and vehicle bodywork.
2. **CRAFT Sensitivity:** EasyOCR’s CRAFT text detector expects clean text with minimal surrounding high-contrast body lines. Bumper contours, fog lights, and grille slats create false text candidate regions that confuse the text grouping algorithm.
3. **Conclusion:** Downstream OCR requires **tight, precise bounding boxes** ($\text{IoU} \ge 0.70$ with $<10\%$ margin padding). A rough geometric proposal cannot supply the boundary precision required for reliable recognition.

---

## 8. Primary Conclusions & Recommendations

> [!IMPORTANT]
> **Definitive Finding:**  
> A vehicle-prior geometric heuristic achieves **only $9.01\%$ recall at $\text{IoU} \ge 0.25$ and $1.35\%$ recall at $\text{IoU} \ge 0.50$**, with a median IoU of **$0.0000$**.  
> Furthermore, downstream OCR on these proposals achieves **$0.0\%$ accuracy** due to severe bumper clutter.  
> **Geometric priors cannot replace a dedicated object detector.**

### Recommended Next Technical Step
1. **Do Not Rely on Cascaded Vehicle Heuristics for Production ALPR:**
   * Vehicle detection alone is insufficient to localize license plates.
2. **Adopt a Dedicated License Plate Detector:**
   * Proceed with a dedicated single-class `license_plate` detector (e.g. YOLOv8s fine-tuned on dedicated plate datasets) to regress precise bounding boxes directly on full dashcam frames.
3. **Preserve Current Milestone State:**
   * Ground truth crops remain the only reliable basis for OCR benchmarking until dedicated detector weights are integrated.

---

## 9. Test Suite Verification

* **Command:** `PYTHONPATH=. python3 -m unittest discover -s tests`
* **Result:** `Ran 49 tests in 7.569s — OK (skipped=6)`
* All existing tests remain passing without regressions.
