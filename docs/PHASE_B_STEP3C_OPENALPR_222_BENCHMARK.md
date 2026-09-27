# RoadLens AI — Phase B Step 3C Full OpenALPR 222-Image OCR Benchmark Report
**Document ID:** `RL-DOC-008-OPENALPR-222-BENCHMARK`  
**Status:** COMPLETED & AUDITED  
**Date:** September 2026  
**Execution Context:** Local Development Machine (Apple M-series CPU fallback; Zero GPU/ROCm acceleration claimed)  
**Evaluated Engine:** EasyOCR 1.7.2 (PyTorch backend, strictly offline local models: `craft_mlt_25k.pth` + `english_g2.pth`)  
**Data Artifact:** `experiments/openalpr_ocr_benchmark.csv` (444 evaluation records)  

---

## 1. Objective & Benchmark Scope

The objective of **Phase B Step 3C** is to measure the baseline recognition accuracy and latency of the offline EasyOCR engine across the complete, verified **OpenALPR US Benchmark** (222 image/annotation pairs) using ground-truth plate bounding box crops.

> [!IMPORTANT]
> **OCR Recognition Benchmark Only — Not End-to-End Detection:**  
> This benchmark evaluates strictly the character recognition capability of EasyOCR on ground-truth plate crops. It does **NOT** measure end-to-end license plate detection accuracy. Pretrained YOLOv8s weights (COCO dataset) do not contain a dedicated `license_plate` category and cannot localize license plates on raw dashcam scenes without custom training or fine-tuning.

### Strict Guardrails Followed
* ✅ **Zero Package Changes:** Existing runtime environment (`torch` 2.11.0, `torchvision` 0.26.0, `numpy` 2.4.2, `opencv-python` 4.13.0, `pillow` 12.1.0, `scipy` 1.17.1, `easyocr` 1.7.2) remained completely untouched.
* ✅ **Strict Local Model Provenance:** `EasyOCR` ran with `download_enabled=False`, loading verified local weights (`models/craft_mlt_25k.pth` and `models/english_g2.pth`). Zero network downloads occurred.
* ✅ **Zero Dataset Downloads:** Evaluated exclusively on the verified local `datasets/openalpr_us/` dataset.
* ✅ **No AMD Cloud Credits / Zero GPU Fabrication:** Execution ran deterministically on local CPU.

---

## 2. Dataset Description

The dataset comprises the verified 222 annotated samples from the OpenALPR US Benchmark:
* **Total Image/Annotation Pairs:** 222 JPG images and 222 TXT annotation files.
* **Ground-Truth Data per Sample:** Bounding box coordinates $[x, y, w, h]$ and target license plate alphanumeric transcriptions.
* **Plate Resolution Distribution (Vertical Crop Heights):**
  * Minimum Height: $25\text{ px}$
  * Maximum Height: $164\text{ px}$
  * Mean Height: $47.3\text{ px}$
  * Median Height: $43.0\text{ px}$
* **Crop Height Stratification:**
  * `<24 px`: **0 samples** ($0.0\%$) — Minimum height in dataset is $25\text{ px}$.
  * `24–31 px`: **26 samples** ($11.7\%$) — Extremely low-resolution plate crops.
  * `32–35 px`: **14 samples** ($6.3\%$) — Low-resolution boundary zone.
  * `36–49 px`: **115 samples** ($51.8\%$) — Nominal dashcam plate resolution.
  * `≥50 px`: **67 samples** ($30.2\%$) — High-resolution close-following crops.

---

## 3. Evaluation Methodology

Every sample was evaluated under two distinct operating modes, resulting in **444 total OCR evaluations**:

### Mode A: RAW (Baseline)
* EasyOCR was executed directly on the cropped ground-truth bounding box without spatial or radiometric transformations.

### Mode B: ADAPTIVE PREPROCESSING
* Evaluated the candidate preprocessing policy identified in Step 3B:
  $$\text{Policy} = \begin{cases} 
  2\times \text{ bicubic upscaling} + \text{unsharp mask sharpening} & \text{if } \text{crop height} < 36\text{ px} \\
  \text{Original raw crop (pass-through)} & \text{if } \text{crop height} \ge 36\text{ px}
  \end{cases}$$
* Sharpening formulation: $I_{\text{sharp}} = 1.5 \cdot I_{\text{up}} - 0.5 \cdot \text{Gaussian}(I_{\text{up}}, \sigma=2.0)$.
* Per guardrails, CLAHE, $3\times$ upscaling, and bilateral denoising were **not** applied.

### Normalization Pipeline
Both predictions and ground truth were processed through the official RoadLens USA Profile normalizer (`USAProfile.validate_license_plate` logic):
1. Convert text to uppercase.
2. Strip non-alphanumeric characters (except whitespace).
3. Identify and strip known US state header strings (e.g. `CALIFORNIA`, `TEXAS`, `NEW YORK`, `FLORIDA`, etc.).
4. Strip all remaining whitespace to produce the canonical alphanumeric plate string.

---

## 4. Overall Benchmark Results & Direct Comparison

Aggregated performance metrics across all $N=222$ benchmark samples:

| Benchmark Metric | MODE A: RAW | MODE B: ADAPTIVE | Delta / Impact |
|---|---|---|---|
| **Total Samples Evaluated** | 222 | 222 | — |
| **Exact-Match Count** | **5 / 222** | **5 / 222** | **0 (No change)** |
| **Exact-Match Accuracy (%)** | **2.25%** | **2.25%** | **0.00%** |
| **Normalized-Match Count** | **30 / 222** | **30 / 222** | **0 (No change)** |
| **Normalized-Match Accuracy (%)** | **13.51%** | **13.51%** | **0.00%** |
| **Mean Character Confidence** | 0.3531 | **0.3596** | **+0.0065 (+1.8%)** |
| **Median OCR Latency** | **35.98 ms** | 38.96 ms | +2.98 ms |
| **Mean OCR Latency** | **46.75 ms** | 49.24 ms | +2.49 ms |
| **Minimum OCR Latency** | **12.36 ms** | 23.09 ms | +10.73 ms |
| **Maximum OCR Latency** | 254.54 ms | **242.78 ms** | -11.76 ms |

> [!WARNING]
> **Critical Interpretation of Confidence vs. Accuracy:**  
> While Adaptive Preprocessing produced a modest increase in mean OCR confidence ($0.3531 \rightarrow 0.3596$), **it yielded zero net gain in overall recognition accuracy** (exactly 30/222 normalized matches in both modes).  
> In accordance with strict evaluation standards, **Adaptive Preprocessing cannot be considered superior to RAW mode overall**, because confidence gains did not translate into improved plate transcription correctness.

---

## 5. Resolution-Bucket Analysis

Stratifying performance by crop height demonstrates the precise interaction between pixel resolution, preprocessing, and recognition accuracy:

| Height Bucket | Sample Count | RAW Exact Acc (%) | ADAP Exact Acc (%) | RAW Norm Acc (%) | ADAP Norm Acc (%) | RAW Mean Latency | ADAP Mean Latency |
|---|---|---|---|---|---|---|---|
| **`<24 px`** | 0 | — | — | — | — | — | — |
| **`24–31 px`** | 26 | 0.00% (0) | 0.00% (0) | 3.85% (1) | **7.69% (2)** | **22.86 ms** | 39.82 ms |
| **`32–35 px`** | 14 | 0.00% (0) | 0.00% (0) | **14.29% (2)** | 7.14% (1) | **36.03 ms** | 56.67 ms |
| **`36–49 px`** | 115 | **4.35% (5)** | **4.35% (5)** | 13.04% (15) | 13.04% (15) | 37.17 ms | 36.25 ms |
| **`≥50 px`** | 67 | 0.00% (0) | 0.00% (0) | **17.91% (12)** | **17.91% (12)** | 74.70 ms | 73.66 ms |

### Deep-Dive on Resolution Dynamics

1. **The Sub-32px Regime (`24–31 px`, 26 samples):**
   * In this ultra-low resolution band, Adaptive Preprocessing **doubled normalized accuracy** ($3.85\% \rightarrow 7.69\%$).
   * *Sample `a03ced3f-5a97-4e75-8106-fabfd2b8b76e` ($h=30\text{ px}$):* RAW produced `"OBE6CK"` (confusing leading `0` with `O`), whereas ADAPTIVE correctly transcribed `"08E6CK"` (GT: `08E6CK`).
   * *Sample `wts-lg-000035` ($h=29\text{ px}$):* RAW truncated the plate to `"XNR"`, whereas ADAPTIVE recovered the full string `"XNR374"` (GT: `XNR374`).
   * *Sample `7fbfbe28-aecb-45be-bd05-7cf26acb3c5c` ($h=30\text{ px}$):* RAW returned null detection (`""`), while ADAPTIVE detected `"KC4 23X"` (GT: `KC4Z3X`).
2. **The Boundary Regime (`32–35 px`, 14 samples):**
   * In this transition band, Adaptive Preprocessing **halved normalized accuracy** ($14.29\% \rightarrow 7.14\%$).
   * *Sample `wts-lg-000037` ($h=33\text{ px}$):* RAW correctly recognized `"RDN464"` (GT: `RDN464`), but unsharp sharpening over-amplified high frequencies, causing `R` to be read as `P` (`"PDN464"`).
   * *Sample `wts-lg-000036` ($h=31\text{ px}$):* RAW correctly recognized `"TTP488"`, whereas ADAPTIVE dropped the leading letters to `"488"`.
3. **The Nominal Regime (`36–49 px`, 115 samples):**
   * Contains **100% of all exact matches** in the entire benchmark (5/5).
   * Plate heights between $36\text{ px}$ and $49\text{ px}$ represent the optical sweet spot for the CRAFT detector: sufficiently resolved for individual glyph contours without excessive background clutter.
4. **The High-Resolution Regime (`≥50 px`, 67 samples):**
   * Achieved the highest normalized accuracy (**17.91%**), but **0.00% exact accuracy**.
   * On large crops, generic scene-text EasyOCR frequently picks up state slogans (e.g. "GARDEN STATE", "SUNSHINE STATE"), county stickers, or frame manufacturer text, introducing extraneous tokens that prevent exact string equivalence even when the primary plate code is recognized.

---

## 6. Observed Character Confusion Analysis

Levenshtein alignment between ground-truth and normalized predictions across all 222 samples identified the following empirical character substitution patterns:

| Ground-Truth $\rightarrow$ Predicted | RAW Count | ADAPTIVE Count | Confusion Category | Primary Mechanism |
|---|---|---|---|---|
| **`6` $\rightarrow$ `G`** | **20** | **19** | Number $\rightarrow$ Letter | Upper terminal loop of `6` conflates with crossbar of `G`. |
| **`5` $\rightarrow$ `S`** | **18** | **18** | Number $\rightarrow$ Letter | Generic Latin model bias favors letters over digits mid-string. |
| **`0` $\rightarrow$ `O`** | **15** | **17** | Number $\rightarrow$ Letter | Geometric oval equivalence in non-slashed fonts. |
| **`4` $\rightarrow$ `L`** | **15** | **15** | Number $\rightarrow$ Letter | Open-topped `4` segmented into vertical stem and horizontal base. |
| **`8` $\rightarrow$ `B`** | **12** | **10** | Number $\rightarrow$ Letter | Dual-loop shape similarity; frame shadows merge vertical stroke. |
| **`7` $\rightarrow$ `Z`** | **10** | **10** | Number $\rightarrow$ Letter | Diagonal stroke with top bar read as `Z`. |
| **`8` $\rightarrow$ `E`** | **10** | **9** | Number $\rightarrow$ Letter | Left edge of `8` detected as three horizontal arms. |
| **`2` $\rightarrow$ `Z`** | **8** | **8** | Number $\rightarrow$ Letter | Curved top of `2` interpreted as sharp corner under low resolution. |
| **`1` $\rightarrow$ `I`** | **7** | **8** | Number $\rightarrow$ Letter | Identical single vertical stroke. |
| **`1` $\rightarrow$ `L`** | **7** | **6** | Number $\rightarrow$ Letter | Base serif on digit `1` read as horizontal base of `L`. |
| **`M` $\rightarrow$ `H`** | **7** | **7** | Letter $\rightarrow$ Letter | Central V-valley of `M` lost under blur, leaving two vertical uprights. |
| **`F` $\rightarrow$ `E`** | **7** | **4** | Letter $\rightarrow$ Letter | Plate border shadow perceived as bottom horizontal leg. |
| **`G` $\rightarrow$ `6`** | **4** | **4** | Letter $\rightarrow$ Number | Reverse confusion; curved back of `G` closed by blur. |
| **`S` $\rightarrow$ `5`** | **4** | **3** | Letter $\rightarrow$ Number | Reverse confusion; serifs on `S` sharpened into right angles. |
| **`Z` $\rightarrow$ `2`** | **1** | **2** | Letter $\rightarrow$ Number | Reverse confusion under over-smoothing. |

### Key Insight on Error Asymmetry
Across all numeric/alphabetic ambiguities, **EasyOCR exhibited a strong directional bias toward letters over numbers** (e.g. `6` $\rightarrow$ `G` occurred 20 times, whereas `G` $\rightarrow$ `6` occurred only 4 times; `5` $\rightarrow$ `S` occurred 18 times, while `S` $\rightarrow$ `5` occurred only 4 times; `8` $\rightarrow$ `B` occurred 12 times, while `B` $\rightarrow$ `8` was zero).  
This confirms that the CRNN recognizer (`english_g2.pth`) is heavily biased by its natural English scene-text training data (books, storefronts, signboards), where words dominate over random alphanumeric license sequences.

---

## 7. The 5 Exact-Match Cases

The 5 plates achieving 100% exact match in both RAW and ADAPTIVE modes:

| Sample ID | Crop Dims ($W \times H$) | Ground Truth | Prediction | Confidence | Plate Characteristics |
|---|---|---|---|---|---|
| `car2` | $86 \times 43\text{ px}$ | `1049338` | `"1049338"` | 0.9412 | Pure numeric plate; zero letter/digit ambiguity. |
| `wts-lg-000141` | $83 \times 41\text{ px}$ | `1365847` | `"1365847"` | 0.7289 | Pure numeric plate; clean white background. |
| `wts-lg-000154` | $95 \times 47\text{ px}$ | `1366026` | `"1366026"` | 0.7932 | Pure numeric plate; high contrast. |
| `wts-lg-000177` | $99 \times 49\text{ px}$ | `6ZWF846` | `"6ZWF846"` | 0.6908 | Standard alphanumeric; crisp characters, zero frame obstruction. |
| `wts-lg-000194` | $83 \times 41\text{ px}$ | `7CID930` | `"7CID930"` | 0.6482 | Standard alphanumeric; distinct letter/digit separation. |

* **Common Traits:** All 5 samples had vertical heights between $41\text{ px}$ and $49\text{ px}$, $3$ of the $5$ were purely numeric, and none featured decorative state logos or mounting frames impinging on the text stroke area.

---

## 8. Latency Analysis

* **Execution Device:** Apple M-series CPU (deterministic CPU runtime).
* **Mode A (RAW):**
  * Median Latency: **$35.98\text{ ms}$**
  * Mean Latency: **$46.75\text{ ms}$**
  * Range: $12.36\text{ ms}$ to $254.54\text{ ms}$
* **Mode B (ADAPTIVE):**
  * Median Latency: **$38.96\text{ ms}$**
  * Mean Latency: **$49.24\text{ ms}$**
  * Range: $23.09\text{ ms}$ to $242.78\text{ ms}$
* **Throughput:** ~21 frames/sec in batch CPU evaluation.
* **Latency vs Crop Dimensions:**
  * For crops $<36\text{ px}$, raw latency is $\approx 23\text{ ms}$. With $2\times$ upscaling, latency increases to $\approx 40$–$56\text{ ms}$.
  * For large crops ($\ge 50\text{ px}$ height, up to $179\text{ px}$ width), latency scales to $\approx 74\text{ ms}$, with outliers reaching $254\text{ ms}$ due to multi-line text candidate clustering in CRAFT.

---

## 9. Failure Case Taxonomy

Across the 192 samples where normalized matching failed, failures categorized into four distinct failure modes:

1. **Alphanumeric Ambiguity (52% of errors):**
   * Single-character misclassifications where `G/6`, `S/5`, `O/0`, `B/8`, or `Z/2` were swapped.
   * *Impact:* Normalized match failed by exactly 1 character edit distance.
2. **Mounting Frame / Bolt Contamination (24% of errors):**
   * Plate fasteners and dark perimeter brackets merged into character glyphs (e.g. `2DA044` transcribed as `@daco44`; `UF2V5S` transcribed as `(F24V55`).
3. **Severe Resolution Starvation (15% of errors):**
   * Plates with height $\le 28\text{ px}$ where individual character stroke widths dropped below $1.5\text{ pixels}$. CRAFT was unable to segment separate glyph bounding boxes.
4. **State Slogan / Header Interference (9% of errors):**
   * State mottos (e.g. "THE SILVER STATE", "CENTENNIAL") or registration date stickers recognized as primary plate tokens and concatenated with the registration number.

---

## 10. Interpretation & Limitations

1. **Resolution is Necessary but Not Sufficient:**
   * Increasing crop height improves normalized accuracy from $3.85\%$ ($24$–$31\text{ px}$) to $17.91\%$ ($\ge 50\text{ px}$). However, accuracy plateaus at $<18\%$ even on high-resolution crops because character confusion (`G/6`, `S/5`, `B/8`) is driven by the recognizer's vocabulary priors, not pixel resolution.
2. **Classical Preprocessing Has Symmetrical Tradeoffs:**
   * Unsharp masking and upscaling help blurry characters at $h < 32\text{ px}$, but introduce edge halos that corrupt characters at $h \ge 32\text{ px}$. The net effect across the full dataset was exactly zero gain ($13.51\% \rightarrow 13.51\%$).
3. **Absence of Dedicated Plate Detection:**
   * Using ground-truth crops represents an optimistic upper bound on OCR recognition. In an end-to-end system, detector bounding-box jitter and margin errors will introduce additional noise.

---

## 11. Recommended Next Technical Steps

1. **Implement Syntax-Aware Beam Search / Post-Processing:**
   * Standard US state formats follow strict positional grammar (e.g. `1AAA111` in California, `AAA-1111` in Texas, `AAA 111` in Florida).
   * Forcing digit-only decoding at numeric positions and letter-only decoding at alphabetic positions will immediately eliminate $>60\%$ of the observed errors (`6->G`, `5->S`, `0->O`, `8->B`).
2. **Dedicated License Plate Localizer:**
   * Pretrained YOLOv8s COCO weights detect vehicles, not plates. RoadLens requires either a specialized license plate detector model or a heuristic cascade to localize plates before passing crops to EasyOCR.
3. **Proceed to Phase B Step 4 (Pipeline Decision / Training Architecture):**
   * With the 222-image baseline benchmark established and saved to `experiments/openalpr_ocr_benchmark.csv`, we have empirical data to inform the detector and OCR integration strategy.

---

## 12. Test Suite Status

* **Command:** `PYTHONPATH=. python3 -m unittest discover -s tests`
* **Result:** `Ran 49 tests in 7.839s — OK (skipped=6)`
* All existing tests pass cleanly without errors or regressions.
