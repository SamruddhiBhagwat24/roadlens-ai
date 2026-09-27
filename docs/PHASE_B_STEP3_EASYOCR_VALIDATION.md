# RoadLens AI — EasyOCR Installation & Validation Report
**Document ID:** `RL-DOC-006-EASYOCR-VALIDATION`  
**Status:** COMPLETED & VERIFIED  
**Date:** September 2026  
**Execution Context:** Local Development Machine (CPU Execution, Zero AMD Fabrication)

---

## 1. Executive Summary

In strict accordance with the explicit authorization for **Phase B Step 3A**:
* ✅ Only the `easyocr` (v1.7.2) package and its required non-conflicting auxiliary dependencies (`python-bidi`, `pyclipper`, `shapely`, `scikit-image`, `ninja`) were installed.
* ✅ **Zero core dependencies were modified:** `torch` (2.11.0), `torchvision` (0.26.0), `numpy` (2.4.2), `opencv-python` (4.13.0), `pillow` (12.1.0), and `scipy` (1.17.1) remain **100% identical** to their pre-installation versions.
* ✅ **Strict Local Model Provenance:** `easyocr.Reader` was configured with `model_storage_directory="models"`, `user_network_directory="models"`, and `download_enabled=False`. EasyOCR loaded the existing verified local files (`craft_mlt_25k.pth` and `english_g2.pth`) completely offline. Zero internet downloads occurred.
* ✅ **Controlled OCR Validation:** Executed directly on ground-truth plate crops for the 5 representative OpenALPR samples.
* 🚫 **No datasets downloaded** (zero LISA, zero India datasets).
* 🚫 **No model weights downloaded**.
* 🚫 **No training or fine-tuning performed**.
* 🚫 **No AMD Developer Cloud GPU credits consumed**.
* 🚫 **No fabricated plate detections:** This was strictly an OCR recognition evaluation on ground-truth crops. Pretrained YOLOv8s COCO weights do not detect license plates.

---

## 2. Environment & Dependency State Verification

### Core Package Version Audit
| Dependency | Pre-Install Version | Post-Install Version | Status | Integrity Check |
|---|---|---|---|---|
| **EasyOCR** | *Not Installed* | **1.7.2** | ✅ Successfully Installed | Official PyPI package |
| **Python** | 3.11.0 | **3.11.0** | ✅ Preserved | Framework Python 3.11 |
| **PyTorch (`torch`)** | 2.11.0 | **2.11.0** | ✅ **Untouched** | No replacement or upgrade |
| **TorchVision (`torchvision`)** | 0.26.0 | **0.26.0** | ✅ **Untouched** | No replacement or upgrade |
| **OpenCV (`cv2`)** | 4.13.0 | **4.13.0** | ✅ **Untouched** | `opencv-python` preserved |
| **NumPy (`numpy`)** | 2.4.2 | **2.4.2** | ✅ **Untouched** | No replacement or upgrade |
| **Pillow (`PIL`)** | 12.1.0 | **12.1.0** | ✅ **Untouched** | No replacement or upgrade |
| **SciPy (`scipy`)** | 1.17.1 | **1.17.1** | ✅ **Untouched** | No replacement or upgrade |

All imports succeeded cleanly in a standalone verification run. Pre-installation freeze snapshot is preserved at `docs/.pre_install_freeze.txt`.

---

## 3. Local Model Loading Verification

* **Model Directory:** `/Users/samruddhibhagwat/roadlens-ai-local/models`
* **Local Weights Loaded:**
  * Detection Model: `models/craft_mlt_25k.pth` ($83,152,330\text{ bytes}$)
  * Recognition Model: `models/english_g2.pth` ($15,143,997\text{ bytes}$)
* **EasyOCR Configuration:**
  ```python
  reader = easyocr.Reader(
      ["en"],
      gpu=False,
      model_storage_directory="models",
      user_network_directory="models",
      download_enabled=False
  )
  ```
* **Verification Result:**
  * `Detector network:` `<class 'easyocr.craft.CRAFT'>`
  * `Recognizer network:` `<class 'easyocr.model.vgg_model.Model'>`
  * Reader initialization latency: $2,661.3\text{ ms}$ (one-time cold load).
  * `download_enabled=False` was strictly respected; no network connections were attempted.

---

## 4. Controlled 5-Sample OCR Validation Results

The 5 representative test samples from `datasets/openalpr_us/` were evaluated by feeding the exact ground-truth bounding box crop directly into EasyOCR:

| Sample Index | Sample ID | Crop Dims ($W \times H$) | Ground Truth | EasyOCR Prediction | Mean Conf | Latency | Exact Match? | Normalized Match? | Error Analysis |
|---|---|---|---|---|---|---|---|---|---|
| **1** | `0b86cecf-67d1-4fc0-87c9-b36b0ee228bb` | $99 \times 49\text{ px}$ | `YG9X2G` | `"YG9-X2G"` | **0.65** | $93.7\text{ ms}$ | ❌ False | ✅ **True** | Hyphen separator recognized; normalized plate syntax matches ground truth perfectly. |
| **2** | `12c6cb72-3ea3-49e7-b381-e0cdfc5e8960` | $62 \times 31\text{ px}$ | `0SG719` | `"056*719"` | **0.34** | $24.8\text{ ms}$ | ❌ False | ❌ False | Low resolution ($31\text{ px}$ height). Character confusion: `S` $\rightarrow$ `5`, `G` $\rightarrow$ `6`, dot $\rightarrow$ `*`. |
| **3** | `1e241dc8-8f18-4955-8988-03a0ab49f813` | $63 \times 31\text{ px}$ | `CCLVN3` | `"CcLV <"` | **0.11** | $20.7\text{ ms}$ | ❌ False | ❌ False | Low resolution ($31\text{ px}$ height). Trailing characters `N3` occluded/truncated as `<`. |
| **4** | `21d8c31d-3deb-494b-9c63-c0223306fd82` | $58 \times 29\text{ px}$ | `2DA044` | `"@daco44"` | **0.26** | $24.9\text{ ms}$ | ❌ False | ❌ False | Very low resolution ($29\text{ px}$ height). Initial `2` detected as `@`, case lowered (`daco44`). |
| **5** | `22e54a62-57a8-4a0a-88c1-4b9758f67651` | $58 \times 29\text{ px}$ | `SL7C6S` | `"5L7 CGs"` | **0.51** | $20.4\text{ ms}$ | ❌ False | ❌ False | Very low resolution ($29\text{ px}$ height). `S` $\rightarrow$ `5`, space inserted, `6` $\rightarrow$ `G`. |

---

## 5. Performance & Accuracy Metrics Summary

* **Execution Device:** **CPU** (Apple M-series CPU fallback; zero GPU/ROCm acceleration claimed).
* **Mean OCR Inference Latency:** **$36.89\text{ ms}$ per crop** (Cold warm-up: $93.7\text{ ms}$, Steady state: $20.4$–$24.9\text{ ms}$).
* **Exact-Match Accuracy:** **$0.0\%$** ($0 / 5$).
* **Normalized-Match Accuracy:** **$20.0\%$** ($1 / 5$).
* **Character Error Patterns Observed:**
  1. *Resolution Sensitivity:* In crops $< 60\text{ px}$ in width and $< 32\text{ px}$ in height, standard generic Latin scene text models suffer from character blur and ambiguity.
  2. *Digit vs. Letter Confusion:* `S` vs `5`, `G` vs `6`, `O` vs `0`, `2` vs `@`.
  3. *Noise Artifacts:* Mounting screws, plate frames, and state logos occasionally generate punctuation tokens (e.g. `*`, `-`, `<`).

---

## 6. Architectural Insights & Technical Takeaways

1. **OCR Pipeline Operates as Designed:** EasyOCR runs natively in PyTorch, loads local weights cleanly with `download_enabled=False`, and reports real character confidences and millisecond latencies without fabrication.
2. **Need for Plate Resolution Scaling:** Real road vehicle license plates in $720\text{p}$ and $1080\text{p}$ scenes occupy tiny pixel regions ($29$–$49\text{ px}$ vertical height). Applying bicubic upscaling ($2\times$) or super-resolution prior to OCR recognition is a key enhancement for subsequent milestones.
3. **Role of Country Profile Syntax Postprocessing:** In Sample 1, raw OCR returned `YG9-X2G`. The USA Country Profile syntax validator (`USAProfile.validate_license_plate()`) strips hyphens and verifies state alphanumeric formats, transforming `YG9-X2G` $\rightarrow$ `YG9X2G` (100% ground-truth recovery).
4. **Detection Separation Reaffirmed:** Pretrained YOLOv8s COCO weights detect vehicles, not license plates. Plate recognition on end-to-end full images requires either custom fine-tuned weights or heuristic rectangular proposals.

---

## 7. Test Suite Status

* **Command:** `PYTHONPATH=. python3 -m unittest discover -s tests`
* **Total Tests:** **49**
* **Passed:** **43**
* **Failed:** **0**
* **Skipped:** **6** (Integration tests awaiting live web server)
* All existing tests (pipeline, image quality, validation, profiles, models) remain fully passing.
