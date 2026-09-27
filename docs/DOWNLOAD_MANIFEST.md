# RoadLens AI — Download Manifest
**Document ID:** `RL-DOC-004-MANIFEST`  
**Status:** AUDITED & VERIFIED  
**Date:** September 2026  
**Phase:** Phase B Step 2 (Approved Asset Downloads)  
**Execution Context:** Local Development Machine (CPU fallback / AMD ROCm compatible architecture)

---

## 1. Executive Summary

In strict accordance with the explicit authorization for **Phase B Step 2**:
* ✅ Only the approved, verified models and benchmark datasets have been downloaded.
* ✅ All assets were retrieved exclusively from their authoritative primary sources.
* ✅ Model weights are stored exclusively in `models/`.
* ✅ Benchmark dataset files are stored exclusively in `datasets/openalpr_us/`.
* ⚠️ **LISA Traffic Sign Dataset** remains **UNRESOLVED / NOT DOWNLOADED** because the authoritative primary UCSD URL (`https://cvrr.ucsd.edu/lisa/traffic-sign-dataset.html`) returns HTTP 404. No mirror, Kaggle, or alternate university portal was accessed.
* 🚫 **No Indian datasets** were downloaded.
* 🚫 **No unauthorized datasets** (DAWN, BDD100K, MTSD, Unidata, Mendeley NPDS) were downloaded.
* 🚫 **No Python packages** were installed.
* 🚫 **No production source code** was modified.
* 🚫 **No AMD Developer Cloud GPU credits** were consumed.

---

## 2. Master Download Inventory

| Asset Name | Exact Filename / Directory | Source Repository / Direct URL | Release / Version | License | Download Size | Extracted Size | SHA256 Checksum | Destination Path | Verification Status |
|---|---|---|---|---|---|---|---|---|---|
| **YOLOv8s Pretrained Weights** | `yolov8s.pt` | [Ultralytics Assets](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt) | `v8.3.0` | AGPL-3.0 | 22,588,772 bytes (~22.59 MB) | 22,588,772 bytes (~22.59 MB) | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` | `models/yolov8s.pt` | ✅ Verified Intact (PyTorch Zip Container) |
| **EasyOCR CRAFT Text Detector** | `craft_mlt_25k.pth` | [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip) | `pre-v1.1.6` | Apache 2.0 | 77,251,756 bytes (~77.25 MB zip) | 83,152,330 bytes (~83.15 MB pth) | **PTH:** `4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17`<br>**ZIP:** `8dc6a1c703a89ed56308ef742d26ebd45c656248cbbbda6e7fe60e569f873e65` | `models/craft_mlt_25k.pth` | ✅ Verified Intact (PyTorch Pickle State Dict) |
| **EasyOCR CRNN English Recognizer** | `english_g2.pth` | [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip) | `v1.3` | Apache 2.0 | 14,040,947 bytes (~14.04 MB zip) | 15,143,997 bytes (~15.14 MB pth) | **PTH:** `e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8`<br>**ZIP:** `1b5eaebf1c062de6205560c97ffcfa8dc0e6f413c340e8adc5cfc57e159f61ff` | `models/english_g2.pth` | ✅ Verified Intact (PyTorch Torchscript Zip) |
| **OpenALPR US Plate Benchmark** | `datasets/openalpr_us/` (222 JPGs + 222 TXTs) | [openalpr/benchmarks](https://github.com/openalpr/benchmarks/archive/refs/heads/master.zip) | `master` (`endtoend/us/` subset) | Open Source Benchmark (AGPL-3.0 / CC BY 4.0 mirrors) | 217,707,688 bytes (~217.7 MB full zip) | 29,466,473 bytes (~29.47 MB extracted subset) | **ZIP:** `faec6b1814a360749792c7f8932ad78a98fe9d2b27551133635bd090747d0261` | `datasets/openalpr_us/` | ✅ Verified Intact (222 paired JPG + TXT ground truth) |
| **LISA Traffic Sign Dataset** | *Not Downloaded* | `https://cvrr.ucsd.edu/lisa/traffic-sign-dataset.html` | IEEE T-ITS 2012 | Academic Research Agreement | N/A | N/A | N/A | `datasets/lisa/` | ❌ **UNRESOLVED (HTTP 404 on official URL)** |

---

## 3. Asset Details & Integrity Verification

### Asset 1: YOLOv8s Pretrained Weights (`yolov8s.pt`)
* **Primary Role:** Object detector for US traffic signs and license plates.
* **Architecture:** YOLOv8s (11.2M parameters, 28.6B FLOPs).
* **Storage Location:** `models/yolov8s.pt`
* **File Size:** $22,588,772\text{ bytes}$ ($22.59\text{ MB}$).
* **Binary Header:** `PK\x03\x04` (PyTorch Zip Container / standard format).
* **SHA256:** `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`
* **License Compliance:** AGPL-3.0 copyleft. Permitted for hackathons and open-source research.

### Asset 2: EasyOCR CRAFT Text Detector (`craft_mlt_25k.pth`)
* **Primary Role:** Scene text and character region detector.
* **Architecture:** Character-Region Awareness For Text (CRAFT) multi-language model trained on 25k images.
* **Storage Location:** `models/craft_mlt_25k.pth`
* **Compressed Archive Size:** $77,251,756\text{ bytes}$ ($77.25\text{ MB}$).
* **Extracted State Dict Size:** $83,152,330\text{ bytes}$ ($83.15\text{ MB}$).
* **Binary Header:** `\x80\x02\x8a\n` (PyTorch Pickle tensor storage format).
* **SHA256 (Extracted `.pth`):** `4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17`
* **SHA256 (Source `.zip`):** `8dc6a1c703a89ed56308ef742d26ebd45c656248cbbbda6e7fe60e569f873e65`
* **License Compliance:** Apache 2.0 (Permissive).

### Asset 3: EasyOCR CRNN English Recognizer (`english_g2.pth`)
* **Primary Role:** Alphanumeric sequence recognition for road signs and plates.
* **Architecture:** CRNN (ResNet backbone + BiLSTM sequence modeling + CTC loss).
* **Storage Location:** `models/english_g2.pth`
* **Compressed Archive Size:** $14,040,947\text{ bytes}$ ($14.04\text{ MB}$).
* **Extracted State Dict Size:** $15,143,997\text{ bytes}$ ($15.14\text{ MB}$).
* **Binary Header:** `PK\x03\x04` (PyTorch Torchscript Zip format).
* **SHA256 (Extracted `.pth`):** `e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8`
* **SHA256 (Source `.zip`):** `1b5eaebf1c062de6205560c97ffcfa8dc0e6f413c340e8adc5cfc57e159f61ff`
* **License Compliance:** Apache 2.0 (Permissive).

### Asset 4: OpenALPR US License Plate Benchmark (`datasets/openalpr_us/`)
* **Primary Role:** Ground-truth evaluation dataset for US license plate detection and OCR.
* **Extraction Scope:** Only the verified `endtoend/us/` directory was extracted. All other directories (`endtoend/eu`, `endtoend/br`, build scripts, docs) were completely discarded.
* **File Breakdown:**
  * **222** `.jpg` road vehicle images ($29,418,655\text{ bytes}$).
  * **222** `.txt` ground-truth annotation files ($47,818\text{ bytes}$).
  * **Total Files:** Exactly 444 files.
  * **Total Extracted Size:** $29,466,473\text{ bytes}$ ($29.47\text{ MB}$).
* **Sample Annotation Integrity Check:**
  * `0b86cecf-67d1-4fc0-87c9-b36b0ee228bb.jpg` $\rightarrow$ `x: 935, y: 362, w: 99, h: 49, plate: "YG9X2G"`
  * `12c6cb72-3ea3-49e7-b381-e0cdfc5e8960.jpg` $\rightarrow$ `x: 911, y: 136, w: 62, h: 31, plate: "0SG719"`
  * `1e241dc8-8f18-4955-8988-03a0ab49f813.jpg` $\rightarrow$ `x: 569, y: 318, w: 63, h: 31, plate: "CCLVN3"`
* **Cleanup:** The full temporary repository archive ($217.7\text{ MB}$) was purged immediately after extracting `endtoend/us/`.

---

## 4. Unresolved Asset Status: LISA Traffic Sign Dataset

* **Authoritative Source Recorded:** `https://cvrr.ucsd.edu/lisa/traffic-sign-dataset.html` (UC San Diego Laboratory for Intelligent and Safe Automobiles).
* **Failure Mode:** HTTP 404 (Not Found). The host server is running, but the path is gone.
* **Current Status:** **NOT DOWNLOADED.**
* **Guardrail Enforcement:** In accordance with strict instructions:
  * No alternate university portal (e.g. Aalborg University) was used.
  * No secondary mirror, Kaggle re-upload, or Zenodo deposit was used.
  * System remains paused awaiting user instruction regarding the US traffic sign benchmark source.

---

## 5. Storage Directory Verification

```
models/
├── .gitkeep
├── craft_mlt_25k.pth  (83,152,330 bytes)
├── english_g2.pth     (15,143,997 bytes)
└── yolov8s.pt         (22,588,772 bytes)

datasets/
├── .gitkeep
└── openalpr_us/       (222 JPGs + 222 TXTs, 29,466,473 bytes)
```

No extraneous, temporary, or unverified files exist in `models/` or `datasets/`.
All entries are protected by `.gitignore` rules to prevent inadvertent tracking in version control.
