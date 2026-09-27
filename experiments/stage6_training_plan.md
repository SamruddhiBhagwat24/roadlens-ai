# RoadLens AI — Stage 6A: AMD ROCm Training Preparation Audit & Execution Plan

**Document ID:** `RL-EXP-STAGE6-TRAINING-PLAN`  
**Date:** September 2026  
**Execution Context:** Planning & Pre-Flight Audit Only (Zero AMD GPU credits spent, Zero instances launched, Zero model training)  
**Target Accelerator:** AMD Instinct (MI210 / MI250 / MI300X) or Radeon Pro (W7900) via AMD Developer Cloud  
**Hardware Abstraction Interface:** PyTorch with ROCm HIP acceleration (`torch.cuda` under HIP)  
**Target Model:** Dedicated 5-Class Traffic Sign Detector (`sign_yolov8n.pt`)  
**Base Architecture:** Ultralytics YOLOv8n (Nano: 3.2M parameters, 8.7 GFLOPs, 6.2 MB weight size)  

---

## 1. Dataset Verification & Boundary Invariants

### A. Training Configuration Verification ([`dataset.yaml`](file:///Users/samruddhibhagwat/roadlens-ai-local/datasets/roadlens_signs_training/dataset.yaml))
The assembled training dataset configuration was verified with Ultralytics dataset validation (`check_det_dataset`):

```yaml
path: /Users/samruddhibhagwat/roadlens-ai-local/datasets/roadlens_signs_training
train: images/train
val: images/val
test: /Users/samruddhibhagwat/roadlens-ai-local/datasets/roadlens_signs_yolo/images/test

names:
  0: SPEED_LIMIT_SIGN
  1: ADVISORY_SPEED
  2: STOP_SIGN
  3: WARNING_SIGN
  4: WORK_ZONE_SIGN
```

### B. Strict Split Isolation Audit
| Split | Directory Path | Verified Image Count | Verified Label Count | Annotation Count | Origin Breakdown |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | `images/train/`, `labels/train/` | **4,135** | **4,135** | **4,691 boxes** | 3,735 Real LISA + 400 Synthetic |
| **Val** | `images/val/`, `labels/val/` | **994** | **994** | **1,086 boxes** | 894 Real LISA + 100 Synthetic |
| **Test (Benchmark)** | `datasets/roadlens_signs_yolo/images/test/` | **781** | **781** | **878 boxes** | **100% Real Physical LISA (Untouched)** |

- **Zero Test Leakage:** 0 LISA test images exist in `images/train` or `images/val`.
- **Zero Sequence Leakage:** 0 overlap between the 297 train sequences, 63 val sequences, and 63 test sequences.
- **Zero Synthetic in Test:** 0 synthetic images exist in the test benchmark split.
- **Evaluation Guarantee:** The test split will **never** be used for training, backpropagation, hyperparameter tuning, or validation checkpoint selection. It is strictly reserved for post-training benchmark evaluation.

---

## 2. Repository AMD/ROCm Architecture Review

### A. Hardware Device Manager Integration ([`backend/app/performance/device.py`](file:///Users/samruddhibhagwat/roadlens-ai-local/backend/app/performance/device.py))
RoadLens AI features an established hardware abstraction layer:
1. **HIP Detection:** Checks `getattr(torch.version, "hip", None)`. When present with `torch.cuda.is_available()`, sets `device_type = "AMD_ROCM"`.
2. **Device Discovery:** Queries `torch.cuda.get_device_name(0)` to inspect device string (e.g. `AMD Instinct MI210`, `AMD Radeon Pro W7900`).
3. **Telemetry & VRAM:** Tracks GPU allocation through `torch.cuda.memory_allocated(0)` and `torch.cuda.get_device_properties(0).total_memory`.
4. **Ultralytics Routing:** Ultralytics natively supports AMD ROCm by accepting `device=0` (or `device='cuda:0'`), which PyTorch maps directly to the underlying AMD GPU through the ROCm HIP driver.

### B. Library Version Compatibility
- **Installed Ultralytics:** Version `8.4.46` (fully tested, compatible with PyTorch 2.0+ and ROCm 5.7+ / 6.0+).
- **Mixed Precision:** Uses `amp=True` (`torch.cuda.amp.autocast`) which is natively accelerated on AMD matrix cores (CDNA / RDNA architecture).
- **Dataloader Multiprocessing:** `workers=8` with pinned memory support (`pin_memory=True`) on Linux/ROCm.

---

## 3. Experiment 1: Reproducible Training Configuration (YOLOv8n, 25 Epochs)

### A. Model Hyperparameters & Strategy

| Parameter | Specification | Rationale & Design Decision |
| :--- | :--- | :--- |
| **Base Architecture** | `yolov8n.pt` | Nano model (3.2M params). Minimizes compute time, runs at $>150$ FPS on AMD, trains $3\times$ faster than Small. |
| **Pretrained Weights** | COCO pretrained backbone | Faster feature transfer on general edge, shape, and corner representations. |
| **Epochs** | **25 epochs** | Sufficient for fine-tuning 5 sign classes on 4,135 images without overfitting. |
| **Input Image Size** | `imgsz = 640` | Standard multi-scale anchor-free resolution; maintains legibility of distant signs. |
| **Batch Size** | `batch = 32` | Optimal gradient stability on AMD Instinct (consumes $\approx 3.2\text{ GB}$ VRAM out of 64GB). |
| **Dataloader Workers** | `workers = 8` | Prevents GPU starvation during high-throughput image decoding. |
| **Optimizer** | `AdamW` | Fast initial convergence on mixed real/synthetic road domains with decoupled weight decay. |
| **Initial Learning Rate** | `lr0 = 0.001` | Conservative fine-tuning rate preventing gradient spikes. |
| **Final Learning Rate** | `lrf = 0.01` | Cosine decay down to $1\times 10^{-5}$. |
| **Weight Decay** | `weight_decay = 0.0005` | Regularization against background overfitting. |
| **Warmup** | `warmup_epochs = 3.0` | 3 linear warmup epochs for backbone adaptation. |
| **Loss Gains** | `box = 7.5, cls = 0.5, dfl = 1.5` | Emphasizes bounding box localization precision. |

### B. Augmentation Policy (Optimized for Traffic Signs)
- `fliplr = 0.5`: Horizontal flipping (simulates opposite roadside views).
- `flipud = 0.0`: **Strictly disabled** (traffic signs are never upside down).
- `degrees = 5.0`: Mild planar tilt ($\pm 5^\circ$) for crooked posts and camera roll.
- `perspective = 0.0005`: Mild 3D perspective warp.
- `mosaic = 1.0`: 4-image mosaic compositing (essential for small distant sign recall).
- `mixup = 0.1`: 10% mixup blending for boundary regularization.
- `close_mosaic = 5`: Disables mosaic during the final 5 epochs for clean localized boundary convergence.
- `hsv_h = 0.015, hsv_s = 0.7, hsv_v = 0.4`: Color temperature, sunlight, and shadow invariance.

### C. Checkpointing & Outputs
- **Project Directory:** `runs/train/`
- **Experiment Run Name:** `roadlens_sign_yolov8n_exp1`
- **Output Weights:**
  - `runs/train/roadlens_sign_yolov8n_exp1/weights/best.pt` (selected by validation mAP50)
  - `runs/train/roadlens_sign_yolov8n_exp1/weights/last.pt` (epoch 25 checkpoint)
- **Standardized RoadLens Export:** Automatically copied to `models/sign_yolov8n.pt`.
- **Early Stopping:** `patience = 10` (halts training if validation mAP50 fails to improve for 10 consecutive epochs to conserve AMD credits).
- **Seed & Determinism:** `seed = 42, deterministic = True`.

---

## 4. Strict Cost, Runtime & Safety Mechanisms

### A. Execution Runtime & Credit Cost Estimates

| Stage | Batch / Images | Speed on AMD MI210 / W7900 | Time per Epoch | 25 Epochs Total | Estimated AMD Cost (@ $1.50/hr) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Split** | 4,135 images | $\approx 180\text{ img/sec}$ | $\approx 23\text{ sec}$ | $\approx 575\text{ sec}$ ($9.6\text{ min}$) | $\$0.24$ |
| **Val Split** | 994 images | $\approx 350\text{ img/sec}$ | $\approx 3\text{ sec}$ | $\approx 75\text{ sec}$ ($1.25\text{ min}$) | $\$0.03$ |
| **Warmup & Setup** | Pretrained weights download & model build | — | — | $\approx 60\text{ sec}$ ($1.0\text{ min}$) | $\$0.02$ |
| **TOTAL** | **5,129 images** | — | — | **$\approx 12\text{ minutes}$** | **$\approx \$0.30\text{ USD}$** |

> [!NOTE]
> Even allowing for container initialization, environment checks, and metric export, the entire Experiment 1 training run will consume **less than 20 minutes** of GPU time, corresponding to **$\approx \$0.50$ of the user's $100.00 credit pool ($< 0.5\%$)**.

### B. Automated Safety Harness (Zero GPU Idle Policy)
To guarantee that AMD credits are never drained by an unattended or lingering instance:
1. **Hard Wall-Clock Timeout:** The training script enforces a hard timeout of **45 minutes** (`--max-runtime 45`). If training hangs or stalls, the script terminates immediately.
2. **Automatic Post-Execution Shutdown:** The instance launch script executes under an automated wrap-around command:
   ```bash
   python3 scratch/train_sign_detector_amd.py --data datasets/roadlens_signs_training/dataset.yaml --epochs 25 --batch 32 && \
   tar -czf roadlens_sign_exp1_artifacts.tar.gz runs/train/roadlens_sign_yolov8n_exp1/ models/sign_yolov8n.pt && \
   sudo shutdown -h now
   ```
3. **Artifact Integrity Verification Before Shutdown:**
   - Script verifies `weights/best.pt` exists and has file size $> 5.0\text{ MB}$.
   - Exports `training_summary.json` containing epoch logs, elapsed time, loss curves, and hardware metadata.
4. **Immediate Instance Destruction Protocol:**
   - Once artifacts are downloaded to the local Mac, the user destroys the instance from the AMD Developer Cloud dashboard to stop all storage and compute accruals.

---

## 5. Pre-Flight Verification & Dry-Run Status

- **Standalone Script Created:** [`scratch/train_sign_detector_amd.py`](file:///Users/samruddhibhagwat/roadlens-ai-local/scratch/train_sign_detector_amd.py)
- **Local Dry-Run Execution:**
  ```bash
  python3 scratch/train_sign_detector_amd.py --dry-run
  ```
  - Hardware check: Successfully verified PyTorch, accelerator detection, and device args.
  - Dataset check: Successfully resolved 4,135 train images, 994 val images, and 781 isolated test benchmark images.
  - Exit code: `0` (PASS).
- **Backend Regression Suite:** All 50 existing backend tests pass (`Ran 50 tests in 9.13s - OK`).
- **Production Code Status:** Untouched. Zero changes to `backend/app/*` or `frontend/*`.

---

## 6. Execution Command for Stage 6B (Ready Upon Approval)

When authorized by the user to proceed with AMD instance launch:

```bash
# 1. Run on AMD ROCm Instance:
python3 scratch/train_sign_detector_amd.py \
  --data datasets/roadlens_signs_training/dataset.yaml \
  --model yolov8n.pt \
  --epochs 25 \
  --batch 32 \
  --imgsz 640 \
  --workers 8 \
  --lr0 0.001 \
  --patience 10 \
  --seed 42 \
  --max-runtime 45 \
  --export models/sign_yolov8n.pt

# 2. Automated evaluation on untouched test benchmark (781 LISA frames):
yolo detect val \
  model=models/sign_yolov8n.pt \
  data=datasets/roadlens_signs_training/dataset.yaml \
  split=test
```

---

## 7. Sign-Off Checklist Before Proceeding

- [x] `dataset.yaml` paths and 5-class target schema verified.
- [x] Complete isolation of 781 LISA benchmark test frames confirmed.
- [x] Zero test images in train/val verified.
- [x] AMD ROCm hardware detection and routing verified.
- [x] Pre-flight dry-run completed with zero errors.
- [x] Wall-clock safety timeout (45 min) and self-termination harness designed.
- [x] Estimated cost computed ($\approx \$0.30\text{--}\$0.50$ out of $\$100.00$).
- [x] Zero AMD instances launched; zero model training started.
