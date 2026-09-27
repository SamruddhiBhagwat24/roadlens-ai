# RoadLens AI — Stage 6B: AMD Remote Provisioning & Environment Validation Diagnostic Report

**Document ID:** `RL-EXP-STAGE6B-ENV-DIAGNOSTIC`  
**Date:** September 27, 2026  
**Execution Status:** DIAGNOSIS COMPLETE — AWAITING INSTANCE DETAILS  
**Safety Protocol:** Strict Zero-Fabrication, Zero Unauthorized Spend, Read-Only Audit  

---

## 1. Executive Summary & Root Cause Analysis

During Stage 6B remote environment audit, the agent session experienced a socket disconnection (Gemini API / Antigravity network broken pipe). This was an **infrastructure connection drop** between the IDE interface and the assistant engine, not a failure of RoadLens code, local scripts, or AMD hardware.

### Key Forensic Findings:
1. **Command Active at Disconnection:**
   - The last command executed was `ls -la datasets/roadlens_signs_training/` at 11:52:28 IST.
   - It exited cleanly with code `0`.
   - No GPU training command was started. No remote command crashed.
2. **AMD Instance Accessibility:**
   - **No active AMD Developer Cloud instance is currently accessible.**
   - `~/.ssh/config` does not exist.
   - The only entry in `~/.ssh/known_hosts` is `129.212.182.192`, which WHOIS confirms belongs to **DigitalOcean LLC** (connected on Sept 25 for unrelated tasks) and timed out on port 22.
   - Zero AMD environment variables, SSH hosts, or Developer Cloud credentials exist in the local workspace or environment.
3. **Status of the Large Download (8.57 GB LISA Dataset):**
   - The large download observed in the terminal was `datasets/lisa_signs/lisa_traffic_signs.zip` (Task `task-2112`).
   - **It completed 100% successfully** at 10:47:18 IST (8,573,963,212 bytes verified, zero corruption, zero partial files).
   - All extraction, parsing, synthetic augmentation, and train/val split assembly were completed and verified in Stages 5B, 5C, and 5D.
4. **Remote File Status:**
   - Because no remote AMD Developer Cloud instance is currently active, no files have been uploaded remotely yet.
   - All dataset files (`datasets/roadlens_signs_training/`) and training scripts (`scratch/train_sign_detector_amd.py`) are fully validated and stored locally on disk.
5. **AMD ROCm Environment Validation:**
   - Remote AMD GPU, ROCm/HIP driver, and hourly rate could **not** be validated because no active AMD instance is provisioned.
   - Local training script dry-run (`scratch/train_sign_detector_amd.py --dry-run`) passed with exit code `0` (Task `task-2636`), verifying dataset paths, class mappings, and PyTorch/Ultralytics orchestration.

---

## 2. Forensic Diagnostic Checklist

| Audit Question | Verified Result | Evidence / Details |
|---|---|---|
| **What command was running during failure?** | `ls -la datasets/roadlens_signs_training/` | Exited code 0 at 11:52:28 IST. Broken pipe occurred in client-server transport before the subsequent response streamed. |
| **Was AMD instance actually reached?** | **NO** | No AMD instance host/IP is configured or running. Known host `129.212.182.192` is DigitalOcean and is offline. |
| **Did the large download succeed?** | **YES (100% complete)** | `lisa_traffic_signs.zip` (8.57 GB) completed in `task-2112`. Verified intact on disk. |
| **Does an AMD SSH connection exist?** | **NO** | `ps aux` shows zero active SSH tunnels or remote connections. `manage_task` lists 0 running background tasks. |
| **Was AMD ROCm validated?** | **PARTIAL (Local only)** | Local dry-run passed (exit 0). Remote ROCm driver/GPU unverified pending instance launch. |
| **Are there partially downloaded files?** | **NO** | All dataset zips and generated files are complete with valid checksums and manifests. |

---

## 3. Dataset Integrity & Benchmark Isolation Verification

- **Combined Training Set (`datasets/roadlens_signs_training/`):**
  - **Train:** 4,135 images (3,735 Real LISA + 400 Synthetic), 4,691 bounding boxes.
  - **Val:** 994 images (894 Real LISA + 100 Synthetic), 1,086 bounding boxes.
  - **Classes (5):**
    - `0: SPEED_LIMIT_SIGN`
    - `1: ADVISORY_SPEED`
    - `2: STOP_SIGN`
    - `3: WARNING_SIGN`
    - `4: WORK_ZONE_SIGN`
- **Benchmark Test Isolation:**
  - **Test Benchmark:** 781 images (100% Real LISA, 0% synthetic, 878 bounding boxes) located in `datasets/roadlens_signs_yolo/images/test`.
  - **Isolation Confirmation:** `scratch/train_sign_detector_amd.py` invokes `model.train(data=..., val=True)`, which strictly trains on `images/train` and validates on `images/val`. The test set is **never** loaded, referenced, or evaluated during training or hyperparameter tuning.

---

## 4. Cost & Guardrail Projection (For Instance Launch)

When an AMD Developer Cloud instance is provisioned:
- **Typical AMD Developer Cloud Rate:** ~\$1.00 – \$2.00 / hour (AMD Instinct MI210 or Radeon Pro W7900).
- **Estimated Experiment 1 Training Time:** 11 to 13 minutes (25 epochs @ ~26s/epoch).
- **Hard Timeout Guardrail:** `--max-runtime 45` minutes embedded in `scratch/train_sign_detector_amd.py`.
- **Maximum Financial Exposure (if 45-min timeout reached):**
  $$\text{Max Cost} = \frac{45\text{ min}}{60\text{ min}} \times \$2.00/\text{hr} = \mathbf{\$1.50\text{ USD}}$$
  Consumes $\le 1.5\%$ of the user's $\$100.00$ credit allocation.

---

## 5. Safest Next Action

Per instructions to **stop after diagnosis and not provision without explicit authorization**:
1. **DO NOT** attempt to guess instance credentials or connect to non-AMD hosts.
2. **DO NOT** re-download the 8.57 GB dataset (it is already fully downloaded and assembled locally).
3. **DO NOT** start YOLO training.
4. **Action Required from User:**
   Log into the AMD Developer Cloud dashboard and provide:
   - **Instance IP / Hostname** and **SSH Port** (or Web Terminal access instructions)
   - **SSH User** (typically `ubuntu` or `root`)
   - **Confirmed GPU model** (e.g., Instinct MI210 / W7900)
   - **Hourly rate** displayed on the instance card
