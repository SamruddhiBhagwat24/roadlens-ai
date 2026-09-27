# RoadLens AI — AI-Powered Road Perception & Road Intelligence

> **Milestone 1 Release** • Research Prototype for ADAS & Automated-Driving Perception Pipelines

RoadLens AI transforms raw road-camera imagery into structured, machine-readable road intelligence. It detects traffic signs, warning signs, advisory plaques, and vehicle license plates, extracts text under adverse visual conditions (blur, glare, noise, low light, perspective skew), and converts visual extractions into structured road intelligence.

---

## Architecture Overview

```
Raw Road Image (JPG / PNG / WEBP)
              │
              ▼
  [ Module 2: Image Quality Intelligence ]
  (Laplacian Blur, Noise Std, Glare Saturation, Brightness, Perspective)
              │
              ▼
  [ Module 3: Adaptive Conditional Preprocessing ]
  (CLAHE, Denoising, Gamma Correction, Unsharp Sharpening)
              │
              ▼
  [ Module 4: Road Object & Region Detection ]
  (STOP_SIGN, SPEED_LIMIT_SIGN, ADVISORY_SPEED, WARNING_SIGN, LICENSE_PLATE)
              │
              ▼
  [ Module 5: Replaceable OCR Service Engine ]
  (HeuristicContourCV, EasyOCR, PyTesseract)
              │
              ▼
  [ Module 6: Semantic Reasoning & Road Intelligence ]
  (Typed ADAS entities, advisory limits, multi-stage confidence)
              │
              ▼
  [ Fast API Backend /api/analyze ] ────► [ React + Vite + Tailwind Dashboard ]
```

---

## Key Capabilities (Milestone 1)

1. **Target Object Recognition & OCR (Mini Challenge 2):**
   - **Regulatory Signs:** Stop signs (mandatory stop inference) and Speed Limit signs.
   - **Advisory Speed Plaques:** Number-only advisory speed plaques (e.g. 25 MPH, 35 MPH, 45 MPH).
   - **Work-Zone & Warning Signs:** Multi-line text alerts and construction hazard warnings.
   - **US License Plates:** Vehicle license plate region segmentation and character pattern extraction.

2. **Adverse Visual Condition Intelligence:**
   - Pre-recognition diagnostic analyzer evaluating **blur**, **illumination**, **sensor noise**, **glare**, and **perspective skew**.
   - Conditional preprocessing that applies enhancements *only* when adverse triggers are flagged.

3. **Pluggable OCR Engine Interface:**
   - `BaseOCREngine` contract allows swapping between:
     - `HeuristicContourOCREngine`: Built-in OpenCV baseline (zero external binary dependency).
     - `EasyOCREngine`: Deep learning PyTorch engine.
     - `TesseractOCREngine`: Traditional Tesseract engine.

4. **AMD GPU / ROCm Acceleration Architecture:**
   - Detects AMD ROCm / HIP environments automatically (`torch.version.hip`).
   - Seamless CPU fallback when AMD hardware is unavailable (safe for local development).
   - **Zero-Fabrication Policy:** Never outputs synthetic or fabricated latency, FPS, or GPU memory metrics. Unmeasured parameters remain `null` / `N/A`.

5. **Multi-Stage Confidence Scoring:**
   - Localized detection confidence, text OCR confidence, and semantic interpretation confidence.

---

## Project Structure

```
roadlens-ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes.py            # /api/analyze, /api/health, /api/config
│   │   │   └── validation.py        # Magic byte signature, size & MIME security validation
│   │   ├── intelligence/
│   │   │   ├── interpreter.py       # Semantic translation into structured ADAS entities
│   │   │   └── road_state.py        # Road state model
│   │   ├── performance/
│   │   │   ├── benchmark.py         # Precision stage-by-stage latency profiler
│   │   │   └── device.py            # AMD ROCm / CUDA / MPS / CPU detection & fallback
│   │   ├── vision/
│   │   │   ├── detector.py          # Color, geometry & contour region detector
│   │   │   ├── image_quality.py     # Blur, noise, glare, brightness & skew metrics
│   │   │   ├── ocr.py               # Replaceable BaseOCREngine & modular OCR service
│   │   │   ├── postprocessing.py    # Coordinate clamping and token sanitization
│   │   │   └── preprocessing.py     # Adaptive conditional image enhancement
│   │   ├── config.py                # Configuration management (.env / defaults)
│   │   ├── main.py                  # FastAPI application entry point
│   │   └── schemas.py               # Pydantic response data schemas
│   └── requirements.txt             # Backend dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── DetectionCanvas.jsx  # Visual image preview & bounding box projection
│   │   │   ├── DetectionsList.jsx   # Tabular list of localized targets & OCR extractions
│   │   │   ├── ImageQualityCard.jsx # Adverse condition metrics & severity badges
│   │   │   ├── ImageUpload.jsx      # Drag & drop upload with client-side validation
│   │   │   ├── Navbar.jsx           # Header with real-time ROCm/CPU backend status
│   │   │   ├── PerformanceCard.jsx  # Stage latency breakdown & hardware telemetry
│   │   │   └── RoadIntelligenceCard.jsx # Structured road intelligence output
│   │   ├── services/
│   │   │   └── api.js               # API service client
│   │   ├── App.jsx                  # Main dashboard layout
│   │   └── main.jsx                 # React root
│   ├── package.json                 # Frontend dependencies
│   ├── tailwind.config.js           # Tailwind CSS configuration
│   └── vite.config.js               # Vite config with backend proxy
├── models/                          # Storage for weights (git-kept)
├── datasets/                        # Dataset folders for benchmarks (git-kept)
├── experiments/                     # Experiment scripts (git-kept)
├── tests/                           # Automated test suite (validation, vision, API)
├── docs/                            # In-depth architectural documentation
├── .env.example                     # Environment template
└── README.md
```

---

## Setup & Execution

### 1. Environment Configuration

Copy the template environment file:
```bash
cp .env.example .env
```

Key environment variables:
| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | API bind address |
| `PORT` | `8000` | API port |
| `MAX_UPLOAD_SIZE_MB` | `15` | Maximum image size |
| `ALLOWED_EXTENSIONS` | `jpg,jpeg,png,webp` | Permitted image file types |
| `DEVICE_PREFERENCE` | `auto` | Compute backend (`auto`, `cpu`, `rocm`) |
| `GEMINI_API_KEY` | `""` | Optional server-side key for advanced semantic reasoning |

### 2. Backend Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Run the FastAPI server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup

```bash
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```

Navigate to `http://localhost:5173` to access the dashboard.

---

## API Specification

### `POST /api/analyze`
Submits a road scene image for perception and intelligence analysis.

**Request:** `multipart/form-data`
- `file`: Image file (`.jpg`, `.jpeg`, `.png`, `.webp` up to 15MB)

**Response Schema:**
```json
{
  "success": true,
  "filename": "highway_scene.jpg",
  "image_dimensions": { "width": 1920, "height": 1080, "channels": 3 },
  "detections": [
    {
      "class_name": "SPEED_LIMIT_SIGN",
      "confidence": 0.94,
      "bounding_box": [320, 180, 480, 390],
      "label": "Speed Limit Sign"
    }
  ],
  "ocr_results": [
    {
      "text": "45",
      "confidence": 0.91,
      "bounding_box": [340, 240, 460, 360],
      "engine": "HeuristicContourCV"
    }
  ],
  "road_intelligence": {
    "object_type": "speed_limit_sign",
    "meaning": "Speed limit 45 MPH",
    "value": 45,
    "unit": "mph",
    "detection_confidence": 0.94,
    "ocr_confidence": 0.91,
    "interpretation_confidence": 0.95,
    "advisory_notes": null
  },
  "image_quality": {
    "blur": "low",
    "blur_score": 182.4,
    "noise": "low",
    "noise_score": 2.8,
    "glare": "low",
    "glare_score": 0.3,
    "brightness": "normal",
    "brightness_score": 134.2,
    "perspective": "normal",
    "perspective_score": 1.8,
    "adverse_conditions": []
  },
  "performance": {
    "device_type": "CPU",
    "device_name": "Host CPU",
    "rocm_enabled": false,
    "latency_ms": {
      "quality_check": 4.1,
      "preprocessing": 1.2,
      "detection": 11.5,
      "ocr": 6.8,
      "intelligence": 0.4,
      "total": 24.0
    },
    "gpu_utilization": null,
    "gpu_memory_mb": null,
    "status_note": "No GPU accelerator detected; executing on CPU"
  }
}
```

### `GET /api/health`
Returns compute device availability and system health:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "device": "Host CPU",
  "rocm_available": false
}
```

---

## Automated Test Suite

Run the unit and integration tests:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Tests cover:
- Binary magic byte signature and corrupted image validation (`test_validation.py`)
- Conditional adaptive preprocessing (`test_preprocessing.py`)
- Image quality metrics: blur, noise, glare, brightness (`test_image_quality.py`)
- Modular OCR service interface and swappability (`test_ocr_interface.py`)
- AMD ROCm detection and CPU fallback guarantees (`test_device.py`)
- Output JSON schema adherence (`test_response_schema.py`)
- FastAPI endpoints integration (`test_api_endpoints.py`)

---

## Safety & Disclaimer

RoadLens AI is an ADAS and automated-driving **research prototype**. It does **not** provide vehicle actuation (steering, braking, throttle), claims no automotive safety certification, and is not affiliated with or integrated with Tesla or any specific automotive manufacturer.
