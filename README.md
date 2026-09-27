# 🚗 RoadLens AI – AI-Powered Road Perception & Road Intelligence

An AI-powered road perception and intelligence system that transforms road-camera images into **structured, machine-readable road information** using computer vision, OCR, semantic interpretation, and AMD ROCm acceleration.

Built as a **research prototype for ADAS and automated-driving perception pipelines**.

---

## 🚀 Features

### 🛣️ Road Perception

- Traffic sign detection

- Speed-limit sign recognition

- Stop-sign detection

- Warning and work-zone sign detection

- Vehicle detection

- License-plate region detection


### 🔤 OCR & Intelligence

- OCR-based road-sign text extraction

- Spatial association between detected regions and OCR

- Multi-stage confidence scoring

- Structured road intelligence generation

- India license-plate format validation

- Bharat Series (`BH`) format validation


### 🌎 Country Profiles

- 🇺🇸 USA road perception profile

- 🇮🇳 India road perception profile

- Country-specific road-sign interpretation

- Country-specific units and registration formats


### 🌧️ Image Quality Analysis

- Blur detection

- Noise analysis

- Brightness analysis

- Glare detection

- Perspective analysis

- Adaptive image preprocessing


### ⚡ AMD Acceleration

- AMD Instinct MI300X support

- AMD ROCm / HIP acceleration

- PyTorch GPU inference

- CPU fallback for local environments

- Real hardware telemetry without fabricated metrics

---

## 🧠 AI Pipeline


Road Image
    ↓
Image Quality Analysis
    ↓
Adaptive Preprocessing
    ↓
Object Detection
    ↓
OCR
    ↓
Semantic Interpretation
    ↓
Structured Road Intelligence
    ↓
React Dashboard

---

## 📊 Model Performance

| Metric | Validation | Held-Out Test |
|---|---:|---:|
| Precision | 93.1% | 92.2% |
| Recall | 90.4% | 85.4% |
| mAP@50 | 95.4% | 91.6% |
| mAP@50-95 | 81.9% | 73.5% |

**Held-out benchmark:** 781 images with 878 annotated instances.

---

## 📁 Project Structure

~~~text
RoadLens-AI/
│
├── backend/
│   └── app/
│       ├── api/
│       ├── intelligence/
│       ├── performance/
│       └── vision/
│
├── frontend/
│   └── src/
│       ├── components/
│       └── services/
│
├── training/
├── experiments/
├── tests/
├── docs/
├── models/
├── .env.example
├── .gitignore
└── README.md
~~~

---

## 🛠️ Tech Stack

### Frontend

- React
- Vite
- Tailwind CSS

### Backend

- Python
- FastAPI
- Pydantic

### AI / Computer Vision

- PyTorch
- YOLO
- OpenCV
- EasyOCR

### Hardware / Infrastructure

- AMD Instinct MI300X
- AMD ROCm
- GitHub

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend and hardware health |
| GET | `/api/config` | System configuration |
| GET | `/api/profiles` | Available country profiles |
| POST | `/api/analyze` | Analyze road image |

### Analyze Image

~~~text
POST /api/analyze
~~~

Upload an image using `multipart/form-data`.

**Supported country profiles:**

~~~text
country_code=usa
country_code=india
~~~

---

## 🧪 Testing

Run the automated test suite:

~~~bash
python3 -m unittest discover -s tests -p "test_*.py" -v
~~~

The test suite covers:

- API endpoints
- Image validation
- Image quality analysis
- Adaptive preprocessing
- OCR interface
- Device detection
- Country profiles
- Response schemas
- Pipeline behavior

---

## 👨‍💻 Developer

**Samruddhi Bhagwat**

B.E. Artificial Intelligence & Data Science

D. Y. Patil College of Engineering, Akurdi, Pune

**GitHub:**  
https://github.com/SamruddhiBhagwat24

---

## ⭐ RoadLens AI

**Turning road images into intelligent, structured road perception.**

