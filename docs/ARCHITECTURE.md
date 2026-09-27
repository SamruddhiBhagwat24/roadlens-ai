# RoadLens AI — Architecture & Technical Design

## 1. Perception Pipeline Flow

RoadLens AI executes a multi-stage sequential visual perception pipeline designed for automated-driving research:

```
[Raw Image Upload]
       │
       ▼
1. Validation Layer (Magic bytes, MIME, Dimensions, Integrity)
       │
       ▼
2. Image Quality Intelligence (Pre-recognition diagnostic)
   ├── Blur: Laplacian operator variance
   ├── Brightness: Mean luminance distribution
   ├── Noise: Gaussian difference residual std
   ├── Glare: Pixel saturation percentage (> 245)
   └── Perspective: Line orientation angle histogram
       │
       ▼
3. Adaptive Preprocessing (Strictly conditional execution)
   ├── Low Light ──► Adaptive Gamma Correction
   ├── Glare / High Contrast ──► CLAHE (LAB color space)
   ├── Sensor Noise ──► Bilateral Edge-Preserving Filter
   ├── Blur ──► Unsharp Masking (Frequency enhancement)
   └── Nominal ──► Zero-overhead Passthrough
       │
       ▼
4. Road Object / Candidate Region Detector
   ├── Stop Signs (Octagon geometry / red hue masks)
   ├── Speed Limit Signs (Rectangular aspect ratio / high-contrast gradients)
   ├── Advisory Speed Plaques (Yellow hue / aspect ratio)
   ├── Warning & Work Zone Signs (Diamond / orange masks)
   └── License Plates (Horizontal aspect ratio 1.8-2.5)
   └── NMS (Non-Maximum Suppression IoU >= 0.35)
       │
       ▼
5. Modular OCR Service
   ├── Abstract Interface: BaseOCREngine
   ├── HeuristicContourCV (Default zero-dependency CV engine)
   ├── EasyOCR (Deep learning engine with GPU support)
   └── PyTesseract (Classic OCR engine)
       │
       ▼
6. Semantic Reasoning & Road Intelligence
   ├── Translation into typed ADAS entities
   ├── Numerical speed extraction & unit tagging
   ├── Mandatory action directives
   └── Multi-stage confidence scoring
       │
       ▼
7. Performance Telemetry & Hardware Profiler
   ├── Monotonic high-precision timers per stage
   └── AMD ROCm / HIP detection with seamless CPU fallback
```

---

## 2. AMD GPU / ROCm Strategy & CPU Fallback

RoadLens AI is architected from day one to support AMD Developer Cloud acceleration via ROCm/HIP:

* **Detection:** The `DeviceManager` interrogates `torch.version.hip` and `torch.cuda.is_available()`.
* **ROCm Environment:** When running on AMD hardware with ROCm PyTorch installed, operations dispatch to the AMD GPU (`cuda:0` alias in HIP).
* **CPU Fallback:** On standard machines (development laptops, macOS, or non-accelerated cloud nodes), execution falls back gracefully to CPU with zero crashes and zero fake metrics.
* **Zero Fabrication Policy:**
  * GPU memory allocation (`gpu_memory_mb`) is only queried when `cuda_available` is True.
  * GPU utilization is left `null` unless an actual hardware profiler (like ROCm SMI) is actively attached.
  * Latency numbers are measured using `time.perf_counter()`.

---

## 3. Replaceable OCR Architecture

To prevent vendor lock-in and evaluate different models in future milestones, OCR is decoupled behind an abstract base class:

```python
class BaseOCREngine(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def recognize_crop(self, image_crop: np.ndarray, offset_box: Optional[List[int]]) -> List[OCRResultItem]: ...
```

The system automatically selects the highest-capability engine available:
1. `EasyOCR` (Deep learning)
2. `PyTesseract` (OCR engine)
3. `HeuristicContourCV` (OpenCV baseline with zero external dependencies)
