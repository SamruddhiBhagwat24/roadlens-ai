"""API routes for RoadLens AI.

Implements:
- POST /api/analyze: Full perception pipeline on uploaded road image
- GET /api/health: System and compute device health check
- GET /api/config: Public operational configuration
"""
import io
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.app.schemas import (
    AnalyzeResponse,
    HealthResponse,
    ImageDimensions,
    ImageQualityDetails,
    PerformanceDetails,
    ProfilesResponse,
    CountryProfileSummary,
)
from backend.app.api.validation import validate_image_upload, ValidationError
from backend.app.vision.image_quality import quality_analyzer
from backend.app.vision.preprocessing import preprocessor
from backend.app.vision.detector import road_detector
from backend.app.vision.ocr import ocr_service
from backend.app.vision.postprocessing import postprocessor
from backend.app.intelligence.interpreter import interpreter
from backend.app.intelligence.profiles import profile_registry
from backend.app.performance.benchmark import PipelineProfiler
from backend.app.performance.device import device_manager
from backend.app.config import settings

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns system status, active hardware compute backend, and ROCm availability."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        device=device_manager.device_name,
        rocm_available=device_manager.is_rocm,
    )


@router.get("/config")
async def get_public_config():
    """Returns public constraints and upload settings."""
    return {
        "max_upload_size_mb": settings.max_upload_size_mb,
        "allowed_extensions": settings.allowed_extensions,
        "device_type": device_manager.device_type,
        "rocm_enabled": device_manager.is_rocm,
        "default_country_profile": settings.default_country_profile,
    }


@router.get("/profiles", response_model=ProfilesResponse)
async def list_profiles():
    """Returns registered national road-standard and regulatory profiles."""
    return ProfilesResponse(
        active_default=settings.default_country_profile,
        profiles=[CountryProfileSummary(**p) for p in profile_registry.list_profiles()],
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    country_code: str = Query(default="usa", description="Country road standard profile ('usa', 'india')"),
):
    """Analyzes a road scene image through the complete perception pipeline.

    Steps:
    1. Validation: Size, MIME, magic signature, and integrity verification.
    2. Image Quality: Quantifies blur, noise, glare, brightness, perspective.
    3. Adaptive Preprocessing: Conditionally enhances adverse condition images.
    4. Object Detection: Detects signs, plates, plaques, and bounding boxes.
    5. Modular OCR: Extracts character text from candidate crops.
    6. Postprocessing: Cleans tokens and bounds coordinates.
    7. Semantic Understanding: Converts extractions into structured road intelligence.
    8. Performance Telemetry: Measures real stage timings and hardware metrics.
    """
    profiler = PipelineProfiler()
    profiler.start_pipeline()

    # 1. Read & Validate Upload
    try:
        content = await file.read()
        pil_img, _ = validate_image_upload(
            filename=file.filename or "uploaded_image",
            content=content,
            content_type=file.content_type or "",
        )
    except ValidationError as ve:
        raise HTTPException(status_code=ve.status_code, detail=ve.message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {str(e)}")

    # Convert PIL Image (RGB) to NumPy OpenCV format (BGR)
    img_rgb = np.array(pil_img)
    img_bgr = img_rgb[:, :, ::-1].copy()
    h, w, c = img_bgr.shape

    # 2. Image Quality Intelligence
    profiler.start_stage("quality_check")
    quality_dict = quality_analyzer.analyze(img_bgr)
    profiler.end_stage("quality_check")

    # 3. Conditional Adaptive Preprocessing
    profiler.start_stage("preprocessing")
    processed_bgr, applied_transforms = preprocessor.adaptively_preprocess(img_bgr, quality_dict)
    profiler.end_stage("preprocessing")

    # 4. Object Detection & Region Extraction
    profiler.start_stage("detection")
    detections = road_detector.detect(processed_bgr)
    profiler.end_stage("detection")

    # 5. Modular OCR Extraction
    profiler.start_stage("ocr")
    ocr_target_classes = {
        "LICENSE_PLATE",
        "SPEED_LIMIT_SIGN",
        "ADVISORY_SPEED",
        "STOP_SIGN",
        "WARNING_SIGN",
        "WORK_ZONE_SIGN",
    }
    candidate_boxes = [d.bounding_box for d in detections if d.class_name in ocr_target_classes]
    plate_boxes = [d.bounding_box for d in detections if d.class_name == "LICENSE_PLATE"]
    raw_ocr_results = ocr_service.extract_from_regions(processed_bgr, candidate_boxes, plate_boxes=plate_boxes)
    # Postprocess OCR tokens and bounds
    ocr_results = postprocessor.postprocess_ocr(raw_ocr_results, max_width=w, max_height=h)
    profiler.end_stage("ocr")

    # 6. Semantic Interpretation & Road Intelligence
    profiler.start_stage("intelligence")
    road_intelligence = interpreter.interpret(
        detections, ocr_results, quality_dict, country_code=country_code
    )
    profiler.end_stage("intelligence")

    # 7. Hardware & Latency Telemetry
    perf_dict = profiler.finish_pipeline()

    return AnalyzeResponse(
        success=True,
        filename=file.filename or "uploaded_image",
        image_dimensions=ImageDimensions(width=w, height=h, channels=c),
        detections=detections,
        ocr_results=ocr_results,
        road_intelligence=road_intelligence,
        image_quality=ImageQualityDetails(**quality_dict),
        performance=PerformanceDetails(**perf_dict),
    )
