"""Pydantic schemas for RoadLens AI structured API responses.

Complies strictly with the RoadLens AI Master Project Plan specification:
- detections: list of recognized object regions
- ocr_results: list of OCR extractions with confidence
- road_intelligence: semantic interpretation of road state
- image_quality: adverse condition analysis
- performance: actual hardware device execution metrics
"""
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """Detected road sign, plaque, or license plate region."""
    class_name: str = Field(..., description="Object category, e.g. SPEED_LIMIT_SIGN, STOP_SIGN, LICENSE_PLATE")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    bounding_box: List[int] = Field(..., description="[x1, y1, x2, y2] pixel coordinates")
    label: Optional[str] = Field(default=None, description="Human readable label")


class OCRResultItem(BaseModel):
    """Text recognized from a candidate image crop."""
    text: str = Field(..., description="Recognized character text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR recognition confidence")
    bounding_box: List[int] = Field(..., description="[x1, y1, x2, y2] bounding box within image")
    engine: str = Field(..., description="Engine used for recognition (e.g. HeuristicContour, EasyOCR, Tesseract)")


class ImageQualityDetails(BaseModel):
    """Pre-recognition adverse condition evaluation."""
    blur: str = Field(default="low", description="Blur severity: low, medium, high")
    blur_score: float = Field(default=0.0, description="Laplacian variance sharpness metric")
    noise: str = Field(default="low", description="Noise severity: low, medium, high")
    noise_score: float = Field(default=0.0, description="Estimated noise standard deviation")
    glare: str = Field(default="low", description="Glare severity: low, medium, high")
    glare_score: float = Field(default=0.0, description="Percentage of saturated/specular pixels")
    brightness: str = Field(default="normal", description="Brightness evaluation: low, normal, high")
    brightness_score: float = Field(default=0.0, description="Mean luminance value (0-255)")
    perspective: str = Field(default="normal", description="Perspective skew: normal, moderate_skew, high_skew")
    perspective_score: float = Field(default=0.0, description="Estimated off-axis perspective angle in degrees")
    adverse_conditions: List[str] = Field(default_factory=list, description="List of detected adverse condition flags")


class RoadIntelligenceDetails(BaseModel):
    """Semantic understanding of road scene."""
    country_profile: str = Field(default="usa", description="Country/standard profile applied ('usa', 'india')")
    regulatory_standard: Optional[str] = Field(default=None, description="Regulatory standard reference (e.g. 'FHWA MUTCD R2-1', 'IRC:67')")
    object_type: Optional[str] = Field(default=None, description="Primary road entity class")
    meaning: Optional[str] = Field(default=None, description="Contextual interpretation")
    value: Optional[Union[int, float, str]] = Field(default=None, description="Parsed numeric value if present (e.g. 45)")
    unit: Optional[str] = Field(default=None, description="Unit if applicable (e.g. mph, km/h)")
    detection_confidence: Optional[float] = Field(default=None, description="Region detection confidence")
    ocr_confidence: Optional[float] = Field(default=None, description="Text extraction confidence")
    interpretation_confidence: Optional[float] = Field(default=None, description="Semantic reasoning confidence")
    advisory_notes: Optional[str] = Field(default=None, description="Advisory or safety notes")
    plate_validation: Optional[Dict[str, Any]] = Field(default=None, description="Jurisdictional license plate validation details")
    all_states: List[Dict[str, Any]] = Field(default_factory=list, description="Structured items identified in scene")


class PerformanceDetails(BaseModel):
    """Hardware acceleration and measured latency metrics.
    Strictly zero-fabrication: unmeasured values are null.
    """
    device_type: str = Field(..., description="Compute device: CPU, AMD_ROCM, CUDA, APPLE_MPS")
    device_name: str = Field(..., description="Hardware device identifier")
    rocm_enabled: bool = Field(default=False, description="True if executing on AMD ROCm/HIP")
    latency_ms: Dict[str, float] = Field(..., description="Breakdown of measured execution times in milliseconds")
    gpu_utilization: Optional[float] = Field(default=None, description="Measured GPU utilization (null if CPU)")
    gpu_memory_mb: Optional[float] = Field(default=None, description="Measured GPU memory usage (null if CPU)")
    status_note: str = Field(default="Execution completed successfully")


class ImageDimensions(BaseModel):
    width: int
    height: int
    channels: int


class AnalyzeResponse(BaseModel):
    """Complete structured response for POST /api/analyze."""
    success: bool = True
    filename: str
    image_dimensions: ImageDimensions
    detections: List[DetectionItem] = Field(default_factory=list)
    ocr_results: List[OCRResultItem] = Field(default_factory=list)
    road_intelligence: RoadIntelligenceDetails = Field(default_factory=RoadIntelligenceDetails)
    image_quality: ImageQualityDetails = Field(default_factory=ImageQualityDetails)
    performance: PerformanceDetails


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    rocm_available: bool


class CountryProfileSummary(BaseModel):
    """Metadata summary of a supported country / road-standard profile."""
    code: str = Field(..., description="Country code identifier (e.g. 'usa', 'india')")
    name: str = Field(..., description="Full profile name")
    speed_unit: str = Field(..., description="National legal speed unit ('mph' or 'km/h')")
    status: str = Field(..., description="'production_active' or 'staged_specification'")
    standards: List[str] = Field(default_factory=list, description="Applicable road and sign standards")


class ProfilesResponse(BaseModel):
    """Response model for GET /api/profiles."""
    active_default: str = Field(default="usa", description="Default country profile code")
    profiles: List[CountryProfileSummary] = Field(..., description="List of supported and staged profiles")
