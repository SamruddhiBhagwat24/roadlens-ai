"""Data models and schemas for RoadLens AI synthetic license-plate dataset generation."""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import json
from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Integer pixel bounding box coordinates [x_min, y_min, x_max, y_max]."""

    x_min: int = Field(..., description="Minimum x pixel coordinate (left)")
    y_min: int = Field(..., description="Minimum y pixel coordinate (top)")
    x_max: int = Field(..., description="Maximum x pixel coordinate (right)")
    y_max: int = Field(..., description="Maximum y pixel coordinate (bottom)")

    @field_validator("x_max")
    @classmethod
    def validate_x_max(cls, v: int, info) -> int:
        if "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError(f"x_max ({v}) must be greater than x_min ({info.data['x_min']})")
        return v

    @field_validator("y_max")
    @classmethod
    def validate_y_max(cls, v: int, info) -> int:
        if "y_min" in info.data and v <= info.data["y_min"]:
            raise ValueError(f"y_max ({v}) must be greater than y_min ({info.data['y_min']})")
        return v

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def x_center(self) -> float:
        return self.x_min + self.width / 2.0

    @property
    def y_center(self) -> float:
        return self.y_min + self.height / 2.0

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_yolo(self, img_width: int, img_height: int, class_id: int = 0) -> "YOLOBoundingBox":
        """Converts pixel bounding box to normalized YOLO format."""
        if img_width <= 0 or img_height <= 0:
            raise ValueError("Image dimensions must be positive")
        norm_xc = self.x_center / float(img_width)
        norm_yc = self.y_center / float(img_height)
        norm_w = self.width / float(img_width)
        norm_h = self.height / float(img_height)
        return YOLOBoundingBox(
            class_id=class_id,
            x_center=round(max(0.0, min(1.0, norm_xc)), 6),
            y_center=round(max(0.0, min(1.0, norm_yc)), 6),
            width=round(max(0.0, min(1.0, norm_w)), 6),
            height=round(max(0.0, min(1.0, norm_h)), 6),
        )

    def to_list(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def to_dict(self) -> Dict[str, int]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "width": self.width,
            "height": self.height,
        }


class YOLOBoundingBox(BaseModel):
    """Normalized YOLO format bounding box: class_id x_center y_center width height."""

    class_id: int = Field(default=0, description="YOLO class index (0 for license_plate)")
    x_center: float = Field(..., ge=0.0, le=1.0, description="Normalized x center coordinate")
    y_center: float = Field(..., ge=0.0, le=1.0, description="Normalized y center coordinate")
    width: float = Field(..., ge=0.0, le=1.0, description="Normalized width")
    height: float = Field(..., ge=0.0, le=1.0, description="Normalized height")

    def to_yolo_line(self) -> str:
        """Formats string for standard YOLO annotation file."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_pixel_bbox(self, img_width: int, img_height: int) -> BoundingBox:
        """Denormalizes YOLO format back to pixel coordinates."""
        abs_w = self.width * img_width
        abs_h = self.height * img_height
        abs_xc = self.x_center * img_width
        abs_yc = self.y_center * img_height

        xmin = int(round(abs_xc - abs_w / 2.0))
        ymin = int(round(abs_yc - abs_h / 2.0))
        xmax = int(round(abs_xc + abs_w / 2.0))
        ymax = int(round(abs_yc + abs_h / 2.0))

        xmin = max(0, min(img_width - 1, xmin))
        ymin = max(0, min(img_height - 1, ymin))
        xmax = max(xmin + 1, min(img_width, xmax))
        ymax = max(ymin + 1, min(img_height, ymax))

        return BoundingBox(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax)


class PlateMetadata(BaseModel):
    """Metadata schema recorded for each synthetic sample."""

    sample_id: str = Field(..., description="Unique sample identifier (e.g. syn_us_000001)")
    image_filename: str = Field(..., description="Image filename (e.g. syn_us_000001.jpg)")
    label_filename: str = Field(..., description="YOLO label filename (e.g. syn_us_000001.txt)")
    plate_text: str = Field(..., min_length=1, description="Ground truth alphanumeric license-plate text")
    synthetic_layout_id: str = Field(..., description="Synthetic layout template identifier")
    state_header: str = Field(..., description="Header text appearing on the plate (e.g. CALIFORNIA)")
    image_width: int = Field(..., gt=0, description="Output image width in pixels")
    image_height: int = Field(..., gt=0, description="Output image height in pixels")
    plate_bbox_pixels: Dict[str, int] = Field(..., description="Bounding box in pixel coordinates")
    yolo_bbox: Dict[str, Any] = Field(..., description="Normalized YOLO bounding box")
    applied_degradations: List[str] = Field(default_factory=list, description="List of applied degradation functions")
    scale_bucket: str = Field(default="medium", description="Scale classification: large, medium, small")
    difficulty_profile: str = Field(default="moderate", description="Difficulty level: clean, moderate, adverse")
    camera_angle: str = Field(default="standard", description="Camera perspective: standard, high_angle, left_skew, right_skew, low_angle")
    occlusion_type: Optional[str] = Field(default=None, description="Applied occlusion type (e.g. tow_hitch, bracket_lip, mud_splatter) or None")
    rotation_deg: float = Field(default=0.0, description="Planar rotation applied in degrees")
    random_seed: Optional[int] = Field(default=None, description="Random seed used for generation")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp",
    )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent)


class SyntheticPlateSample:
    """In-memory composite container holding the generated image and all annotations."""

    def __init__(
        self,
        image_np,
        pixel_bbox: BoundingBox,
        yolo_bbox: YOLOBoundingBox,
        metadata: PlateMetadata,
    ):
        self.image_np = image_np
        self.pixel_bbox = pixel_bbox
        self.yolo_bbox = yolo_bbox
        self.metadata = metadata
