"""Synthetic USA license-plate dataset generation modules for RoadLens AI."""
from training.synthetic.dataset_schema import (
    BoundingBox,
    YOLOBoundingBox,
    PlateMetadata,
    SyntheticPlateSample,
)
from training.synthetic.plate_layouts import (
    PlateLayout,
    PlateStyle,
    SYNTHETIC_LAYOUTS,
    render_synthetic_plate_image,
    generate_valid_plate_text,
)
from training.synthetic.degradations import (
    apply_adverse_degradations,
    DegradationConfig,
)

__all__ = [
    "BoundingBox",
    "YOLOBoundingBox",
    "PlateMetadata",
    "SyntheticPlateSample",
    "PlateLayout",
    "PlateStyle",
    "SYNTHETIC_LAYOUTS",
    "render_synthetic_plate_image",
    "generate_valid_plate_text",
    "apply_adverse_degradations",
    "DegradationConfig",
]
