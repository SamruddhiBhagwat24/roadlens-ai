"""Abstract Base Country Profile contract for RoadLens AI.

Defines the interface for jurisdiction-specific road intelligence:
- Speed measurement units (mph vs km/h)
- National traffic sign standards (FHWA MUTCD, IRC:67, Vienna Convention)
- License plate syntax, aspect ratio, and regional header validation
- Regional hazard and road-work semantics

Ensures the Universal Perception Layer remains strictly country-agnostic.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from backend.app.schemas import DetectionItem, OCRResultItem, RoadIntelligenceDetails


class BaseCountryProfile(ABC):
    """Abstract base class for all national road-standard and regulatory profiles."""

    @property
    @abstractmethod
    def code(self) -> str:
        """Unique lowercase profile identifier (e.g. 'usa', 'india')."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Full human-readable country profile name."""
        pass

    @property
    @abstractmethod
    def speed_unit(self) -> str:
        """National legal speed measurement unit ('mph' or 'km/h')."""
        pass

    @property
    @abstractmethod
    def status(self) -> str:
        """Implementation status: 'production_active' or 'staged_specification'."""
        pass

    @property
    @abstractmethod
    def standards(self) -> List[str]:
        """List of applicable road and signage legal standards."""
        pass

    @abstractmethod
    def interpret_scene(
        self,
        detections: List[DetectionItem],
        ocr_results: List[OCRResultItem],
        image_quality: Dict[str, Any],
    ) -> RoadIntelligenceDetails:
        """Translates localized road entities and extracted tokens into structured road state."""
        pass

    @abstractmethod
    def validate_license_plate(
        self,
        plate_text: str,
        bounding_box: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Validates alphanumeric plate code against national format rules."""
        pass

    @abstractmethod
    def interpret_speed_sign(
        self,
        detection: DetectionItem,
        ocr_results: List[OCRResultItem],
    ) -> Dict[str, Any]:
        """Interprets regulatory maximum or advisory speed plaques according to local laws."""
        pass

    @abstractmethod
    def interpret_warning_sign(
        self,
        detection: DetectionItem,
        ocr_results: List[OCRResultItem],
    ) -> Dict[str, Any]:
        """Maps hazard/construction signage into national warning categories."""
        pass

    def get_summary(self) -> Dict[str, Any]:
        """Returns metadata dictionary describing this profile."""
        return {
            "code": self.code,
            "name": self.name,
            "speed_unit": self.speed_unit,
            "status": self.status,
            "standards": self.standards,
        }
