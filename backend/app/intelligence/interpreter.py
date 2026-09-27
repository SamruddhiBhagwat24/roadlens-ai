"""Semantic Understanding and Road Intelligence Interpreter for RoadLens AI.

Orchestrates semantic interpretation of universal vision extractions by dispatching
to the designated Country / Road-Standard Profile (e.g. USA MUTCD, India IRC:67).

Ensures complete country-agnostic design in the Universal Perception Layer:
- Units (mph vs km/h), sign standards, and plate syntax rules are encapsulated within profiles.
- Supports optional server-side Gemini enrichment without violating jurisdictional rules.
"""
from typing import List, Dict, Any, Optional
from backend.app.schemas import DetectionItem, OCRResultItem, RoadIntelligenceDetails
from backend.app.intelligence.profiles import profile_registry
from backend.app.config import settings


class RoadIntelligenceInterpreter:
    """Interprets raw detections and OCR tokens into structured road state via Country Profiles."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_key = gemini_api_key or settings.gemini_api_key

    def interpret(
        self,
        detections: List[DetectionItem],
        ocr_results: List[OCRResultItem],
        image_quality: Dict[str, Any],
        country_code: Optional[str] = None,
    ) -> RoadIntelligenceDetails:
        """Translates localized entities and tokens into structured road intelligence.

        Args:
            detections: Bounding boxes and object classes from the universal detector.
            ocr_results: Raw text tokens and character confidences from OCR.
            image_quality: Universal adverse visual condition metrics.
            country_code: Country standard to apply (e.g. 'usa', 'india').
        """
        # Resolve target country profile (defaults to configured system default, e.g. 'usa')
        target_country = country_code or settings.default_country_profile
        profile = profile_registry.get(target_country)

        # Delegate semantic reasoning to the active profile
        base_intelligence = profile.interpret_scene(detections, ocr_results, image_quality)

        # Optional higher-level multimodal enrichment via Gemini if key is provided
        if self.gemini_key:
            try:
                enriched = self._enrich_with_gemini(base_intelligence, detections, ocr_results)
                if enriched:
                    return enriched
            except Exception:
                pass

        return base_intelligence

    def _enrich_with_gemini(
        self,
        base_intelligence: RoadIntelligenceDetails,
        detections: List[DetectionItem],
        ocr_results: List[OCRResultItem],
    ) -> Optional[RoadIntelligenceDetails]:
        """Optional higher-level interpretation enrichment via server-side Gemini API."""
        # Kept modular for subsequent milestones
        return None


# Global interpreter singleton
interpreter = RoadIntelligenceInterpreter()
