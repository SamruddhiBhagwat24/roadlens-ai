"""Vision and OCR postprocessing utilities for RoadLens AI.

Cleans extracted text, filters artifacts, verifies coordinate boundaries,
and structures raw vision detections.
"""
import re
from typing import List, Dict, Any, Optional
from backend.app.schemas import DetectionItem, OCRResultItem


class VisionPostprocessor:
    """Cleans up detection and OCR output streams."""

    @staticmethod
    def clean_text_token(raw_text: str) -> str:
        """Removes noise artifacts and standardizes recognized character strings."""
        if not raw_text:
            return ""
        # Remove non-printable or unwanted symbols while keeping alphanumeric and hyphens
        cleaned = re.sub(r"[^\w\s\-]", "", raw_text).strip()
        return cleaned

    @staticmethod
    def clamp_box(box: List[int], max_width: int, max_height: int) -> List[int]:
        """Clamps bounding box coordinates strictly within image bounds."""
        x1, y1, x2, y2 = box
        return [
            max(0, min(int(x1), max_width)),
            max(0, min(int(y1), max_height)),
            max(0, min(int(x2), max_width)),
            max(0, min(int(y2), max_height)),
        ]

    def postprocess_ocr(
        self, ocr_results: List[OCRResultItem], max_width: int, max_height: int
    ) -> List[OCRResultItem]:
        cleaned = []
        for r in ocr_results:
            text = self.clean_text_token(r.text)
            if not text:
                continue
            box = self.clamp_box(r.bounding_box, max_width, max_height)
            cleaned.append(
                OCRResultItem(
                    text=text,
                    confidence=r.confidence,
                    bounding_box=box,
                    engine=r.engine,
                )
            )
        return cleaned


# Global postprocessor singleton
postprocessor = VisionPostprocessor()
