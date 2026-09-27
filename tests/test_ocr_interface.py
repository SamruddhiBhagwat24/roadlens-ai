"""Unit tests for replaceable OCR service and engine contracts."""
import unittest
import numpy as np

from backend.app.vision.ocr import (
    BaseOCREngine,
    HeuristicContourOCREngine,
    OCRService,
    ocr_service,
)
from backend.app.schemas import OCRResultItem


class DummyTestEngine(BaseOCREngine):
    """Custom pluggable test engine to verify OCR swappability."""
    def name(self) -> str:
        return "CustomTestEngine"

    def is_available(self) -> bool:
        return True

    def recognize_crop(self, image_crop, offset_box=None):
        return [
            OCRResultItem(
                text="TEST_SPEED_55",
                confidence=0.99,
                bounding_box=offset_box or [0, 0, 10, 10],
                engine=self.name(),
            )
        ]


class TestOCRModularInterface(unittest.TestCase):
    """Validates pluggable OCR interface and factory dispatch."""

    def test_heuristic_engine_availability(self):
        engine = HeuristicContourOCREngine()
        self.assertTrue(engine.is_available())
        self.assertEqual(engine.name(), "HeuristicContourCV")

    def test_heuristic_engine_execution(self):
        crop = np.zeros((80, 80, 3), dtype=np.uint8)
        # Draw some character-like patterns
        crop[20:60, 20:30] = 255
        crop[20:60, 40:50] = 255
        engine = HeuristicContourOCREngine()
        results = engine.recognize_crop(crop, offset_box=[10, 10, 90, 90])
        self.assertIsInstance(results, list)

    def test_custom_engine_plug_and_play(self):
        service = OCRService()
        custom_engine = DummyTestEngine()
        service.engines["custom"] = custom_engine
        service.preferred_engine = "custom"

        active = service.get_active_engine()
        self.assertEqual(active.name(), "CustomTestEngine")

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        extracted = service.extract_from_regions(dummy_img, [[10, 10, 50, 50]])
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].text, "TEST_SPEED_55")


if __name__ == "__main__":
    unittest.main()
