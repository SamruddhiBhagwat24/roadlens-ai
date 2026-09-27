"""Unit tests for Phase B model integration (YOLOv8s and EasyOCR interfaces)."""
import os
import unittest
import numpy as np

from backend.app.schemas import DetectionItem, OCRResultItem
from backend.app.vision.detector import (
    BaseDetector,
    YOLOv8RoadDetector,
    HeuristicRoadDetector,
    RoadDetectorService,
)
from backend.app.vision.ocr import (
    BaseOCREngine,
    EasyOCREngine,
    HeuristicContourOCREngine,
    OCRService,
)


class TestModelIntegration(unittest.TestCase):
    """Verifies that downloaded model weights and detector abstractions interface cleanly."""

    def setUp(self):
        # Create a synthetic 300x300 BGR test image
        self.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw a red rectangle in center
        self.test_img[100:200, 100:200] = [0, 0, 255]

    def test_yolov8_detector_availability(self):
        detector = YOLOv8RoadDetector(weights_path="models/yolov8s.pt")
        # If models/yolov8s.pt exists and ultralytics is installed, is_available is True
        self.assertEqual(detector.name(), "YOLOv8s")
        if os.path.exists("models/yolov8s.pt"):
            self.assertTrue(detector.is_available())

    def test_yolov8_detector_inference_schema(self):
        detector = YOLOv8RoadDetector(weights_path="models/yolov8s.pt")
        if not detector.is_available():
            self.skipTest("YOLOv8s weights not available")

        detections = detector.detect(self.test_img)
        self.assertIsInstance(detections, list)
        for det in detections:
            self.assertIsInstance(det, DetectionItem)
            self.assertIsInstance(det.class_name, str)
            self.assertGreaterEqual(det.confidence, 0.0)
            self.assertLessEqual(det.confidence, 1.0)
            self.assertEqual(len(det.bounding_box), 4)

    def test_yolov8_class_mapping(self):
        # Verify class mapping behavior
        cls_name, label = YOLOv8RoadDetector._map_class("stop sign")
        self.assertEqual(cls_name, "STOP_SIGN")
        self.assertEqual(label, "Stop Sign")

        cls_name, label = YOLOv8RoadDetector._map_class("car")
        self.assertEqual(cls_name, "VEHICLE")
        self.assertEqual(label, "Car")

        # Unknown classes should map to None
        cls_name, label = YOLOv8RoadDetector._map_class("couch")
        self.assertIsNone(cls_name)

    def test_heuristic_detector_fallback(self):
        detector = HeuristicRoadDetector()
        self.assertTrue(detector.is_available())
        self.assertEqual(detector.name(), "HeuristicContourCV")
        detections = detector.detect(self.test_img)
        self.assertIsInstance(detections, list)

    def test_road_detector_service_orchestration(self):
        service = RoadDetectorService(preferred_detector="auto")
        active = service.get_active_detector()
        self.assertIsInstance(active, BaseDetector)
        self.assertIn("plate_yolo", service.detectors)
        self.assertEqual(service.detectors["plate_yolo"].name(), "plate_yolo")
        detections = service.detect(self.test_img)
        self.assertIsInstance(detections, list)

    def test_plate_yolo_dedicated_detection(self):
        detector = YOLOv8RoadDetector(weights_path="models/plate_yolov8.pt", detector_name="plate_yolo")
        self.assertEqual(detector.name(), "plate_yolo")
        if not detector.is_available():
            self.skipTest("models/plate_yolov8.pt not available")
        self.assertTrue(detector.is_available())
        detections = detector.detect(self.test_img)
        self.assertIsInstance(detections, list)
        for det in detections:
            self.assertIsInstance(det, DetectionItem)
            self.assertEqual(det.class_name, "LICENSE_PLATE")
            self.assertGreaterEqual(det.confidence, 0.0)
            self.assertLessEqual(det.confidence, 1.0)
            self.assertEqual(len(det.bounding_box), 4)

    def test_easyocr_engine_interface(self):
        engine = EasyOCREngine(model_storage_directory="models")
        self.assertEqual(engine.name(), "EasyOCR")
        # If easyocr package is not installed, is_available is False
        try:
            import easyocr
            has_pkg = True
        except ImportError:
            has_pkg = False

        if has_pkg and os.path.exists("models/craft_mlt_25k.pth"):
            self.assertTrue(engine.is_available())
        else:
            self.assertFalse(engine.is_available())

    def test_ocr_service_fallback(self):
        service = OCRService()
        active = service.get_active_engine()
        self.assertIsInstance(active, BaseOCREngine)
        # Empty or nominal crop returns list of OCRResultItem
        results = service.extract_from_regions(self.test_img, [[50, 50, 250, 250]])
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
