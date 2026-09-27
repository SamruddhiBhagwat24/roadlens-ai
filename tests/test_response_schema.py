"""Unit tests for response schema validation and serialization."""
import unittest
from backend.app.schemas import (
    AnalyzeResponse,
    DetectionItem,
    OCRResultItem,
    RoadIntelligenceDetails,
    ImageQualityDetails,
    PerformanceDetails,
    ImageDimensions,
)


class TestResponseSchema(unittest.TestCase):
    """Ensures JSON responses strictly match RoadLens AI specification."""

    def test_schema_serialization(self):
        resp = AnalyzeResponse(
            success=True,
            filename="sample_road.jpg",
            image_dimensions=ImageDimensions(width=1920, height=1080, channels=3),
            detections=[
                DetectionItem(
                    class_name="SPEED_LIMIT_SIGN",
                    confidence=0.95,
                    bounding_box=[100, 150, 300, 350],
                    label="Speed Limit Sign",
                )
            ],
            ocr_results=[
                OCRResultItem(
                    text="45",
                    confidence=0.92,
                    bounding_box=[120, 200, 280, 320],
                    engine="HeuristicContourCV",
                )
            ],
            road_intelligence=RoadIntelligenceDetails(
                object_type="speed_limit_sign",
                meaning="Speed limit 45 MPH",
                value=45,
                unit="mph",
                detection_confidence=0.95,
                ocr_confidence=0.92,
            ),
            image_quality=ImageQualityDetails(
                blur="low",
                blur_score=145.2,
                noise="low",
                noise_score=3.1,
                glare="low",
                glare_score=0.4,
                brightness="normal",
                brightness_score=128.5,
                perspective="normal",
                perspective_score=2.1,
                adverse_conditions=[],
            ),
            performance=PerformanceDetails(
                device_type="CPU",
                device_name="Intel CPU",
                rocm_enabled=False,
                latency_ms={"quality_check": 4.2, "preprocessing": 1.1, "detection": 12.5, "ocr": 8.4, "total": 26.2},
                gpu_utilization=None,
                gpu_memory_mb=None,
                status_note="Executed on CPU",
            ),
        )

        d = resp.model_dump()
        self.assertIn("detections", d)
        self.assertIn("ocr_results", d)
        self.assertIn("road_intelligence", d)
        self.assertIn("image_quality", d)
        self.assertIn("performance", d)
        self.assertEqual(d["road_intelligence"]["value"], 45)
        self.assertEqual(d["road_intelligence"]["unit"], "mph")


if __name__ == "__main__":
    unittest.main()
