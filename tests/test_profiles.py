"""Unit tests for Country / Road-Standard Profile architecture (Phase A)."""
import unittest
from backend.app.intelligence.profiles import (
    BaseCountryProfile,
    ProfileRegistry,
    USAProfile,
    IndiaProfile,
    profile_registry,
)
from backend.app.schemas import (
    DetectionItem,
    OCRResultItem,
    ProfilesResponse,
    CountryProfileSummary,
)
from backend.app.intelligence.interpreter import interpreter


class TestCountryProfiles(unittest.TestCase):
    """Verifies BaseCountryProfile contract, ProfileRegistry, and country implementations."""

    def setUp(self):
        self.registry = ProfileRegistry()
        self.usa = self.registry.get("usa")
        self.india = self.registry.get("india")

    # 1. Profile Registry Tests
    def test_registry_contains_usa_and_india(self):
        self.assertIsInstance(self.usa, USAProfile)
        self.assertIsInstance(self.india, IndiaProfile)
        self.assertTrue(self.registry.is_supported("usa"))
        self.assertTrue(self.registry.is_supported("india"))

    def test_registry_fallback_to_usa(self):
        fallback = self.registry.get("non_existent_country")
        self.assertIsInstance(fallback, USAProfile)
        self.assertEqual(fallback.code, "usa")

        none_fallback = self.registry.get(None)
        self.assertIsInstance(none_fallback, USAProfile)

    def test_registry_list_profiles_schema(self):
        summaries = self.registry.list_profiles()
        self.assertGreaterEqual(len(summaries), 2)
        codes = [s["code"] for s in summaries]
        self.assertIn("usa", codes)
        self.assertIn("india", codes)

        resp = ProfilesResponse(
            active_default="usa",
            profiles=[CountryProfileSummary(**s) for s in summaries],
        )
        self.assertEqual(resp.active_default, "usa")

    # 2. USA Profile Tests (Mini Challenge 2 Target)
    def test_usa_profile_attributes(self):
        self.assertEqual(self.usa.code, "usa")
        self.assertEqual(self.usa.speed_unit, "mph")
        self.assertEqual(self.usa.status, "production_active")
        self.assertTrue(any("MUTCD" in std for std in self.usa.standards))

    def test_usa_speed_limit_sign(self):
        det = DetectionItem(
            class_name="SPEED_LIMIT_SIGN",
            confidence=0.95,
            bounding_box=[100, 100, 300, 400],
        )
        ocr = [
            OCRResultItem(
                text="SPEED LIMIT 45",
                confidence=0.92,
                bounding_box=[120, 150, 280, 350],
                engine="UniversalOCR",
            )
        ]
        intel = self.usa.interpret_speed_sign(det, ocr)
        self.assertEqual(intel["value"], 45)
        self.assertEqual(intel["unit"], "mph")
        self.assertEqual(intel["regulatory_standard"], "FHWA MUTCD R2-1")

    def test_usa_advisory_speed_plaque(self):
        det = DetectionItem(
            class_name="ADVISORY_SPEED",
            confidence=0.91,
            bounding_box=[100, 100, 250, 250],
        )
        ocr = [
            OCRResultItem(
                text="35",
                confidence=0.89,
                bounding_box=[120, 120, 230, 230],
                engine="UniversalOCR",
            )
        ]
        intel = self.usa.interpret_advisory_speed(det, ocr)
        self.assertEqual(intel["value"], 35)
        self.assertEqual(intel["unit"], "mph")
        self.assertEqual(intel["regulatory_standard"], "FHWA MUTCD W13-1P")
        self.assertIn("cautionary", intel["advisory_notes"].lower())

    def test_usa_work_zone_sign(self):
        det = DetectionItem(
            class_name="WORK_ZONE_SIGN",
            confidence=0.93,
            bounding_box=[100, 100, 300, 300],
        )
        ocr = [
            OCRResultItem(
                text="ROAD WORK AHEAD",
                confidence=0.94,
                bounding_box=[120, 120, 280, 280],
                engine="UniversalOCR",
            )
        ]
        intel = self.usa.interpret_warning_sign(det, ocr)
        self.assertEqual(intel["regulatory_standard"], "FHWA MUTCD W20 Series (Work Zone)")
        self.assertIn("ROAD WORK", intel["meaning"].upper())

    def test_usa_license_plate_validation(self):
        # Case 1: Valid California plate
        val = self.usa.validate_license_plate("CALIFORNIA 7XYZ123", bounding_box=[50, 50, 250, 150])
        self.assertTrue(val["valid"])
        self.assertEqual(val["detected_state"], "CALIFORNIA")
        self.assertEqual(val["plate_code"], "7XYZ123")
        self.assertTrue(val["aspect_ratio_valid"])

        # Case 2: Texas plate
        val_tx = self.usa.validate_license_plate("TEXAS ABC1234", bounding_box=[50, 50, 250, 150])
        self.assertTrue(val_tx["valid"])
        self.assertEqual(val_tx["detected_state"], "TEXAS")

    # 3. India Staged Profile Tests
    def test_india_profile_attributes(self):
        self.assertEqual(self.india.code, "india")
        self.assertEqual(self.india.speed_unit, "km/h")
        self.assertEqual(self.india.status, "staged_specification")
        self.assertTrue(any("IRC:67" in std for std in self.india.standards))

    def test_india_speed_sign_kmh(self):
        det = DetectionItem(
            class_name="SPEED_LIMIT_SIGN",
            confidence=0.94,
            bounding_box=[100, 100, 300, 300],
        )
        ocr = [
            OCRResultItem(
                text="60",
                confidence=0.90,
                bounding_box=[120, 120, 280, 280],
                engine="UniversalOCR",
            )
        ]
        intel = self.india.interpret_speed_sign(det, ocr)
        self.assertEqual(intel["value"], 60)
        self.assertEqual(intel["unit"], "km/h")
        self.assertEqual(intel["regulatory_standard"], "IRC:67-2022 Mandatory")

    def test_india_license_plate_syntax(self):
        # Case 1: Standard Maharashtra plate
        val = self.india.validate_license_plate("MH 12 DE 1432")
        self.assertTrue(val["valid"])
        self.assertEqual(val["detected_state_code"], "MH")
        self.assertFalse(val["is_bh_series"])

        # Case 2: Bharat series
        val_bh = self.india.validate_license_plate("22 BH 1234 AA")
        self.assertTrue(val_bh["valid"])
        self.assertTrue(val_bh["is_bh_series"])

    # 4. Country Routing in Interpreter
    def test_country_routing_usa_vs_india(self):
        dets = [
            DetectionItem(
                class_name="SPEED_LIMIT_SIGN",
                confidence=0.95,
                bounding_box=[100, 100, 300, 400],
            )
        ]
        ocr = [
            OCRResultItem(
                text="50",
                confidence=0.90,
                bounding_box=[150, 150, 250, 250],
                engine="UniversalOCR",
            )
        ]
        quality = {"adverse_conditions": []}

        # USA routing -> mph
        usa_intel = interpreter.interpret(dets, ocr, quality, country_code="usa")
        self.assertEqual(usa_intel.country_profile, "usa")
        self.assertEqual(usa_intel.unit, "mph")
        self.assertEqual(usa_intel.value, 50)
        self.assertEqual(usa_intel.regulatory_standard, "FHWA MUTCD R2-1")

        # India routing -> km/h
        india_intel = interpreter.interpret(dets, ocr, quality, country_code="india")
        self.assertEqual(india_intel.country_profile, "india")
        self.assertEqual(india_intel.unit, "km/h")
        self.assertEqual(india_intel.value, 50)
        self.assertEqual(india_intel.regulatory_standard, "IRC:67-2022 Mandatory")


if __name__ == "__main__":
    unittest.main()
