"""API integration tests for RoadLens AI endpoints."""
import io
import unittest
import numpy as np
from PIL import Image

try:
    from fastapi.testclient import TestClient
    from backend.app.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class TestAPIEndpoints(unittest.TestCase):
    """Tests FastAPI /api/health, /api/config, and /api/analyze endpoints."""

    def setUp(self):
        if not FASTAPI_AVAILABLE:
            raise unittest.SkipTest("FastAPI test dependencies not installed yet.")
        self.client = TestClient(app)

    def _create_sample_jpeg(self) -> bytes:
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_health_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("device", data)

    def test_config_endpoint(self):
        resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("allowed_extensions", data)

    def test_analyze_endpoint_valid_image(self):
        img_bytes = self._create_sample_jpeg()
        resp = self.client.post(
            "/api/analyze",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("detections", data)
        self.assertIn("ocr_results", data)
        self.assertIn("road_intelligence", data)
        self.assertIn("image_quality", data)
        self.assertIn("performance", data)

    def test_analyze_endpoint_invalid_file(self):
        resp = self.client.post(
            "/api/analyze",
            files={"file": ("test.txt", b"plain text", "text/plain")},
        )
        self.assertIn(resp.status_code, [400, 415])

    def test_profiles_endpoint(self):
        resp = self.client.get("/api/profiles")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["active_default"], "usa")
        self.assertTrue(any(p["code"] == "usa" for p in data["profiles"]))
        self.assertTrue(any(p["code"] == "india" for p in data["profiles"]))

    def test_analyze_endpoint_with_country_code(self):
        img_bytes = self._create_sample_jpeg()
        resp = self.client.post(
            "/api/analyze?country_code=india",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["road_intelligence"]["country_profile"], "india")


if __name__ == "__main__":
    unittest.main()
