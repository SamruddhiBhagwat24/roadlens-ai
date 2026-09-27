"""Unit tests for Image Quality Intelligence."""
import unittest
import numpy as np
import cv2

from backend.app.vision.image_quality import quality_analyzer


class TestImageQuality(unittest.TestCase):
    """Verifies blur, noise, brightness, and glare metric computations."""

    def test_sharp_vs_blurry_detection(self):
        # Create a high-contrast sharp grid
        sharp = np.zeros((200, 200, 3), dtype=np.uint8)
        sharp[::10, :] = 255
        sharp[:, ::10] = 255

        # Create a heavily blurred version
        blurry = cv2.GaussianBlur(sharp, (31, 31), 10)

        sharp_res = quality_analyzer.analyze(sharp)
        blurry_res = quality_analyzer.analyze(blurry)

        self.assertGreater(sharp_res["blur_score"], blurry_res["blur_score"])
        self.assertEqual(blurry_res["blur"], "high")

    def test_low_light_detection(self):
        dark_img = np.full((100, 100, 3), 20, dtype=np.uint8)
        res = quality_analyzer.analyze(dark_img)
        self.assertEqual(res["brightness"], "low")
        self.assertIn("low_light", res["adverse_conditions"])

    def test_glare_detection(self):
        glare_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Saturated specular reflection
        glare_img[20:60, 20:60] = 255
        res = quality_analyzer.analyze(glare_img)
        self.assertGreater(res["glare_score"], 0)


if __name__ == "__main__":
    unittest.main()
