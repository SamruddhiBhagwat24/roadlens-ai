"""Unit tests for adaptive preprocessing operations."""
import unittest
import numpy as np

from backend.app.vision.preprocessing import preprocessor


class TestPreprocessing(unittest.TestCase):
    """Tests preprocessing methods and conditional pipeline logic."""

    def setUp(self):
        # Create standard test synthetic RGB image
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.img[25:75, 25:75] = 200

    def test_to_grayscale(self):
        gray = preprocessor.to_grayscale(self.img)
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (100, 100))

    def test_enhance_low_light(self):
        dark_img = np.full((50, 50, 3), 30, dtype=np.uint8)
        enhanced = preprocessor.enhance_low_light(dark_img, gamma=1.8)
        self.assertGreater(np.mean(enhanced), np.mean(dark_img))

    def test_apply_clahe(self):
        clahe_out = preprocessor.apply_clahe(self.img)
        self.assertEqual(clahe_out.shape, self.img.shape)

    def test_denoise(self):
        noisy_img = self.img.copy()
        noise = np.random.normal(0, 20, self.img.shape).astype(np.int16)
        noisy_img = np.clip(noisy_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        denoised = preprocessor.denoise(noisy_img)
        self.assertEqual(denoised.shape, self.img.shape)

    def test_sharpen(self):
        sharpened = preprocessor.sharpen(self.img)
        self.assertEqual(sharpened.shape, self.img.shape)

    def test_adaptive_pipeline_conditional(self):
        # Case 1: Nominal conditions -> no transform applied (passthrough)
        nominal_quality = {"adverse_conditions": []}
        out, ops = preprocessor.adaptively_preprocess(self.img, nominal_quality)
        self.assertIn("passthrough(nominal_conditions)", ops)

        # Case 2: Low-light condition -> enhance applied
        low_light_quality = {"adverse_conditions": ["low_light"]}
        out, ops = preprocessor.adaptively_preprocess(self.img, low_light_quality)
        self.assertTrue(any("enhance_low_light" in op for op in ops))

        # Case 3: Multiple conditions -> chained conditional transforms
        multi_quality = {"adverse_conditions": ["severe_noise", "severe_glare", "severe_blur"]}
        out, ops = preprocessor.adaptively_preprocess(self.img, multi_quality)
        self.assertEqual(len(ops), 3)


if __name__ == "__main__":
    unittest.main()
