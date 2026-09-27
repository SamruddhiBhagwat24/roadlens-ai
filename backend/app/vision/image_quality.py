"""Image Quality Intelligence for RoadLens AI.

Evaluates road scene imagery before recognition to quantify adverse visual conditions:
- Blur (sharpness estimation via Laplacian variance)
- Brightness (underexposure / overexposure detection)
- Noise (sensor noise standard deviation)
- Glare (specular highlight / saturation detection)
- Perspective (off-axis skew estimation)
"""
import numpy as np
import cv2
from typing import Dict, Any, List


class ImageQualityAnalyzer:
    """Pre-recognition image quality and adverse condition evaluator."""

    def __init__(
        self,
        blur_threshold_high: float = 80.0,
        blur_threshold_medium: float = 160.0,
        low_light_threshold: float = 65.0,
        overexposed_threshold: float = 200.0,
        noise_threshold_medium: float = 8.0,
        noise_threshold_high: float = 18.0,
        glare_threshold_medium: float = 2.0,
        glare_threshold_high: float = 6.0,
    ):
        self.blur_threshold_high = blur_threshold_high
        self.blur_threshold_medium = blur_threshold_medium
        self.low_light_threshold = low_light_threshold
        self.overexposed_threshold = overexposed_threshold
        self.noise_threshold_medium = noise_threshold_medium
        self.noise_threshold_high = noise_threshold_high
        self.glare_threshold_medium = glare_threshold_medium
        self.glare_threshold_high = glare_threshold_high

    def analyze(self, image_np: np.ndarray) -> Dict[str, Any]:
        """Analyzes an RGB or BGR numpy image array.

        Returns structured image quality assessment with adverse condition flags.
        """
        if image_np is None or image_np.size == 0:
            raise ValueError("Input image is empty or invalid")

        # Convert to grayscale for frequency and luminance metrics
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image_np

        adverse_conditions: List[str] = []

        # 1. Blur Analysis (Laplacian Variance)
        # Low variance signifies weak edge gradients (blurry image)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.blur_threshold_high:
            blur_level = "high"
            adverse_conditions.append("severe_blur")
        elif laplacian_var < self.blur_threshold_medium:
            blur_level = "medium"
            adverse_conditions.append("moderate_blur")
        else:
            blur_level = "low"

        # 2. Brightness & Illumination
        mean_brightness = float(np.mean(gray))
        if mean_brightness < self.low_light_threshold:
            brightness_level = "low"
            adverse_conditions.append("low_light")
        elif mean_brightness > self.overexposed_threshold:
            brightness_level = "high"
            adverse_conditions.append("overexposed")
        else:
            brightness_level = "normal"

        # 3. Noise Estimation (High-frequency residual)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_residual = cv2.absdiff(gray, blurred)
        noise_score = float(np.std(noise_residual))
        if noise_score > self.noise_threshold_high:
            noise_level = "high"
            adverse_conditions.append("severe_noise")
        elif noise_score > self.noise_threshold_medium:
            noise_level = "medium"
            adverse_conditions.append("moderate_noise")
        else:
            noise_level = "low"

        # 4. Glare / Saturated Pixels
        saturated_pixels = np.count_nonzero(gray >= 245)
        glare_score = float((saturated_pixels / gray.size) * 100.0)
        if glare_score > self.glare_threshold_high:
            glare_level = "high"
            adverse_conditions.append("severe_glare")
        elif glare_score > self.glare_threshold_medium:
            glare_level = "medium"
            adverse_conditions.append("moderate_glare")
        else:
            glare_level = "low"

        # 5. Perspective Skew Estimation
        perspective_level, perspective_angle = self._estimate_perspective_skew(gray)
        if perspective_level != "normal":
            adverse_conditions.append("perspective_skew")

        return {
            "blur": blur_level,
            "blur_score": round(laplacian_var, 2),
            "noise": noise_level,
            "noise_score": round(noise_score, 2),
            "glare": glare_level,
            "glare_score": round(glare_score, 2),
            "brightness": brightness_level,
            "brightness_score": round(mean_brightness, 2),
            "perspective": perspective_level,
            "perspective_score": round(perspective_angle, 2),
            "adverse_conditions": adverse_conditions,
        }

    def _estimate_perspective_skew(self, gray: np.ndarray) -> tuple[str, float]:
        """Estimates dominant line angles using Canny edges and Hough transform."""
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=10)
            if lines is None or len(lines) < 4:
                return "normal", 0.0

            angles = []
            for line in lines[:30]:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Map to deviation from horizontal/vertical
                deviation = abs(angle) % 90
                if deviation > 45:
                    deviation = 90 - deviation
                angles.append(deviation)

            median_deviation = float(np.median(angles))
            if median_deviation > 20.0:
                return "high_skew", median_deviation
            elif median_deviation > 8.0:
                return "moderate_skew", median_deviation
            return "normal", median_deviation
        except Exception:
            return "normal", 0.0


# Default singleton analyzer
quality_analyzer = ImageQualityAnalyzer()
