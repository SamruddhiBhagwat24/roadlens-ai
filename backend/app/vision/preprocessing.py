"""Modular adaptive preprocessing pipeline for RoadLens AI.

Applies conditional enhancements targeted specifically at identified adverse visual conditions:
- Low-light enhancement (adaptive gamma correction)
- Glare reduction (CLAHE contrast limited adaptive histogram equalization)
- Denoising (bilateral edge-preserving filtering)
- Blur mitigation (unsharp masking / Laplacian sharpening)
- Perspective rectification

Strict Rule: Preprocessing is conditional. Filters are only applied when
the pre-analysis triggers adverse condition flags.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


class ImagePreprocessor:
    """Modular conditional image preprocessor."""

    @staticmethod
    def to_grayscale(image_np: np.ndarray) -> np.ndarray:
        if len(image_np.shape) == 2:
            return image_np
        if image_np.shape[2] == 4:
            return cv2.cvtColor(image_np, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def enhance_low_light(image_np: np.ndarray, gamma: float = 1.6) -> np.ndarray:
        """Applies adaptive gamma correction to boost luminance in underexposed regions."""
        inv_gamma = 1.0 / max(gamma, 0.1)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image_np, table)

    @staticmethod
    def apply_clahe(image_np: np.ndarray, clip_limit: float = 2.5, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        if len(image_np.shape) == 3:
            # Convert to LAB to equalize luminance channel without distorting chromaticity
            lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return clahe.apply(image_np)

    @staticmethod
    def denoise(image_np: np.ndarray, strength: int = 5) -> np.ndarray:
        """Applies bilateral filtering to suppress sensor noise while preserving sharp edge boundaries."""
        if len(image_np.shape) == 3:
            return cv2.bilateralFilter(image_np, d=strength, sigmaColor=50, sigmaSpace=50)
        return cv2.bilateralFilter(image_np, d=strength, sigmaColor=50, sigmaSpace=50)

    @staticmethod
    def sharpen(image_np: np.ndarray, strength: float = 1.2) -> np.ndarray:
        """Applies unsharp masking to restore high-frequency character edge detail."""
        gaussian = cv2.GaussianBlur(image_np, (0, 0), sigmaX=2.0)
        unsharp = cv2.addWeighted(image_np, 1.0 + strength, gaussian, -strength, 0)
        return unsharp

    def adaptively_preprocess(
        self, image_np: np.ndarray, quality_metrics: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """Conditionally applies targeted transforms based strictly on quality triggers.

        Returns (processed_image, list_of_applied_operations).
        """
        processed = image_np.copy()
        applied_ops: List[str] = []

        adverse = quality_metrics.get("adverse_conditions", [])

        # 1. Noise reduction first if noisy
        if "severe_noise" in adverse or "moderate_noise" in adverse:
            strength = 9 if "severe_noise" in adverse else 5
            processed = self.denoise(processed, strength=strength)
            applied_ops.append(f"denoise(strength={strength})")

        # 2. Low-light enhancement
        if "low_light" in adverse:
            processed = self.enhance_low_light(processed, gamma=1.6)
            applied_ops.append("enhance_low_light(gamma=1.6)")

        # 3. Glare & contrast correction
        if "severe_glare" in adverse or "moderate_glare" in adverse:
            clip = 3.0 if "severe_glare" in adverse else 2.0
            processed = self.apply_clahe(processed, clip_limit=clip)
            applied_ops.append(f"clahe(clip_limit={clip})")

        # 4. Sharpening if blurred
        if "severe_blur" in adverse or "moderate_blur" in adverse:
            strength = 1.5 if "severe_blur" in adverse else 1.0
            processed = self.sharpen(processed, strength=strength)
            applied_ops.append(f"sharpen(strength={strength})")

        if not applied_ops:
            applied_ops.append("passthrough(nominal_conditions)")

        return processed, applied_ops


# Default preprocessor singleton
preprocessor = ImagePreprocessor()
