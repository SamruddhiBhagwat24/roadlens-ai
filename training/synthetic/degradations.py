"""Adverse environmental degradation and augmentation functions for RoadLens AI (V2).

Provides mathematically sound, realistic road-condition degradations:
- Gaussian defocus blur
- Directional linear motion blur
- Sensor/electronic noise (CMOS ISO grain)
- Brightness and contrast variation
- Glare (headlight / direct sun specular flare)
- Low-light / night conditions (underexposure + non-linear gamma attenuation)
- Lossy JPEG compression artifacts
- Spatial perspective homography distortion
- Small planar rotations
- Angled and lip cast shadows
- Mild partial occlusions (tow hitch ball, mounting bracket tab, mud splatter)

Supports difficulty profiles: 'clean', 'moderate', 'adverse'.
All operations accept a deterministic RandomState to guarantee exact reproducibility.
"""
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import cv2


@dataclass
class DegradationConfig:
    """Configuration governing degradation probabilities and intensity ranges."""

    degradation_prob: float = 0.70  # default for backwards compatibility
    max_simultaneous_degradations: int = 3
    gaussian_blur_prob: float = 0.30
    motion_blur_prob: float = 0.30
    sensor_noise_prob: float = 0.35
    brightness_contrast_prob: float = 0.40
    glare_prob: float = 0.25
    low_light_prob: float = 0.30
    jpeg_compression_prob: float = 0.40
    shadow_prob: float = 0.30
    occlusion_prob: float = 0.25


def apply_gaussian_blur(img: np.ndarray, rng: np.random.RandomState, ksize: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """Applies Gaussian defocus blur with random or specified kernel size and sigma."""
    if ksize is None:
        ksize = int(rng.choice([3, 5, 7]))
    sigma = float(rng.uniform(0.8, 2.5))
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return blurred, f"gaussian_blur(k={ksize},sigma={sigma:.2f})"


def apply_motion_blur(img: np.ndarray, rng: np.random.RandomState, length: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """Applies directional linear motion blur mimicking vehicle vibration/speed."""
    if length is None:
        length = int(rng.choice([5, 7, 9, 11]))
    angle_deg = float(rng.uniform(-30.0, 30.0))

    # Construct linear kernel
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    cv2.line(
        kernel,
        (0, center),
        (length - 1, center),
        color=1.0,
        thickness=1,
    )
    # Rotate kernel
    rot_mat = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (length, length))
    ksum = np.sum(kernel)
    if ksum > 0:
        kernel /= ksum

    blurred = cv2.filter2D(img, -1, kernel)
    return blurred, f"motion_blur(len={length},ang={angle_deg:.1f})"


def apply_sensor_noise(img: np.ndarray, rng: np.random.RandomState, std: Optional[float] = None) -> Tuple[np.ndarray, str]:
    """Applies additive Gaussian sensor noise (CMOS ISO grain)."""
    if std is None:
        std = float(rng.uniform(8.0, 24.0))
    noise = rng.normal(0, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, f"sensor_noise(std={std:.1f})"


def apply_brightness_contrast(
    img: np.ndarray, rng: np.random.RandomState, alpha: Optional[float] = None, beta: Optional[float] = None
) -> Tuple[np.ndarray, str]:
    """Applies realistic exposure scaling and contrast adjustment."""
    if alpha is None:
        alpha = float(rng.uniform(0.70, 1.35))  # Contrast multiplier
    if beta is None:
        beta = float(rng.uniform(-35.0, 35.0))  # Brightness bias
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted, f"brightness_contrast(alpha={alpha:.2f},beta={beta:.1f})"


def apply_glare(
    img: np.ndarray, rng: np.random.RandomState, bbox: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[np.ndarray, str]:
    """Simulates localized high-intensity specular highlight from sun or headlights."""
    h, w = img.shape[:2]
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        cx = int(rng.uniform(max(0, xmin - 20), min(w, xmax + 20)))
        cy = int(rng.uniform(max(0, ymin - 20), min(h, ymax + 20)))
        radius = int(rng.uniform(max(20, (xmax - xmin) * 0.3), max(40, (xmax - xmin) * 0.8)))
    else:
        cx = int(rng.uniform(w * 0.2, w * 0.8))
        cy = int(rng.uniform(h * 0.2, h * 0.8))
        radius = int(rng.uniform(min(h, w) * 0.15, min(h, w) * 0.45))

    intensity = float(rng.uniform(0.45, 0.85))

    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    mask = np.clip(1.0 - (dist_from_center / float(max(1, radius))), 0.0, 1.0)
    mask = (mask ** 1.8) * intensity

    glare_color = np.array([255, 250, 235], dtype=np.float32)  # Warm white
    out = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1.0 - mask) + glare_color[c] * mask
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, f"glare(cx={cx},cy={cy},r={radius})"


def apply_low_light(img: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, str]:
    """Simulates nighttime / dusk adverse lighting via gamma curve and brightness drop."""
    gamma = float(rng.uniform(1.6, 2.5))
    attenuation = float(rng.uniform(0.35, 0.65))

    lut = np.array(
        [np.clip(((i / 255.0) ** gamma) * 255.0 * attenuation, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    darkened = cv2.LUT(img, lut)
    return darkened, f"low_light(gamma={gamma:.2f},att={attenuation:.2f})"


def apply_jpeg_compression(img: np.ndarray, rng: np.random.RandomState, quality: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """Simulates compression block artifacts from dashcam encoders."""
    if quality is None:
        quality = int(rng.randint(20, 55))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, enc = cv2.imencode(".jpg", img, encode_param)
    if not success:
        return img, "jpeg_compression_failed"
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return decoded, f"jpeg_compression(q={quality})"


def apply_cast_shadow(
    img: np.ndarray, rng: np.random.RandomState, bbox: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[np.ndarray, str]:
    """Simulates cast shadows across the vehicle or plate (bumper lip or diagonal pole shadow)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    shadow_style = int(rng.choice([0, 1]))
    attenuation = float(rng.uniform(0.35, 0.55))

    if bbox is not None and shadow_style == 0:
        # Bumper overhang / trunk lip shadow covering upper portion of the plate
        xmin, ymin, xmax, ymax = bbox
        ph = ymax - ymin
        lip_bottom = int(ymin + ph * rng.uniform(0.20, 0.45))
        cv2.rectangle(mask, (max(0, xmin - 20), max(0, ymin - 20)), (min(w, xmax + 20), lip_bottom), 1.0, -1)
        desc = "cast_shadow(type=bumper_lip)"
    else:
        # Diagonal band shadow (tree branch, signpost, side vehicle shadow)
        p1 = (int(rng.uniform(0, w * 0.6)), 0)
        p2 = (int(p1[0] + rng.uniform(80, 220)), 0)
        p3 = (int(rng.uniform(p2[0] - 100, w)), h)
        p4 = (int(max(0, p3[0] - rng.uniform(80, 220))), h)
        poly = np.array([p1, p2, p3, p4], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 1.0)
        desc = "cast_shadow(type=diagonal_band)"

    # Soft feathering
    mask = cv2.GaussianBlur(mask, (21, 21), 6.0)
    shadowed = (img.astype(np.float32) * (1.0 - attenuation * mask[:, :, None])).astype(np.uint8)
    return shadowed, desc


def apply_partial_occlusion(
    img: np.ndarray, rng: np.random.RandomState, bbox: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[np.ndarray, str]:
    """Applies realistic mild partial occlusion (tow hitch ball, bracket tab, or mud splatter).

    Coverage is strictly mild (5-15% of plate area), ensuring the plate remains detectable.
    """
    h, w = img.shape[:2]
    if bbox is None:
        # Fallback to random position in lower center
        xmin, ymin, xmax, ymax = int(w * 0.4), int(h * 0.5), int(w * 0.6), int(h * 0.65)
    else:
        xmin, ymin, xmax, ymax = bbox

    pw = max(10, xmax - xmin)
    ph = max(10, ymax - ymin)
    occ_style = int(rng.choice([0, 1, 2]))
    out = img.copy()

    if occ_style == 0:
        # Tow hitch ball at bottom-center of the plate
        cx = int(xmin + pw * rng.uniform(0.42, 0.58))
        cy = int(ymax - ph * rng.uniform(0.02, 0.12))
        r = int(ph * rng.uniform(0.18, 0.28))
        cv2.circle(out, (cx, cy), r, (28, 28, 30), -1)
        cv2.circle(out, (cx, cy), r, (55, 55, 60), 2)
        # Specular ball highlight
        cv2.circle(out, (cx - int(r * 0.25), cy - int(r * 0.25)), max(2, int(r * 0.2)), (95, 95, 100), -1)
        desc = "partial_occlusion(type=tow_hitch)"

    elif occ_style == 1:
        # License plate mounting bracket tab along top or bottom rim
        on_top = bool(rng.rand() > 0.5)
        tab_w = int(pw * rng.uniform(0.10, 0.18))
        tab_h = int(ph * rng.uniform(0.12, 0.20))
        tx = int(xmin + pw * rng.uniform(0.20, 0.80 - tab_w / float(pw)))
        ty = ymin if on_top else (ymax - tab_h)
        cv2.rectangle(out, (tx, ty), (tx + tab_w, ty + tab_h), (20, 20, 22), -1)
        cv2.rectangle(out, (tx, ty), (tx + tab_w, ty + tab_h), (45, 45, 48), 1)
        desc = "partial_occlusion(type=bracket_tab)"

    else:
        # Cluster of road grime / mud splatter on one corner
        corner_left = bool(rng.rand() > 0.5)
        cx = int(xmin + pw * (0.10 if corner_left else 0.90))
        cy = int(ymax - ph * rng.uniform(0.10, 0.35))
        num_splatters = int(rng.randint(5, 10))
        for _ in range(num_splatters):
            sx = int(cx + rng.normal(0, pw * 0.05))
            sy = int(cy + rng.normal(0, ph * 0.08))
            sr = int(rng.randint(2, max(3, int(ph * 0.08))))
            cv2.circle(out, (sx, sy), sr, (45, 40, 32), -1)
        desc = "partial_occlusion(type=mud_splatter)"

    return out, desc


def apply_adverse_degradations(
    img: np.ndarray,
    rng: np.random.RandomState,
    config: Optional[DegradationConfig] = None,
    difficulty_profile: str = "moderate",
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Applies adverse conditions calibrated to a specific difficulty profile.

    Difficulty Profiles:
    - 'clean': 100% clean, crisp, unmodified output.
    - 'moderate': 1 to 2 mild degradations (mild blur, JPEG, exposure tweak).
    - 'adverse': 2 to 3 severe degradations (glare, low light, heavy blur, noise, shadow, occlusion).

    Returns:
        Tuple of (degraded_image_np, list_of_applied_degradations).
    """
    if config is None:
        config = DegradationConfig()

    applied_degradations: List[str] = []

    # 1. Clean Profile
    if difficulty_profile == "clean":
        applied_degradations.append("clean_unmodified")
        return img.copy(), applied_degradations

    out = img.copy()

    # 2. Moderate Profile
    if difficulty_profile == "moderate":
        moderate_ops = [
            lambda m: apply_gaussian_blur(m, rng, ksize=3),
            lambda m: apply_motion_blur(m, rng, length=5),
            lambda m: apply_brightness_contrast(m, rng, alpha=float(rng.uniform(0.85, 1.15)), beta=float(rng.uniform(-15, 15))),
            lambda m: apply_jpeg_compression(m, rng, quality=int(rng.randint(45, 65))),
            lambda m: apply_sensor_noise(m, rng, std=float(rng.uniform(5.0, 10.0))),
        ]
        num_ops = int(rng.choice([1, 2]))
        perm = rng.permutation(len(moderate_ops))[:num_ops]
        for idx in perm:
            out, desc = moderate_ops[idx](out)
            applied_degradations.append(desc)
        return out, applied_degradations

    # 3. Adverse Profile (Severe conditions)
    adverse_candidates = [
        (0.35, lambda m: apply_glare(m, rng, bbox=bbox)),
        (0.35, lambda m: apply_low_light(m, rng)),
        (0.35, lambda m: apply_cast_shadow(m, rng, bbox=bbox)),
        (0.30, lambda m: apply_partial_occlusion(m, rng, bbox=bbox)),
        (0.35, lambda m: apply_motion_blur(m, rng, length=int(rng.choice([9, 11])))),
        (0.35, lambda m: apply_sensor_noise(m, rng, std=float(rng.uniform(14.0, 26.0)))),
        (0.40, lambda m: apply_jpeg_compression(m, rng, quality=int(rng.randint(20, 38)))),
        (0.30, lambda m: apply_gaussian_blur(m, rng, ksize=5)),
    ]

    perm = rng.permutation(len(adverse_candidates))
    count = 0
    max_ops = config.max_simultaneous_degradations

    for idx in perm:
        prob, op = adverse_candidates[idx]
        if rng.rand() < prob:
            out, desc = op(out)
            applied_degradations.append(desc)
            count += 1
            if count >= max_ops:
                break

    if not applied_degradations:
        out, desc = apply_motion_blur(out, rng, length=7)
        applied_degradations.append(desc)

    return out, applied_degradations
