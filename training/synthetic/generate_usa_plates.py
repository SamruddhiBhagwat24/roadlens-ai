"""Synthetic USA License Plate Dataset Generator for RoadLens AI (V2).

Generates realistic USA-compatible license-plate images with YOLO-format annotations
and rich JSON metadata. Incorporates:
- Controlled plate scale buckets (large, medium, small)
- Realistic perspective / skew / rotation and camera angles
- Adverse environmental conditions (motion blur, defocus blur, sensor noise,
  exposure shifts, low light, specular glare, cast shadows, and mild occlusions)
- Rich procedural vehicle/road-like mounting backgrounds
- Realistic difficulty profiles (clean, moderate, adverse)
- Strictly deterministic random seeds

Output structure:
datasets/synthetic_usa_plates_pilot_1000/
├── images/     (JPEG image files)
├── labels/     (Single-class YOLO bounding-box annotations: 0 xc yc w h)
├── metadata/   (Per-image JSON audit metadata)
└── dataset_summary.json (Aggregated dataset statistics)
"""
from typing import List, Tuple, Dict, Any, Optional
import os
import sys
import argparse
import json
import time
from dataclasses import dataclass, field
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Ensure repository root is on sys.path for standalone script execution
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.synthetic.dataset_schema import (
    BoundingBox,
    YOLOBoundingBox,
    PlateMetadata,
    SyntheticPlateSample,
)
from training.synthetic.plate_layouts import (
    PlateLayout,
    SYNTHETIC_LAYOUTS,
    generate_valid_plate_text,
    render_synthetic_plate_image,
)
from training.synthetic.degradations import (
    DegradationConfig,
    apply_adverse_degradations,
)


@dataclass
class GeneratorConfig:
    """Configuration for synthetic dataset generation (V2)."""

    output_dir: str = "datasets/synthetic_usa_plates_pilot_1000"
    count: int = 1000
    seed: int = 42
    scale_distribution: Dict[str, float] = field(
        default_factory=lambda: {"large": 0.25, "medium": 0.50, "small": 0.25}
    )
    scale_ratios: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "large": (0.30, 0.44),    # Close vehicle (~1-3m)
            "medium": (0.18, 0.30),   # Standard distance (~4-8m)
            "small": (0.10, 0.18),    # Distant vehicle (~10-20m)
        }
    )
    difficulty_distribution: Dict[str, float] = field(
        default_factory=lambda: {"clean": 0.25, "moderate": 0.45, "adverse": 0.30}
    )
    camera_angles: List[str] = field(
        default_factory=lambda: [
            "standard",
            "left_skew",
            "right_skew",
            "high_angle",
            "low_angle",
        ]
    )
    resolutions: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (1280, 720),
            (1920, 1080),
            (800, 600),
            (640, 480),
            (960, 540),
        ]
    )


class SyntheticPlateGenerator:
    """Generates synthetic USA license-plate scenes with bounding-box annotations (V2)."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.layout_keys = sorted(list(SYNTHETIC_LAYOUTS.keys()))
        self.degradation_config = DegradationConfig()

    def _generate_scene_background(
        self, width: int, height: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Synthesizes rich vehicle and road mounting scenes."""
        bg_style = int(rng.choice([0, 1, 2, 3, 4, 5]))

        if bg_style == 0:
            # 1. BUMPER_MATTE_TEXTURE: Textured dark polymer bumper with trim & light shroud
            base_val = int(rng.uniform(28, 48))
            scene = np.full((height, width, 3), base_val, dtype=np.uint8)
            texture = rng.normal(0, 5, (height, width, 3)).astype(np.float32)
            scene = np.clip(scene.astype(np.float32) + texture, 0, 255).astype(np.uint8)
            trim_y = int(height * rng.uniform(0.38, 0.52))
            cv2.line(scene, (0, trim_y), (width, trim_y), (base_val + 20, base_val + 20, base_val + 22), 4)
            # License plate light shroud above plate area
            shroud_x = int(width * 0.40)
            shroud_w = int(width * 0.20)
            shroud_y = max(10, trim_y - 35)
            cv2.rectangle(scene, (shroud_x, shroud_y), (shroud_x + shroud_w, shroud_y + 14), (20, 20, 22), -1)
            cv2.circle(scene, (shroud_x + int(shroud_w * 0.25), shroud_y + 7), 3, (240, 240, 210), -1)
            cv2.circle(scene, (shroud_x + int(shroud_w * 0.75), shroud_y + 7), 3, (240, 240, 210), -1)

        elif bg_style == 1:
            # 2. METALLIC_CAR_BODY: Metallic car paint with clearcoat horizon reflection
            scene = np.zeros((height, width, 3), dtype=np.uint8)
            col_l = int(rng.uniform(130, 175))
            col_r = int(rng.uniform(165, 215))
            for x in range(width):
                ratio = x / float(width)
                v = int(col_l * (1.0 - ratio) + col_r * ratio)
                scene[:, x, :] = (v, v, v + 2)
            horizon_y = int(height * rng.uniform(0.30, 0.45))
            for dy in range(-25, 25):
                y = horizon_y + dy
                if 0 <= y < height:
                    fade = 1.0 - abs(dy) / 25.0
                    boost = int(40 * fade)
                    scene[y, :, :] = np.clip(scene[y, :, :].astype(np.int32) + boost, 0, 255).astype(np.uint8)

        elif bg_style == 2:
            # 3. RICH AUTOMOTIVE PAINT: Deep Red, Midnight Blue, Charcoal, Pearl White, Forest Green
            paints = [
                (125, 20, 25),    # Crimson Red
                (15, 30, 110),    # Deep Metallic Blue
                (30, 50, 35),     # Hunter Green
                (42, 42, 46),     # Charcoal Metallic
                (220, 220, 225),  # Pearl White
                (185, 140, 45),   # Sunset Gold
            ]
            base_color = paints[int(rng.choice(len(paints)))]
            scene = np.full((height, width, 3), base_color, dtype=np.uint8)
            crease_y = int(height * rng.uniform(0.20, 0.35))
            cv2.line(scene, (0, crease_y), (width, crease_y), (min(255, base_color[0] + 45), min(255, base_color[1] + 45), min(255, base_color[2] + 45)), 3)
            cv2.line(scene, (0, crease_y + 4), (width, crease_y + 4), (max(0, base_color[0] - 30), max(0, base_color[1] - 30), max(0, base_color[2] - 30)), 2)

        elif bg_style == 3:
            # 4. FRONT_GRILLE_DUAL: Radiator grille mesh with chrome trim strip
            scene = np.full((height, width, 3), (22, 22, 24), dtype=np.uint8)
            slat_sp = int(rng.uniform(16, 26))
            for y in range(0, height, slat_sp):
                cv2.line(scene, (0, y), (width, y), (55, 55, 60), 3)
            # Chrome trim strip
            mid_y = int(height * 0.48)
            cv2.line(scene, (0, mid_y), (width, mid_y), (190, 190, 195), 5)
            cv2.line(scene, (0, mid_y - 2), (width, mid_y - 2), (245, 245, 250), 2)

        elif bg_style == 4:
            # 5. TAILGATE_RECESS: Rear SUV/truck liftgate with license-plate cavity pocket
            scene = np.full((height, width, 3), (160, 160, 168), dtype=np.uint8)
            rw = int(width * rng.uniform(0.38, 0.52))
            rh = int(height * rng.uniform(0.38, 0.50))
            rx = (width - rw) // 2
            ry = (height - rh) // 2
            recess = np.full((rh, rw, 3), (38, 38, 42), dtype=np.uint8)
            scene[ry:ry + rh, rx:rx + rw] = recess
            # Cast shadow inside recess top and left
            cv2.rectangle(scene, (rx, ry), (rx + rw, ry + 12), (18, 18, 20), -1)
            cv2.rectangle(scene, (rx, ry), (rx + 10, ry + rh), (18, 18, 20), -1)
            cv2.rectangle(scene, (rx, ry), (rx + rw, ry + rh), (100, 100, 105), 3)

        else:
            # 6. ROAD_ASPHALT_LINE: Pavement texture with vehicle chassis shadow & lane marking
            scene = np.full((height, width, 3), (55, 55, 58), dtype=np.uint8)
            asphalt_noise = rng.normal(0, 8, (height, width, 3)).astype(np.float32)
            scene = np.clip(scene.astype(np.float32) + asphalt_noise, 0, 255).astype(np.uint8)
            # Vehicle shadow across upper 60%
            shadow_mask = np.zeros((height, width), dtype=np.float32)
            shadow_h = int(height * 0.65)
            shadow_mask[:shadow_h, :] = 1.0
            shadow_mask = cv2.GaussianBlur(shadow_mask, (41, 41), 15.0)
            scene = (scene.astype(np.float32) * (1.0 - 0.55 * shadow_mask[:, :, None])).astype(np.uint8)
            # White road lane marking in lower portion
            lane_y = int(height * 0.85)
            cv2.line(scene, (0, lane_y), (width, lane_y), (210, 210, 210), 12)

        return scene

    def _sample_scale_bucket(self, rng: np.random.RandomState) -> str:
        """Samples a scale bucket according to the configured distribution."""
        dist = self.config.scale_distribution
        buckets = list(dist.keys())
        probs = [dist[b] for b in buckets]
        return str(rng.choice(buckets, p=probs))

    def _sample_difficulty_profile(self, rng: np.random.RandomState) -> str:
        """Samples a difficulty profile according to the configured distribution."""
        dist = self.config.difficulty_distribution
        profiles = list(dist.keys())
        probs = [dist[p] for p in profiles]
        return str(rng.choice(profiles, p=probs))

    def _compute_perspective_corners(
        self, pw: int, ph: int, camera_angle: str, rng: np.random.RandomState
    ) -> np.ndarray:
        """Calculates destination corner coordinates for a specific camera angle."""
        src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
        dst = src.copy()

        if camera_angle == "standard":
            # Mild symmetric perspective jitter
            j = float(rng.uniform(1.0, 4.0))
            dst[0] = [j, j]
            dst[1] = [pw - j, j]
            dst[2] = [pw - j, ph - j]
            dst[3] = [j, ph - j]

        elif camera_angle == "left_skew":
            # Camera viewing vehicle from left lane (right edge compressed/further away)
            skew_x = pw * float(rng.uniform(0.06, 0.12))
            skew_y = ph * float(rng.uniform(0.08, 0.16))
            dst[0] = [0, -skew_y * 0.4]
            dst[1] = [pw - skew_x, skew_y]
            dst[2] = [pw - skew_x, ph - skew_y]
            dst[3] = [0, ph + skew_y * 0.4]

        elif camera_angle == "right_skew":
            # Camera viewing vehicle from right lane (left edge compressed/further away)
            skew_x = pw * float(rng.uniform(0.06, 0.12))
            skew_y = ph * float(rng.uniform(0.08, 0.16))
            dst[0] = [skew_x, skew_y]
            dst[1] = [pw, -skew_y * 0.4]
            dst[2] = [pw, ph + skew_y * 0.4]
            dst[3] = [skew_x, ph - skew_y]

        elif camera_angle == "high_angle":
            # High camera looking down (bottom edge narrower/further away)
            taper_x = pw * float(rng.uniform(0.06, 0.12))
            squish_y = ph * float(rng.uniform(0.08, 0.15))
            dst[0] = [-taper_x * 0.5, 0]
            dst[1] = [pw + taper_x * 0.5, 0]
            dst[2] = [pw - taper_x, ph - squish_y]
            dst[3] = [taper_x, ph - squish_y]

        elif camera_angle == "low_angle":
            # Low camera looking up (top edge narrower/further away)
            taper_x = pw * float(rng.uniform(0.06, 0.12))
            dst[0] = [taper_x, 0]
            dst[1] = [pw - taper_x, 0]
            dst[2] = [pw + taper_x * 0.5, ph]
            dst[3] = [-taper_x * 0.5, ph]

        return dst

    def generate_single_sample(self, sample_idx: int) -> SyntheticPlateSample:
        """Generates a single synthetic license-plate scene with bounding-box annotations."""
        sample_id = f"syn_us_{sample_idx:06d}"
        img_filename = f"{sample_id}.jpg"
        lbl_filename = f"{sample_id}.txt"

        # 1. Deterministic local RNG seeded by global seed + sample_idx
        local_seed = (self.config.seed * 10007 + sample_idx * 37) & 0x7FFFFFFF
        local_rng = np.random.RandomState(local_seed)

        # 2. Select image resolution
        res_idx = int(local_rng.choice(len(self.config.resolutions)))
        img_w, img_h = self.config.resolutions[res_idx]

        # 3. Select scale bucket & difficulty profile
        scale_bucket = self._sample_scale_bucket(local_rng)
        difficulty_profile = self._sample_difficulty_profile(local_rng)

        # 4. Select camera angle
        if difficulty_profile == "clean":
            camera_angle = "standard"
            rot_deg = float(local_rng.uniform(-3.0, 3.0))
        elif difficulty_profile == "moderate":
            camera_angle = str(local_rng.choice(self.config.camera_angles))
            rot_deg = float(local_rng.uniform(-6.0, 6.0))
        else:  # adverse
            camera_angle = str(local_rng.choice(self.config.camera_angles))
            rot_deg = float(local_rng.uniform(-11.0, 11.0))

        # 5. Select synthetic layout and generate text
        layout_key = self.layout_keys[int(local_rng.choice(len(self.layout_keys)))]
        layout: PlateLayout = SYNTHETIC_LAYOUTS[layout_key]
        display_text, clean_text = generate_valid_plate_text(
            layout.format_pattern, local_rng, separator=layout.separator_char
        )

        # 6. Render raw plate image at high resolution (400x200)
        plate_pil = render_synthetic_plate_image(layout, display_text, local_rng, plate_width=400, plate_height=200)
        plate_np = np.array(plate_pil)

        # 7. Scale plate based on scale_bucket
        scale_min, scale_max = self.config.scale_ratios[scale_bucket]
        target_plate_w = int(img_w * local_rng.uniform(scale_min, scale_max))
        target_plate_h = max(20, int(target_plate_w * 0.50))  # Standard 2:1 ratio

        plate_resized = cv2.resize(plate_np, (target_plate_w, target_plate_h), interpolation=cv2.INTER_AREA)

        # Rounded alpha mask for the plate
        mask_pil = Image.new("L", (target_plate_w, target_plate_h), 0)
        mask_draw = ImageDraw.Draw(mask_pil)
        corner_r = max(3, int(target_plate_h * 0.08))
        mask_draw.rounded_rectangle([0, 0, target_plate_w - 1, target_plate_h - 1], radius=corner_r, fill=255)
        mask_np = np.array(mask_pil)

        # 8. Perspective & Rotation Transformation
        pw, ph = target_plate_w, target_plate_h
        src_corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
        dst_corners = self._compute_perspective_corners(pw, ph, camera_angle, local_rng)

        # Apply rotation around center
        rot_rad = np.radians(rot_deg)
        cos_a = np.cos(rot_rad)
        sin_a = np.sin(rot_rad)
        center_x, center_y = pw / 2.0, ph / 2.0

        for i in range(4):
            rel_x = dst_corners[i, 0] - center_x
            rel_y = dst_corners[i, 1] - center_y
            dst_corners[i, 0] = rel_x * cos_a - rel_y * sin_a + center_x
            dst_corners[i, 1] = rel_x * sin_a + rel_y * cos_a + center_y

        # Shift destination corners to ensure non-negative coordinates
        min_x = float(np.min(dst_corners[:, 0]))
        min_y = float(np.min(dst_corners[:, 1]))
        shift_x = -min_x if min_x < 0 else 2.0
        shift_y = -min_y if min_y < 0 else 2.0
        dst_corners[:, 0] += shift_x
        dst_corners[:, 1] += shift_y

        box_w = int(np.ceil(np.max(dst_corners[:, 0]))) + 4
        box_h = int(np.ceil(np.max(dst_corners[:, 1]))) + 4

        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        warped_plate = cv2.warpPerspective(
            plate_resized, M, (box_w, box_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        warped_mask = cv2.warpPerspective(
            mask_np, M, (box_w, box_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # 9. Generate scene background & composite plate
        scene = self._generate_scene_background(img_w, img_h, local_rng)

        max_px = max(1, img_w - box_w - 8)
        max_py = max(1, img_h - box_h - 8)

        # Placement bias: centered horizontally, mid-to-lower vertically
        px = int(local_rng.uniform(max(5, int(img_w * 0.12)), min(max_px, int(img_w * 0.68))))
        py = int(local_rng.uniform(max(5, int(img_h * 0.22)), min(max_py, int(img_h * 0.78))))

        # Alpha composite warped plate onto scene
        alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, None]
        roi = scene[py : py + box_h, px : px + box_w]
        blended = (warped_plate.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        scene[py : py + box_h, px : px + box_w] = blended

        # 10. Compute exact bounding box of the plate in the scene
        plate_xmin = int(round(px + np.min(dst_corners[:, 0])))
        plate_ymin = int(round(py + np.min(dst_corners[:, 1])))
        plate_xmax = int(round(px + np.max(dst_corners[:, 0])))
        plate_ymax = int(round(py + np.max(dst_corners[:, 1])))

        # Strict clamping within image dimensions
        plate_xmin = max(0, min(img_w - 2, plate_xmin))
        plate_ymin = max(0, min(img_h - 2, plate_ymin))
        plate_xmax = max(plate_xmin + 2, min(img_w, plate_xmax))
        plate_ymax = max(plate_ymin + 2, min(img_h, plate_ymax))

        pixel_bbox = BoundingBox(
            x_min=plate_xmin, y_min=plate_ymin, x_max=plate_xmax, y_max=plate_ymax
        )
        yolo_bbox = pixel_bbox.to_yolo(img_width=img_w, img_height=img_h, class_id=0)

        # 11. Apply environmental degradations based on difficulty profile
        degraded_scene, applied_degradations = apply_adverse_degradations(
            scene,
            local_rng,
            self.degradation_config,
            difficulty_profile=difficulty_profile,
            bbox=(plate_xmin, plate_ymin, plate_xmax, plate_ymax),
        )

        # Detect if partial occlusion was applied
        occlusion_type = None
        for deg in applied_degradations:
            if "partial_occlusion" in deg:
                occlusion_type = deg.split("=")[-1].rstrip(")")
                break

        # 12. Construct rich V2 metadata
        metadata = PlateMetadata(
            sample_id=sample_id,
            image_filename=img_filename,
            label_filename=lbl_filename,
            plate_text=clean_text,
            synthetic_layout_id=layout.layout_id,
            state_header=layout.state_name,
            image_width=img_w,
            image_height=img_h,
            plate_bbox_pixels=pixel_bbox.to_dict(),
            yolo_bbox=yolo_bbox.model_dump(),
            applied_degradations=applied_degradations,
            scale_bucket=scale_bucket,
            difficulty_profile=difficulty_profile,
            camera_angle=camera_angle,
            occlusion_type=occlusion_type,
            rotation_deg=round(rot_deg, 2),
            random_seed=local_seed,
        )

        return SyntheticPlateSample(
            image_np=degraded_scene,
            pixel_bbox=pixel_bbox,
            yolo_bbox=yolo_bbox,
            metadata=metadata,
        )

    def generate_dataset(self) -> List[PlateMetadata]:
        """Generates the full dataset and writes all files to disk.

        Creates:
        - <output_dir>/images/<sample_id>.jpg
        - <output_dir>/labels/<sample_id>.txt
        - <output_dir>/metadata/<sample_id>.json
        - <output_dir>/dataset_summary.json
        """
        out_dir = self.config.output_dir
        images_dir = os.path.join(out_dir, "images")
        labels_dir = os.path.join(out_dir, "labels")
        metadata_dir = os.path.join(out_dir, "metadata")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        all_metadata: List[PlateMetadata] = []
        t0 = time.perf_counter()

        layout_counts: Dict[str, int] = {}
        degradation_counts: Dict[str, int] = {}
        scale_counts: Dict[str, int] = {}
        difficulty_counts: Dict[str, int] = {}
        angle_counts: Dict[str, int] = {}

        for idx in range(1, self.config.count + 1):
            sample = self.generate_single_sample(idx)
            meta = sample.metadata

            # Save JPEG image
            img_path = os.path.join(images_dir, meta.image_filename)
            cv2.imwrite(img_path, cv2.cvtColor(sample.image_np, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            # Save YOLO text label
            lbl_path = os.path.join(labels_dir, meta.label_filename)
            with open(lbl_path, "w") as f:
                f.write(sample.yolo_bbox.to_yolo_line() + "\n")

            # Save per-image JSON metadata
            meta_path = os.path.join(metadata_dir, f"{meta.sample_id}.json")
            with open(meta_path, "w") as f:
                f.write(meta.to_json(indent=2))

            all_metadata.append(meta)

            # Tally distributions
            layout_counts[meta.synthetic_layout_id] = layout_counts.get(meta.synthetic_layout_id, 0) + 1
            scale_counts[meta.scale_bucket] = scale_counts.get(meta.scale_bucket, 0) + 1
            difficulty_counts[meta.difficulty_profile] = difficulty_counts.get(meta.difficulty_profile, 0) + 1
            angle_counts[meta.camera_angle] = angle_counts.get(meta.camera_angle, 0) + 1

            for deg in meta.applied_degradations:
                deg_name = deg.split("(")[0]
                degradation_counts[deg_name] = degradation_counts.get(deg_name, 0) + 1

        elapsed = time.perf_counter() - t0

        summary = {
            "dataset_name": "RoadLens AI Synthetic USA License Plate Dataset (V2)",
            "total_samples": len(all_metadata),
            "random_seed": self.config.seed,
            "generation_time_seconds": round(elapsed, 2),
            "generation_fps": round(len(all_metadata) / max(0.001, elapsed), 2),
            "scale_distribution": scale_counts,
            "difficulty_distribution": difficulty_counts,
            "camera_angle_distribution": angle_counts,
            "layout_distribution": layout_counts,
            "degradation_frequency": degradation_counts,
            "directories": {
                "images": images_dir,
                "labels": labels_dir,
                "metadata": metadata_dir,
            },
        }

        summary_path = os.path.join(out_dir, "dataset_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return all_metadata


def main():
    parser = argparse.ArgumentParser(description="RoadLens AI Synthetic USA License Plate Generator (V2)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/synthetic_usa_plates_pilot_1000",
        help="Output directory (default: datasets/synthetic_usa_plates_pilot_1000)",
    )
    parser.add_argument("--count", type=int, default=1000, help="Number of synthetic samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    args = parser.parse_args()

    config = GeneratorConfig(output_dir=args.output_dir, count=args.count, seed=args.seed)
    generator = SyntheticPlateGenerator(config)

    if not args.quiet:
        print(f"=== RoadLens AI Synthetic USA Plate Generator (V2) ===")
        print(f"Target count: {config.count}")
        print(f"Random seed:  {config.seed}")
        print(f"Output dir:   {config.output_dir}")

    metadata = generator.generate_dataset()

    if not args.quiet:
        print(f"Successfully generated {len(metadata)} samples in {config.output_dir}")
        print(f"YOLO labels written to {os.path.join(config.output_dir, 'labels')}")
        print(f"Metadata written to {os.path.join(config.output_dir, 'metadata')}")


if __name__ == "__main__":
    main()
