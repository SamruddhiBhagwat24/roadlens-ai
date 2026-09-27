"""Unit tests for RoadLens AI synthetic license-plate generation pipeline."""
import unittest
import os
import shutil
import json
import numpy as np
import cv2

from training.synthetic.dataset_schema import (
    BoundingBox,
    YOLOBoundingBox,
    PlateMetadata,
)
from training.synthetic.plate_layouts import (
    SYNTHETIC_LAYOUTS,
    PlateLayout,
    generate_valid_plate_text,
    render_synthetic_plate_image,
)
from training.synthetic.degradations import (
    DegradationConfig,
    apply_gaussian_blur,
    apply_motion_blur,
    apply_sensor_noise,
    apply_brightness_contrast,
    apply_glare,
    apply_low_light,
    apply_jpeg_compression,
    apply_cast_shadow,
    apply_partial_occlusion,
    apply_adverse_degradations,
)
from training.synthetic.generate_usa_plates import (
    SyntheticPlateGenerator,
    GeneratorConfig,
)


class TestSyntheticPlates(unittest.TestCase):
    """Test suite covering schema, layouts, degradations, and full dataset generation."""

    def setUp(self):
        self.test_output_dir = "datasets/synthetic_usa_plates_test"

    def test_bounding_box_and_yolo_schema(self):
        """Tests pixel bounding box and YOLO normalized conversion."""
        bbox = BoundingBox(x_min=100, y_min=200, x_max=300, y_max=300)
        self.assertEqual(bbox.width, 200)
        self.assertEqual(bbox.height, 100)
        self.assertEqual(bbox.x_center, 200.0)
        self.assertEqual(bbox.y_center, 250.0)

        # Convert to YOLO (image 1000x500)
        yolo = bbox.to_yolo(img_width=1000, img_height=500, class_id=0)
        self.assertEqual(yolo.class_id, 0)
        self.assertAlmostEqual(yolo.x_center, 0.200000, places=5)
        self.assertAlmostEqual(yolo.y_center, 0.500000, places=5)
        self.assertAlmostEqual(yolo.width, 0.200000, places=5)
        self.assertAlmostEqual(yolo.height, 0.200000, places=5)
        self.assertEqual(yolo.to_yolo_line(), "0 0.200000 0.500000 0.200000 0.200000")

        # Convert back from YOLO
        reconstructed = yolo.to_pixel_bbox(img_width=1000, img_height=500)
        self.assertEqual(reconstructed.x_min, 100)
        self.assertEqual(reconstructed.y_min, 200)
        self.assertEqual(reconstructed.x_max, 300)
        self.assertEqual(reconstructed.y_max, 300)

    def test_plate_layouts_and_rendering(self):
        """Tests all defined synthetic layouts and verifies rendered plate images."""
        rng = np.random.RandomState(42)
        self.assertGreaterEqual(len(SYNTHETIC_LAYOUTS), 6)

        for layout_id, layout in SYNTHETIC_LAYOUTS.items():
            self.assertIsInstance(layout, PlateLayout)
            display_text, clean_text = generate_valid_plate_text(layout.format_pattern, rng, separator=layout.separator_char)
            self.assertGreater(len(clean_text), 0)
            self.assertTrue(clean_text.isalnum())

            plate_img = render_synthetic_plate_image(layout, display_text, rng, plate_width=400, plate_height=200)
            plate_np = np.array(plate_img)
            self.assertEqual(plate_np.shape, (200, 400, 3))
            self.assertEqual(plate_np.dtype, np.uint8)

    def test_degradations_do_not_crash(self):
        """Tests that all individual adverse degradation operators execute cleanly."""
        rng = np.random.RandomState(101)
        test_img = np.full((300, 400, 3), 120, dtype=np.uint8)

        ops = [
            apply_gaussian_blur,
            apply_motion_blur,
            apply_sensor_noise,
            apply_brightness_contrast,
            apply_glare,
            apply_low_light,
            apply_jpeg_compression,
        ]

        for op in ops:
            out, desc = op(test_img, rng)
            self.assertEqual(out.shape, test_img.shape, f"Operation {op.__name__} modified image shape")
            self.assertEqual(out.dtype, np.uint8, f"Operation {op.__name__} corrupted dtype")
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)

        # Composite adverse degradation pipeline
        cfg = DegradationConfig(degradation_prob=1.0)
        deg_img, deg_descs = apply_adverse_degradations(test_img, rng, cfg)
        self.assertEqual(deg_img.shape, test_img.shape)
        self.assertGreater(len(deg_descs), 0)

    def test_deterministic_generation_with_fixed_seed(self):
        """Verifies that generator produces identical outputs given the same seed."""
        cfg1 = GeneratorConfig(count=3, seed=999, output_dir="scratch/det1")
        cfg2 = GeneratorConfig(count=3, seed=999, output_dir="scratch/det2")

        gen1 = SyntheticPlateGenerator(cfg1)
        gen2 = SyntheticPlateGenerator(cfg2)

        s1 = gen1.generate_single_sample(1)
        s2 = gen2.generate_single_sample(1)

        self.assertEqual(s1.metadata.plate_text, s2.metadata.plate_text)
        self.assertEqual(s1.metadata.synthetic_layout_id, s2.metadata.synthetic_layout_id)
        self.assertEqual(s1.pixel_bbox.to_list(), s2.pixel_bbox.to_list())
        self.assertEqual(s1.yolo_bbox.to_yolo_line(), s2.yolo_bbox.to_yolo_line())
        np.testing.assert_array_equal(s1.image_np, s2.image_np)

    def test_generated_sample_validity(self):
        """Tests bounding box validity, non-empty plate text, and metadata integrity."""
        gen = SyntheticPlateGenerator(GeneratorConfig(seed=777))
        sample = gen.generate_single_sample(42)

        meta = sample.metadata
        self.assertGreater(len(meta.plate_text), 0)
        self.assertTrue(meta.plate_text.isalnum())
        self.assertGreater(meta.image_width, 0)
        self.assertGreater(meta.image_height, 0)

        # Pixel bbox must be strictly inside image
        px = sample.pixel_bbox
        self.assertGreaterEqual(px.x_min, 0)
        self.assertGreaterEqual(px.y_min, 0)
        self.assertLessEqual(px.x_max, meta.image_width)
        self.assertLessEqual(px.y_max, meta.image_height)
        self.assertGreater(px.width, 10)
        self.assertGreater(px.height, 10)

        # YOLO coordinates must be strictly within [0, 1]
        yolo = sample.yolo_bbox
        self.assertEqual(yolo.class_id, 0)
        self.assertGreater(yolo.x_center, 0.0)
        self.assertLess(yolo.x_center, 1.0)
        self.assertGreater(yolo.y_center, 0.0)
        self.assertLess(yolo.y_center, 1.0)
        self.assertGreater(yolo.width, 0.0)
        self.assertLess(yolo.width, 1.0)
        self.assertGreater(yolo.height, 0.0)
        self.assertLess(yolo.height, 1.0)

    def test_dataset_generation_produces_100_samples(self):
        """Tests full generator producing exactly 100 samples in datasets/synthetic_usa_plates_test/."""
        config = GeneratorConfig(
            output_dir=self.test_output_dir,
            count=100,
            seed=42,
        )
        generator = SyntheticPlateGenerator(config)
        metadata_list = generator.generate_dataset()

        self.assertEqual(len(metadata_list), 100)

        images_dir = os.path.join(self.test_output_dir, "images")
        labels_dir = os.path.join(self.test_output_dir, "labels")
        metadata_dir = os.path.join(self.test_output_dir, "metadata")
        summary_path = os.path.join(self.test_output_dir, "dataset_summary.json")

        self.assertTrue(os.path.isdir(images_dir))
        self.assertTrue(os.path.isdir(labels_dir))
        self.assertTrue(os.path.isdir(metadata_dir))
        self.assertTrue(os.path.isfile(summary_path))

        image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
        meta_files = [f for f in os.listdir(metadata_dir) if f.endswith(".json")]

        self.assertEqual(len(image_files), 100, f"Expected 100 image files, found {len(image_files)}")
        self.assertEqual(len(label_files), 100, f"Expected 100 label files, found {len(label_files)}")
        self.assertEqual(len(meta_files), 100, f"Expected 100 metadata files, found {len(meta_files)}")

        # Verify summary JSON
        with open(summary_path) as f:
            summary = json.load(f)
        self.assertEqual(summary["total_samples"], 100)
        self.assertEqual(summary["random_seed"], 42)
        self.assertIn("layout_distribution", summary)
        self.assertIn("degradation_frequency", summary)

        # Spot check 5 random samples
        for check_idx in [1, 25, 50, 75, 100]:
            sid = f"syn_us_{check_idx:06d}"
            img_p = os.path.join(images_dir, f"{sid}.jpg")
            lbl_p = os.path.join(labels_dir, f"{sid}.txt")
            meta_p = os.path.join(metadata_dir, f"{sid}.json")

            self.assertTrue(os.path.exists(img_p), f"Missing {img_p}")
            self.assertTrue(os.path.exists(lbl_p), f"Missing {lbl_p}")
            self.assertTrue(os.path.exists(meta_p), f"Missing {meta_p}")

            # Check image can be loaded by OpenCV
            loaded = cv2.imread(img_p)
            self.assertIsNotNone(loaded)
            self.assertGreater(loaded.shape[0], 0)
            self.assertGreater(loaded.shape[1], 0)

            # Check label format
            with open(lbl_p) as f:
                line = f.readline().strip()
            parts = line.split()
            self.assertEqual(len(parts), 5)
            self.assertEqual(parts[0], "0")
            for val in parts[1:]:
                fval = float(val)
                self.assertGreaterEqual(fval, 0.0)
                self.assertLessEqual(fval, 1.0)

    def test_v2_shadow_and_occlusion_degradations(self):
        """Tests V2 cast shadow and partial occlusion degradation operators."""
        rng = np.random.RandomState(42)
        test_img = np.full((300, 400, 3), 140, dtype=np.uint8)
        bbox = (100, 100, 300, 200)

        # Cast shadow
        shadowed, s_desc = apply_cast_shadow(test_img, rng, bbox=bbox)
        self.assertEqual(shadowed.shape, test_img.shape)
        self.assertEqual(shadowed.dtype, np.uint8)
        self.assertIn("cast_shadow", s_desc)

        # Partial occlusion
        occluded, o_desc = apply_partial_occlusion(test_img, rng, bbox=bbox)
        self.assertEqual(occluded.shape, test_img.shape)
        self.assertEqual(occluded.dtype, np.uint8)
        self.assertIn("partial_occlusion", o_desc)

    def test_v2_difficulty_profiles(self):
        """Tests V2 difficulty profiles: clean, moderate, and adverse."""
        rng = np.random.RandomState(123)
        test_img = np.full((300, 400, 3), 150, dtype=np.uint8)
        bbox = (100, 100, 300, 200)

        # Clean profile produces unmodified image
        clean_img, clean_descs = apply_adverse_degradations(
            test_img, rng, difficulty_profile="clean", bbox=bbox
        )
        self.assertEqual(clean_descs, ["clean_unmodified"])
        np.testing.assert_array_equal(clean_img, test_img)

        # Moderate profile produces 1-2 mild degradations
        mod_img, mod_descs = apply_adverse_degradations(
            test_img, rng, difficulty_profile="moderate", bbox=bbox
        )
        self.assertGreaterEqual(len(mod_descs), 1)
        self.assertLessEqual(len(mod_descs), 2)

        # Adverse profile produces 1-3 severe degradations
        adv_img, adv_descs = apply_adverse_degradations(
            test_img, rng, difficulty_profile="adverse", bbox=bbox
        )
        self.assertGreaterEqual(len(adv_descs), 1)
        self.assertLessEqual(len(adv_descs), 3)

    def test_v2_scale_buckets_and_camera_angles(self):
        """Tests V2 scale buckets (large, medium, small) and camera angles."""
        gen = SyntheticPlateGenerator(GeneratorConfig(seed=555))

        for idx in range(1, 15):
            sample = gen.generate_single_sample(idx)
            meta = sample.metadata

            # Verify V2 metadata fields exist and are valid
            self.assertIn(meta.scale_bucket, ["large", "medium", "small"])
            self.assertIn(meta.difficulty_profile, ["clean", "moderate", "adverse"])
            self.assertIn(meta.camera_angle, ["standard", "left_skew", "right_skew", "high_angle", "low_angle"])
            self.assertIsInstance(meta.rotation_deg, float)

            # Bounding box must be strictly valid inside image
            self.assertGreaterEqual(sample.pixel_bbox.x_min, 0)
            self.assertGreaterEqual(sample.pixel_bbox.y_min, 0)
            self.assertLessEqual(sample.pixel_bbox.x_max, meta.image_width)
            self.assertLessEqual(sample.pixel_bbox.y_max, meta.image_height)

            # Scale bucket checks
            plate_w = sample.pixel_bbox.width
            ratio = plate_w / float(meta.image_width)
            if meta.scale_bucket == "large":
                self.assertGreaterEqual(ratio, 0.22)
            elif meta.scale_bucket == "small":
                self.assertLessEqual(ratio, 0.25)


if __name__ == "__main__":
    unittest.main()
