"""Unit tests for image upload security and format validation."""
import io
import unittest
import numpy as np
from PIL import Image

from backend.app.api.validation import validate_image_upload, ValidationError


class TestImageValidation(unittest.TestCase):
    """Verifies file type, size, MIME, magic signature, and integrity checks."""

    def _create_synthetic_image_bytes(self, fmt: str = "JPEG", size=(100, 100)) -> bytes:
        img = Image.fromarray(np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    def test_valid_jpeg(self):
        content = self._create_synthetic_image_bytes("JPEG")
        pil_img, detected_fmt = validate_image_upload("road_scene.jpg", content, "image/jpeg")
        self.assertEqual(detected_fmt, "jpeg")
        self.assertEqual(pil_img.size, (100, 100))

    def test_valid_png(self):
        content = self._create_synthetic_image_bytes("PNG")
        pil_img, detected_fmt = validate_image_upload("stop_sign.png", content, "image/png")
        self.assertEqual(detected_fmt, "png")
        self.assertEqual(pil_img.size, (100, 100))

    def test_valid_webp(self):
        content = self._create_synthetic_image_bytes("WEBP")
        pil_img, detected_fmt = validate_image_upload("plate.webp", content, "image/webp")
        self.assertEqual(detected_fmt, "webp")
        self.assertEqual(pil_img.size, (100, 100))

    def test_empty_file_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_image_upload("empty.jpg", b"")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_unsupported_extension_rejected(self):
        content = self._create_synthetic_image_bytes("JPEG")
        with self.assertRaises(ValidationError) as ctx:
            validate_image_upload("malicious.exe", content)
        self.assertIn("unsupported file extension", str(ctx.exception).lower())

    def test_spoofed_extension_rejected(self):
        # A text file renamed to .jpg
        fake_content = b"This is not a real jpeg file"
        with self.assertRaises(ValidationError) as ctx:
            validate_image_upload("fake.jpg", fake_content)
        self.assertIn("signature", str(ctx.exception).lower())

    def test_mismatched_extension_and_content_rejected(self):
        # Real PNG saved with .jpg extension
        png_content = self._create_synthetic_image_bytes("PNG")
        with self.assertRaises(ValidationError) as ctx:
            validate_image_upload("actually_png.jpg", png_content)
        self.assertIn("does not match", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
