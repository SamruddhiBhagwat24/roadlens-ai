"""Image upload security and integrity validation for RoadLens AI.

Enforces:
1. File extension validation (.jpg, .jpeg, .png, .webp)
2. MIME-type validation
3. Binary magic byte signature verification
4. Maximum payload size boundaries (default 15MB)
5. Integrity check to detect truncated, corrupt, or malicious files
"""
import io
import os
from typing import Tuple
from PIL import Image
from backend.app.config import settings


class ValidationError(Exception):
    """Raised when uploaded file fails security or format validation."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# Magic byte signatures for authorized image formats
MAGIC_SIGNATURES = {
    "jpeg": [(0, b"\xff\xd8\xff")],
    "png": [(0, b"\x89PNG\r\n\x1a\n")],
    "webp": [(0, b"RIFF"), (8, b"WEBP")],
}


def validate_image_upload(filename: str, content: bytes, content_type: str = "") -> Tuple[Image.Image, str]:
    """Validates uploaded image bytes for security, size, and integrity.

    Returns:
        Tuple of (PIL.Image.Image, normalized_format_string)
    Raises:
        ValidationError if any security check fails.
    """
    if not filename:
        raise ValidationError("Missing filename in upload request", 400)

    # 1. File size validation
    size_bytes = len(content)
    if size_bytes == 0:
        raise ValidationError("Uploaded file is empty (0 bytes)", 400)

    if size_bytes > settings.max_upload_size_bytes:
        max_mb = settings.max_upload_size_mb
        raise ValidationError(
            f"File size ({round(size_bytes / (1024*1024), 2)} MB) exceeds maximum limit of {max_mb} MB",
            413,
        )

    # 2. File extension check
    _, ext = os.path.splitext(filename.lower())
    clean_ext = ext.lstrip(".")
    if clean_ext not in settings.allowed_extensions:
        raise ValidationError(
            f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(settings.allowed_extensions)}",
            415,
        )

    # 3. Binary Magic Byte Signature Check
    detected_format = None
    if content.startswith(b"\xff\xd8\xff"):
        detected_format = "jpeg"
    elif content.startswith(b"\x89PNG\r\n\x1a\n"):
        detected_format = "png"
    elif content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"WEBP":
        detected_format = "webp"
    else:
        raise ValidationError(
            "File header does not match valid image signature (corrupt or spoofed file)",
            400,
        )

    # Cross-check detected binary format with file extension
    if detected_format == "jpeg" and clean_ext not in ["jpg", "jpeg"]:
        raise ValidationError(f"File extension '.{clean_ext}' does not match JPEG binary content", 400)
    elif detected_format == "png" and clean_ext != "png":
        raise ValidationError(f"File extension '.{clean_ext}' does not match PNG binary content", 400)
    elif detected_format == "webp" and clean_ext != "webp":
        raise ValidationError(f"File extension '.{clean_ext}' does not match WEBP binary content", 400)

    # 4. MIME-type validation if provided
    if content_type:
        expected_mimes = ["image/jpeg", "image/jpg", "image/png", "image/webp", "application/octet-stream"]
        if content_type.lower() not in expected_mimes:
            raise ValidationError(f"Invalid Content-Type header: '{content_type}'", 415)

    # 5. Image Integrity & Decompression Verification
    try:
        image_stream = io.BytesIO(content)
        pil_img = Image.open(image_stream)
        pil_img.verify()  # Verifies file integrity without full decompression
    except Exception as e:
        raise ValidationError(f"Corrupt or unreadable image file: {str(e)}", 400)

    # Re-open after verify() closes the stream
    try:
        image_stream.seek(0)
        final_img = Image.open(image_stream)
        # Convert to RGB to sanitize palette/alpha complexities for CV
        final_img = final_img.convert("RGB")
        return final_img, detected_format
    except Exception as e:
        raise ValidationError(f"Failed to decode image pixels: {str(e)}", 400)
