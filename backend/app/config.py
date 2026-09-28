"""Configuration management for RoadLens AI backend.

Supports loading from environment variables and provides sensible defaults.
Works with or without pydantic-settings.
"""
import os
from typing import List

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class Settings(BaseSettings):
        host: str = Field(default="0.0.0.0", alias="HOST")
        port: int = Field(default=8000, alias="PORT")
        environment: str = Field(default="development", alias="ENVIRONMENT")
        log_level: str = Field(default="INFO", alias="LOG_LEVEL")

        # CORS
        allowed_origins_raw: str = Field(
            default="http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,https://frontend-chi-ashen-3d1tf04kdq.vercel.app",
            alias="ALLOWED_ORIGINS",
        )

        # Uploads
        max_upload_size_mb: int = Field(default=15, alias="MAX_UPLOAD_SIZE_MB")
        allowed_extensions_raw: str = Field(
            default="jpg,jpeg,png,webp", alias="ALLOWED_EXTENSIONS"
        )

        # Device & Acceleration
        device_preference: str = Field(default="auto", alias="DEVICE_PREFERENCE")

        # OCR Engine
        ocr_engine: str = Field(default="auto", alias="OCR_ENGINE")

        # Country Profile
        default_country_profile: str = Field(default="usa", alias="DEFAULT_COUNTRY_PROFILE")

        # Semantic Intelligence
        gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
        gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"

        @property
        def allowed_origins(self) -> List[str]:
            return [o.strip() for o in self.allowed_origins_raw.split(",") if o.strip()]

        @property
        def allowed_extensions(self) -> List[str]:
            return [
                e.strip().lower()
                for e in self.allowed_extensions_raw.split(",")
                if e.strip()
            ]

        @property
        def max_upload_size_bytes(self) -> int:
            return self.max_upload_size_mb * 1024 * 1024

    settings = Settings()

except ImportError:
    # Graceful fallback without pydantic-settings
    class FallbackSettings:
        def __init__(self):
            self.host = os.getenv("HOST", "0.0.0.0")
            self.port = int(os.getenv("PORT", "8000"))
            self.environment = os.getenv("ENVIRONMENT", "development")
            self.log_level = os.getenv("LOG_LEVEL", "INFO")
            self.allowed_origins_raw = os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,https://frontend-chi-ashen-3d1tf04kdq.vercel.app",
            )
            self.max_upload_size_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))
            self.allowed_extensions_raw = os.getenv(
                "ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp"
            )
            self.device_preference = os.getenv("DEVICE_PREFERENCE", "auto")
            self.ocr_engine = os.getenv("OCR_ENGINE", "auto")
            self.default_country_profile = os.getenv("DEFAULT_COUNTRY_PROFILE", "usa").lower().strip()
            self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
            self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        @property
        def allowed_origins(self) -> List[str]:
            return [o.strip() for o in self.allowed_origins_raw.split(",") if o.strip()]

        @property
        def allowed_extensions(self) -> List[str]:
            return [
                e.strip().lower()
                for e in self.allowed_extensions_raw.split(",")
                if e.strip()
            ]

        @property
        def max_upload_size_bytes(self) -> int:
            return self.max_upload_size_mb * 1024 * 1024

    settings = FallbackSettings()
