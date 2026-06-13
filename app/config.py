"""Application configuration using pydantic-settings.
Settings are loaded from a .env file and/or environment variables.
Environment variables take precedence over .env file values.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings loaded from .env file and environment variables.
    All fields use the prefix CHESTXPERT_ for environment variable loading.
    For example, CHESTXPERT_MODEL_PATH overrides model_path.
    A .env file in the project root is loaded automatically if present.
    """

    model_path: str = Field(
        default="models/chest-xpert-model.onnx",
        max_length=255,
        description="Path to the ONNX model file",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:4200"],
        description="Allowed CORS origins",
    )
    server_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Server port number",
    )
    rgb_diff_threshold: float = Field(
        default=5.0,
        description="RGB-Diff filter threshold (0.0–255.0)",
    )
    target_classes: list[str] = Field(
        default=[
            "Cardiomegaly",
            "Edema",
            "Consolidation",
            "Atelectasis",
            "Pleural Effusion",
        ],
        description="Target pathology class names",
    )

    model_config = SettingsConfigDict(
        env_prefix="CHEST_XPERT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("rgb_diff_threshold", mode="before")
    @classmethod
    def clamp_threshold(cls, v: object) -> float:
        """Clamp rgb_diff_threshold to [0.0, 255.0]; fall back to 5.0 if out of range."""
        try:
            val = float(v)  # type: ignore[arg-type]
            if val < 0.0 or val > 255.0:
                return 5.0
            return val
        except TypeError, ValueError:
            return 5.0
