"""Edge case tests to improve code coverage.

Covers: config validators, error handlers, content-type validation,
file size limits, and model-not-found scenarios.
"""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from app.config import Settings
from app.services.inference import InferenceService


class TestConfigValidator:
    """Tests for Settings.clamp_threshold validator."""

    def test_threshold_below_zero_returns_default(self):
        """Negative threshold is clamped to 5.0."""
        settings = Settings(rgb_diff_threshold=-10.0)
        assert settings.rgb_diff_threshold == 5.0

    def test_threshold_above_255_returns_default(self):
        """Threshold above 255 is clamped to 5.0."""
        settings = Settings(rgb_diff_threshold=300.0)
        assert settings.rgb_diff_threshold == 5.0

    def test_threshold_invalid_type_returns_default(self):
        """Non-numeric threshold returns 5.0."""
        settings = Settings(rgb_diff_threshold="invalid")
        assert settings.rgb_diff_threshold == 5.0

    def test_threshold_valid_passes(self):
        """A valid threshold within range is kept."""
        settings = Settings(rgb_diff_threshold=10.0)
        assert settings.rgb_diff_threshold == 10.0


class TestInferenceServiceErrors:
    """Tests for InferenceService error handling."""

    def test_model_not_found_raises(self):
        """FileNotFoundError raised when model path doesn't exist."""
        with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
            InferenceService("/nonexistent/path/model.onnx")


class TestPredictEdgeCases:
    """Edge cases for the /predict endpoint."""

    def test_invalid_content_type_returns_400(self, test_client):
        """Non-image content type is rejected with 400."""
        response = test_client.post(
            "/predict",
            files={"file": ("doc.pdf", b"fake pdf content", "application/pdf")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "Invalid image format" in data["error"]

    def test_file_too_large_returns_413(self, test_client):
        """File exceeding 10MB returns 413."""
        # Create bytes just over 10MB
        large_bytes = b"x" * (10 * 1024 * 1024 + 1)
        response = test_client.post(
            "/predict",
            files={"file": ("big.png", large_bytes, "image/png")},
        )
        assert response.status_code == 413
        data = response.json()
        assert data["success"] is False
        assert "10MB" in data["error"]

    def test_preprocessing_value_error_returns_400(self, test_client):
        """An image that passes filter but fails preprocessing returns 400."""
        # Create a very small valid PNG that PIL can open but is unusual
        # Actually, we mock preprocessing to raise ValueError
        with patch(
            "app.routers.predict.PreprocessingService.preprocess",
            side_effect=ValueError("Test preprocessing failure"),
        ):
            # Need a grayscale image that passes the filter
            img = Image.new("L", (10, 10), color=128)
            buf = io.BytesIO()
            img.save(buf, format="PNG")

            response = test_client.post(
                "/predict",
                files={"file": ("xray.png", buf.getvalue(), "image/png")},
            )
            assert response.status_code == 400
            data = response.json()
            assert data["success"] is False
            assert "Invalid image" in data["error"]
