"""Integration tests for the POST /predict endpoint.

These tests require the ONNX model to be available.
They are automatically skipped if the model file is not found
(via the test_client fixture which depends on model_path).
"""

import io

import numpy as np
import pytest
from PIL import Image


class TestPredictEndpoint:
    """Integration tests for POST /predict — requires ONNX model."""

    def test_predict_grayscale_returns_200(self, test_client, grayscale_image_bytes):
        """A valid grayscale image returns 200 with predictions."""
        response = test_client.post(
            "/predict",
            files={"file": ("xray.png", grayscale_image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["predictions"]) == 5

    def test_predict_response_structure(self, test_client, grayscale_image_bytes):
        """Response has correct schema: success + predictions list."""
        response = test_client.post(
            "/predict",
            files={"file": ("xray.png", grayscale_image_bytes, "image/png")},
        )
        data = response.json()
        assert "success" in data
        assert "predictions" in data

        for item in data["predictions"]:
            assert "pathology" in item
            assert "probability" in item
            assert 0.0 <= item["probability"] <= 1.0

    def test_predict_returns_five_pathologies(self, test_client, grayscale_image_bytes):
        """Response contains exactly 5 pathology predictions."""
        response = test_client.post(
            "/predict",
            files={"file": ("xray.png", grayscale_image_bytes, "image/png")},
        )
        data = response.json()
        pathologies = [p["pathology"] for p in data["predictions"]]
        expected = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
        assert pathologies == expected

    def test_predict_colorful_image_rejected(self, test_client, colorful_image_bytes):
        """A colorful image is rejected by the RGB-Diff filter."""
        response = test_client.post(
            "/predict",
            files={"file": ("photo.png", colorful_image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "Chromatic artifacts" in data["error"]

    def test_predict_corrupt_image_raises_error(self, test_client, corrupt_image_bytes):
        """Corrupt/invalid image bytes trigger an error in the filter service.

        Note: The FilterService currently doesn't handle PIL.UnidentifiedImageError
        for corrupt bytes. This test documents the current behavior. A future fix
        should add try/except in FilterService.validate() to return a 400 response.
        """
        from PIL import UnidentifiedImageError

        with pytest.raises(UnidentifiedImageError):
            test_client.post(
                "/predict",
                files={"file": ("bad.png", corrupt_image_bytes, "image/png")},
            )

    def test_predict_no_file_returns_422(self, test_client):
        """Missing file field returns 422 validation error."""
        response = test_client.post("/predict")
        assert response.status_code == 422

    def test_predict_jpeg_accepted(self, test_client):
        """A valid JPEG grayscale image is accepted."""
        img = Image.new("L", (224, 224), color=100)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        response = test_client.post(
            "/predict",
            files={"file": ("xray.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_predict_large_image_resized(self, test_client):
        """A large image (1024x1024) is resized and processed correctly."""
        arr = np.random.default_rng(42).integers(0, 256, (1024, 1024), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        response = test_client.post(
            "/predict",
            files={"file": ("big_xray.png", buf.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["predictions"]) == 5
