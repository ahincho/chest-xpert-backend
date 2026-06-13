"""Shared test fixtures and Hypothesis configuration for chest-xpert-backend tests.
Provides:
- model_path: Path to the real ONNX model (skips if not found)
- test_client: FastAPI TestClient wired with the real model
- grayscale_image_bytes: Valid 224x224 grayscale PNG as bytes
- colorful_image_bytes: High channel-variance RGB image (should be rejected by filter)
- corrupt_image_bytes: Invalid bytes that cannot be decoded as an image
"""

import io
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import settings as hypothesis_settings
from PIL import Image

# ---------------------------------------------------------------------------
# Hypothesis configuration
# ---------------------------------------------------------------------------

hypothesis_settings.register_profile(
    "default",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
hypothesis_settings.register_profile(
    "ci",
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
hypothesis_settings.load_profile("default")

# ---------------------------------------------------------------------------
# Model path fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def model_path() -> Path:
    """Return the path to the ONNX model, skipping if not found."""
    # First: check if model exists in the backend repo itself (production layout)
    local_path = Path(__file__).resolve().parent.parent / "models" / "chest-xpert-model.onnx"
    if local_path.exists():
        return local_path

    # Fallback: check sibling repo (development layout)
    sibling_path = (
        Path(__file__).resolve().parent.parent
        / ".."
        / "chest-xpert-ai"
        / "models"
        / "chest-xpert-model.onnx"
    ).resolve()
    if sibling_path.exists():
        return sibling_path

    pytest.skip(f"ONNX model not found at {local_path} or {sibling_path}")
    return local_path  # unreachable, satisfies type checker


# ---------------------------------------------------------------------------
# Test client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def test_client(model_path: Path):
    """Create a FastAPI TestClient with the real ONNX model loaded.
    Overrides get_settings to point model_path to the actual model location.
    Uses the `with TestClient(app)` pattern to trigger lifespan events
    (model loading on startup, cleanup on shutdown).
    """
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.dependencies import get_settings
    from app.main import create_app

    def _override_settings() -> Settings:
        return Settings(model_path=str(model_path))

    app = create_app()
    app.dependency_overrides[get_settings] = _override_settings
    # Override app.state.settings so lifespan uses the correct model_path
    app.state.settings = _override_settings()

    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Sample image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grayscale_image_bytes() -> bytes:
    """Create a valid 224x224 grayscale PNG image as bytes.
    Generates a simple gradient pattern to simulate a medical X-ray-like image.
    """
    # Create a gradient pattern (0-255 across width)
    arr = np.tile(np.linspace(0, 255, 224, dtype=np.uint8), (224, 1))
    img = Image.fromarray(arr, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def colorful_image_bytes() -> bytes:
    """Create a colorful RGB image with high channel variance as bytes.
    This image has large differences between R, G, and B channels,
    so it should be rejected by the RGB-Diff filter.
    """
    # Create an image with very different channel values
    height, width = 224, 224
    r_channel = np.full((height, width), 200, dtype=np.uint8)
    g_channel = np.full((height, width), 50, dtype=np.uint8)
    b_channel = np.full((height, width), 100, dtype=np.uint8)

    arr = np.stack([r_channel, g_channel, b_channel], axis=-1)
    img = Image.fromarray(arr, mode="RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def corrupt_image_bytes() -> bytes:
    """Return invalid bytes that cannot be decoded as an image."""
    return b"this is not a valid image file \x00\x01\x02\xff\xfe\xfd"
