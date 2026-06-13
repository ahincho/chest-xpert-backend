"""Unit tests for PreprocessingService."""

import io

import numpy as np
import pytest
from PIL import Image

from app.services.preprocessing import PreprocessingService


class TestPreprocessingService:
    """Tests for PreprocessingService.preprocess()."""

    def _make_image_bytes(
        self, mode: str = "RGB", size: tuple = (100, 100), fmt: str = "PNG"
    ) -> bytes:
        """Helper to create image bytes for testing."""
        img = Image.new(mode, size, color=128)
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    def test_output_shape(self):
        """Output tensor has shape (1, 1, 224, 224)."""
        image_bytes = self._make_image_bytes()
        result = PreprocessingService.preprocess(image_bytes)
        assert result.shape == (1, 1, 224, 224)

    def test_output_dtype(self):
        """Output tensor has dtype float32."""
        image_bytes = self._make_image_bytes()
        result = PreprocessingService.preprocess(image_bytes)
        assert result.dtype == np.float32

    def test_output_value_range(self):
        """Output values are in [0, 255] without normalization."""
        image_bytes = self._make_image_bytes()
        result = PreprocessingService.preprocess(image_bytes)
        assert result.min() >= 0.0
        assert result.max() <= 255.0

    def test_grayscale_conversion(self):
        """RGB image is converted to grayscale (single channel)."""
        img = Image.new("RGB", (50, 50), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = PreprocessingService.preprocess(buf.getvalue())
        # Single channel output
        assert result.shape[1] == 1

    def test_resize_to_224x224(self):
        """Images of various sizes are resized to 224x224."""
        for size in [(10, 10), (500, 300), (1, 1), (1024, 1024)]:
            image_bytes = self._make_image_bytes(size=size)
            result = PreprocessingService.preprocess(image_bytes)
            assert result.shape == (1, 1, 224, 224)

    def test_jpeg_format(self):
        """JPEG images are processed correctly."""
        image_bytes = self._make_image_bytes(mode="RGB", fmt="JPEG")
        result = PreprocessingService.preprocess(image_bytes)
        assert result.shape == (1, 1, 224, 224)
        assert result.dtype == np.float32

    def test_grayscale_input(self):
        """Already-grayscale images are handled correctly."""
        image_bytes = self._make_image_bytes(mode="L")
        result = PreprocessingService.preprocess(image_bytes)
        assert result.shape == (1, 1, 224, 224)

    def test_round_trip_fidelity(self):
        """Extracted [0,0,:,:] as uint8 differs from resized original by at most 1 per pixel."""
        # Fill with a gradient pattern (arange wraps naturally with uint8)
        pixels = np.arange(300 * 300, dtype=np.uint8).reshape(300, 300)
        img = Image.fromarray(pixels, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = PreprocessingService.preprocess(buf.getvalue())

        # Extract and cast back to uint8
        reconstructed = result[0, 0, :, :].astype(np.uint8)

        # Compare with directly resized original
        expected = np.array(img.resize((224, 224)), dtype=np.uint8)

        diff = np.abs(reconstructed.astype(np.int16) - expected.astype(np.int16))
        assert diff.max() <= 1

    def test_invalid_bytes_raises_value_error(self):
        """Invalid/corrupt bytes raise ValueError."""
        with pytest.raises(ValueError, match="Cannot decode image"):
            PreprocessingService.preprocess(b"not an image at all")

    def test_empty_bytes_raises_value_error(self):
        """Empty bytes raise ValueError."""
        with pytest.raises(ValueError, match="Cannot decode image"):
            PreprocessingService.preprocess(b"")

    def test_no_normalization_applied(self):
        """Pixel values are preserved without division or scaling."""
        # Create a white image (255 in grayscale)
        img = Image.new("L", (224, 224), color=255)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = PreprocessingService.preprocess(buf.getvalue())
        # All values should be 255.0, not 1.0 (no /255 normalization)
        assert np.allclose(result, 255.0)
