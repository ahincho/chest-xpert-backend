"""Unit tests for the RGB-Diff FilterService."""

import io

from PIL import Image

from app.services.filter import FilterResult, FilterService


class TestFilterService:
    """Tests for FilterService.validate() — no model required."""

    def test_grayscale_image_passes(self, grayscale_image_bytes):
        """A grayscale image should pass the filter."""
        service = FilterService(threshold=5.0)
        result = service.validate(grayscale_image_bytes)
        assert result.passed is True
        assert result.error is None

    def test_colorful_image_rejected(self, colorful_image_bytes):
        """A colorful RGB image should be rejected."""
        service = FilterService(threshold=5.0)
        result = service.validate(colorful_image_bytes)
        assert result.passed is False
        assert result.error is not None
        assert "Chromatic artifacts" in result.error

    def test_pure_white_passes(self):
        """A pure white image (R=G=B=255) should pass."""
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        service = FilterService(threshold=5.0)
        result = service.validate(buf.getvalue())
        assert result.passed is True

    def test_pure_black_passes(self):
        """A pure black image (R=G=B=0) should pass."""
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        service = FilterService(threshold=5.0)
        result = service.validate(buf.getvalue())
        assert result.passed is True

    def test_subtle_color_passes_default_threshold(self):
        """An image with slight color variance (< 5.0) should pass."""
        # R=130, G=128, B=129 → diff_rg=2, diff_gb=1
        img = Image.new("RGB", (100, 100), color=(130, 128, 129))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        service = FilterService(threshold=5.0)
        result = service.validate(buf.getvalue())
        assert result.passed is True

    def test_strong_red_rejected(self):
        """A strong red image (R=255, G=0, B=0) should be rejected."""
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        service = FilterService(threshold=5.0)
        result = service.validate(buf.getvalue())
        assert result.passed is False

    def test_custom_threshold_strict(self):
        """A strict threshold (1.0) rejects even slight color differences."""
        # R=130, G=128, B=128 → diff_rg=2.0 > 1.0
        img = Image.new("RGB", (100, 100), color=(130, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        service = FilterService(threshold=1.0)
        result = service.validate(buf.getvalue())
        assert result.passed is False

    def test_custom_threshold_permissive(self):
        """A permissive threshold (200.0) allows most images."""
        img = Image.new("RGB", (100, 100), color=(200, 50, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        service = FilterService(threshold=200.0)
        result = service.validate(buf.getvalue())
        assert result.passed is True

    def test_grayscale_mode_l_passes(self):
        """A mode 'L' (grayscale) image converted to RGB should pass."""
        img = Image.new("L", (100, 100), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        service = FilterService(threshold=5.0)
        result = service.validate(buf.getvalue())
        assert result.passed is True

    def test_result_dataclass_structure(self):
        """FilterResult has correct fields."""
        result = FilterResult(passed=True, error=None)
        assert result.passed is True
        assert result.error is None

        result_fail = FilterResult(passed=False, error="test error")
        assert result_fail.passed is False
        assert result_fail.error == "test error"
