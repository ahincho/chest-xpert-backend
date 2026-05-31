"""RGB-Diff security filter for rejecting non-medical images."""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

@dataclass
class FilterResult:
    """Result of the RGB-Diff filter validation."""
    passed: bool
    error: str | None = None

class FilterService:
    """Validates images by checking chromatic variance between RGB channels.
    Medical X-ray images should be grayscale. Color images (screenshots,
    photographs) are rejected based on mean absolute differences between
    R-G and G-B channels exceeding a configurable threshold.
    """
    def __init__(self, threshold: float = 5.0) -> None:
        self.threshold = threshold

    def validate(self, image_bytes: bytes) -> FilterResult:
        """Check RGB channel variance to detect non-grayscale images.
        Args:
            image_bytes: Raw image file bytes.
        Returns:
            FilterResult with passed=True if image is acceptable,
            or passed=False with error message if chromatic artifacts detected.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float64)

        diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1]).mean()
        diff_gb = np.abs(arr[:, :, 1] - arr[:, :, 2]).mean()

        if diff_rg > self.threshold or diff_gb > self.threshold:
            return FilterResult(
                passed=False,
                error=(
                    "SAMPLE REJECTED: Chromatic artifacts detected. "
                    "Please upload a valid grayscale medical image (DICOM exported to JPG/PNG)."
                ),
            )

        return FilterResult(passed=True)
