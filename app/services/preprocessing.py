"""Preprocessing service for converting raw image bytes to ONNX model input tensors."""

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

class PreprocessingService:
    """Handles image preprocessing for the chest X-ray classification model."""

    @staticmethod
    def preprocess(image_bytes: bytes) -> NDArray[np.float32]:
        """Convert raw image bytes to model input tensor.
        Decodes the image, converts to grayscale, resizes to 224x224,
        and reshapes to a float32 tensor with shape (1, 1, 224, 224).
        Pixel values are kept in the original [0, 255] range without normalization.
        Args:
            image_bytes: Raw bytes of the input image (JPEG, PNG, etc.)
        Returns:
            A numpy float32 array with shape (1, 1, 224, 224) and values in [0, 255].
        Raises:
            ValueError: If the image bytes cannot be decoded or are invalid.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()  # Force full decode to catch truncated/corrupt images
        except UnidentifiedImageError:
            raise ValueError("Cannot decode image: file is not a valid image format.")
        except Exception as e:
            raise ValueError(f"Cannot decode image: {e}")

        img_gray = img.convert("L")
        img_resized = img_gray.resize((224, 224))

        arr = np.array(img_resized, dtype=np.float32)
        return arr[np.newaxis, np.newaxis, :, :]  # (1, 1, 224, 224)
