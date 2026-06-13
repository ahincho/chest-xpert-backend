"""Inference service for running predictions via ONNX Runtime."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray


class InferenceService:
    """Manages the ONNX Runtime inference session for chest X-ray classification.
    Loads the ONNX model on construction and provides a predict method
    that accepts a preprocessed tensor and returns pathology probabilities.
    """

    def __init__(self, model_path: str) -> None:
        """Load the ONNX model from the given path.
        Args:
            model_path: Path to the .onnx model file.
        Raises:
            FileNotFoundError: If the model file does not exist at the given path.
            ort.InvalidGraph: If the model file is corrupt or invalid (propagated from onnxruntime).
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")
        self.session = ort.InferenceSession(model_path)

    def predict(self, tensor: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run inference on a preprocessed image tensor.
        Args:
            tensor: A float32 numpy array with shape (1, 1, 224, 224)
                    representing a grayscale chest X-ray with pixel values in [0, 255].
        Returns:
            A float32 numpy array of 5 probability values corresponding to
            [Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion].
        """
        results = self.session.run(["predictions"], {"image": tensor})
        return results[0][0]

    def close(self) -> None:
        """Release ONNX session resources."""
        del self.session
