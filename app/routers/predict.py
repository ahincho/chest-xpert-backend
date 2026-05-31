"""Prediction router — POST /predict endpoint.
Accepts a chest X-ray image via multipart/form-data, validates it,
runs the RGB-Diff filter, preprocesses, and returns pathology predictions.
"""

import logging

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse

from app.config import Settings
from app.dependencies import (
    get_filter_service,
    get_inference_service,
    get_preprocessing_service,
    get_settings,
)
from app.errors import FileTooLargeError
from app.schemas.prediction import (
    ErrorResponse,
    PredictionItem,
    PredictionSuccessResponse,
)
from app.services.filter import FilterService
from app.services.inference import InferenceService
from app.services.preprocessing import PreprocessingService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])

_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

@router.post(
    "/predict",
    response_model=PredictionSuccessResponse,
    summary="Predict pathologies from chest X-ray",
    description=(
        "Upload a chest X-ray image (JPEG or PNG, max 10MB) and receive "
        "probability scores for 5 thoracic pathologies: Cardiomegaly, Edema, "
        "Consolidation, Atelectasis, and Pleural Effusion."
    ),
    responses={
        200: {
            "description": "Prediction result (success or filter rejection)",
            "model": PredictionSuccessResponse,
        },
        400: {
            "description": "Invalid or corrupt image",
            "model": ErrorResponse,
        },
        413: {
            "description": "File exceeds 10MB limit",
            "model": ErrorResponse,
        },
        422: {
            "description": "Missing file or validation error",
            "model": ErrorResponse,
        },
    },
)
async def predict(
    file: UploadFile = File(...),
    inference_service: InferenceService = Depends(get_inference_service),
    filter_service: FilterService = Depends(get_filter_service),
    preprocessing_service: PreprocessingService = Depends(get_preprocessing_service),
    settings: Settings = Depends(get_settings),
) -> PredictionSuccessResponse | JSONResponse:
    """Process a chest X-ray image and return pathology predictions."""
    # Read file bytes
    file_bytes = await file.read()

    # Validate file size
    if len(file_bytes) > _MAX_FILE_SIZE:
        raise FileTooLargeError()

    # Validate image format via content type
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Invalid image format. Only JPEG and PNG are accepted.",
            },
        )

    # Run RGB-Diff filter
    filter_result = filter_service.validate(file_bytes)
    if not filter_result.passed:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": filter_result.error,
            },
        )

    # Preprocess image to tensor
    try:
        tensor = preprocessing_service.preprocess(file_bytes)
    except ValueError as e:
        logger.warning("Image preprocessing failed: %s", str(e))
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Invalid image: {e}",
            },
        )

    # Run inference
    predictions = inference_service.predict(tensor)

    # Build response
    prediction_items = [
        PredictionItem(pathology=pathology, probability=float(prob))
        for pathology, prob in zip(settings.target_classes, predictions)
    ]

    return PredictionSuccessResponse(
        success=True,
        predictions=prediction_items,
    )
