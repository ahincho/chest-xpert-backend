"""Pydantic schemas for prediction request/response models."""

from pydantic import BaseModel, Field

class PredictionItem(BaseModel):
    """Single pathology prediction result."""
    pathology: str = Field(..., description="Pathology name")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability [0,1]")

class PredictionSuccessResponse(BaseModel):
    """Successful prediction response with 5 pathology probabilities."""
    success: bool = Field(default=True)
    predictions: list[PredictionItem] = Field(..., min_length=5, max_length=5)

class ErrorResponse(BaseModel):
    """Unified error response schema for all 4xx/5xx responses."""
    success: bool = Field(default=False)
    error: str = Field(..., max_length=500)
    detail: str | None = Field(default=None, max_length=1000)
