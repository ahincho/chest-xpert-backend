"""FastAPI dependency injection providers.
Provides Depends()-compatible callables for injecting configuration
and services into router endpoints.
"""

from functools import lru_cache

from fastapi import Request

from app.config import Settings
from app.services.filter import FilterService
from app.services.inference import InferenceService
from app.services.preprocessing import PreprocessingService


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.
    Uses @lru_cache so the Settings object is created once and reused
    across all requests, avoiding repeated environment variable parsing.
    """
    return Settings()


def get_inference_service(request: Request) -> InferenceService:
    """Retrieve the InferenceService from application state.
    The InferenceService is created during the app lifespan startup
    and stored on app.state for shared access across all requests.
    """
    return request.app.state.inference_service


def get_filter_service(request: Request) -> FilterService:
    """Retrieve the FilterService from application state.
    The FilterService is created during the app lifespan startup
    and stored on app.state for shared access across all requests.
    """
    return request.app.state.filter_service


def get_preprocessing_service() -> PreprocessingService:
    """Return a new PreprocessingService instance.
    PreprocessingService is stateless (uses only static methods),
    so a fresh instance is returned on each call.
    """
    return PreprocessingService()
