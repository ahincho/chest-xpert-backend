"""Application factory and lifespan management.
Defines the FastAPI application with ONNX model lifecycle management,
CORS middleware, router registration, and error handlers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.errors import register_exception_handlers
from app.routers import health, predict
from app.services.filter import FilterService
from app.services.inference import InferenceService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: load model on startup, release on shutdown."""
    settings: Settings = app.state.settings

    # Startup: load ONNX model and create services
    try:
        logger.info("Loading ONNX model from %s", settings.model_path)
        app.state.inference_service = InferenceService(settings.model_path)
        logger.info("ONNX model loaded successfully")
    except FileNotFoundError as exc:
        logger.critical("Model file not found: %s", exc)
        raise SystemExit(1)
    except Exception as exc:
        logger.critical("Failed to load ONNX model: %s", exc)
        raise SystemExit(1)

    app.state.filter_service = FilterService(threshold=settings.rgb_diff_threshold)
    logger.info("FilterService initialized with threshold=%.1f", settings.rgb_diff_threshold)

    yield

    # Shutdown: release resources
    app.state.inference_service.close()
    logger.info("ONNX session released")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = Settings()

    app = FastAPI(
        title="Chest-Xpert AI Inference API",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Store settings on app state for lifespan access
    app.state.settings = settings

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(predict.router)

    # Error handlers
    register_exception_handlers(app)

    return app

app = create_app()
