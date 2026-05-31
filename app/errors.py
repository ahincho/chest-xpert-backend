"""Exception handlers for the FastAPI application.
Registers handlers that ensure all error responses conform to the
unified ErrorResponse schema, with appropriate logging levels and
no information leakage (stack traces, file paths, class names).
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class FileTooLargeError(Exception):
    """Raised when an uploaded file exceeds the maximum allowed size (10MB)."""
    pass

def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the FastAPI application."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors → 422 with ErrorResponse schema."""
        detail_str = str(exc.errors())[:1000]
        logger.warning("Validation error on %s %s: %s", request.method, request.url.path, detail_str)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Request validation failed",
                "detail": detail_str,
            },
        )

    @app.exception_handler(FileTooLargeError)
    async def file_too_large_handler(
        request: Request, exc: FileTooLargeError
    ) -> JSONResponse:
        """Handle file size exceeding 10MB → 413."""
        logger.warning("File too large on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "success": False,
                "error": "File exceeds maximum size of 10MB",
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle all unhandled exceptions → 500 with generic message.
        Logs the full exception with traceback at ERROR level but never
        exposes stack traces, file paths, or internal class names in the response.
        """
        logger.error(
            "Unexpected error on %s %s: %s",
            request.method,
            request.url.path,
            str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal processing error",
            },
        )
