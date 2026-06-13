"""OpenTelemetry instrumentation for the Chest-Xpert backend.

Configures traces, metrics, and logs export via OTLP to an OpenTelemetry Collector.
The collector endpoint is configurable via OTEL_EXPORTER_OTLP_ENDPOINT env var.

When OTel packages are not installed (production without observability),
the setup is silently skipped and the app runs without instrumentation.
"""

import logging
import os

from fastapi import FastAPI

logger = logging.getLogger(__name__)

OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() in ("true", "1", "yes")


def setup_telemetry(app: FastAPI) -> None:
    """Configure OpenTelemetry instrumentation for the FastAPI app.

    Only activates if OTEL_ENABLED=true and OTel packages are installed.
    Silently no-ops otherwise, allowing the app to run without observability deps.
    """
    if not OTEL_ENABLED:
        logger.info("OpenTelemetry disabled (set OTEL_ENABLED=true to activate)")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "OpenTelemetry packages not installed. "
            "Install with: uv sync --extra otel"
        )
        return

    # Configure resource (service identity)
    resource = Resource.create(
        {
            "service.name": "chest-xpert-backend",
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("OTEL_ENVIRONMENT", "development"),
        }
    )

    # Configure trace provider with OTLP exporter
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # Instrument FastAPI (auto-generates spans for all HTTP requests)
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health",  # Don't trace health checks
    )

    logger.info(
        "OpenTelemetry initialized — exporting to %s",
        otlp_endpoint,
    )
