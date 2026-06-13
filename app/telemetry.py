"""OpenTelemetry instrumentation for the Chest-Xpert backend.

Configures the three pillars of observability:
- Traces → Tempo (via OTLP gRPC)
- Logs → Loki (via OTLP gRPC)
- Metrics → Mimir/Prometheus (via OTLP gRPC)

Plus custom business metrics for inference monitoring.

When OTel packages are not installed or OTEL_ENABLED != true,
the setup is silently skipped and the app runs without instrumentation.
"""

import logging
import os

from fastapi import FastAPI

logger = logging.getLogger(__name__)

OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() in ("true", "1", "yes")

# Custom metrics instruments (initialized when OTel is enabled)
predictions_counter = None
inference_duration_histogram = None


def setup_telemetry(app: FastAPI) -> None:
    """Configure OpenTelemetry instrumentation for the FastAPI app.

    Only activates if OTEL_ENABLED=true and OTel packages are installed.
    Silently no-ops otherwise, allowing the app to run without observability deps.
    """
    global predictions_counter, inference_duration_histogram  # noqa: PLW0603

    if not OTEL_ENABLED:
        logger.info("OpenTelemetry disabled (set OTEL_ENABLED=true to activate)")
        return

    try:
        from opentelemetry import metrics, trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OpenTelemetry packages not installed. Install with: uv sync --extra otel")
        return

    # Configure resource (service identity)
    resource = Resource.create(
        {
            "service.name": "chest-xpert-backend",
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("OTEL_ENVIRONMENT", "development"),
        }
    )

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    # =========================================================================
    # TRACES → Tempo
    # =========================================================================
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # =========================================================================
    # LOGS → Loki
    # =========================================================================
    logger_provider = LoggerProvider(resource=resource)
    log_exporter = OTLPLogExporter(endpoint=otlp_endpoint, insecure=True)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(logger_provider)

    # Attach OTel handler to Python's root logger
    otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    logging.getLogger().addHandler(otel_handler)

    # =========================================================================
    # METRICS → Mimir/Prometheus
    # =========================================================================
    metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=15000,  # Export every 15 seconds
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Custom business metrics
    meter = metrics.get_meter("chest-xpert-backend", "1.0.0")

    predictions_counter = meter.create_counter(
        name="chest_xpert.predictions_total",
        description="Total number of predictions made",
        unit="1",
    )

    inference_duration_histogram = meter.create_histogram(
        name="chest_xpert.inference_duration_seconds",
        description="Duration of ONNX model inference (seconds)",
        unit="s",
    )

    # =========================================================================
    # Instrument FastAPI (auto-generates traces + metrics for HTTP requests)
    # =========================================================================
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health",
        meter_provider=meter_provider,
    )

    logger.info(
        "OpenTelemetry initialized (traces + logs + metrics) — exporting to %s",
        otlp_endpoint,
    )


def record_prediction(pathology_count: int = 5) -> None:
    """Record a successful prediction in custom metrics."""
    if predictions_counter is not None:
        predictions_counter.add(1, {"pathology_count": pathology_count})


def record_inference_duration(duration_seconds: float) -> None:
    """Record inference duration in the histogram."""
    if inference_duration_histogram is not None:
        inference_duration_histogram.record(duration_seconds)
