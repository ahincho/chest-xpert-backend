# ============================================================
# Chest-Xpert Backend — Production Multi-Stage Docker Build
# ============================================================
# Best practices applied (2026):
#   - Multi-stage build (builder → runtime) for minimal image size
#   - Python 3.14 slim-bookworm base for glibc compatibility
#   - Pinned UV version for reproducible builds
#   - UV_COMPILE_BYTECODE for faster startup in production
#   - UV_LINK_MODE=copy for cache-mount compatibility
#   - Layer ordering: deps → source → project (cache efficiency)
#   - Non-root user (chest-xpert) — principle of least privilege
#   - ONNX model embedded for single-command startup
#   - OCI metadata labels for traceability (docker inspect)
#   - HEALTHCHECK in exec form for orchestrator integration
#   - PYTHONFAULTHANDLER for debugging segfaults in C extensions
#   - No shell in CMD (exec form) — proper signal handling
#   - --no-editable install for production (no source dependency)
# ============================================================

# --- Stage 1: Build dependencies ---
FROM python:3.14-slim-bookworm AS builder

# Install UV — pinned to specific version for reproducibility
# See: https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

# Compile bytecode for faster cold starts in production
ENV UV_COMPILE_BYTECODE=1
# Use copy mode to avoid hard link issues across Docker layers
ENV UV_LINK_MODE=copy

WORKDIR /build

# Layer 1: Copy only dependency manifests (changes rarely → cached)
COPY pyproject.toml uv.lock ./

# Layer 2: Install deps without the project itself (intermediate layer optimization)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra otel --no-install-project

# Layer 3: Copy application source (changes frequently)
COPY app/ ./app/

# Layer 4: Install the project in non-editable mode (no source dependency in .venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra otel --no-editable


# --- Stage 2: Production runtime (minimal) ---
FROM python:3.14-slim-bookworm AS runtime

# OCI image metadata — enables `docker inspect` traceability
LABEL org.opencontainers.image.title="Chest-Xpert AI Inference API" \
      org.opencontainers.image.description="Multi-label chest pathology classifier — FastAPI + ONNX Runtime" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.authors="Angel Hincho Jove <ahincho@unsa.edu.pe>" \
      org.opencontainers.image.source="https://github.com/ahincho/chest-xpert-backend" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.base.name="python:3.14-slim-bookworm"

# Security: create dedicated non-root user and group
RUN groupadd -r chest-xpert \
    && useradd -r -g chest-xpert -d /app -s /sbin/nologin chest-xpert

WORKDIR /app

# Copy virtual environment from builder (compiled bytecode included)
COPY --from=builder --chown=chest-xpert:chest-xpert /build/.venv /app/.venv

# Copy application source
COPY --from=builder --chown=chest-xpert:chest-xpert /build/app ./app/

# Copy ONNX model into image — enables single `docker run` without volumes
COPY --chown=chest-xpert:chest-xpert models/ ./models/

# Copy .env.example as documentation reference
COPY --chown=chest-xpert:chest-xpert .env.example ./.env.example

# Runtime environment configuration
ENV PATH="/app/.venv/bin:$PATH" \
    # Performance and debugging
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    # Application defaults (overridable at runtime via -e)
    CHEST_XPERT_MODEL_PATH="models/chest-xpert-model.onnx" \
    CHEST_XPERT_SERVER_PORT="8000" \
    # OTel disabled by default — activate with -e OTEL_ENABLED=true
    OTEL_ENABLED="false"

# Switch to non-root user — all subsequent commands run as chest-xpert
USER chest-xpert

# Expose service port
EXPOSE 8000

# Health check — exec form (no shell) for proper signal handling
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

# Entrypoint: uvicorn in exec form for proper PID 1 signal handling
# Single worker — appropriate for CPU-bound ONNX inference
# Uses python -m to avoid shebang path issues from multi-stage builds
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
