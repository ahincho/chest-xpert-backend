# ============================================================
# Chest-Xpert Backend — Multi-stage Docker Build
# ============================================================
# Uses UV for fast, reproducible dependency installation.
# Python 3.14 slim image for minimal footprint.
# ============================================================

# --- Stage 1: Build dependencies ---
FROM python:3.14-slim AS builder

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY app/ ./app/

# Install the project itself
RUN uv sync --frozen --no-dev

# --- Stage 2: Production runtime ---
FROM python:3.14-slim AS runtime

# Security: run as non-root user
RUN groupadd -r chest-xpert && useradd -r -g chest-xpert -d /app -s /sbin/nologin chest-xpert

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY --from=builder /app/app ./app/

# Copy .env.example as reference (actual .env should be mounted or env vars set)
COPY .env.example ./.env.example

# Create models directory (model files should be mounted as volume)
RUN mkdir -p /app/models && chown -R chest-xpert:chest-xpert /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER chest-xpert

# Expose port (configurable via CHESTXPERT_SERVER_PORT)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
