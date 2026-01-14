# Multi-stage Dockerfile for Bookmark Validation and Enhancement Tool
# Optimized for small image size and efficient AI model caching

# ============================================================================
# Stage 1: Builder - Install dependencies and compile packages
# ============================================================================
FROM python:3.12-slim as builder

# Set build arguments for optimization
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ============================================================================
FROM python:3.12-slim

# Set metadata labels
LABEL maintainer="Troy Davis"
LABEL description="Bookmark Validation and Enhancement Tool for raindrop.io"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model cache directory (mounted as volume)
    TRANSFORMERS_CACHE=/app/cache/models \
    # Checkpoint directory (mounted as volume)
    BOOKMARK_CHECKPOINT_DIR=/app/checkpoints \
    # Default working directory for data
    DATA_DIR=/app/data \
    # Set PATH for virtual environment
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for some Python packages
    libgomp1 \
    # Useful utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash bookmarkuser && \
    mkdir -p /app/data /app/cache/models /app/checkpoints /app/logs && \
    chown -R bookmarkuser:bookmarkuser /app

# Copy virtual environment from builder
COPY --from=builder --chown=bookmarkuser:bookmarkuser /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=bookmarkuser:bookmarkuser bookmark_processor/ ./bookmark_processor/
COPY --chown=bookmarkuser:bookmarkuser pyproject.toml setup.py ./
COPY --chown=bookmarkuser:bookmarkuser README.md LICENSE ./

# Install the package in editable mode
USER bookmarkuser
RUN pip install -e .

# Create volume mount points
VOLUME ["/app/data", "/app/cache/models", "/app/checkpoints", "/app/logs"]

# Set working directory to data directory for convenience
WORKDIR /app/data

# Health check - verify the CLI is accessible
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m bookmark_processor --help || exit 1

# Default command - show help if no arguments provided
ENTRYPOINT ["python", "-m", "bookmark_processor"]
CMD ["--help"]
