# Bulk Transcribe - Docker Image (CPU or GPU)
# CPU: docker compose up -d --build
# GPU: docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

FROM python:3.11-slim-bookworm

LABEL maintainer="Bulk Transcribe"
LABEL description="Audio transcription using OpenAI Whisper"

# Build argument to control CPU vs GPU dependencies
ARG DEVICE=cpu

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user (non-root)
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-server.txt* constraints.txt* ./

# Install Python dependencies based on DEVICE build arg
# CPU: use PyTorch CPU-only wheels (no CUDA dependencies)
# GPU: use default PyTorch wheels with CUDA support
RUN if [ "$DEVICE" = "cpu" ]; then \
        echo "Installing CPU-only torch/torchaudio..." && \
        pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cpu \
            torch==2.1.2 torchaudio==2.1.2 && \
        echo "Installing remaining requirements with constraints..." && \
        pip install --no-cache-dir -r requirements-server.txt -c constraints.txt; \
    else \
        echo "Installing GPU/CUDA torch/torchaudio..." && \
        pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cu121 \
            torch==2.1.2 torchaudio==2.1.2 && \
        echo "Installing remaining requirements with constraints..." && \
        pip install --no-cache-dir -r requirements-server.txt -c constraints.txt; \
    fi

# Guardrail: verify no CUDA packages on CPU build
RUN if [ "$DEVICE" = "cpu" ]; then \
        echo "Checking for unwanted CUDA packages..." && \
        if pip freeze | grep -qE '^nvidia-'; then \
            echo "ERROR: CUDA packages found in CPU build!" && \
            pip freeze | grep -E '^nvidia-' && \
            exit 1; \
        else \
            echo "OK: No CUDA packages installed"; \
        fi; \
    fi

# Copy application source (relies on .dockerignore to exclude junk)
COPY . .

# Ensure Python can find local modules
ENV PYTHONPATH=/app

# Create data directories
RUN mkdir -p /data/input /data/output && \
    chown -R appuser:appuser /data /app

# Switch to non-root user
USER appuser

# Environment defaults
ENV PORT=8476
ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output
ENV DEVICE=cpu

# Expose port
EXPOSE 8476

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8476/healthz || exit 1

# Run with Gunicorn
# - workers: 2 (reasonable for CPU transcription)
# - threads: 4 (handle concurrent requests)
# - timeout: 0 (disable timeout - transcription jobs can be long)
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "0", "--bind", "0.0.0.0:8476", "app:app"]
