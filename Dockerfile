# Bulk Transcribe - CPU Docker Image
# For GPU support, use docker-compose.gpu.yml override

FROM python:3.11-slim-bookworm

LABEL maintainer="Bulk Transcribe"
LABEL description="Audio transcription using OpenAI Whisper"

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
COPY requirements.txt requirements-server.txt* ./

# Install Python dependencies
# Use requirements-server.txt if it exists (excludes pywebview/pyinstaller)
RUN pip install --no-cache-dir gunicorn && \
    if [ -f requirements-server.txt ]; then \
        pip install --no-cache-dir -r requirements-server.txt; \
    else \
        pip install --no-cache-dir flask openai-whisper; \
    fi

# Copy application code
COPY app.py transcribe_options.py ./
COPY templates/ templates/
COPY static/ static/

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
