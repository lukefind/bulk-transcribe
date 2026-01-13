# GPU Worker Dockerfile for Bulk Transcribe
# Optimized for NVIDIA GPU with CUDA support

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY worker/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy worker code
COPY worker/ /app/worker/
COPY diarization.py /app/diarization.py
COPY diarization_policy.py /app/diarization_policy.py
COPY audio_utils.py /app/audio_utils.py
COPY logger.py /app/logger.py

# Create temp directory
RUN mkdir -p /tmp/bt-worker

# Environment defaults
ENV WORKER_PORT=8477
ENV WORKER_TMP_DIR=/tmp/bt-worker
ENV WORKER_MAX_FILE_MB=2000
ENV WORKER_MODEL=large-v3
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8477

# Run worker
CMD ["python3", "-m", "worker.app"]
