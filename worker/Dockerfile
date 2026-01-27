# GPU Worker Dockerfile for Bulk Transcribe
# Optimized for NVIDIA GPU with CUDA support
# Builds for linux/amd64 (CUDA) and linux/arm64 (CPU fallback for Mac)

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Build arguments for version tracking
# These are injected at build time by scripts/build_worker.sh
ARG BUILD_COMMIT=unknown
ARG BUILD_TIME=unknown
# IMAGE_DIGEST must be set at deploy/run time since it's only known after push
# Example: docker run -e IMAGE_DIGEST=sha256:abc123... 
# RunPod: Set in pod environment variables after pulling image

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Pip hardening
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and stable pip temp dir
WORKDIR /app
RUN mkdir -p /var/tmp/pip && chmod 777 /var/tmp/pip
ENV TMPDIR=/var/tmp/pip

# Install Python dependencies
COPY worker/requirements.txt /app/requirements.txt

# Step 1: Install numpy FIRST to avoid "Numpy is not available" errors
# This ensures numpy is present before torch tries to use it during install
RUN pip3 install --upgrade pip && \
    pip3 install numpy==1.26.4

# Step 2: Install torch/torchaudio with CUDA 12.1 wheels
# These must be installed from PyTorch index to get CUDA-enabled builds
RUN pip3 install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.2 torchaudio==2.2.2 || \
    pip3 install torch==2.2.2 torchaudio==2.2.2

# Step 3: Install remaining dependencies
# numpy is already installed, pip will skip it
RUN pip3 install -r requirements.txt

# Step 4: Verify critical imports work at build time
# This catches dependency issues early rather than at runtime
RUN python3 -c "import numpy; print(f'numpy {numpy.__version__}')" && \
    python3 -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')" && \
    python3 -c "import whisper; print('whisper OK')"

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
ENV BUILD_COMMIT=${BUILD_COMMIT}
ENV BUILD_TIME=${BUILD_TIME}

# Expose port
EXPOSE 8477

# Run worker
CMD ["python3", "-m", "worker.app"]
