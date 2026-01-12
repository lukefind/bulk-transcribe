#!/bin/bash
# Deploy script for bulk-transcribe
# Ensures container runs the latest code with proper env propagation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "=== Bulk Transcribe Deploy ==="
echo "Repo: $REPO_ROOT"
echo ""

# Get current git info
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Git commit: $GIT_COMMIT"
echo "Git branch: $GIT_BRANCH"
echo "Build time: $BUILD_TIME"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Copy .env.example to .env and configure HF_TOKEN for diarization."
    echo ""
fi

# Check HF_TOKEN in .env
if [ -f ".env" ]; then
    if grep -q "^HF_TOKEN=" .env && ! grep -q "^HF_TOKEN=$" .env && ! grep -q "^HF_TOKEN=your_" .env; then
        echo "OK: HF_TOKEN appears to be set in .env"
    else
        echo "WARNING: HF_TOKEN not set in .env - diarization will be disabled"
    fi
fi
echo ""

# Stop existing container
echo "Stopping existing container..."
docker compose down || true

# Build with version info
echo ""
echo "Building image with version info..."
docker compose build --no-cache \
    --build-arg BUILD_COMMIT="$GIT_COMMIT" \
    --build-arg BUILD_TIME="$BUILD_TIME"

# Start container
echo ""
echo "Starting container..."
docker compose up -d

# Wait for health check
echo ""
echo "Waiting for container to be healthy..."
for i in {1..30}; do
    if curl -sf http://localhost:8476/healthz > /dev/null 2>&1; then
        echo "Container is healthy!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Container failed to become healthy after 30 seconds"
        docker compose logs --tail=50
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo ""

# Verify deployment
echo ""
echo "=== Deployment Verification ==="

# Check container info
echo ""
echo "Container info:"
docker inspect bulk-transcribe --format 'Image={{.Image}}' | head -c 80
echo ""
docker inspect bulk-transcribe --format 'Created={{.Created}}'

# Check env vars in container
echo ""
echo "Environment check (inside container):"
docker exec bulk-transcribe sh -c 'echo "BUILD_COMMIT=$BUILD_COMMIT"; echo "BUILD_TIME=$BUILD_TIME"'
docker exec bulk-transcribe sh -c 'if [ -n "$HF_TOKEN" ]; then echo "HF_TOKEN=SET"; else echo "HF_TOKEN=NOT_SET"; fi'

# Check diarization vars
echo ""
echo "Diarization policy env vars:"
docker exec bulk-transcribe sh -c 'env | grep -E "^DIARIZATION_" | head -10 || echo "(using defaults)"'

# Check runtime API
echo ""
echo "Runtime API check:"
curl -s http://localhost:8476/api/runtime | jq '{
    diarizationAvailable: .diarizationAvailable,
    hfTokenPresent: .hfTokenPresent,
    pyannoteAvailable: .pyannoteAvailable,
    buildCommit: .buildCommit,
    buildTime: .buildTime,
    serverMaxDuration: .diarizationPolicy.serverMaxDurationSeconds
}'

echo ""
echo "=== Deploy Complete ==="
echo "Access UI at: http://localhost:8476"
