#!/bin/bash
# Build the GPU Worker Docker image
# Usage: ./scripts/build_worker.sh [tag]

set -e

cd "$(dirname "$0")/.."

TAG="${1:-latest}"
IMAGE_NAME="bulk-transcribe-worker"

# Get git commit for version tagging
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Building GPU Worker image..."
echo "  Image: ${IMAGE_NAME}:${TAG}"
echo "  Commit: ${GIT_COMMIT}"
echo "  Time: ${BUILD_TIME}"
echo ""

docker build \
    -f worker/Dockerfile \
    --build-arg BUILD_COMMIT="${GIT_COMMIT}" \
    --build-arg BUILD_TIME="${BUILD_TIME}" \
    -t "${IMAGE_NAME}:${TAG}" \
    -t "${IMAGE_NAME}:${GIT_COMMIT}" \
    .

echo ""
echo "Build complete!"
echo ""
echo "To run locally:"
echo "  docker run -p 8477:8477 -e WORKER_TOKEN=your-token ${IMAGE_NAME}:${TAG}"
echo ""
echo "To push to registry:"
echo "  ./scripts/push_worker.sh ${TAG}"
