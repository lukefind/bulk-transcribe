#!/bin/bash
# Build the GPU Worker Docker image
# Usage: ./scripts/build_worker.sh [tag] [--platform linux/amd64]
#
# Examples:
#   ./scripts/build_worker.sh                    # Build :latest for local arch
#   ./scripts/build_worker.sh main-amd64         # Build :main-amd64 for local arch
#   ./scripts/build_worker.sh main-amd64 --amd64 # Build :main-amd64 for linux/amd64

set -e

cd "$(dirname "$0")/.."

TAG="${1:-latest}"
PLATFORM_FLAG=""

# Check for --amd64 flag
if [[ "$2" == "--amd64" ]] || [[ "$TAG" == *"-amd64" ]]; then
    PLATFORM_FLAG="--platform linux/amd64"
    echo "Building for linux/amd64 (cross-compile if on ARM)"
fi

IMAGE_NAME="bulk-transcribe-worker"

# Get git commit for version tagging
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo ""
echo "============================================"
echo "Building GPU Worker image"
echo "============================================"
echo "  Tag:      ${IMAGE_NAME}:${TAG}"
echo "  Commit:   ${GIT_COMMIT}"
echo "  Time:     ${BUILD_TIME}"
echo "  Platform: ${PLATFORM_FLAG:-native}"
echo ""

docker build \
    -f worker/Dockerfile \
    ${PLATFORM_FLAG} \
    --build-arg BUILD_COMMIT="${GIT_COMMIT}" \
    --build-arg BUILD_TIME="${BUILD_TIME}" \
    -t "${IMAGE_NAME}:${TAG}" \
    -t "${IMAGE_NAME}:${GIT_COMMIT}" \
    .

echo ""
echo "============================================"
echo "Build complete!"
echo "============================================"
echo ""
echo "Image tags created:"
echo "  ${IMAGE_NAME}:${TAG}"
echo "  ${IMAGE_NAME}:${GIT_COMMIT}"
echo ""
echo "To run locally:"
echo "  docker run -p 8477:8477 -e WORKER_TOKEN=your-token ${IMAGE_NAME}:${TAG}"
echo ""
echo "To push to registry:"
echo "  ./scripts/push_worker.sh ${TAG} ghcr.io/lukefind"
echo ""
echo "For RunPod deployment (main branch):"
echo "  ./scripts/build_worker.sh main-amd64"
echo "  ./scripts/push_worker.sh main-amd64 ghcr.io/lukefind"
