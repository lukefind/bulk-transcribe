#!/bin/bash
# Push the GPU Worker Docker image to a registry
# Usage: ./scripts/push_worker.sh [tag] [registry]
#
# Examples:
#   ./scripts/push_worker.sh latest dockerhub-username
#   ./scripts/push_worker.sh v1.0.0 ghcr.io/username

set -e

cd "$(dirname "$0")/.."

TAG="${1:-latest}"
REGISTRY="${2:-}"

IMAGE_NAME="bulk-transcribe-worker"
LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"

if [ -z "$REGISTRY" ]; then
    echo "Usage: ./scripts/push_worker.sh [tag] [registry]"
    echo ""
    echo "Examples:"
    echo "  ./scripts/push_worker.sh latest your-dockerhub-username"
    echo "  ./scripts/push_worker.sh v1.0.0 ghcr.io/your-username"
    echo ""
    echo "Make sure to:"
    echo "  1. Build the image first: ./scripts/build_worker.sh ${TAG}"
    echo "  2. Login to your registry: docker login [registry]"
    exit 1
fi

REMOTE_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Pushing GPU Worker image..."
echo "  Local:  ${LOCAL_IMAGE}"
echo "  Remote: ${REMOTE_IMAGE}"
echo ""

# Tag for remote registry
docker tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"

# Push
docker push "${REMOTE_IMAGE}"

# Also push with git commit tag if available
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "")
if [ -n "$GIT_COMMIT" ]; then
    COMMIT_IMAGE="${REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT}"
    docker tag "${LOCAL_IMAGE}" "${COMMIT_IMAGE}"
    docker push "${COMMIT_IMAGE}"
    echo ""
    echo "Also pushed: ${COMMIT_IMAGE}"
fi

echo ""
echo "Push complete!"
echo ""
echo "To use this image on RunPod/Lambda/Vast.ai:"
echo "  Image: ${REMOTE_IMAGE}"
echo ""
echo "Required environment variables:"
echo "  WORKER_TOKEN=your-shared-secret"
echo "  HF_TOKEN=your-huggingface-token (for diarization)"
