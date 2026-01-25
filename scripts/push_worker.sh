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

# Get the image digest (sha256) for provable identity
IMAGE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "${REMOTE_IMAGE}" 2>/dev/null | sed 's/.*@//')
if [ -z "$IMAGE_DIGEST" ]; then
    # Fallback: try to get digest from push output or manifest
    IMAGE_DIGEST=$(docker manifest inspect "${REMOTE_IMAGE}" 2>/dev/null | grep -o '"digest": "sha256:[^"]*"' | head -1 | sed 's/"digest": "//;s/"//')
fi

echo ""
echo "Push complete!"
echo ""
echo "============================================"
echo "IMAGE IDENTITY (for provable verification)"
echo "============================================"
echo "  Image:       ${REMOTE_IMAGE}"
echo "  Git Commit:  ${GIT_COMMIT:-unknown}"
if [ -n "$IMAGE_DIGEST" ]; then
    echo "  Digest:      ${IMAGE_DIGEST}"
else
    echo "  Digest:      (run 'docker manifest inspect ${REMOTE_IMAGE}' to get digest)"
fi
echo ""
echo "============================================"
echo "RUNPOD ENVIRONMENT VARIABLES"
echo "============================================"
echo "Required:"
echo "  WORKER_TOKEN=your-shared-secret"
echo ""
echo "For provable identity (IMPORTANT):"
if [ -n "$IMAGE_DIGEST" ]; then
    echo "  IMAGE_DIGEST=${IMAGE_DIGEST}"
else
    echo "  IMAGE_DIGEST=sha256:... (get from 'docker manifest inspect ${REMOTE_IMAGE}')"
fi
echo ""
echo "Optional:"
echo "  HF_TOKEN=your-huggingface-token (for diarization)"
echo ""
echo "The IMAGE_DIGEST is the only immutable identifier."
echo "Set it in RunPod pod environment to enable provable worker identity."
