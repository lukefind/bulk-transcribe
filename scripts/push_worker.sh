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
echo "IMAGE IDENTITY"
echo "============================================"
echo "  Tag:         ${REMOTE_IMAGE}"
echo "  Git Commit:  ${GIT_COMMIT:-unknown}"
if [ -n "$IMAGE_DIGEST" ]; then
    echo "  Digest:      ${IMAGE_DIGEST}"
    DIGEST_PINNED_REF="${REGISTRY}/${IMAGE_NAME}@${IMAGE_DIGEST}"
else
    echo "  Digest:      (run 'docker manifest inspect ${REMOTE_IMAGE}' to get digest)"
    DIGEST_PINNED_REF="(digest not available)"
fi
echo ""
echo "============================================"
echo "ROLLBACK REFERENCE (digest-pinned)"
echo "============================================"
if [ -n "$IMAGE_DIGEST" ]; then
    echo "  ${DIGEST_PINNED_REF}"
    echo ""
    echo "  To rollback, set RunPod image to this digest-pinned reference."
else
    echo "  (digest not available - run 'docker manifest inspect ${REMOTE_IMAGE}')"
fi
echo ""
echo "============================================"
echo "RUNPOD ENVIRONMENT VARIABLES (copy-paste)"
echo "============================================"
echo ""
echo "# Required"
echo "WORKER_TOKEN=your-shared-secret"
echo ""
echo "# Identity tracking (for controller mismatch detection)"
if [ -n "$GIT_COMMIT" ]; then
    echo "EXPECTED_WORKER_GIT_COMMIT=${GIT_COMMIT}"
fi
if [ -n "$IMAGE_DIGEST" ]; then
    echo "EXPECTED_WORKER_IMAGE_DIGEST=${IMAGE_DIGEST}"
    echo "IMAGE_DIGEST=${IMAGE_DIGEST}"
else
    echo "# EXPECTED_WORKER_IMAGE_DIGEST=sha256:... (get from manifest inspect)"
    echo "# IMAGE_DIGEST=sha256:... (optional, for worker to echo back)"
fi
echo ""
echo "# Optional"
echo "HF_TOKEN=your-huggingface-token"
echo ""
echo "============================================"
echo "CONTROLLER ENVIRONMENT (for mismatch detection)"
echo "============================================"
echo ""
echo "# Add to controller .env to enable identity mismatch warnings:"
if [ -n "$GIT_COMMIT" ]; then
    echo "EXPECTED_WORKER_GIT_COMMIT=${GIT_COMMIT}"
fi
if [ -n "$IMAGE_DIGEST" ]; then
    echo "EXPECTED_WORKER_IMAGE_DIGEST=${IMAGE_DIGEST}"
fi
echo ""
echo "============================================"
echo "VERIFICATION (after RunPod redeploy)"
echo "============================================"
echo ""
echo "curl \$CONTROLLER_URL/api/runtime | jq '.remoteWorker | {identity, expectedIdentity, identityMatches, identityMismatchReason}'"
echo ""
echo "Expected when matching:"
echo "  {"
echo "    \"identity\": {"
echo "      \"gitCommit\": \"${GIT_COMMIT:-<commit>}\","
echo "      \"buildTime\": \"<build-time>\","
echo "      \"declaredImageDigest\": \"${IMAGE_DIGEST:-null}\","
echo "      \"imageDigestSource\": \"env\""
echo "    },"
echo "    \"expectedIdentity\": {"
echo "      \"gitCommit\": \"${GIT_COMMIT:-<commit>}\","
echo "      \"imageDigest\": \"${IMAGE_DIGEST:-null}\""
echo "    },"
echo "    \"identityMatches\": true,"
echo "    \"identityMismatchReason\": null"
echo "  }"
