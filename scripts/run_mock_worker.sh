#!/bin/bash
# Run a mock GPU worker for testing remote worker integration
# This starts the worker service locally without GPU requirements

set -e

cd "$(dirname "$0")/.."

# Set test environment
export WORKER_TOKEN="${WORKER_TOKEN:-test-token-12345}"
export WORKER_PORT="${WORKER_PORT:-8477}"
export WORKER_TMP_DIR="${WORKER_TMP_DIR:-/tmp/bt-worker-test}"
export WORKER_MODEL="${WORKER_MODEL:-base}"  # Use small model for testing

# Create temp directory
mkdir -p "$WORKER_TMP_DIR"

echo "Starting mock GPU worker..."
echo "  Port: $WORKER_PORT"
echo "  Token: ${WORKER_TOKEN:0:10}..."
echo "  Temp dir: $WORKER_TMP_DIR"
echo ""
echo "To test, set these on your controller:"
echo "  REMOTE_WORKER_URL=http://localhost:$WORKER_PORT"
echo "  REMOTE_WORKER_TOKEN=$WORKER_TOKEN"
echo "  REMOTE_WORKER_MODE=optional"
echo ""

# Run worker
python -m worker.app
