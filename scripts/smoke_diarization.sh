#!/bin/bash
# Diarization smoke test script
# Usage: ./scripts/smoke_diarization.sh <audio_file_path>
# Example: ./scripts/smoke_diarization.sh ./samples/test.wav

set -e

# Configuration
BASE_URL="${BASE_URL:-http://localhost:8476}"
AUDIO_FILE="$1"

# Check if audio file is provided
if [ -z "$AUDIO_FILE" ]; then
    echo "ERROR: Audio file path required"
    echo "Usage: $0 <audio_file_path>"
    echo "Example: $0 ./samples/test.wav"
    exit 2
fi

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "ERROR: Audio file not found: $AUDIO_FILE"
    exit 2
fi

echo "=== Diarization Smoke Test ==="
echo "Base URL: $BASE_URL"
echo "Audio file: $AUDIO_FILE"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
fail() {
    echo -e "${RED}FAIL: $1${NC}"
    exit 1
}

pass() {
    echo -e "${GREEN}PASS: $1${NC}"
}

warn() {
    echo -e "${YELLOW}WARN: $1${NC}"
}

info() {
    echo "INFO: $1"
}

# Step 1: Check /api/runtime diarization readiness
info "Checking /api/runtime diarization readiness..."
RUNTIME_RESP=$(curl -s "$BASE_URL/api/runtime")

# Check required fields
echo "$RUNTIME_RESP" | jq -e '.hfTokenPresent' > /dev/null || fail "Missing hfTokenPresent field"
echo "$RUNTIME_RESP" | jq -e '.hfAccessOk' > /dev/null || fail "Missing hfAccessOk field"
echo "$RUNTIME_RESP" | jq -e '.diarizationAvailable' > /dev/null || fail "Missing diarizationAvailable field"

# Check values
HF_TOKEN_PRESENT=$(echo "$RUNTIME_RESP" | jq -r '.hfTokenPresent')
HF_ACCESS_OK=$(echo "$RUNTIME_RESP" | jq -r '.hfAccessOk')
DIARIZATION_AVAILABLE=$(echo "$RUNTIME_RESP" | jq -r '.diarizationAvailable')

if [ "$HF_TOKEN_PRESENT" != "true" ]; then
    fail "HF_TOKEN not present"
fi

if [ "$HF_ACCESS_OK" != "true" ]; then
    MISSING_REPOS=$(echo "$RUNTIME_RESP" | jq -r '.hfAccessMissingRepos[]?')
    fail "HF access not OK. Missing repos: $MISSING_REPOS"
fi

if [ "$DIARIZATION_AVAILABLE" != "true" ]; then
    REASON=$(echo "$RUNTIME_RESP" | jq -r '.diarizationReason // "Unknown reason"')
    fail "Diarization not available: $REASON"
fi

pass "Diarization is ready"
echo

# Step 2: Get session and CSRF token
info "Getting session and CSRF token..."
curl -s -c /tmp/cookies.txt "$BASE_URL/" > /dev/null
CSRF=$(curl -s -b /tmp/cookies.txt "$BASE_URL/api/session" | jq -r '.csrfToken')

if [ -z "$CSRF" ] || [ "$CSRF" = "null" ]; then
    fail "Failed to get CSRF token"
fi

pass "Session established"
echo

# Step 3: Upload audio file
info "Uploading audio file..."
UPLOAD_RESP=$(curl -s -b /tmp/cookies.txt -X POST \
    -H "x-csrf-token: $CSRF" \
    -F "files=@$AUDIO_FILE" \
    "$BASE_URL/api/uploads")

# Extract file ID
FILE_ID=$(echo "$UPLOAD_RESP" | jq -r '.uploads[0].id // empty')

if [ -z "$FILE_ID" ]; then
    echo "Upload response:"
    echo "$UPLOAD_RESP" | jq .
    fail "Failed to upload file"
fi

pass "File uploaded (ID: $FILE_ID)"
echo

# Step 4: Create job with diarization
info "Creating diarization job..."
JOB_RESP=$(curl -s -b /tmp/cookies.txt -X POST \
    -H "Content-Type: application/json" \
    -H "x-csrf-token: $CSRF" \
    -d "{
        \"uploadIds\": [\"$FILE_ID\"],
        \"options\": {
            \"model\": \"tiny\",
            \"language\": \"en\",
            \"backend\": \"cpu\",
            \"diarizationEnabled\": true
        }
    }" \
    "$BASE_URL/api/jobs")

# Extract job ID
JOB_ID=$(echo "$JOB_RESP" | jq -r '.jobId // empty')

if [ -z "$JOB_ID" ]; then
    echo "Job creation response:"
    echo "$JOB_RESP" | jq .
    fail "Failed to create job"
fi

pass "Job created (ID: $JOB_ID)"
echo

# Step 5: Monitor job progress
info "Monitoring job progress..."
MAX_WAIT=300  # 5 minutes
WAIT_INTERVAL=5
elapsed=0

while [ $elapsed -lt $MAX_WAIT ]; do
    STATUS_RESP=$(curl -s -b /tmp/cookies.txt "$BASE_URL/api/jobs/$JOB_ID")
    STATUS=$(echo "$STATUS_RESP" | jq -r '.status')
    PROGRESS=$(echo "$STATUS_RESP" | jq -r '.progress.currentFile // "Unknown"')
    
    echo "  Status: $STATUS | Progress: $PROGRESS"
    
    if [ "$STATUS" = "complete" ] || [ "$STATUS" = "complete_with_errors" ]; then
        break
    fi
    
    if [ "$STATUS" = "failed" ] || [ "$STATUS" = "canceled" ]; then
        ERROR=$(echo "$STATUS_RESP" | jq -r '.error.message // "Unknown error"')
        fail "Job $STATUS: $ERROR"
    fi
    
    sleep $WAIT_INTERVAL
    elapsed=$((elapsed + WAIT_INTERVAL))
done

if [ $elapsed -ge $MAX_WAIT ]; then
    fail "Job timed out after $MAX_WAIT seconds"
fi

pass "Job completed"
echo

# Step 6: Verify outputs
info "Verifying outputs..."
OUTPUTS_RESP=$(curl -s -b /tmp/cookies.txt "$BASE_URL/api/jobs/$JOB_ID")

# Check for required output types
HAS_SPEAKER_MD=$(echo "$OUTPUTS_RESP" | jq -e '.outputs[] | select(.type == "speaker-markdown")' > /dev/null && echo "true" || echo "false")
HAS_DIARIZATION_JSON=$(echo "$OUTPUTS_RESP" | jq -e '.outputs[] | select(.type == "diarization-json")' > /dev/null && echo "true" || echo "false")
HAS_RTTM=$(echo "$OUTPUTS_RESP" | jq -e '.outputs[] | select(.type == "rttm")' > /dev/null && echo "true" || echo "false")

if [ "$HAS_SPEAKER_MD" != "true" ]; then
    fail "Missing speaker-markdown output"
fi

if [ "$HAS_DIARIZATION_JSON" != "true" ]; then
    fail "Missing diarization-json output"
fi

if [ "$HAS_RTTM" != "true" ]; then
    fail "Missing RTTM output"
fi

pass "All required outputs present"
echo

# Step 7: Check for errors
ERROR_COUNT=$(echo "$OUTPUTS_RESP" | jq '[.outputs[] | select(.error)] | length')
if [ "$ERROR_COUNT" -gt 0 ]; then
    warn "Job completed with $ERROR_COUNT errors"
    echo "$OUTPUTS_RESP" | jq -r '.outputs[] | select(.error) | "  - \(.filename): \(.error.code) - \(.error.message)"'
else
    pass "No errors in outputs"
fi

echo
echo "=== Diarization Smoke Test PASSED ==="
echo "All checks passed successfully!"
