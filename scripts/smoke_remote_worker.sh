#!/bin/bash
# Smoke test for Remote GPU Worker
# Tests end-to-end job execution against a real worker
#
# Usage: ./scripts/smoke_remote_worker.sh
#
# Required environment variables:
#   REMOTE_WORKER_URL - URL of the GPU worker (e.g., https://worker.example.com)
#   REMOTE_WORKER_TOKEN - Shared authentication token
#
# Optional:
#   CONTROLLER_URL - Controller URL (default: http://localhost:8476)
#   TEST_AUDIO_FILE - Path to test audio file (default: test.wav in repo)

set -e

cd "$(dirname "$0")/.."

# Source .env if it exists (for WORKER_URL, WORKER_TOKEN, etc.)
if [ -f .env ]; then
    # Export variables from .env, handling quotes
    set -a
    source .env
    set +a
fi

# Map WORKER_* to REMOTE_WORKER_* if not already set
: "${REMOTE_WORKER_URL:=$WORKER_URL}"
: "${REMOTE_WORKER_TOKEN:=$WORKER_TOKEN}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Remote GPU Worker Smoke Test"
echo "=========================================="
echo ""

# Check required environment variables
if [ -z "$REMOTE_WORKER_URL" ]; then
    echo -e "${RED}ERROR: REMOTE_WORKER_URL not set${NC}"
    echo "Set it to your GPU worker URL, e.g.:"
    echo "  export REMOTE_WORKER_URL=https://your-worker.example.com"
    exit 1
fi

if [ -z "$REMOTE_WORKER_TOKEN" ]; then
    echo -e "${RED}ERROR: REMOTE_WORKER_TOKEN not set${NC}"
    echo "Set it to your shared worker token, e.g.:"
    echo "  export REMOTE_WORKER_TOKEN=your-secret-token"
    exit 1
fi

CONTROLLER_URL="${CONTROLLER_URL:-http://localhost:8476}"
TEST_AUDIO="${TEST_AUDIO_FILE:-test.wav}"

echo "Configuration:"
echo "  Worker URL:     $REMOTE_WORKER_URL"
echo "  Controller URL: $CONTROLLER_URL"
echo "  Test audio:     $TEST_AUDIO"
echo ""

# Step 1: Check worker health
echo -e "${YELLOW}Step 1: Checking worker health...${NC}"
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "$REMOTE_WORKER_URL/health" 2>/dev/null || echo -e "\n000")
HEALTH_CODE=$(echo "$HEALTH_RESPONSE" | tail -1)
HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | sed '$d')

if [ "$HEALTH_CODE" != "200" ]; then
    echo -e "${RED}FAILED: Worker health check failed (HTTP $HEALTH_CODE)${NC}"
    echo "Response: $HEALTH_BODY"
    exit 1
fi
echo -e "${GREEN}OK: Worker is healthy${NC}"
echo "  $HEALTH_BODY"
echo ""

# Step 2: Check worker capabilities via /v1/ping
echo -e "${YELLOW}Step 2: Checking worker capabilities...${NC}"
PING_RESPONSE=$(curl -s -w "\n%{http_code}" -H "Authorization: Bearer $REMOTE_WORKER_TOKEN" "$REMOTE_WORKER_URL/v1/ping" 2>/dev/null || echo -e "\n000")
PING_CODE=$(echo "$PING_RESPONSE" | tail -1)
PING_BODY=$(echo "$PING_RESPONSE" | sed '$d')

if [ "$PING_CODE" != "200" ]; then
    echo -e "${YELLOW}WARNING: /v1/ping not available (HTTP $PING_CODE), using basic health${NC}"
else
    echo -e "${GREEN}OK: Worker capabilities retrieved${NC}"
    echo "  $PING_BODY"
fi
echo ""

# Step 3: Check controller is running
echo -e "${YELLOW}Step 3: Checking controller...${NC}"
CTRL_RESPONSE=$(curl -s -w "\n%{http_code}" "$CONTROLLER_URL/api/runtime" 2>/dev/null || echo -e "\n000")
CTRL_CODE=$(echo "$CTRL_RESPONSE" | tail -1)

if [ "$CTRL_CODE" != "200" ]; then
    echo -e "${RED}FAILED: Controller not reachable at $CONTROLLER_URL (HTTP $CTRL_CODE)${NC}"
    echo "Make sure the controller is running:"
    echo "  python app.py"
    exit 1
fi
echo -e "${GREEN}OK: Controller is running${NC}"
echo ""

# Step 4: Get CSRF token
echo -e "${YELLOW}Step 4: Getting session and CSRF token...${NC}"
# First hit any endpoint to establish session cookie
curl -s -c cookies.tmp -b cookies.tmp "$CONTROLLER_URL/api/runtime" > /dev/null 2>&1
# Then get CSRF token from /api/session
CSRF_RESPONSE=$(curl -s -c cookies.tmp -b cookies.tmp "$CONTROLLER_URL/api/session" 2>/dev/null)
CSRF_TOKEN=$(echo "$CSRF_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('csrfToken',''))" 2>/dev/null || echo "")

if [ -z "$CSRF_TOKEN" ]; then
    echo -e "${RED}FAILED: Could not get CSRF token${NC}"
    echo "Response: $CSRF_RESPONSE"
    exit 1
else
    echo -e "${GREEN}OK: Got CSRF token${NC}"
fi
echo ""

# Step 5: Upload test audio file
echo -e "${YELLOW}Step 5: Uploading test audio file...${NC}"
if [ ! -f "$TEST_AUDIO" ]; then
    echo -e "${RED}FAILED: Test audio file not found: $TEST_AUDIO${NC}"
    exit 1
fi

UPLOAD_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -b cookies.tmp \
    -H "X-CSRF-Token: $CSRF_TOKEN" \
    -F "files=@$TEST_AUDIO" \
    "$CONTROLLER_URL/api/uploads" 2>/dev/null)
UPLOAD_CODE=$(echo "$UPLOAD_RESPONSE" | tail -1)
UPLOAD_BODY=$(echo "$UPLOAD_RESPONSE" | sed '$d')

if [ "$UPLOAD_CODE" != "200" ]; then
    echo -e "${RED}FAILED: Upload failed (HTTP $UPLOAD_CODE)${NC}"
    echo "Response: $UPLOAD_BODY"
    rm -f cookies.tmp
    exit 1
fi

UPLOAD_ID=$(echo "$UPLOAD_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uploads',[])[0].get('id',''))" 2>/dev/null || echo "")
if [ -z "$UPLOAD_ID" ]; then
    echo -e "${RED}FAILED: Could not get upload ID${NC}"
    echo "Response: $UPLOAD_BODY"
    rm -f cookies.tmp
    exit 1
fi
echo -e "${GREEN}OK: File uploaded (ID: $UPLOAD_ID)${NC}"
echo ""

# Step 6: Create job with remote worker
echo -e "${YELLOW}Step 6: Creating job with useRemoteWorker=true...${NC}"
JOB_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -b cookies.tmp \
    -H "Content-Type: application/json" \
    -H "X-CSRF-Token: $CSRF_TOKEN" \
    -d "{\"uploadIds\":[\"$UPLOAD_ID\"],\"options\":{\"model\":\"base\",\"useRemoteWorker\":true}}" \
    "$CONTROLLER_URL/api/jobs" 2>/dev/null)
JOB_CODE=$(echo "$JOB_RESPONSE" | tail -1)
JOB_BODY=$(echo "$JOB_RESPONSE" | sed '$d')

if [ "$JOB_CODE" != "200" ]; then
    echo -e "${RED}FAILED: Job creation failed (HTTP $JOB_CODE)${NC}"
    echo "Response: $JOB_BODY"
    rm -f cookies.tmp
    exit 1
fi

JOB_ID=$(echo "$JOB_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('jobId',''))" 2>/dev/null || echo "")
EXEC_MODE=$(echo "$JOB_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('executionMode','local'))" 2>/dev/null || echo "local")

if [ -z "$JOB_ID" ]; then
    echo -e "${RED}FAILED: Could not get job ID${NC}"
    echo "Response: $JOB_BODY"
    rm -f cookies.tmp
    exit 1
fi

echo -e "${GREEN}OK: Job created (ID: $JOB_ID, Mode: $EXEC_MODE)${NC}"

if [ "$EXEC_MODE" != "remote" ]; then
    echo -e "${YELLOW}WARNING: Job is running locally, not on remote worker${NC}"
    echo "Check that REMOTE_WORKER_MODE is set to 'optional' or 'required' on the controller"
fi
echo ""

# Step 7: Poll for job completion
echo -e "${YELLOW}Step 7: Waiting for job completion...${NC}"
MAX_WAIT=300  # 5 minutes
WAITED=0
POLL_INTERVAL=3

while [ $WAITED -lt $MAX_WAIT ]; do
    STATUS_RESPONSE=$(curl -s -b cookies.tmp "$CONTROLLER_URL/api/jobs/$JOB_ID" 2>/dev/null)
    STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
    STAGE=$(echo "$STATUS_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('progress',{}).get('stage',''))" 2>/dev/null || echo "")
    PERCENT=$(echo "$STATUS_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('progress',{}).get('percent',0))" 2>/dev/null || echo "0")
    
    echo "  Status: $STATUS | Stage: $STAGE | Progress: $PERCENT%"
    
    if [ "$STATUS" = "complete" ] || [ "$STATUS" = "complete_with_errors" ]; then
        echo -e "${GREEN}OK: Job completed successfully${NC}"
        break
    elif [ "$STATUS" = "failed" ]; then
        ERROR=$(echo "$STATUS_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error',{}).get('message','Unknown error'))" 2>/dev/null || echo "Unknown error")
        echo -e "${RED}FAILED: Job failed - $ERROR${NC}"
        rm -f cookies.tmp
        exit 1
    elif [ "$STATUS" = "canceled" ]; then
        echo -e "${RED}FAILED: Job was canceled${NC}"
        rm -f cookies.tmp
        exit 1
    fi
    
    sleep $POLL_INTERVAL
    WAITED=$((WAITED + POLL_INTERVAL))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}FAILED: Job timed out after ${MAX_WAIT}s${NC}"
    rm -f cookies.tmp
    exit 1
fi
echo ""

# Step 8: Verify outputs exist
echo -e "${YELLOW}Step 8: Verifying outputs...${NC}"
OUTPUTS_RESPONSE=$(curl -s -b cookies.tmp "$CONTROLLER_URL/api/jobs/$JOB_ID/outputs" 2>/dev/null)
OUTPUT_COUNT=$(echo "$OUTPUTS_RESPONSE" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('outputs',[])))" 2>/dev/null || echo "0")

if [ "$OUTPUT_COUNT" -eq "0" ]; then
    echo -e "${RED}FAILED: No outputs found${NC}"
    rm -f cookies.tmp
    exit 1
fi
echo -e "${GREEN}OK: Found $OUTPUT_COUNT output(s)${NC}"
echo ""

# Cleanup
rm -f cookies.tmp

echo "=========================================="
echo -e "${GREEN}SMOKE TEST PASSED${NC}"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Worker URL: $REMOTE_WORKER_URL"
echo "  Job ID: $JOB_ID"
echo "  Execution Mode: $EXEC_MODE"
echo "  Outputs: $OUTPUT_COUNT"
echo ""
exit 0
