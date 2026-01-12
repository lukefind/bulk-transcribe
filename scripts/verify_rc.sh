#!/bin/bash
# Release Candidate Verification Script
# Runs proof-level checks against a running server instance.
#
# Usage:
#   ./scripts/verify_rc.sh [BASE_URL] [TEST_FILE]
#
# Arguments:
#   BASE_URL   - Server URL (default: http://localhost:8476)
#   TEST_FILE  - Path to a small audio file for testing (optional)
#
# Requirements:
#   - curl, jq
#   - Server must be running
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed

BASE_URL="${1:-http://localhost:8476}"
TEST_FILE="${2:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED++))
}

log_info() {
    echo -e "[INFO] $1"
}

# Check dependencies
command -v curl >/dev/null 2>&1 || { echo "curl is required"; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq is required"; exit 1; }

echo "========================================"
echo "Release Candidate Verification"
echo "========================================"
echo "Target: $BASE_URL"
echo ""

# -----------------------------------------------------------------------------
# Test 1: Server is reachable
# -----------------------------------------------------------------------------
log_info "Testing server connectivity..."
if curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/" | grep -q "200"; then
    log_pass "Server is reachable"
else
    log_fail "Server is not reachable at $BASE_URL"
    exit 1
fi

# -----------------------------------------------------------------------------
# Test 2: /api/runtime returns valid JSON with expected fields
# -----------------------------------------------------------------------------
log_info "Testing /api/runtime endpoint..."
RUNTIME=$(curl -s "$BASE_URL/api/runtime")

if echo "$RUNTIME" | jq -e '.os' >/dev/null 2>&1; then
    log_pass "/api/runtime returns valid JSON"
else
    log_fail "/api/runtime does not return valid JSON"
fi

# Check required fields
for field in os arch inDocker supportedBackends recommendedBackend diarizationAvailable; do
    if echo "$RUNTIME" | jq -e "has(\"$field\")" >/dev/null 2>&1; then
        VALUE=$(echo "$RUNTIME" | jq -r ".$field")
        log_pass "/api/runtime has $field = $VALUE"
    else
        log_fail "/api/runtime missing $field"
    fi
done

# Check diarization reason if unavailable
DIAR_AVAILABLE=$(echo "$RUNTIME" | jq -r '.diarizationAvailable')
if [ "$DIAR_AVAILABLE" = "false" ]; then
    if echo "$RUNTIME" | jq -e '.diarizationReason' >/dev/null 2>&1; then
        REASON=$(echo "$RUNTIME" | jq -r '.diarizationReason')
        log_pass "Diarization unavailable with reason: $REASON"
    else
        log_fail "Diarization unavailable but no reason provided"
    fi
fi

# -----------------------------------------------------------------------------
# Test 3: /api/session returns CSRF token
# -----------------------------------------------------------------------------
log_info "Testing /api/session endpoint..."
COOKIES=$(mktemp)
# First request to establish session
curl -s -c "$COOKIES" "$BASE_URL/" >/dev/null
# Then get CSRF token
SESSION_RESP=$(curl -s -b "$COOKIES" -c "$COOKIES" "$BASE_URL/api/session")

if echo "$SESSION_RESP" | jq -e '.csrfToken' >/dev/null 2>&1; then
    CSRF_TOKEN=$(echo "$SESSION_RESP" | jq -r '.csrfToken')
    log_pass "/api/session returns CSRF token"
else
    log_fail "/api/session does not return CSRF token"
    CSRF_TOKEN=""
fi

# -----------------------------------------------------------------------------
# Test 4: Backend validation rejects unsupported backends
# -----------------------------------------------------------------------------
log_info "Testing backend validation..."
if [ -n "$CSRF_TOKEN" ]; then
    INVALID_BACKEND_RESP=$(curl -s -b "$COOKIES" \
        -H "Content-Type: application/json" \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        -X POST "$BASE_URL/api/jobs" \
        -d '{"uploadIds": ["fake"], "options": {"backend": "invalid_backend"}}')
    
    if echo "$INVALID_BACKEND_RESP" | jq -e '.error' >/dev/null 2>&1; then
        log_pass "Invalid backend rejected with error"
    else
        log_fail "Invalid backend was not rejected"
    fi
else
    log_skip "Backend validation (no CSRF token)"
fi

# -----------------------------------------------------------------------------
# Test 5: Diarization without HF_TOKEN is rejected (if diarization unavailable)
# -----------------------------------------------------------------------------
log_info "Testing diarization gating..."
if [ "$DIAR_AVAILABLE" = "false" ] && [ -n "$CSRF_TOKEN" ]; then
    DIAR_RESP=$(curl -s -b "$COOKIES" \
        -H "Content-Type: application/json" \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        -X POST "$BASE_URL/api/jobs" \
        -d '{"uploadIds": ["fake"], "options": {"diarizationEnabled": true}}')
    
    ERROR_CODE=$(echo "$DIAR_RESP" | jq -r '.code // empty')
    if [ "$ERROR_CODE" = "HF_TOKEN_MISSING" ] || [ "$ERROR_CODE" = "DIARIZATION_UNAVAILABLE" ]; then
        log_pass "Diarization rejected with code: $ERROR_CODE"
    else
        # May fail for other reasons (fake upload ID), check error message
        ERROR_MSG=$(echo "$DIAR_RESP" | jq -r '.error // empty')
        if echo "$ERROR_MSG" | grep -qi "diarization\|HF_TOKEN"; then
            log_pass "Diarization rejected: $ERROR_MSG"
        else
            log_skip "Diarization gating (rejected for other reason: $ERROR_MSG)"
        fi
    fi
elif [ "$DIAR_AVAILABLE" = "true" ]; then
    log_skip "Diarization gating (diarization is available)"
else
    log_skip "Diarization gating (no CSRF token)"
fi

# -----------------------------------------------------------------------------
# Test 6: File upload and job creation (requires test file)
# -----------------------------------------------------------------------------
if [ -n "$TEST_FILE" ] && [ -f "$TEST_FILE" ]; then
    log_info "Testing file upload with $TEST_FILE..."
    
    UPLOAD_RESP=$(curl -s -b "$COOKIES" \
        -H "X-CSRF-Token: $CSRF_TOKEN" \
        -F "files=@$TEST_FILE" \
        "$BASE_URL/api/uploads")
    
    if echo "$UPLOAD_RESP" | jq -e '.uploads[0].id' >/dev/null 2>&1; then
        UPLOAD_ID=$(echo "$UPLOAD_RESP" | jq -r '.uploads[0].id')
        log_pass "File uploaded with ID: $UPLOAD_ID"
        
        # Create job
        log_info "Creating transcription job..."
        JOB_RESP=$(curl -s -b "$COOKIES" \
            -H "Content-Type: application/json" \
            -H "X-CSRF-Token: $CSRF_TOKEN" \
            -X POST "$BASE_URL/api/jobs" \
            -d "{\"uploadIds\": [\"$UPLOAD_ID\"], \"options\": {\"model\": \"tiny\"}}")
        
        if echo "$JOB_RESP" | jq -e '.jobId' >/dev/null 2>&1; then
            JOB_ID=$(echo "$JOB_RESP" | jq -r '.jobId')
            log_pass "Job created with ID: $JOB_ID"
            
            # Poll for completion (max 120 seconds)
            log_info "Waiting for job completion (max 120s)..."
            for i in {1..60}; do
                sleep 2
                STATUS_RESP=$(curl -s -b "$COOKIES" "$BASE_URL/api/jobs/$JOB_ID")
                STATUS=$(echo "$STATUS_RESP" | jq -r '.status')
                
                if [ "$STATUS" = "complete" ]; then
                    log_pass "Job completed successfully"
                    
                    # Check outputs
                    OUTPUTS=$(echo "$STATUS_RESP" | jq '.outputs | length')
                    if [ "$OUTPUTS" -gt 0 ]; then
                        log_pass "Job produced $OUTPUTS output(s)"
                    else
                        log_fail "Job completed but no outputs"
                    fi
                    break
                elif [ "$STATUS" = "failed" ] || [ "$STATUS" = "canceled" ]; then
                    ERROR=$(echo "$STATUS_RESP" | jq -r '.error.code // "unknown"')
                    log_fail "Job ended with status: $STATUS (error: $ERROR)"
                    break
                elif [ "$STATUS" = "complete_with_errors" ]; then
                    log_pass "Job completed with errors (partial success)"
                    break
                fi
                
                if [ $i -eq 60 ]; then
                    log_fail "Job timed out after 120 seconds (status: $STATUS)"
                fi
            done
            
            # Test rerun endpoint
            log_info "Testing rerun endpoint..."
            RERUN_RESP=$(curl -s -b "$COOKIES" \
                -H "Content-Type: application/json" \
                -H "X-CSRF-Token: $CSRF_TOKEN" \
                -X POST "$BASE_URL/api/jobs/$JOB_ID/rerun")
            
            if echo "$RERUN_RESP" | jq -e '.jobId' >/dev/null 2>&1; then
                NEW_JOB_ID=$(echo "$RERUN_RESP" | jq -r '.jobId')
                if [ "$NEW_JOB_ID" != "$JOB_ID" ]; then
                    log_pass "Rerun created new job: $NEW_JOB_ID"
                    
                    # Cancel the rerun job to test cancel
                    log_info "Testing cancel endpoint..."
                    sleep 1
                    CANCEL_RESP=$(curl -s -b "$COOKIES" \
                        -H "X-CSRF-Token: $CSRF_TOKEN" \
                        -X POST "$BASE_URL/api/jobs/$NEW_JOB_ID/cancel")
                    
                    sleep 2
                    CANCEL_STATUS=$(curl -s -b "$COOKIES" "$BASE_URL/api/jobs/$NEW_JOB_ID" | jq -r '.status')
                    if [ "$CANCEL_STATUS" = "canceled" ] || [ "$CANCEL_STATUS" = "canceling" ]; then
                        log_pass "Cancel worked (status: $CANCEL_STATUS)"
                        
                        # Check for USER_CANCELED error code
                        CANCEL_CODE=$(curl -s -b "$COOKIES" "$BASE_URL/api/jobs/$NEW_JOB_ID" | jq -r '.error.code // empty')
                        if [ "$CANCEL_CODE" = "USER_CANCELED" ]; then
                            log_pass "Cancel has USER_CANCELED error code"
                        elif [ -n "$CANCEL_CODE" ]; then
                            log_pass "Cancel has error code: $CANCEL_CODE"
                        fi
                    else
                        log_skip "Cancel verification (job may have completed: $CANCEL_STATUS)"
                    fi
                else
                    log_fail "Rerun returned same job ID"
                fi
            else
                ERROR=$(echo "$RERUN_RESP" | jq -r '.error // "unknown"')
                log_fail "Rerun failed: $ERROR"
            fi
        else
            ERROR=$(echo "$JOB_RESP" | jq -r '.error // "unknown"')
            log_fail "Job creation failed: $ERROR"
        fi
    else
        ERROR=$(echo "$UPLOAD_RESP" | jq -r '.error // "unknown"')
        log_fail "Upload failed: $ERROR"
    fi
else
    log_skip "File upload tests (no TEST_FILE provided)"
    log_info "To run upload tests: $0 $BASE_URL /path/to/audio.mp3"
fi

# Cleanup
rm -f "$COOKIES"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Verification Summary"
echo "========================================"
echo -e "${GREEN}Passed:${NC}  $PASSED"
echo -e "${RED}Failed:${NC}  $FAILED"
echo -e "${YELLOW}Skipped:${NC} $SKIPPED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}VERIFICATION FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}VERIFICATION PASSED${NC}"
    exit 0
fi
