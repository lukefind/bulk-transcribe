#!/bin/bash
# Batch runner for Bulk Transcribe
# Uploads files and creates jobs with resumability support
#
# Usage: ./scripts/run_batch.sh --input ./myfiles [options]
#
# Options:
#   --input DIR         Input directory containing audio files (required)
#   --controller URL    Controller URL (default: http://localhost:8476)
#   --remote            Request remote GPU worker execution
#   --diarization       Enable speaker diarization
#   --model MODEL       Whisper model (default: large-v3)
#   --state FILE        State file for resumability (default: batch_state.json in input dir)
#   --parallel N        Max parallel jobs (default: 1)
#   --help              Show this help

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
CONTROLLER_URL="${CONTROLLER_URL:-http://localhost:8476}"
USE_REMOTE=false
USE_DIARIZATION=false
MODEL="large-v3"
PARALLEL=1
INPUT_DIR=""
STATE_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --controller)
            CONTROLLER_URL="$2"
            shift 2
            ;;
        --remote)
            USE_REMOTE=true
            shift
            ;;
        --diarization)
            USE_DIARIZATION=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --state)
            STATE_FILE="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --help)
            head -20 "$0" | tail -18
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate input
if [ -z "$INPUT_DIR" ]; then
    echo -e "${RED}ERROR: --input is required${NC}"
    echo "Usage: ./scripts/run_batch.sh --input ./myfiles [options]"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}ERROR: Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

# Default state file
if [ -z "$STATE_FILE" ]; then
    STATE_FILE="$INPUT_DIR/batch_state.json"
fi

# Find audio files
AUDIO_EXTENSIONS="mp3 wav m4a flac ogg webm mp4 mov avi"
AUDIO_FILES=()
for ext in $AUDIO_EXTENSIONS; do
    while IFS= read -r -d '' file; do
        AUDIO_FILES+=("$file")
    done < <(find "$INPUT_DIR" -maxdepth 1 -type f -iname "*.$ext" -print0 2>/dev/null)
done

if [ ${#AUDIO_FILES[@]} -eq 0 ]; then
    echo -e "${RED}ERROR: No audio files found in $INPUT_DIR${NC}"
    exit 1
fi

echo "=========================================="
echo "Bulk Transcribe Batch Runner"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Input directory: $INPUT_DIR"
echo "  Controller:      $CONTROLLER_URL"
echo "  Model:           $MODEL"
echo "  Remote worker:   $USE_REMOTE"
echo "  Diarization:     $USE_DIARIZATION"
echo "  Parallel jobs:   $PARALLEL"
echo "  State file:      $STATE_FILE"
echo "  Files found:     ${#AUDIO_FILES[@]}"
echo ""

# Initialize or load state
init_state() {
    if [ -f "$STATE_FILE" ]; then
        echo -e "${BLUE}Loading existing state from $STATE_FILE${NC}"
    else
        echo '{"files": {}, "session": null, "csrfToken": null}' > "$STATE_FILE"
        echo -e "${BLUE}Created new state file${NC}"
    fi
}

get_state_value() {
    python3 -c "import json; print(json.load(open('$STATE_FILE')).get('$1', ''))" 2>/dev/null || echo ""
}

set_state_value() {
    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
state['$1'] = '$2'
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
" 2>/dev/null
}

get_file_state() {
    local filename="$1"
    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
file_state = state.get('files', {}).get('$filename', {})
print(json.dumps(file_state))
" 2>/dev/null || echo "{}"
}

set_file_state() {
    local filename="$1"
    local key="$2"
    local value="$3"
    python3 -c "
import json
state = json.load(open('$STATE_FILE'))
if 'files' not in state:
    state['files'] = {}
if '$filename' not in state['files']:
    state['files']['$filename'] = {}
state['files']['$filename']['$key'] = '$value'
json.dump(state, open('$STATE_FILE', 'w'), indent=2)
" 2>/dev/null
}

# Get session and CSRF token
get_session() {
    echo -e "${YELLOW}Getting session...${NC}"
    
    local response
    response=$(curl -s -c cookies.tmp -b cookies.tmp "$CONTROLLER_URL/api/session" 2>/dev/null)
    
    local csrf_token
    csrf_token=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('csrfToken',''))" 2>/dev/null || echo "")
    
    if [ -z "$csrf_token" ]; then
        echo -e "${RED}ERROR: Could not get session${NC}"
        return 1
    fi
    
    set_state_value "csrfToken" "$csrf_token"
    echo -e "${GREEN}Session established${NC}"
}

# Upload a file
upload_file() {
    local filepath="$1"
    local filename=$(basename "$filepath")
    
    # Check if already uploaded
    local file_state
    file_state=$(get_file_state "$filename")
    local upload_id
    upload_id=$(echo "$file_state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uploadId',''))" 2>/dev/null || echo "")
    
    if [ -n "$upload_id" ]; then
        echo -e "${BLUE}  Already uploaded: $filename (ID: $upload_id)${NC}"
        echo "$upload_id"
        return 0
    fi
    
    local csrf_token
    csrf_token=$(get_state_value "csrfToken")
    
    local response
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -b cookies.tmp \
        -H "X-CSRF-Token: $csrf_token" \
        -F "files=@$filepath" \
        "$CONTROLLER_URL/api/uploads" 2>/dev/null)
    
    local http_code
    http_code=$(echo "$response" | tail -1)
    local body
    body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" != "200" ]; then
        echo -e "${RED}  Upload failed: $filename (HTTP $http_code)${NC}"
        return 1
    fi
    
    upload_id=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uploads',[])[0].get('id',''))" 2>/dev/null || echo "")
    
    if [ -z "$upload_id" ]; then
        echo -e "${RED}  Upload failed: $filename (no ID returned)${NC}"
        return 1
    fi
    
    set_file_state "$filename" "uploadId" "$upload_id"
    echo -e "${GREEN}  Uploaded: $filename (ID: $upload_id)${NC}"
    echo "$upload_id"
}

# Create a job for a file
create_job() {
    local filename="$1"
    local upload_id="$2"
    
    # Check if job already exists
    local file_state
    file_state=$(get_file_state "$filename")
    local job_id
    job_id=$(echo "$file_state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('jobId',''))" 2>/dev/null || echo "")
    
    if [ -n "$job_id" ]; then
        echo -e "${BLUE}  Job exists: $filename (ID: $job_id)${NC}"
        echo "$job_id"
        return 0
    fi
    
    local csrf_token
    csrf_token=$(get_state_value "csrfToken")
    
    # Build options JSON
    local options="{\"model\":\"$MODEL\""
    if [ "$USE_REMOTE" = true ]; then
        options="$options,\"useRemoteWorker\":true"
    fi
    if [ "$USE_DIARIZATION" = true ]; then
        options="$options,\"diarizationEnabled\":true"
    fi
    options="$options}"
    
    local response
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -b cookies.tmp \
        -H "Content-Type: application/json" \
        -H "X-CSRF-Token: $csrf_token" \
        -d "{\"uploadIds\":[\"$upload_id\"],\"options\":$options}" \
        "$CONTROLLER_URL/api/jobs" 2>/dev/null)
    
    local http_code
    http_code=$(echo "$response" | tail -1)
    local body
    body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" != "200" ]; then
        local error
        error=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','Unknown error'))" 2>/dev/null || echo "HTTP $http_code")
        echo -e "${RED}  Job creation failed: $filename - $error${NC}"
        set_file_state "$filename" "error" "$error"
        return 1
    fi
    
    job_id=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('jobId',''))" 2>/dev/null || echo "")
    local exec_mode
    exec_mode=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin).get('executionMode','local'))" 2>/dev/null || echo "local")
    
    set_file_state "$filename" "jobId" "$job_id"
    set_file_state "$filename" "executionMode" "$exec_mode"
    set_file_state "$filename" "status" "running"
    
    echo -e "${GREEN}  Job created: $filename (ID: $job_id, Mode: $exec_mode)${NC}"
    echo "$job_id"
}

# Poll job status
poll_job() {
    local filename="$1"
    local job_id="$2"
    
    local response
    response=$(curl -s -b cookies.tmp "$CONTROLLER_URL/api/jobs/$job_id" 2>/dev/null)
    
    local status
    status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
    
    set_file_state "$filename" "status" "$status"
    
    echo "$status"
}

# Print summary table
print_summary() {
    echo ""
    echo "=========================================="
    echo "Batch Summary"
    echo "=========================================="
    printf "%-40s %-12s %-10s %-8s\n" "FILE" "JOB_ID" "STATUS" "MODE"
    echo "------------------------------------------"
    
    for filepath in "${AUDIO_FILES[@]}"; do
        local filename=$(basename "$filepath")
        local file_state
        file_state=$(get_file_state "$filename")
        
        local job_id
        job_id=$(echo "$file_state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('jobId','')[:12] if json.load(open('/dev/stdin')).get('jobId') else '-')" 2>/dev/null || echo "-")
        local status
        status=$(echo "$file_state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','-'))" 2>/dev/null || echo "-")
        local mode
        mode=$(echo "$file_state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('executionMode','-'))" 2>/dev/null || echo "-")
        
        # Truncate filename if too long
        local display_name="$filename"
        if [ ${#display_name} -gt 38 ]; then
            display_name="${display_name:0:35}..."
        fi
        
        # Color status
        local status_color="$NC"
        case "$status" in
            complete) status_color="$GREEN" ;;
            failed) status_color="$RED" ;;
            running|queued*) status_color="$YELLOW" ;;
        esac
        
        printf "%-40s %-12s ${status_color}%-10s${NC} %-8s\n" "$display_name" "$job_id" "$status" "$mode"
    done
    echo ""
}

# Main execution
init_state
get_session

echo ""
echo "Uploading files..."
for filepath in "${AUDIO_FILES[@]}"; do
    upload_file "$filepath" > /dev/null
done

echo ""
echo "Creating jobs..."
for filepath in "${AUDIO_FILES[@]}"; do
    filename=$(basename "$filepath")
    file_state=$(get_file_state "$filename")
    upload_id=$(echo "$file_state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uploadId',''))" 2>/dev/null || echo "")
    
    if [ -n "$upload_id" ]; then
        # Check if already complete
        status=$(echo "$file_state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
        if [ "$status" = "complete" ]; then
            echo -e "${BLUE}  Skipping (complete): $filename${NC}"
            continue
        fi
        
        create_job "$filename" "$upload_id" > /dev/null
    fi
done

echo ""
echo "Polling job status..."
POLL_INTERVAL=5
MAX_POLLS=720  # 1 hour at 5s intervals
poll_count=0

while [ $poll_count -lt $MAX_POLLS ]; do
    all_done=true
    
    for filepath in "${AUDIO_FILES[@]}"; do
        filename=$(basename "$filepath")
        file_state=$(get_file_state "$filename")
        job_id=$(echo "$file_state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('jobId',''))" 2>/dev/null || echo "")
        status=$(echo "$file_state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
        
        # Skip if no job or already terminal
        if [ -z "$job_id" ]; then
            continue
        fi
        if [ "$status" = "complete" ] || [ "$status" = "failed" ] || [ "$status" = "canceled" ]; then
            continue
        fi
        
        all_done=false
        new_status=$(poll_job "$filename" "$job_id")
        
        if [ "$new_status" != "$status" ]; then
            echo -e "  $filename: $status -> $new_status"
        fi
    done
    
    if [ "$all_done" = true ]; then
        break
    fi
    
    sleep $POLL_INTERVAL
    poll_count=$((poll_count + 1))
done

print_summary

# Count results
complete_count=0
failed_count=0
for filepath in "${AUDIO_FILES[@]}"; do
    filename=$(basename "$filepath")
    file_state=$(get_file_state "$filename")
    status=$(echo "$file_state" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
    
    if [ "$status" = "complete" ]; then
        complete_count=$((complete_count + 1))
    elif [ "$status" = "failed" ]; then
        failed_count=$((failed_count + 1))
    fi
done

echo "Results: $complete_count complete, $failed_count failed, $((${#AUDIO_FILES[@]} - complete_count - failed_count)) other"

# Cleanup
rm -f cookies.tmp

if [ $failed_count -gt 0 ]; then
    exit 1
fi
exit 0
