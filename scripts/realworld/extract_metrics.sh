#!/usr/bin/env bash
set -euo pipefail

# Extract A/B test metrics from client logs
# Usage: ./extract_metrics.sh <log_dir> [sync|async|both]

LOG_DIR="${1:-.}"
MODE="${2:-both}"

echo "=== Metrics Extraction ==="
echo "Log dir: $LOG_DIR"
echo ""
extract_sync() {
    local dir="$LOG_DIR/sync_baseline_"*/
    dir=$(echo "$dir" | head -1)
    if [ ! -d "$dir" ]; then
        echo "[SYNC] No log dir found"
        return
    fi
    
    local client_log="$dir/client.log"
    if [ ! -f "$client_log" ]; then
        echo "[SYNC] No client.log found"
        return
    fi
    
    echo "--- SYNC Baseline ---"
    local traj=$(grep -c "Received Trajectory" "$client_log" 2>/dev/null || echo "0")
    local no_traj=$(grep -c "No trajectory" "$client_log" 2>/dev/null || echo "0")
    local discrete=$(grep -c "Received Discrete" "$client_log" 2>/dev/null || echo "0")
    local http_count=$(grep -c "\[HTTP\]" "$client_log" 2>/dev/null || echo "0")
    local total_time=$(grep "bag play" "$client_log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "0")
    
    echo "  traj_count: $traj"
    echo "  no_traj_count: $no_traj"  
    echo "  discrete_count: $discrete"
    echo "  http_requests: $http_count"
    
    if [ "$http_count" -gt 0 ] 2>/dev/null; then
        local req_hz=$(echo "scale=3; $http_count / 120" | bc 2>/dev/null || echo "N/A")
        echo "  req_hz (est): $req_hz"
    fi
}

extract_async() {
    local dir="$LOG_DIR/async_test_"*/
    dir=$(echo "$dir" | head -1)
    if [ ! -d "$dir" ]; then
        echo "[ASYNC] No log dir found"
        return
    fi
    
    local client_log="$dir/client.log"
    if [ ! -f "$client_log" ]; then
        echo "[ASYNC] No client.log found"
        return
    fi
    
    echo "--- ASYNC Test ---"
    local traj=$(grep -c "Received Trajectory" "$client_log" 2>/dev/null || echo "0")
    local no_traj=$(grep -c "No trajectory" "$client_log" 2>/dev/null || echo "0")
    local discrete=$(grep -c "Received Discrete" "$client_log" 2>/dev/null || echo "0")
    local http_count=$(grep -c "\[HTTP\]" "$client_log" 2>/dev/null || echo "0")
    
    echo "  traj_count: $traj"
    echo "  no_traj_count: $no_traj"
    echo "  discrete_count: $discrete"
    echo "  http_requests: $http_count"
    
    if [ "$http_count" -gt 0 ] 2>/dev/null; then
        local req_hz=$(echo "scale=3; $http_count / 120" | bc 2>/dev/null || echo "N/A")
        echo "  req_hz (est): $req_hz"
    fi
}

if [ "$MODE" = "both" ] || [ "$MODE" = "sync" ]; then
    extract_sync
    echo ""
fi

if [ "$MODE" = "both" ] || [ "$MODE" = "async" ]; then
    extract_async
fi

echo ""
echo "Done."