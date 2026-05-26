#!/usr/bin/env bash
# ============================================================================
# gate_check.sh — automated gate verification for InternNav research phases
#
# Tests the current server against numeric gate conditions from PLAN.md.
# Run from W0:P0 (Claude Main) or W3:monitor after an experiment.
#
# Usage:
#   bash scripts/gate_check.sh              # check current gate (auto-detect)
#   bash scripts/gate_check.sh 0            # Gate 0: true async latency
#   bash scripts/gate_check.sh 1            # Gate 1: trajectory ratio ≥ 63%
#   bash scripts/gate_check.sh stats <dir>  # check gate from saved stats JSON
#   bash scripts/gate_check.sh all          # run all gate checks
#
# Gate numbers match PLAN.md:
#   Gate 0 — joint_latency_ms < 20ms, trajectory_ratio ≥ 50%, req_hz ≥ 10
#   Gate 1a — trajectory_ratio ≥ 63% at any temperature
#   Gate 2c — enhanced agent ≥ Phase 1 baseline on all 3 bags
# ============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_PORT="${SERVER_PORT:-5000}"
SERVER_URL="http://localhost:${SERVER_PORT}"
N_REQUESTS="${N_REQUESTS:-10}"   # requests to send for latency measurement

# ── colors ─────────────────────────────────────────────────────────────────────
R='\033[0;31m'; G='\033[0;32m'; Y='\033[0;33m'; B='\033[0;34m'
BOLD='\033[1m'; NC='\033[0m'
PASS="${G}${BOLD}PASS${NC}"; FAIL="${R}${BOLD}FAIL${NC}"; WARN="${Y}${BOLD}WARN${NC}"

# ── helpers ────────────────────────────────────────────────────────────────────
hr()  { printf '%0.s─' {1..68}; echo; }
hdr() { echo; printf "${BOLD}%-68s${NC}\n" "$1"; hr; }
check() {
    # check <label> <value> <op> <threshold> [unit]
    local label="$1" val="$2" op="$3" thresh="$4" unit="${5:-}"
    local result
    if python3 -c "import sys; sys.exit(0 if float('$val') $op float('$thresh') else 1)" 2>/dev/null; then
        result="$PASS"
    else
        result="$FAIL"
    fi
    printf "  %-40s  %s %s  [threshold: %s %s %s]\n" \
        "$label" "$val$unit" "$result" "$op" "$thresh" "$unit"
    # Return exit code so callers can track failures
    python3 -c "import sys; sys.exit(0 if float('$val') $op float('$thresh') else 1)" 2>/dev/null
}

# ── measure actual HTTP latency against /eval_dual_async ───────────────────────
measure_latency() {
    echo "Sending $N_REQUESTS requests to $SERVER_URL/eval_dual_async ..."
    local total=0 count=0 min=99999 max=0 trajectory=0 waiting=0 discrete=0
    local dummy_b64
    dummy_b64=$(python3 -c "import base64,numpy as np; img=np.zeros((256,256,3),dtype=np.uint8); import cv2; ok,buf=cv2.imencode('.jpg',img); print(base64.b64encode(buf).decode())" 2>/dev/null \
        || python3 -c "import base64; print(base64.b64encode(b'fake_img').decode())")

    for i in $(seq 1 $N_REQUESTS); do
        local t0 t1 ms rtype
        t0=$(python3 -c "import time; print(int(time.time()*1000))")
        resp=$(curl -sf -X POST "$SERVER_URL/eval_dual_async" \
            -H "Content-Type: application/json" \
            -d "{\"image\":\"$dummy_b64\",\"instruction\":\"go to the kitchen\"}" \
            2>/dev/null || echo '{"status":"error"}')
        t1=$(python3 -c "import time; print(int(time.time()*1000))")
        ms=$((t1 - t0))
        total=$((total + ms))
        count=$((count + 1))
        [ $ms -lt $min ] && min=$ms
        [ $ms -gt $max ] && max=$ms
        rtype=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print('trajectory' if 'trajectory' in d else ('discrete' if 'discrete_action' in d else 'waiting'))" 2>/dev/null || echo "error")
        case "$rtype" in
            trajectory) trajectory=$((trajectory+1)) ;;
            discrete)   discrete=$((discrete+1)) ;;
            *)          waiting=$((waiting+1)) ;;
        esac
        printf "  req %2d: %3d ms  [%s]\n" "$i" "$ms" "$rtype"
    done

    local mean=$((total / count))
    local traj_pct traj_float
    traj_float=$(python3 -c "print(f'{$trajectory/$count:.4f}')" 2>/dev/null || echo "0")
    traj_pct=$(python3 -c "print(f'{$trajectory/$count*100:.1f}')" 2>/dev/null || echo "0")

    echo ""
    echo "  Mean:  $mean ms  |  Min: $min ms  |  Max: $max ms"
    echo "  trajectory: $trajectory/$count ($traj_pct%)  discrete: $discrete  waiting: $waiting"

    # Return values via global for caller
    _MEAN_MS=$mean
    _MIN_MS=$min
    _TRAJ_RATIO=$traj_float
    _TRAJ_COUNT=$trajectory
    _TOTAL=$count
}

# ── gate check functions ────────────────────────────────────────────────────────

gate_0() {
    hdr "GATE 0 — TRUE ASYNC CONFIRMATION"
    echo "  Conditions: joint_latency_ms < 20 | trajectory_ratio ≥ 50% | req_hz ≥ 10"
    hr

    # 1. Check server is up
    if ! curl -sf "$SERVER_URL/health" >/dev/null 2>&1; then
        echo -e "  ${FAIL}: Server not responding on port $SERVER_PORT"
        echo "  → Start server in W1 first (./scripts/model/start.sh + exec.sh)"
        return 1
    fi
    echo -e "  Server:  ${G}UP${NC} ($SERVER_URL)"

    # 2. Check ASYNC_BACKGROUND_INFERENCE flag
    bg_inf=$(curl -sf "$SERVER_URL/stats" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('async_background_inference','unknown'))" \
        2>/dev/null || echo "unknown")
    printf "  %-40s  %s\n" "async_background_inference" "$bg_inf"
    if [ "$bg_inf" = "False" ] || [ "$bg_inf" = "false" ]; then
        echo -e "  ${WARN}: ASYNC_BACKGROUND_INFERENCE is False — background thread not running"
        echo "  → Fix http_internvla_server_debug.py line ~68 (set to True)"
    fi

    # 3. Measure latency
    echo ""
    measure_latency

    echo ""
    hdr "GATE 0 RESULT"
    local passed=0 failed=0
    check "joint_latency_ms (mean)"  "$_MEAN_MS"    "<"  "20"   " ms" && ((passed++)) || ((failed++)) || true
    check "trajectory_ratio"         "$_TRAJ_RATIO" ">=" "0.50" ""    && ((passed++)) || ((failed++)) || true
    # req_hz requires knowing total experiment time — estimate from N_REQUESTS and mean_ms
    local est_hz
    est_hz=$(python3 -c "print(f'{1000/$_MEAN_MS:.2f}')" 2>/dev/null || echo "N/A")
    check "req_hz (estimated from latency)" "$est_hz" ">=" "10" " Hz" && ((passed++)) || ((failed++)) || true

    echo ""
    if [ $failed -eq 0 ]; then
        echo -e "  ${PASS} Gate 0 PASSED — true async confirmed"
        echo "  → Proceed to GATE 1 (temperature sweep, Task #3)"
    else
        echo -e "  ${FAIL} Gate 0 FAILED ($failed condition(s) not met)"
        echo "  → Fix: remove agent.step() from /eval_dual_async handler (line ~437)"
        echo "  → See PLAN.md Gate Failure Protocol"
    fi
}

gate_1() {
    hdr "GATE 1a — QUALITY OPTIMIZATION (temperature sweep)"
    echo "  Condition: trajectory_ratio ≥ 63% at best temperature, req_hz ≥ 10 Hz"
    hr

    # Read from saved stats files
    local best_temp best_ratio best_hz found=0
    best_ratio=0; best_hz=0; best_temp="none"

    for summary in "$REPO_DIR/stats/"*/summary.json; do
        [ -f "$summary" ] || continue
        temp=$(python3 -c "import json; d=json.load(open('$summary')); print(d.get('control_vars',{}).get('temperature','?'))" 2>/dev/null || continue)
        ratio=$(python3 -c "import json; d=json.load(open('$summary')); print(d.get('trajectory_ratio',0))" 2>/dev/null || echo 0)
        hz=$(python3 -c "import json; d=json.load(open('$summary')); print(d.get('joint_req_hz',0))" 2>/dev/null || echo 0)
        found=1
        printf "  temp=%-5s  trajectory_ratio=%.1f%%  req_hz=%.2f Hz\n" \
            "$temp" "$(python3 -c "print($ratio*100)")" "$hz"
        if python3 -c "sys.exit(0 if float('$ratio')>float('$best_ratio') else 1)" 2>/dev/null; then
            best_ratio=$ratio; best_temp=$temp; best_hz=$hz
        fi
    done

    if [ $found -eq 0 ]; then
        echo "  No stats/*/summary.json found."
        echo "  → Run temperature sweep experiments and save with:"
        echo "    python scripts/realworld/stats_recorder.py parse <log> --tag <name> --temperature <T>"
        return 0
    fi

    echo ""
    hdr "GATE 1a RESULT (best config: temp=$best_temp)"
    local failed=0
    check "best trajectory_ratio" "$best_ratio" ">=" "0.63" "" || ((failed++)) || true
    check "req_hz at best config"  "$best_hz"   ">=" "10"   " Hz" || ((failed++)) || true
    echo ""
    if [ $failed -eq 0 ]; then
        echo -e "  ${PASS} Gate 1a PASSED — optimal temperature: $best_temp"
    else
        echo -e "  ${FAIL} Gate 1a FAILED"
        echo "  → Continue temperature sweep or investigate quality regression"
    fi
}

gate_stats() {
    local stats_dir="$1"
    [ -f "$stats_dir/summary.json" ] || { echo "No summary.json in $stats_dir"; return 1; }
    hdr "GATE CHECK — $stats_dir"
    python3 -m json.tool "$stats_dir/summary.json" | grep -E '"tag"|"joint_req_hz"|"trajectory_ratio"|"joint_latency"|"s2_latency"'
}

# ── main ───────────────────────────────────────────────────────────────────────
CMD="${1:-0}"

case "$CMD" in
    0)          gate_0 ;;
    1)          gate_1 ;;
    stats)      gate_stats "${2:-.}" ;;
    all)        gate_0; echo; gate_1 ;;
    *)          echo "Usage: bash gate_check.sh [0|1|all|stats <dir>]" ;;
esac

echo ""
