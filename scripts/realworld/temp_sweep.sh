#!/usr/bin/env bash
# Phase 1 Task #3 — Temperature sweep on bag 073623
# Usage: docker exec vlnav_internvla_server bash scripts/realworld/temp_sweep.sh
# Runs temps {0.70, 0.75, 0.80, 0.85} and prints a summary table.
set -e
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

BAG="my_camera_bag_20260317_073623"
# BAG_PATH: rosbag played manually from gds container via FastDDS
CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
LOG_BASE="/tmp/temp_sweep"
mkdir -p "$LOG_BASE"

TEMPS=(0.70 0.75 0.80 0.85)

declare -A RES_TRAJ RES_HZ RES_LAT RES_BG

for TEMP in "${TEMPS[@]}"; do
    TAG=$(echo "$TEMP" | tr '.' '_')
    LOG_DIR="$LOG_BASE/temp_${TAG}"
    mkdir -p "$LOG_DIR"

    echo ""
    echo "=============================================="
    echo "  TEMP = $TEMP"
    echo "=============================================="

    # Reset metrics
    curl -sf http://localhost:5802/reset_metrics > "$LOG_DIR/reset.json"
    echo "  Metrics reset OK"
    sleep 2

    # Start client
    cd "$REPO_DIR" && python3.12 scripts/realworld/http_internvla_client_debug.py \
        --mode async \
        --kv-cache \
        --temperature "$TEMP" \
        --jpeg-quality 95 \
        --depth-png-compress 6 \
        --calib "$CALIB" \
        > "$LOG_DIR/client.log" 2>&1 &
    CLIENT_PID=$!
    echo "  Client pid=$CLIENT_PID — waiting 3s..."
    sleep 3

    # Play bag at rate=0.5
    echo "  Playing bag at rate=0.5..."
#     ros2 bag play "$BAG_PATH" --rate 0.5 > "$LOG_DIR/bag.log" 2>&1  # played manually from gds container
    echo "  Bag done. Collecting metrics..."
    sleep 2

    # Collect metrics
    curl -sf http://localhost:5802/async_metrics > "$LOG_DIR/metrics.json"
    kill $CLIENT_PID 2>/dev/null || true

    # Parse
    python3 - <<PYEOF
import json
with open("$LOG_DIR/metrics.json") as f:
    d = json.load(f)
traj = d.get('trajectory_ratio', 0)
hz   = d.get('joint_req_hz', 0)
lat  = d.get('joint_latency_ms', 9999)
bg   = d.get('background_s2_runs', 0)
reqs = d.get('total_requests', 0)
print(f"  trajectory_ratio : {traj:.1f}%")
print(f"  joint_req_hz     : {hz:.2f} Hz")
print(f"  joint_latency_ms : {lat:.2f} ms")
print(f"  background_s2_runs: {bg}")
print(f"  total_requests   : {reqs}")
gate_traj = traj >= 63.0
gate_hz   = hz >= 10.0
print(f"  Gate 1a traj >=63%: {'PASS' if gate_traj else 'FAIL'} ({traj:.1f}%)")
print(f"  Gate 1a hz  >=10  : {'PASS' if gate_hz   else 'FAIL'} ({hz:.2f} Hz)")
PYEOF

    echo "  Logs: $LOG_DIR"
    sleep 5  # cool-down between runs
done

echo ""
echo "=============================================="
echo "  TEMPERATURE SWEEP SUMMARY"
echo "  Bag: $BAG"
echo "=============================================="
echo ""
printf "  %-8s  %-18s  %-14s  %-14s  %-10s\n" "Temp" "trajectory_ratio" "joint_req_hz" "latency_ms" "bg_s2_runs"
printf "  %-8s  %-18s  %-14s  %-14s  %-10s\n" "--------" "------------------" "--------------" "--------------" "----------"

for TEMP in "${TEMPS[@]}"; do
    TAG=$(echo "$TEMP" | tr '.' '_')
    LOG_DIR="$LOG_BASE/temp_${TAG}"
    python3 - <<PYEOF
import json
with open("$LOG_DIR/metrics.json") as f:
    d = json.load(f)
traj = d.get('trajectory_ratio', 0)
hz   = d.get('joint_req_hz', 0)
lat  = d.get('joint_latency_ms', 9999)
bg   = d.get('background_s2_runs', 0)
gate = 'PASS' if (traj >= 63.0 and hz >= 10.0) else 'FAIL'
print(f"  {$TEMP:<8.2f}  {traj:<18.1f}  {hz:<14.2f}  {lat:<14.2f}  {bg:<10}  {gate}")
PYEOF
done

echo ""
echo "Gate 1a: at least one temp with trajectory_ratio >= 63% AND joint_req_hz >= 10 Hz"
