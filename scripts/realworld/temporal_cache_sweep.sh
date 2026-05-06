#!/usr/bin/env bash
# Phase 3 Task #9 — Temporal S2 cache threshold sweep
# Bag: 073623 | Temp: 0.75 (Phase 1 baseline)
# Thresholds: 0.0 (control), 0.85, 0.90, 0.95, 0.97
set -e
source /opt/ros/jazzy/setup.bash

BAG="my_camera_bag_20260317_073623"
BAG_PATH="/workspace/rosbag/${BAG}"
CALIB="/workspace/InternNav/scripts/realworld/calib/calib_scout.txt"
LOG_BASE="/tmp/temporal_sweep"
mkdir -p "$LOG_BASE"

THRESHOLDS=(0.0 0.92 0.95 0.97 0.99)

echo "=================================================="
echo " Phase 3 Task #9 — Temporal S2 cache threshold sweep"
echo " Bag: $BAG | Temp: 0.75 (Phase 1 baseline)"
echo "=================================================="

for T in "${THRESHOLDS[@]}"; do
    TAG=$(echo "$T" | tr '.' '_')
    LOG_DIR="$LOG_BASE/thr_${TAG}"
    mkdir -p "$LOG_DIR"

    echo ""
    echo "--- threshold = $T ---"

    # Set threshold
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=${T}" > "$LOG_DIR/set.json"
    echo "  set: $(cat $LOG_DIR/set.json)"

    # Reset metrics
    curl -sf http://localhost:5802/reset_metrics > "$LOG_DIR/reset.json"
    sleep 2

    # Start client
    python3.12 /workspace/InternNav/scripts/realworld/http_internvla_client_debug.py \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        > "$LOG_DIR/client.log" 2>&1 &
    CLIENT_PID=$!
    sleep 3

    # Play bag at rate=0.5
    echo "  playing bag at rate=0.5..."
    ros2 bag play "$BAG_PATH" --rate 0.5 > "$LOG_DIR/bag.log" 2>&1
    sleep 2

    # Collect metrics
    curl -sf http://localhost:5802/async_metrics > "$LOG_DIR/metrics.json"
    kill $CLIENT_PID 2>/dev/null || true

    python3 - <<PYEOF
import json
with open("$LOG_DIR/metrics.json") as f: d = json.load(f)
print(f"  threshold        : {d.get('temporal_cache_threshold')}")
print(f"  trajectory_ratio : {d.get('trajectory_ratio',0):.1f}%")
print(f"  joint_req_hz     : {d.get('joint_req_hz',0):.2f} Hz")
print(f"  joint_latency_ms : {d.get('joint_latency_ms',0):.2f} ms")
print(f"  bg_s2_runs       : {d.get('background_s2_runs',0)}")
print(f"  cache_skips      : {d.get('temporal_cache_skips',0)}")
print(f"  skip_ratio       : {d.get('temporal_cache_skip_ratio',0):.1f}%")
print(f"  total_requests   : {d.get('total_requests',0)}")
PYEOF

    sleep 5
done

# Reset to 0.0 after sweep
curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.0" > /dev/null

echo ""
echo "=================================================="
echo " SUMMARY (Bag 073623, temp=0.75)"
echo "=================================================="
printf "  %-10s  %-12s  %-10s  %-12s  %-10s  %-10s\n" \
    "Threshold" "skip_ratio" "bg_runs" "traj_ratio" "hz" "latency"
printf "  %-10s  %-12s  %-10s  %-12s  %-10s  %-10s\n" \
    "----------" "------------" "----------" "------------" "----------" "----------"
for T in "${THRESHOLDS[@]}"; do
    TAG=$(echo "$T" | tr '.' '_')
    LOG_DIR="$LOG_BASE/thr_${TAG}"
    python3 - <<PYEOF
import json
with open("$LOG_DIR/metrics.json") as f: d = json.load(f)
sk = d.get('temporal_cache_skip_ratio',0)
bg = d.get('background_s2_runs',0)
tr = d.get('trajectory_ratio',0)
hz = d.get('joint_req_hz',0)
lat = d.get('joint_latency_ms',0)
print(f"  {$T:<10.2f}  {sk:<12.1f}  {bg:<10}  {tr:<12.1f}  {hz:<10.2f}  {lat:<10.3f}")
PYEOF
done

echo ""
echo "GATE 3b conditions:"
echo "  S2 invocations >= 20% fewer (bg_runs <= 0.8 * baseline_runs ~= 377)"
echo "  trajectory_ratio within 5pp of Phase 1 baseline (>=56.5%)"
echo "  Document one threshold value for deployment"
