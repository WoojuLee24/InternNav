#!/usr/bin/env bash
# Phase 1 Task #3 — Replicate trials at the peak temperature
# Runs N trials at the given temp on bag 073623 to estimate variance.
# Usage: docker exec vlnav_internvla_server bash /workspace/InternNav/scripts/realworld/temp_replicates.sh <temp> <n_trials>
set -e
source /opt/ros/jazzy/setup.bash

TEMP="${1:-0.75}"
N="${2:-3}"
BAG="my_camera_bag_20260317_073623"
BAG_PATH="/workspace/rosbag/${BAG}"
CALIB="/workspace/InternNav/scripts/realworld/calib/calib_scout.txt"
TAG=$(echo "$TEMP" | tr '.' '_')
LOG_BASE="/tmp/temp_repl_${TAG}"
mkdir -p "$LOG_BASE"

echo "=================================================="
echo "  TEMP=$TEMP | N=$N trials | bag=$BAG"
echo "=================================================="

for i in $(seq 1 $N); do
    LOG_DIR="$LOG_BASE/trial_$i"
    mkdir -p "$LOG_DIR"
    echo ""
    echo "----- Trial $i / $N -----"

    curl -sf http://localhost:5802/reset_metrics > "$LOG_DIR/reset.json"
    sleep 2

    python3.12 /workspace/InternNav/scripts/realworld/http_internvla_client_debug.py \
        --mode async --kv-cache --temperature "$TEMP" \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        > "$LOG_DIR/client.log" 2>&1 &
    CLIENT_PID=$!
    sleep 3

    echo "  Playing bag at rate=0.5..."
    ros2 bag play "$BAG_PATH" --rate 0.5 > "$LOG_DIR/bag.log" 2>&1
    sleep 2

    curl -sf http://localhost:5802/async_metrics > "$LOG_DIR/metrics.json"
    kill $CLIENT_PID 2>/dev/null || true

    python3 - <<PYEOF
import json
with open("$LOG_DIR/metrics.json") as f: d = json.load(f)
print(f"  trial $i: traj={d.get('trajectory_ratio',0):.1f}%  hz={d.get('joint_req_hz',0):.2f}  bg={d.get('background_s2_runs',0)}")
PYEOF

    sleep 5
done

echo ""
echo "=================================================="
echo "  SUMMARY (TEMP=$TEMP, N=$N)"
echo "=================================================="
python3 - <<PYEOF
import json, statistics
trajs = []
hzs = []
for i in range(1, $N+1):
    with open(f"$LOG_BASE/trial_{i}/metrics.json") as f: d = json.load(f)
    trajs.append(d.get('trajectory_ratio', 0))
    hzs.append(d.get('joint_req_hz', 0))
mean_t = statistics.mean(trajs)
sd_t   = statistics.stdev(trajs) if len(trajs) > 1 else 0.0
mean_h = statistics.mean(hzs)
print(f"  trajectory_ratio per trial: {trajs}")
print(f"  trajectory_ratio mean = {mean_t:.2f}%  sd = {sd_t:.2f} pp")
print(f"  joint_req_hz mean     = {mean_h:.2f} Hz")
n = len($N==1 and [1] or list(range(1)))
# 95% CI (rough, t-dist df=N-1)
if len(trajs) > 1:
    from math import sqrt
    se = sd_t / sqrt(len(trajs))
    # t critical for df=2 (n=3), 95% two-tailed = 4.303
    t = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(len(trajs)-1, 2.0)
    print(f"  95% CI ≈ [{mean_t - t*se:.2f}, {mean_t + t*se:.2f}] %")
gate = mean_t >= 63.0 and mean_h >= 10.0
print(f"  Gate 1a (mean >= 63%, hz >= 10): {'PASS' if gate else 'FAIL'}")
PYEOF
