#!/usr/bin/env bash
# Phase 3 Task #9 follow-up — fresh-only metric validation
# Single bag (073623), thr in {0.0, 0.92}. Compares served traj_ratio vs
# fresh_trajectory_ratio to confirm/reject the measurement-artifact hypothesis.
set -e
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

BAG="my_camera_bag_20260317_073623"
# BAG_PATH: rosbag played manually from gds container via FastDDS
CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
LOG_BASE="/tmp/temporal_freshcheck"
mkdir -p "$LOG_BASE"

THRESHOLDS=(0.0 0.92)

echo "=================================================="
echo " Phase 3 Task #9 — fresh-only metric check"
echo " Bag: $BAG | Temp: 0.75"
echo "=================================================="

for T in "${THRESHOLDS[@]}"; do
    TAG=$(echo "$T" | tr '.' '_')
    LOG_DIR="$LOG_BASE/thr_${TAG}"
    mkdir -p "$LOG_DIR"

    echo ""
    echo "--- threshold = $T ---"

    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=${T}" > "$LOG_DIR/set.json"
    curl -sf http://localhost:5802/reset_metrics > "$LOG_DIR/reset.json"
    sleep 2

    cd "$REPO_DIR" && python3.12 scripts/realworld/http_internvla_client_debug.py \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        > "$LOG_DIR/client.log" 2>&1 &
    CLIENT_PID=$!
    sleep 3

    echo "  playing bag at rate=0.5..."
#     ros2 bag play "$BAG_PATH" --rate 0.5 > "$LOG_DIR/bag.log" 2>&1  # played manually from gds container
    sleep 2

    curl -sf http://localhost:5802/async_metrics > "$LOG_DIR/metrics.json"
    kill $CLIENT_PID 2>/dev/null || true

    python3 - <<PYEOF
import json
with open("$LOG_DIR/metrics.json") as f: d = json.load(f)
print(f"  threshold              : {d.get('temporal_cache_threshold')}")
print(f"  bg_runs (fresh S2)     : {d.get('background_s2_runs',0)}")
print(f"  fresh_traj_outputs     : {d.get('fresh_traj_outputs',0)}")
print(f"  fresh_action_outputs   : {d.get('fresh_action_outputs',0)}")
print(f"  fresh_trajectory_ratio : {d.get('fresh_trajectory_ratio',0):.1f}%   (over fresh S2 only)")
print(f"  served trajectory_ratio: {d.get('trajectory_ratio',0):.1f}%   (over all HTTP responses)")
print(f"  skip_ratio             : {d.get('temporal_cache_skip_ratio',0):.1f}%")
print(f"  joint_req_hz           : {d.get('joint_req_hz',0):.2f} Hz")
print(f"  total_requests         : {d.get('total_requests',0)}")
PYEOF
    sleep 5
done

curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.0" > /dev/null

echo ""
echo "=================================================="
echo " VERDICT GUIDE"
echo "=================================================="
echo "  If fresh_trajectory_ratio at thr=0.92 is ~equal to control's"
echo "  fresh_trajectory_ratio, the cache replay is unbiased and the"
echo "  98.9% served traj_ratio is purely a hold-time artifact."
echo "  If fresh ratio at 0.92 is much higher than control, cache is"
echo "  selecting a quality-skewed subset of frames (still suspicious)."
