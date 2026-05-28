#!/usr/bin/env bash
# Phase 3 Task #9 follow-up — 3-bag cross-validation of temporal cache
# Locks threshold=0.92 (sweep winner: 98.9% traj, 98.4% skip on bag 073623)
# Runs control + cached on bags 061841 and 063047
# (bag 073623 already validated in main sweep)
set -e
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
LOG_BASE="/tmp/temporal_3bag"
mkdir -p "$LOG_BASE"

BAGS=("my_camera_bag_20260317_061841" "my_camera_bag_20260317_063047")
THRESHOLDS=(0.0 0.92)

echo "=================================================="
echo " Phase 3 Task #9 — 3-bag cross-validation"
echo "=================================================="

run_one() {
    local bag="$1"
    local thr="$2"
    local tag="${bag#*_}_thr$(echo $thr | tr '.' '_')"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$bag  threshold=$thr ---"

    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=${thr}" > "$log_dir/set.json"
    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 2

    cd "$REPO_DIR" && python3.12 scripts/realworld/http_internvla_client_debug.py \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        > "$log_dir/client.log" 2>&1 &
    local client_pid=$!
    sleep 3

    # echo "  playing /workspace/rosbag/$bag at rate=0.5..."  # gds container
#     ros2 bag play "/workspace/rosbag/$bag" --rate 0.5 > "$log_dir/bag.log" 2>&1  # played manually from gds container
    sleep 2

    curl -sf http://localhost:5802/async_metrics > "$log_dir/metrics.json"
    kill $client_pid 2>/dev/null || true

    python3 - <<PYEOF
import json
with open("$log_dir/metrics.json") as f: d = json.load(f)
print(f"  thr={d.get('temporal_cache_threshold')} skip%={d.get('temporal_cache_skip_ratio',0):.1f} bg={d.get('background_s2_runs',0)} traj={d.get('trajectory_ratio',0):.1f}% hz={d.get('joint_req_hz',0):.2f}")
PYEOF
    sleep 5
}

for bag in "${BAGS[@]}"; do
    for thr in "${THRESHOLDS[@]}"; do
        run_one "$bag" "$thr"
    done
done

curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.0" > /dev/null

echo ""
echo "=================================================="
echo " 3-BAG SUMMARY (temp=0.75, threshold=0.92 vs control)"
echo "=================================================="
printf "  %-10s  %-10s  %-12s  %-10s  %-12s  %-10s\n" \
    "Bag" "Threshold" "skip%" "bg_runs" "traj_ratio" "hz"
printf "  %-10s  %-10s  %-12s  %-10s  %-12s  %-10s\n" \
    "----------" "----------" "------------" "----------" "------------" "----------"
for bag in "${BAGS[@]}"; do
    short="${bag#*_}"
    for thr in "${THRESHOLDS[@]}"; do
        tag="${short}_thr$(echo $thr | tr '.' '_')"
        log_dir="$LOG_BASE/$tag"
        python3 - <<PYEOF
import json
with open("$log_dir/metrics.json") as f: d = json.load(f)
print(f"  {'$short':<10}  {$thr:<10.2f}  {d.get('temporal_cache_skip_ratio',0):<12.1f}  {d.get('background_s2_runs',0):<10}  {d.get('trajectory_ratio',0):<12.1f}  {d.get('joint_req_hz',0):<10.2f}")
PYEOF
    done
done
