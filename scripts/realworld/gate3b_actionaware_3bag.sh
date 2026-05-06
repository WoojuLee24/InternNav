#!/usr/bin/env bash
# Gate 3b RETRY — action-aware temporal cache (I-046 + I-047)
# Conditions per bag:
#   1. control       : thr=0.0   (no cache)
#   2. action-aware  : thr=0.92  max_hold=10   (I-046 + I-047 active)
# Pass criteria (per bag):
#   * chi2 p-value >= 0.05 between control and action-aware fresh action dist (I-110)
#   * fresh_action_rate(test) >= 0.5 * fresh_action_rate(control)
#   * S2 reduction >= 40%  (bg_runs(test) <= 0.6 * bg_runs(control))
#   * joint_req_hz(test) >= 11.0
set -e
source /opt/ros/jazzy/setup.bash

CALIB="/workspace/InternNav/scripts/realworld/calib/calib_scout.txt"
LOG_BASE="/tmp/gate3b_aa"
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)

# Each condition is "name|threshold|max_hold"
CONDITIONS=(
    "ctrl|0.0|0"
    "aa|0.92|10"
)

echo "=================================================="
echo " Gate 3b RETRY — action-aware temporal cache"
echo " I-046 (action-conditional bypass) + I-047 (max-hold)"
echo "=================================================="

run_one() {
    local bag="$1"
    local condition="$2"
    local name="${condition%%|*}"
    local rest="${condition#*|}"
    local thr="${rest%%|*}"
    local mh="${rest##*|}"
    local short="${bag##*_}"
    local tag="${short}_${name}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  cond=$name  thr=$thr  max_hold=$mh ---"

    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=${thr}" > "$log_dir/set_thr.json"
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=${mh}" > "$log_dir/set_mh.json"
    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 2

    python3.12 /workspace/InternNav/scripts/realworld/http_internvla_client_debug.py \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
        > "$log_dir/client.log" 2>&1 &
    local client_pid=$!
    sleep 3

    echo "  playing /workspace/rosbag/$bag at rate=0.5..."
    ros2 bag play "/workspace/rosbag/$bag" --rate 0.5 > "$log_dir/bag.log" 2>&1
    sleep 2

    curl -sf http://localhost:5802/async_metrics > "$log_dir/metrics.json"
    kill $client_pid 2>/dev/null || true

    python3 - <<PYEOF
import json
with open("$log_dir/metrics.json") as f: d = json.load(f)
ft = d.get("fresh_traj_outputs", 0)
fa = d.get("fresh_action_outputs", 0)
nf = ft + fa
print(f"  bg={d.get('background_s2_runs',0)}  skip%={d.get('temporal_cache_skip_ratio',0):.1f}  AA={d.get('action_aware_bypasses',0)}  MH={d.get('max_hold_bypasses',0)}")
print(f"  fresh: traj={ft}  action={fa}  n={nf}  action_rate={(fa/max(1,nf))*100:.1f}%")
print(f"  served: traj_ratio={d.get('trajectory_ratio',0):.1f}%  hz={d.get('joint_req_hz',0):.2f}")
PYEOF
    sleep 5
}

for bag in "${BAGS[@]}"; do
    for cond in "${CONDITIONS[@]}"; do
        run_one "$bag" "$cond"
    done
done

# Reset cache off
curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.0" > /dev/null
curl -sf "http://localhost:5802/set_max_hold_frames?frames=10" > /dev/null

echo ""
echo "=================================================="
echo " GATE 3b RETRY — PASS/FAIL ANALYSIS PER BAG"
echo "=================================================="
for bag in "${BAGS[@]}"; do
    short="${bag##*_}"
    ctrl="$LOG_BASE/${short}_ctrl/metrics.json"
    test_="$LOG_BASE/${short}_aa/metrics.json"
    echo ""
    echo "--- bag $short ---"
    python3 /workspace/InternNav/scripts/viz/chi_squared_action_test.py "$ctrl" "$test_" || true

    python3 - <<PYEOF
import json
with open("$ctrl") as f: c = json.load(f)
with open("$test_") as f: t = json.load(f)
c_bg = c.get("background_s2_runs", 0)
t_bg = t.get("background_s2_runs", 0)
c_act = c.get("fresh_action_outputs", 0)
c_tot = c_act + c.get("fresh_traj_outputs", 0)
t_act = t.get("fresh_action_outputs", 0)
t_tot = t_act + t.get("fresh_traj_outputs", 0)
c_rate = c_act / max(1, c_tot)
t_rate = t_act / max(1, t_tot)
hz = t.get("joint_req_hz", 0)
print(f"")
print(f"AUX checks:")
print(f"  S2 reduction: {(1 - t_bg/max(1,c_bg))*100:.1f}%  (need >= 40%)  {'PASS' if t_bg <= 0.6*c_bg else 'FAIL'}")
print(f"  action_rate retained: {(t_rate/max(1e-9,c_rate))*100:.1f}% of control  (need >= 50%)  {'PASS' if t_rate >= 0.5*c_rate else 'FAIL'}")
print(f"  hz: {hz:.2f}  (need >= 11.0)  {'PASS' if hz >= 11.0 else 'FAIL'}")
PYEOF
done
