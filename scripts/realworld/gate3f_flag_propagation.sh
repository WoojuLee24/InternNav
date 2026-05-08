#!/usr/bin/env bash
# Gate 3f — I-046/I-047 Flag Propagation Fix (I-050)
# Hypothesis: after I-047 forced bypass, propagate fresh_was_action instead of
# resetting to False. This lets I-046 fire on the next frame when I-047 produced
# an action output — fixing AA=0 on stable bag 073623 without adding S2 load.
#
# Compares: D (Gate 3d config, one-shot resets on ALL bypasses)
#       vs  D_3f (I-050 fix: only reset on I-046 bypass)
#
# Run inside container: docker exec vlnav_internvla_server bash
# Usage: bash scripts/realworld/gate3f_flag_propagation.sh
set -e
source /opt/ros/jazzy/setup.bash

CALIB="/workspace/InternNav/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3f_flag"
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)

echo "=================================================="
echo " Gate 3f — I-046/I-047 Flag Propagation Fix (I-050)"
echo " Conditions: D=old(reset on any bypass)  D_3f=fix(only reset on I-046)"
echo "=================================================="

start_server() {
    local logfile="$1"
    pkill -f "$SERVER" 2>/dev/null || true
    sleep 3

    (cd /workspace/InternNav && python3 $SERVER \
        --mode async --temperature 0.75 --kv-cache \
        --calib "$CALIB" \
        --pre-warm-frames 3) \
        > "$logfile" 2>&1 &
    SERVER_PID=$!

    until curl -sf http://localhost:5802/async_metrics > /dev/null 2>&1; do sleep 2; done
    echo "  server up (pid=$SERVER_PID)"
}

configure_D() {
    # Gate 3d condition D: threshold=0.92, max_hold=10, action_aware=on, adaptive=off
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=10" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    echo "  condition=D (Gate3d baseline, reset on all bypasses)"
}

configure_D3f() {
    # Same config as D — the fix is in the server code itself (I-050)
    # I-050 is always active when server runs with this code version
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=10" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    echo "  condition=D_3f (I-050 fix: I-047 propagates fresh_was_action)"
}

run_bag() {
    local bag="$1"
    local cond="$2"
    local short="${bag##*_}"
    local tag="${short}_${cond}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  cond=$cond ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1

    if [ "$cond" = "D" ]; then
        configure_D
    else
        configure_D3f
    fi

    (cd /workspace/InternNav && python3.12 $CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB") \
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
bg = d.get("background_s2_runs", 0)
ft = d.get("fresh_traj_outputs", 0)
fa = d.get("fresh_action_outputs", 0)
aa = d.get("action_aware_bypasses", 0)
mh = d.get("max_hold_bypasses", 0)
skip = d.get("temporal_cache_skip_ratio", 0)
hz = d.get("joint_req_hz", 0)
n = ft + fa
ar = fa / max(1, n) * 100
print(f"  bg={bg}  skip%={skip:.1f}  hz={hz:.2f}")
print(f"  fresh: traj={ft}  action={fa}  n={n}  action_rate={ar:.1f}%")
print(f"  bypasses: AA={aa}  MH={mh}")
PYEOF
}

# Single server — both conditions use same code (I-050 is always in this binary)
# D condition re-runs to confirm baseline consistency; D_3f shows the fix effect
start_server "$LOG_BASE/server.log"

for cond in D D_3f; do
    echo ""
    echo "=== CONDITION $cond ==="
    for bag in "${BAGS[@]}"; do
        run_bag "$bag" "$cond"
    done
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3f — ANALYSIS"
echo "=================================================="
for bag in "${BAGS[@]}"; do
    short="${bag##*_}"
    echo ""
    echo "--- bag $short ---"
    python3 /workspace/InternNav/scripts/viz/chi_squared_action_test.py \
        "/tmp/gate3d_ablation/${short}_A/metrics.json" \
        "$LOG_BASE/${short}_D_3f/metrics.json" \
        2>/dev/null || true

    python3 - <<PYEOF
import json

short_name = "$short"
log_base = "$LOG_BASE"

def load(tag):
    with open(f"{log_base}/{short_name}_{tag}/metrics.json") as f:
        return json.load(f)

def bg(d): return d.get("background_s2_runs", 0)
def ft(d): return d.get("fresh_traj_outputs", 0)
def fa(d): return d.get("fresh_action_outputs", 0)
def aa(d): return d.get("action_aware_bypasses", 0)
def mh(d): return d.get("max_hold_bypasses", 0)
def ar(d): n = ft(d)+fa(d); return fa(d)/max(1,n)*100

D = load("D")
D3f = load("D_3f")

n_D   = ft(D)   + fa(D)
n_D3f = ft(D3f) + fa(D3f)

print(f"\nBag {short_name}: D(old) vs D_3f(I-050 fix)")
print(f"  bg_s2:       {bg(D):5d}  vs  {bg(D3f):5d}")
print(f"  skip%:       {D.get('temporal_cache_skip_ratio',0):.1f}%  vs  {D3f.get('temporal_cache_skip_ratio',0):.1f}%")
print(f"  action_rate: {ar(D):.1f}%   vs  {ar(D3f):.1f}%")
print(f"  AA bypasses: {aa(D):5d}  vs  {aa(D3f):5d}  (I-046)")
print(f"  MH bypasses: {mh(D):5d}  vs  {mh(D3f):5d}  (I-047)")
s2_delta = (bg(D3f) - bg(D)) / max(1, bg(D)) * 100
print(f"  bg change (D_3f vs D): {s2_delta:+.1f}%")
print(f"  I-046 active in D_3f: {'YES' if aa(D3f) > 0 else 'NO'}")
PYEOF
done

echo ""
echo "GATE3F_DONE"
