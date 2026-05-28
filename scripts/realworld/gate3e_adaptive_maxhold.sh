#!/usr/bin/env bash
# Gate 3e — Adaptive max_hold (I-049)
# Compares: D (fixed max_hold=10, Gate 3b config) vs D_adaptive (I-049 adaptive max_hold)
# Hypothesis: adaptive max_hold allows I-046 to fire in stable scenes,
# reducing V(A vs D_adaptive) to ≤0.10 on all bags including bag 073623.
#
# Run inside container: docker exec vlnav_internvla_server bash
# Usage: bash scripts/realworld/gate3e_adaptive_maxhold.sh
set -e
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3e_adaptive"
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)

echo "=================================================="
echo " Gate 3e — Adaptive max_hold Study (I-049)"
echo " Conditions: D=fixed(Gate3b)  D_adaptive=adaptive(I-049)"
echo "=================================================="

start_server() {
    local logfile="$1"
    pkill -f "$SERVER" 2>/dev/null || true
    sleep 3

    (cd "$REPO_DIR" && python3 $SERVER \
        --mode async --temperature 0.75 --kv-cache \
        --calib "$CALIB" \
        --pre-warm-frames 3) \
        > "$logfile" 2>&1 &
    SERVER_PID=$!

    until curl -sf http://localhost:5802/async_metrics > /dev/null 2>&1; do sleep 2; done
    echo "  server up (pid=$SERVER_PID)"
}

configure_condition() {
    local cond="$1"
    # All conditions: threshold=0.92, action_aware=on
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=10" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null

    if [ "$cond" = "D" ]; then
        # Fixed max_hold=10 (Gate 3b config)
        curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    elif [ "$cond" = "D_adaptive" ]; then
        # Adaptive max_hold (I-049)
        curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=true" > /dev/null
    fi
    echo "  condition=$cond configured"
}

run_bag() {
    local bag="$1"
    local cond="$2"
    local short="${bag#*_}"
    local tag="${short}_${cond}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  cond=$cond ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1
    configure_condition "$cond"

    # Start client in background
    (cd "$REPO_DIR" && python3.12 $CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB") \
        > "$log_dir/client.log" 2>&1 &
    local client_pid=$!
    sleep 3

    # echo "  playing /workspace/rosbag/$bag at rate=0.5..."  # gds container
#     ros2 bag play "/workspace/rosbag/$bag" --rate 0.5 > "$log_dir/bag.log" 2>&1  # played manually from gds container
    sleep 2

    # Fetch metrics before killing client
    curl -sf http://localhost:5802/async_metrics > "$log_dir/metrics.json"
    kill $client_pid 2>/dev/null || true

    # Parse and print summary
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
aw = d.get("adaptive_max_hold_enabled", False)
sim_var = d.get("adaptive_sim_variance")
n = ft + fa
ar = fa / max(1, n) * 100
print(f"  bg={bg}  skip%={skip:.1f}  hz={hz:.2f}")
print(f"  fresh: traj={ft}  action={fa}  n={n}  action_rate={ar:.1f}%")
print(f"  bypasses: AA={aa}  MH={mh}  (adaptive={'on' if aw else 'off'})")
if sim_var is not None:
    print(f"  sim_variance={sim_var:.6f}  window={d.get('adaptive_sim_window_size', 0)}")
PYEOF
}

# Single server for both conditions (runtime-mutable via endpoints)
start_server "$LOG_BASE/server.log"

for cond in D D_adaptive; do
    echo ""
    echo "=== CONDITION $cond ==="
    for bag in "${BAGS[@]}"; do
        run_bag "$bag" "$cond"
    done
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3e — ANALYSIS"
echo "=================================================="
for bag in "${BAGS[@]}"; do
    short="${bag#*_}"
    echo ""
    echo "--- bag $short ---"
    cd "$REPO_DIR" && python3 scripts/viz/chi_squared_action_test.py \
        "/tmp/gate3d_ablation/${short}_A/metrics.json" \
        "$LOG_BASE/${short}_D_adaptive/metrics.json" \
        2>/dev/null || true

    python3 - <<PYEOF
import json, math

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
DA = load("D_adaptive")

print(f"\nBag {short_name}: D(fixed max_hold=10) vs D_adaptive(I-049)")
print(f"  bg_s2:       {bg(D):5d}  vs  {bg(DA):5d}")
print(f"  skip%:       {D.get('temporal_cache_skip_ratio',0):.1f}%  vs  {DA.get('temporal_cache_skip_ratio',0):.1f}%")
print(f"  action_rate: {ar(D):.1f}%   vs  {ar(DA):.1f}%")
print(f"  AA bypasses: {aa(D):5d}  vs  {aa(DA):5d}")
print(f"  MH bypasses: {mh(D):5d}  vs  {mh(DA):5d}")
sim_var = DA.get('adaptive_sim_variance')
if sim_var is not None:
    print(f"  sim_variance (adaptive): {sim_var:.6f}")

s2_red = (bg(D) - bg(DA)) / max(1, bg(D)) * 100
print(f"  bg reduction (D_adaptive vs D): {s2_red:.1f}%")
print(f"  I-046 active in D_adaptive: {'YES' if aa(DA) > 0 else 'NO'}")
PYEOF
done

echo ""
echo "GATE3E_DONE"
