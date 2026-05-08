#!/usr/bin/env bash
# Gate 3e — Adaptive max_hold (I-049)
# Compares: full system D (fixed max_hold=10) vs D+I-049 (adaptive max_hold)
# Hypothesis: adaptive max_hold allows I-046 to fire in stable scenes,
# reducing V(A vs D_adaptive) to ≤0.10 on all bags including bag 073623.
#
# Run inside container: docker exec vlnav_internvla_server bash
# Usage: bash scripts/realworld/gate3e_adaptive_maxhold.sh
set -euo pipefail

SERVER="http_internvla_server_debug.py"
CLIENT="http_internvla_client_debug.py"
BAGS=(
    "/workspace/rosbag/my_camera_bag_20260317_073623"
    "/workspace/rosbag/my_camera_bag_20260317_061841"
    "/workspace/rosbag/my_camera_bag_20260317_063047"
)
LOG_BASE="/tmp/gate3e_adaptive"
mkdir -p "$LOG_BASE"

SERVER_ARGS="--mode async --temperature 0.75 --kv-cache --pre-warm-frames 3"
CLIENT_ARGS="--mode async --kv-cache --temperature 0.75 --jpeg-quality 95"
BAG_RATE=0.5
BAG_WAIT=200  # seconds per bag at rate=0.5

start_server() {
    local logfile="$1"
    cd /workspace/InternNav
    source venv/bin/activate 2>/dev/null || true
    python3 scripts/realworld/$SERVER $SERVER_ARGS > "$logfile" 2>&1 &
    SERVER_PID=$!
    echo "  server starting (pid=$SERVER_PID)..."
    until curl -sf http://localhost:5802/async_metrics > /dev/null 2>&1; do sleep 2; done
    echo "  server up (pid=$SERVER_PID)"
}

run_bag() {
    local bag="$1"
    local cond="$2"
    local tag="${bag##*_}_${cond}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=${bag##*_}  cond=$cond ---"
    curl -sf "http://localhost:5802/reset_metrics" > "$log_dir/reset.json"
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=10" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null

    if [ "$cond" = "D" ]; then
        # Fixed max_hold=10, I-046 on (Gate 3b config, same as Gate 3d condition D)
        curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
        echo "  condition=D configured (fixed max_hold=10)"
    elif [ "$cond" = "D_adaptive" ]; then
        # Adaptive max_hold (I-049), I-046 on
        curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=true" > /dev/null
        echo "  condition=D_adaptive configured (adaptive max_hold, I-049 on)"
    fi

    # Play rosbag and capture client output
    cd /workspace/InternNav
    source venv/bin/activate 2>/dev/null || true
    source /opt/ros/jazzy/setup.bash 2>/dev/null || true
    timeout $((BAG_WAIT + 30)) python3.12 scripts/realworld/$CLIENT $CLIENT_ARGS \
        --bag "$bag" --bag-rate $BAG_RATE > "$log_dir/client.log" 2>&1 || true
    echo "  playing $bag at rate=$BAG_RATE..."

    # Fetch metrics
    curl -sf http://localhost:5802/async_metrics > "$log_dir/metrics.json"

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
    print(f"  sim_variance={sim_var:.6f}  (window size={d.get('adaptive_sim_window_size', 0)})")
PYEOF
}

echo "=================================================="
echo " Gate 3e — Adaptive max_hold Study (I-049)"
echo " Conditions: D=fixed(Gate3b)  D_adaptive=adaptive(I-049)"
echo "=================================================="

# Only need one server start for both conditions
start_server "$LOG_BASE/server.log"

for cond in D D_adaptive; do
    echo ""
    echo "=== CONDITION $cond ==="
    for bag in "${BAGS[@]}"; do
        run_bag "$bag" "$cond"
    done
done

kill $SERVER_PID 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3e — ANALYSIS"
echo "=================================================="
for bag in "${BAGS[@]}"; do
    short="${bag##*_}"
    echo ""
    echo "--- bag $short ---"
    python3 /workspace/InternNav/scripts/viz/chi_squared_action_test.py \
        "$LOG_BASE/${short}_D/metrics.json" "$LOG_BASE/${short}_D_adaptive/metrics.json" \
        2>/dev/null || true

    python3 - <<PYEOF
import json, math

def load(cond):
    with open(f"$LOG_BASE/${short}_{cond}/metrics.json") as f:
        return json.load(f)

def bg(d): return d.get("background_s2_runs", 0)
def ft(d): return d.get("fresh_traj_outputs", 0)
def fa(d): return d.get("fresh_action_outputs", 0)
def aa(d): return d.get("action_aware_bypasses", 0)
def mh(d): return d.get("max_hold_bypasses", 0)
def ar(d): n = ft(d)+fa(d); return fa(d)/max(1,n)*100

D = load("D")
DA = load("D_adaptive")

short = "${short}"
print(f"Bag {short}: D(fixed) vs D_adaptive(I-049)")
print(f"  bg_s2:       {bg(D):4d}  vs  {bg(DA):4d}")
print(f"  action_rate: {ar(D):.1f}%  vs  {ar(DA):.1f}%")
print(f"  AA bypasses: {aa(D):4d}  vs  {aa(DA):4d}")
print(f"  MH bypasses: {mh(D):4d}  vs  {mh(DA):4d}")

red = (bg(D) - bg(DA)) / max(1, bg(D)) * 100
print(f"  bg reduction (D_adaptive vs D): {red:.1f}%")
print(f"  => AA>0 in D_adaptive: {'YES (I-046 contributing!)' if aa(DA)>0 else 'NO (I-046 still dormant)'}")
PYEOF
done

echo ""
echo "GATE3E_DONE"
