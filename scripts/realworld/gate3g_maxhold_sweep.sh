#!/usr/bin/env bash
# Gate 3g — max_hold Parameter Sweep
# Hypothesis: with I-050 fix, higher max_hold safely reduces I-047 forced bypasses →
# higher S2 skip% while maintaining V≤0.10.  I-047 accounts for 67-75% of S2 runs
# at max_hold=10; doubling to 20 should push skip from 88% to ~92%.
#
# Sweep: max_hold ∈ {5, 10, 15, 20, 25, 30}
# Fixed:  tau=0.92, action_aware=on, I-050 always active (code-level)
# Pass:   V(A vs MH_x) ≤ 0.10 all 3 bags for each value
#
# Run inside container: docker exec vlnav_internvla_server bash
# Usage:  bash scripts/realworld/gate3g_maxhold_sweep.sh
set -e
source /opt/ros/jazzy/setup.bash

CALIB="/workspace/InternNav/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3g_maxhold"
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)
MH_VALUES=(5 10 15 20 25 30)

echo "=================================================="
echo " Gate 3g — max_hold Sweep (τ=0.92, I-050 active)"
echo " Values: ${MH_VALUES[*]}"
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

configure_mh() {
    local mh="$1"
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=$mh" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    echo "  max_hold=$mh configured"
}

run_bag() {
    local bag="$1"
    local mh="$2"
    local short="${bag##*_}"
    local tag="${short}_MH${mh}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  max_hold=$mh ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1
    configure_mh "$mh"

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
bg   = d.get("background_s2_runs", 0)
ft   = d.get("fresh_traj_outputs", 0)
fa   = d.get("fresh_action_outputs", 0)
aa   = d.get("action_aware_bypasses", 0)
mh_c = d.get("max_hold_bypasses", 0)
skip = d.get("temporal_cache_skip_ratio", 0)
hz   = d.get("joint_req_hz", 0)
n    = ft + fa
ar   = fa / max(1, n) * 100
nat  = bg - aa - mh_c
print(f"  bg={bg}  skip={skip:.1f}%  hz={hz:.2f}")
print(f"  natural={nat}  AA={aa}  MH={mh_c}")
print(f"  action_rate={ar:.1f}%")
PYEOF
}

start_server "$LOG_BASE/server.log"

for mh in "${MH_VALUES[@]}"; do
    echo ""
    echo "=== max_hold=$mh ==="
    for bag in "${BAGS[@]}"; do
        run_bag "$bag" "$mh"
    done
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3g — ANALYSIS"
echo "=================================================="

python3 - <<PYEOF
import json, os

LOG = "/tmp/gate3g_maxhold"
BAGS = ["073623", "061841", "063047"]
MH_VALS = [5, 10, 15, 20, 25, 30]
GATE3D_A = {"073623": "/tmp/gate3d_ablation/073623_A/metrics.json",
            "061841": "/tmp/gate3d_ablation/061841_A/metrics.json",
            "063047": "/tmp/gate3d_ablation/063047_A/metrics.json"}

def load(path):
    with open(path) as f: return json.load(f)

def bg(d):   return d.get("background_s2_runs", 0)
def fa(d):   return d.get("fresh_action_outputs", 0)
def ft(d):   return d.get("fresh_traj_outputs", 0)
def aa(d):   return d.get("action_aware_bypasses", 0)
def mh_c(d): return d.get("max_hold_bypasses", 0)
def skip(d): return d.get("temporal_cache_skip_ratio", 0)
def ar(d):   n=ft(d)+fa(d); return fa(d)/max(1,n)*100

# Per-bag table
for b in BAGS:
    print(f"\n--- Bag {b} ---")
    print(f"{'MH':>4} {'bg_s2':>7} {'skip%':>7} {'AA':>6} {'MH_b':>6} {'nat':>5} {'ar%':>6}")
    for mh in MH_VALS:
        path = f"{LOG}/{b}_MH{mh}/metrics.json"
        if not os.path.exists(path): continue
        d = load(path)
        nat = bg(d) - aa(d) - mh_c(d)
        print(f"{mh:>4} {bg(d):>7} {skip(d):>6.1f}% {aa(d):>6} {mh_c(d):>6} {nat:>5} {ar(d):>5.1f}%")

# Chi-squared V values
print("\n--- Cramér's V (vs Gate-3d Condition A) ---")
print(f"{'MH':>4} {'073623':>10} {'061841':>10} {'063047':>10} {'worst':>10}")
for mh in MH_VALS:
    vs = []
    row = [mh]
    for b in BAGS:
        path = f"{LOG}/{b}_MH{mh}/metrics.json"
        a_path = GATE3D_A.get(b, "")
        if not (os.path.exists(path) and os.path.exists(a_path)):
            vs.append(None); continue
        d = load(path)
        da = load(a_path)
        # 2x2: traj vs action, cached(MHx) vs control(A)
        ft_a = ft(da); fa_a = fa(da); ft_m = ft(d); fa_m = fa(d)
        n_a = ft_a + fa_a; n_m = ft_m + fa_m
        if n_a == 0 or n_m == 0: vs.append(None); continue
        import math
        n = n_a + n_m
        obs = [[ft_a, fa_a], [ft_m, fa_m]]
        row_sum = [n_a, n_m]; col_sum = [ft_a+ft_m, fa_a+fa_m]
        chi2 = 0
        for i in range(2):
            for j in range(2):
                e = row_sum[i]*col_sum[j]/n
                chi2 += (obs[i][j]-e)**2/max(e,1e-9)
        V = math.sqrt(chi2/n)
        vs.append(V)
    vstr = [f"{v:.4f}" if v is not None else "  N/A" for v in vs]
    worst = max((v for v in vs if v is not None), default=None)
    wstr = f"{worst:.4f}" if worst is not None else " N/A"
    status = "PASS" if worst is not None and worst <= 0.10 else "FAIL"
    print(f"{mh:>4}  {'  '.join(vstr)}  {wstr:>8}  {status}")
PYEOF

echo ""
echo "GATE3G_DONE"
