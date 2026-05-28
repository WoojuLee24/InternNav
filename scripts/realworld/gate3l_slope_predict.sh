#!/usr/bin/env bash
# Gate 3l — Similarity Slope Predictive Refresh (I-054)
# Hypothesis: tracking d(sim)/dt (temporal derivative of the MAD similarity score)
# enables predictive cache invalidation — firing a pre-emptive S2 run when the
# similarity is declining rapidly, BEFORE it crosses the threshold τ.
#
# Current system (I-046/I-047/I-050): all mechanisms are reactive.
#   • I-046: fires AFTER an action output (after the decision point)
#   • I-047: fires on a fixed period (may or may not align with decisions)
#   • τ gate: fires AFTER similarity drops below threshold
#
# I-054 is the first PREDICTIVE mechanism:
#   • If sim >= τ BUT d(sim)/dt < -slope_threshold → pre-emptive S2 run
#   • This "early warning" catches doorway transitions and turns 2-3 frames
#     before the τ gate would fire, reducing stale-cache exposure at decisions.
#
# Expected effect: fewer stale outputs at transition frames → better V.
# Possible tradeoff: more bypasses → lower skip%.
#
# Sweep: slope_threshold ∈ {0.005, 0.010, 0.015, 0.020, 0.030}
#   Very small threshold → fires very often (many predictive bypasses, ↓ skip%)
#   Large threshold → fires rarely (only on steep drops)
# Fixed: max_hold=15, tau=0.92, action_aware=on, window=3 frames
#
# Run inside container: docker exec -it vlnav_internvla_server bash
# Usage:  bash scripts/realworld/gate3l_slope_predict.sh
set -e
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3l_slope"
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)
SLOPE_VALUES=(0.005 0.010 0.015 0.020 0.030)

echo "=================================================="
echo " Gate 3l — I-054 Slope Predictive Refresh Sweep"
echo " Thresholds: ${SLOPE_VALUES[*]}"
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

configure_slope() {
    local slope="$1"
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_trajectory_adaptive_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_serve_count_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_ema_fingerprint?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_slope_predict?enabled=true&threshold=$slope&window=3" > /dev/null
    echo "  slope_threshold=$slope, window=3 configured"
}

run_bag() {
    local bag="$1"
    local slope="$2"
    local short="${bag#*_}"
    local slope_tag="${slope/./_}"
    local tag="${short}_SL${slope_tag}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  slope=$slope ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1
    configure_slope "$slope"

    (cd "$REPO_DIR" && python3.12 $CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB") \
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
bg   = d.get("background_s2_runs", 0)
ft   = d.get("fresh_traj_outputs", 0)
fa   = d.get("fresh_action_outputs", 0)
aa   = d.get("action_aware_bypasses", 0)
mh_c = d.get("max_hold_bypasses", 0)
sp_c = d.get("slope_predict_bypasses", 0)
skip = d.get("temporal_cache_skip_ratio", 0)
hz   = d.get("joint_req_hz", 0)
n    = ft + fa
ar   = fa / max(1, n) * 100
nat  = bg - aa - mh_c - sp_c
print(f"  bg={bg}  skip={skip:.1f}%  hz={hz:.2f}")
print(f"  natural={nat}  AA={aa}  MH={mh_c}  SP={sp_c}")
print(f"  action_rate={ar:.1f}%")
PYEOF
}

run_baseline() {
    local bag="$1"
    local short="${bag#*_}"
    local log_dir="$LOG_BASE/${short}_BASELINE"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  BASELINE (MH=15, I-054=off) ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_slope_predict?enabled=false" > /dev/null

    (cd "$REPO_DIR" && python3.12 $CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB") \
        > "$log_dir/client.log" 2>&1 &
    local client_pid=$!
    sleep 3
# 
#     ros2 bag play "/workspace/rosbag/$bag" --rate 0.5 > "$log_dir/bag.log" 2>&1  # played manually from gds container
    sleep 2

    curl -sf http://localhost:5802/async_metrics > "$log_dir/metrics.json"
    kill $client_pid 2>/dev/null || true

    python3 - <<PYEOF
import json
with open("$log_dir/metrics.json") as f: d = json.load(f)
skip = d.get("temporal_cache_skip_ratio", 0)
hz   = d.get("joint_req_hz", 0)
bg   = d.get("background_s2_runs", 0)
ft   = d.get("fresh_traj_outputs", 0)
fa   = d.get("fresh_action_outputs", 0)
aa   = d.get("action_aware_bypasses", 0)
mh_c = d.get("max_hold_bypasses", 0)
n    = ft + fa; ar = fa / max(1, n) * 100
print(f"  BASELINE: bg={bg}  skip={skip:.1f}%  hz={hz:.2f}  AA={aa}  MH={mh_c}  ar={ar:.1f}%")
PYEOF
}

start_server "$LOG_BASE/server.log"

echo ""
echo "=== BASELINE (Gate 3g: MH=15, I-054 off) ==="
for bag in "${BAGS[@]}"; do
    run_baseline "$bag"
done

for slope in "${SLOPE_VALUES[@]}"; do
    echo ""
    echo "=== slope_threshold=$slope ==="
    for bag in "${BAGS[@]}"; do
        run_bag "$bag" "$slope"
    done
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3l — ANALYSIS"
echo "=================================================="

python3 - <<PYEOF
import json, os, math

LOG = "/tmp/gate3l_slope"
BAGS = ["073623", "061841", "063047"]
SLOPES = ["0.005", "0.010", "0.015", "0.020", "0.030"]

def load(path):
    with open(path) as f: return json.load(f)

def bg(d):   return d.get("background_s2_runs", 0)
def fa(d):   return d.get("fresh_action_outputs", 0)
def ft(d):   return d.get("fresh_traj_outputs", 0)
def aa(d):   return d.get("action_aware_bypasses", 0)
def mh_c(d): return d.get("max_hold_bypasses", 0)
def sp_c(d): return d.get("slope_predict_bypasses", 0)
def skip(d): return d.get("temporal_cache_skip_ratio", 0)

# Per-bag table
for b in BAGS:
    print(f"\n--- Bag {b} ---")
    print(f"{'cond':>12} {'bg':>7} {'skip%':>7} {'AA':>6} {'MH':>6} {'SP':>6} {'ar%':>6}")
    base_path = f"{LOG}/{b}_BASELINE/metrics.json"
    if os.path.exists(base_path):
        d = load(base_path)
        n = ft(d) + fa(d); ar = fa(d) / max(1, n) * 100
        print(f"{'BASELINE':>12} {bg(d):>7} {skip(d):>6.1f}% {aa(d):>6} {mh_c(d):>6} {'N/A':>6} {ar:>5.1f}%")
    for sl in SLOPES:
        sl_tag = sl.replace('.', '_')
        path = f"{LOG}/{b}_SL{sl_tag}/metrics.json"
        if not os.path.exists(path): continue
        d = load(path)
        n = ft(d) + fa(d); ar = fa(d) / max(1, n) * 100
        print(f"{'SL='+sl:>12} {bg(d):>7} {skip(d):>6.1f}% {aa(d):>6} {mh_c(d):>6} {sp_c(d):>6} {ar:>5.1f}%")

# Cramér's V vs BASELINE
print("\n--- Cramér's V (vs Gate-3l BASELINE) ---")
print(f"{'slope':>10} {'073623':>10} {'061841':>10} {'063047':>10} {'worst':>10}")
for sl in SLOPES:
    sl_tag = sl.replace('.', '_')
    vs = []
    for b in BAGS:
        path = f"{LOG}/{b}_SL{sl_tag}/metrics.json"
        base_path = f"{LOG}/{b}_BASELINE/metrics.json"
        if not (os.path.exists(path) and os.path.exists(base_path)):
            vs.append(None); continue
        d = load(path); da = load(base_path)
        ft_a = ft(da); fa_a = fa(da); ft_c = ft(d); fa_c = fa(d)
        n_a = ft_a + fa_a; n_c = ft_c + fa_c
        if n_a == 0 or n_c == 0: vs.append(None); continue
        n = n_a + n_c
        obs = [[ft_a, fa_a], [ft_c, fa_c]]
        rs = [n_a, n_c]; cs = [ft_a+ft_c, fa_a+fa_c]
        chi2 = sum((obs[i][j] - rs[i]*cs[j]/n)**2 / max(rs[i]*cs[j]/n, 1e-9)
                   for i in range(2) for j in range(2))
        vs.append(math.sqrt(chi2 / n))
    vstr = [f"{v:.4f}" if v is not None else "  N/A" for v in vs]
    worst = max((v for v in vs if v is not None), default=None)
    wstr = f"{worst:.4f}" if worst is not None else "  N/A"
    status = "PASS" if worst is not None and worst <= 0.10 else "FAIL"
    print(f"{'SL='+sl:>10}  {'  '.join(vstr)}  {wstr:>8}  {status}")
PYEOF

echo ""
echo "GATE3L_DONE"
