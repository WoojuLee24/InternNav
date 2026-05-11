#!/usr/bin/env bash
# Gate 3n — Full Production Stack Validation
#
# Validates that all mechanisms work correctly together:
#   MH=15 (I-047), τ=0.92, action-aware (I-046), TR-EMA α=0.10 (I-055),
#   slope δ_s=0.010 w=3 (I-054)
#
# Baseline (Condition A): no cache (τ=0.0, MH=1, all mechanisms off)
# Test     (Condition P): full production stack
#
# Pass: V(P vs A) ≤ 0.10 all 3 bags
# Key metric: production skip% and V with FULL stack
#
# Run inside container:
#   docker exec -it vlnav_internvla_server bash
#   bash scripts/realworld/gate3n_production_stack.sh
set -e
source /opt/ros/jazzy/setup.bash

CALIB="/workspace/InternNav/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3n_prod"
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)

echo "=================================================="
echo " Gate 3n — Full Production Stack Validation"
echo " Stack: MH=15, τ=0.92, AA, TR-EMA α=0.10, slope δ=0.010"
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

configure_nocache() {
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.0" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=1" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_trajectory_adaptive_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_serve_count_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_ema_fingerprint?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_slope_predict?enabled=false" > /dev/null
    echo "  no-cache configured (τ=0.0)"
}

configure_production() {
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_trajectory_adaptive_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_serve_count_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true" > /dev/null
    curl -sf "http://localhost:5802/set_slope_predict?enabled=true&threshold=0.010&window=3" > /dev/null
    echo "  production stack configured (MH=15, τ=0.92, AA, TR-EMA α=0.10, slope δ=0.010)"
}

run_bag() {
    local bag="$1"
    local tag="$2"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"
    local short="${bag##*_}"

    echo ""
    echo "--- bag=$short  config=$tag ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1

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
mh   = d.get("max_hold_bypasses", 0)
sp   = d.get("slope_predict_bypasses", 0)
skip = d.get("temporal_cache_skip_ratio", 0)
hz   = d.get("joint_req_hz", 0)
n    = ft + fa
ar   = fa / max(1, n) * 100
nat  = bg - aa - mh - sp
print(f"  bg={bg}  skip={skip:.1f}%  hz={hz:.2f}")
print(f"  natural={nat}  AA={aa}  MH={mh}  SP={sp}")
print(f"  action_rate={ar:.1f}%  (ft={ft} fa={fa})")
PYEOF
}

start_server "$LOG_BASE/server.log"

echo ""
echo "=== CONDITION A: No-Cache Baseline ==="
configure_nocache
for bag in "${BAGS[@]}"; do
    short="${bag##*_}"
    run_bag "$bag" "${short}_NOCACHE"
done

echo ""
echo "=== CONDITION P: Full Production Stack ==="
configure_production
for bag in "${BAGS[@]}"; do
    short="${bag##*_}"
    run_bag "$bag" "${short}_PROD"
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3n — ANALYSIS"
echo "=================================================="

python3 - <<PYEOF
import json, os, math

LOG = "/tmp/gate3n_prod"
BAGS = ["073623", "061841", "063047"]

def load(path):
    with open(path) as f: return json.load(f)

def cramers_v(d, da):
    ft_a = da.get("fresh_traj_outputs", 0)
    fa_a = da.get("fresh_action_outputs", 0)
    ft_c = d.get("fresh_traj_outputs", 0)
    fa_c = d.get("fresh_action_outputs", 0)
    n_a = ft_a + fa_a; n_c = ft_c + fa_c
    if n_a == 0 or n_c == 0: return None
    n = n_a + n_c
    obs = [[ft_a, fa_a], [ft_c, fa_c]]
    rs = [n_a, n_c]; cs = [ft_a+ft_c, fa_a+fa_c]
    chi2 = sum((obs[i][j] - rs[i]*cs[j]/n)**2 / max(rs[i]*cs[j]/n, 1e-9)
               for i in range(2) for j in range(2))
    return math.sqrt(chi2 / n)

print("\n--- Condition A (No-Cache) ---")
print(f"{'Bag':>8} {'bg':>6} {'skip%':>7} {'ar%':>7}")
for b in BAGS:
    p = f"{LOG}/{b}_NOCACHE/metrics.json"
    if not os.path.exists(p): print(f"  {b}: NOT FOUND"); continue
    d = load(p)
    ft=d.get("fresh_traj_outputs",0); fa=d.get("fresh_action_outputs",0)
    ar = fa/max(1,ft+fa)*100
    print(f"  {b}: bg={d.get('background_s2_runs',0)}  skip={d.get('temporal_cache_skip_ratio',0):.1f}%  ar={ar:.1f}%")

print("\n--- Condition P (Production Stack) ---")
print(f"{'Bag':>8} {'bg':>6} {'skip%':>7} {'AA':>5} {'MH':>5} {'SP':>5} {'ar%':>7}")
for b in BAGS:
    p = f"{LOG}/{b}_PROD/metrics.json"
    if not os.path.exists(p): print(f"  {b}: NOT FOUND"); continue
    d = load(p)
    ft=d.get("fresh_traj_outputs",0); fa=d.get("fresh_action_outputs",0)
    ar = fa/max(1,ft+fa)*100
    aa=d.get("action_aware_bypasses",0); mh=d.get("max_hold_bypasses",0)
    sp=d.get("slope_predict_bypasses",0); bg=d.get("background_s2_runs",0)
    nat=bg-aa-mh-sp
    print(f"  {b}: bg={bg}  skip={d.get('temporal_cache_skip_ratio',0):.1f}%  nat={nat}  AA={aa}  MH={mh}  SP={sp}  ar={ar:.1f}%")

print("\n--- Cramér's V: Condition P vs Condition A ---")
print(f"{'Bag':>10} {'V':>8} {'pass?':>7}")
vs = []
for b in BAGS:
    p_path = f"{LOG}/{b}_PROD/metrics.json"
    a_path = f"{LOG}/{b}_NOCACHE/metrics.json"
    if not (os.path.exists(p_path) and os.path.exists(a_path)):
        print(f"  {b}: MISSING"); continue
    v = cramers_v(load(p_path), load(a_path))
    if v is None: print(f"  {b}: V=None"); continue
    vs.append(v)
    status = "PASS" if v <= 0.10 else "FAIL"
    print(f"  {b}: V={v:.4f}  {status}")

if vs:
    v_max = max(vs)
    overall = "PASS" if v_max <= 0.10 else "FAIL"
    print(f"\n  V_max = {v_max:.4f}  →  GATE 3n: {overall}")

print("\n--- LaTeX table rows ---")
for b in BAGS:
    p_path = f"{LOG}/{b}_PROD/metrics.json"
    a_path = f"{LOG}/{b}_NOCACHE/metrics.json"
    if not (os.path.exists(p_path) and os.path.exists(a_path)): continue
    d = load(p_path); da = load(a_path)
    v = cramers_v(d, da)
    skip = d.get("temporal_cache_skip_ratio", 0)
    aa=d.get("action_aware_bypasses",0); mh=d.get("max_hold_bypasses",0)
    sp=d.get("slope_predict_bypasses",0)
    if v is not None:
        chk = "\\checkmark" if v <= 0.10 else "\\times"
        print(f"{b} & {skip:.1f}\\% & {aa} & {mh} & {sp} & {v:.4f} & {chk} \\\\")
PYEOF

echo ""
echo "GATE3N_DONE"
