#!/usr/bin/env bash
# Gate 3m — Transition-Reset EMA Scene Fingerprint (I-055)
# Hypothesis: I-053 (standard EMA) fails because slow EMA update after forced
# bypasses (I-047/I-046) causes a post-bypass cascade: for α=0.05, EMA needs
# ~18 frames to converge to the new scene, triggering natural S2 runs at each
# lag frame (EMA still represents the old scene → low similarity).
#
# I-055 fix: hard-reset EMA to the current frame after each forced bypass
# (I-046/I-047/I-052). This eliminates the cascade:
#   • After forced bypass: EMA = current frame (same as point reference) → no lag
#   • During skip sequences: EMA slowly tracks drift (same as I-053)
#   • Result: point-ref accuracy after transitions + slow-drift tracking during stability
#
# Expected: all α values pass V ≤ 0.10 (no cascade), some α values show
# slight skip% improvement from reduced natural misses during stable periods.
#
# Sweep: α ∈ {0.05, 0.10, 0.15, 0.20, 0.30}
# Fixed: max_hold=15, tau=0.92, action_aware=on, transition_reset=true
#
# Run inside container: docker exec -it vlnav_internvla_server bash
# Usage:  bash scripts/realworld/gate3m_tr_ema_sweep.sh
set -e
source /opt/ros/jazzy/setup.bash

CALIB="/workspace/InternNav/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3m_tr_ema"
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)
ALPHA_VALUES=(0.05 0.10 0.15 0.20 0.30)

echo "=================================================="
echo " Gate 3m — I-055 Transition-Reset EMA α Sweep"
echo " Values: ${ALPHA_VALUES[*]}"
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

configure_tr_ema() {
    local alpha="$1"
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_trajectory_adaptive_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_serve_count_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_slope_predict?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_ema_fingerprint?enabled=true&alpha=$alpha&transition_reset=true" > /dev/null
    echo "  TR-EMA alpha=$alpha, transition_reset=true configured"
}

run_bag() {
    local bag="$1"
    local alpha="$2"
    local short="${bag##*_}"
    local alpha_tag="${alpha/./_}"
    local tag="${short}_TREMA${alpha_tag}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  alpha=$alpha ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1
    configure_tr_ema "$alpha"

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

run_baseline() {
    local bag="$1"
    local short="${bag##*_}"
    local log_dir="$LOG_BASE/${short}_BASELINE"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  BASELINE (MH=15, TR-EMA=off) ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_ema_fingerprint?enabled=false" > /dev/null

    (cd /workspace/InternNav && python3.12 $CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB") \
        > "$log_dir/client.log" 2>&1 &
    local client_pid=$!
    sleep 3

    ros2 bag play "/workspace/rosbag/$bag" --rate 0.5 > "$log_dir/bag.log" 2>&1
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
echo "=== BASELINE (Gate 3g: MH=15, TR-EMA off) ==="
for bag in "${BAGS[@]}"; do
    run_baseline "$bag"
done

for alpha in "${ALPHA_VALUES[@]}"; do
    echo ""
    echo "=== alpha=$alpha ==="
    for bag in "${BAGS[@]}"; do
        run_bag "$bag" "$alpha"
    done
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3m — ANALYSIS"
echo "=================================================="

python3 - <<PYEOF
import json, os, math

LOG = "/tmp/gate3m_tr_ema"
BAGS = ["073623", "061841", "063047"]
ALPHAS = ["0.05", "0.10", "0.15", "0.20", "0.30"]

def load(path):
    with open(path) as f: return json.load(f)

def cramers_v(d, da):
    ft_a = da.get("fresh_traj_outputs", 0); fa_a = da.get("fresh_action_outputs", 0)
    ft_c = d.get("fresh_traj_outputs", 0);  fa_c = d.get("fresh_action_outputs", 0)
    n_a = ft_a + fa_a; n_c = ft_c + fa_c
    if n_a == 0 or n_c == 0: return None
    n = n_a + n_c
    obs = [[ft_a, fa_a], [ft_c, fa_c]]
    rs = [n_a, n_c]; cs = [ft_a + ft_c, fa_a + fa_c]
    chi2 = sum((obs[i][j] - rs[i]*cs[j]/n)**2 / max(rs[i]*cs[j]/n, 1e-9)
               for i in range(2) for j in range(2))
    return math.sqrt(chi2 / n)

def bg(d):   return d.get("background_s2_runs", 0)
def fa(d):   return d.get("fresh_action_outputs", 0)
def ft(d):   return d.get("fresh_traj_outputs", 0)
def aa(d):   return d.get("action_aware_bypasses", 0)
def mh_c(d): return d.get("max_hold_bypasses", 0)
def skip(d): return d.get("temporal_cache_skip_ratio", 0)

# Per-bag table
for b in BAGS:
    print(f"\n--- Bag {b} ---")
    print(f"{'cond':>12} {'bg':>7} {'skip%':>7} {'AA':>6} {'MH':>6} {'nat':>6} {'ar%':>6}")
    base_path = f"{LOG}/{b}_BASELINE/metrics.json"
    if os.path.exists(base_path):
        d = load(base_path)
        n = ft(d) + fa(d); ar = fa(d) / max(1, n) * 100
        nat = bg(d) - aa(d) - mh_c(d)
        print(f"{'BASELINE':>12} {bg(d):>7} {skip(d):>6.1f}% {aa(d):>6} {mh_c(d):>6} {nat:>6} {ar:>5.1f}%")
    for al in ALPHAS:
        al_tag = al.replace('.', '_')
        path = f"{LOG}/{b}_TREMA{al_tag}/metrics.json"
        if not os.path.exists(path): continue
        d = load(path)
        n = ft(d) + fa(d); ar = fa(d) / max(1, n) * 100
        nat = bg(d) - aa(d) - mh_c(d)
        print(f"{'TR='+al:>12} {bg(d):>7} {skip(d):>6.1f}% {aa(d):>6} {mh_c(d):>6} {nat:>6} {ar:>5.1f}%")

# Cramér's V vs BASELINE
print("\n--- Cramér's V (vs Gate-3m BASELINE) ---")
print(f"{'alpha':>10} {'073623':>10} {'061841':>10} {'063047':>10} {'worst':>10}")
for al in ALPHAS:
    al_tag = al.replace('.', '_')
    vs = []
    for b in BAGS:
        path = f"{LOG}/{b}_TREMA{al_tag}/metrics.json"
        base_path = f"{LOG}/{b}_BASELINE/metrics.json"
        if not (os.path.exists(path) and os.path.exists(base_path)):
            vs.append(None); continue
        v = cramers_v(load(path), load(base_path))
        vs.append(v)
    vstr = [f"{v:.4f}" if v is not None else "  N/A" for v in vs]
    worst = max((v for v in vs if v is not None), default=None)
    wstr = f"{worst:.4f}" if worst is not None else "  N/A"
    status = "PASS" if worst is not None and worst <= 0.10 else "FAIL"
    print(f"{'TR='+al:>10}  {'  '.join(vstr)}  {wstr:>8}  {status}")
PYEOF

echo ""
echo "GATE3M_DONE"
