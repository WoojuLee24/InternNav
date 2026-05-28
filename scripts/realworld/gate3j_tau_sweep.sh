#!/usr/bin/env bash
# Gate 3j — MAD Threshold (τ) Sweep
# Hypothesis: with max_hold=15 (Gate 3g optimal) and I-050 active, varying τ
# reveals the 2D quality-efficiency frontier (skip%, V) as a function of
# visual similarity tolerance.
#
# Lower τ → more similar frames required to skip → less caching but safer
# Higher τ → more permissive skip → more caching but riskier
# Current production: τ=0.92. Does a higher τ (e.g., 0.95) pass V≤0.10?
#
# Sweep: τ ∈ {0.85, 0.88, 0.92, 0.95, 0.97}
# Fixed:  max_hold=15 (Gate 3g optimal), action_aware=on, I-050 active
# Pass:   V(τ_x vs A) ≤ 0.10 all 3 bags
# Bonus:  skip% > 91% (Gate 3g baseline) at any τ
#
# Run inside container: docker exec -it vlnav_internvla_server bash
# Usage:  bash scripts/realworld/gate3j_tau_sweep.sh
set -e
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3j_tau"
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)
TAU_VALUES=(0.85 0.88 0.92 0.95 0.97)

echo "=================================================="
echo " Gate 3j — τ Sweep (max_hold=15, I-050 active)"
echo " Values: ${TAU_VALUES[*]}"
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

configure_tau() {
    local tau="$1"
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=$tau" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_trajectory_adaptive_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_serve_count_hold?enabled=false" > /dev/null
    echo "  tau=$tau, max_hold=15 configured"
}

run_bag() {
    local bag="$1"
    local tau="$2"
    local short="${bag#*_}"
    local tau_tag="${tau/./_}"
    local tag="${short}_TAU${tau_tag}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  tau=$tau ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1
    configure_tau "$tau"

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

for tau in "${TAU_VALUES[@]}"; do
    echo ""
    echo "=== tau=$tau ==="
    for bag in "${BAGS[@]}"; do
        run_bag "$bag" "$tau"
    done
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3j — ANALYSIS"
echo "=================================================="

python3 - <<PYEOF
import json, os, math

LOG = "/tmp/gate3j_tau"
BAGS = ["073623", "061841", "063047"]
TAUS = ["0.85", "0.88", "0.92", "0.95", "0.97"]
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

# Per-bag table
for b in BAGS:
    print(f"\n--- Bag {b} ---")
    print(f"{'τ':>6} {'bg':>7} {'skip%':>7} {'AA':>6} {'MH':>6} {'nat':>5} {'ar%':>6}")
    for tau in TAUS:
        tau_tag = tau.replace('.', '_')
        path = f"{LOG}/{b}_TAU{tau_tag}/metrics.json"
        if not os.path.exists(path): continue
        d = load(path)
        nat = bg(d) - aa(d) - mh_c(d)
        n = ft(d) + fa(d)
        ar = fa(d) / max(1, n) * 100
        print(f"{tau:>6} {bg(d):>7} {skip(d):>6.1f}% {aa(d):>6} {mh_c(d):>6} {nat:>5} {ar:>5.1f}%")

# Cramér's V vs Gate-3d Condition A
print("\n--- Cramér's V (vs Gate-3d Condition A) ---")
print(f"{'τ':>6} {'073623':>10} {'061841':>10} {'063047':>10} {'worst':>10}")
for tau in TAUS:
    tau_tag = tau.replace('.', '_')
    vs = []
    for b in BAGS:
        path = f"{LOG}/{b}_TAU{tau_tag}/metrics.json"
        a_path = GATE3D_A.get(b, "")
        if not (os.path.exists(path) and os.path.exists(a_path)):
            vs.append(None); continue
        d = load(path); da = load(a_path)
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
    print(f"{tau:>6}  {'  '.join(vstr)}  {wstr:>8}  {status}")
PYEOF

echo ""
echo "GATE3J_DONE"
