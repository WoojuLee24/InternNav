#!/usr/bin/env bash
# Gate 3q — Single-Run Variance Analysis
#
# Empirically characterizes Cramér's V sampling variance by running the
# same configuration (Gate 3n PROD) 3× on bag 073623.
# Also runs 3× no-cache (NOCACHE) on bag 073623 as intra-baseline control.
#
# Computes:
#   V_intra_nocache = V between pairs of independent no-cache runs
#   V_intra_prod    = V between pairs of independent PROD runs
#   V_cross         = V between PROD runs and the shared NOCACHE reference
#
# This establishes:
#   1. The V noise floor from stochastic sampling alone (V_intra_nocache)
#   2. Whether gate pass/fail boundaries are meaningful vs. noise
#   3. Empirical ±σ for all other gates
#
# Pass criterion: V_intra_nocache < 0.05 (noise floor is below half the threshold)
# Key: If V_intra_nocache ≥ 0.05, the V ≤ 0.10 pass criterion is unreliable.
set -e
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3q_variance"
mkdir -p "$LOG_BASE"

BAG="my_camera_bag_20260317_073623"
SHORT="073623"
N_REPS=3

echo "=================================================="
echo " Gate 3q — Single-Run Variance Analysis"
echo " Bag: $SHORT (3 reps each of NOCACHE and PROD)"
echo " Computes: V_intra_nocache, V_intra_prod, V_cross"
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

configure_nocache() {
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.0" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=0" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_ema_fingerprint?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_slope_predict?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_odom_progress_hold?enabled=false" > /dev/null
    echo "  Config: NOCACHE (τ=0.0, no cache)"
}

configure_prod() {
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_trajectory_adaptive_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_serve_count_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true" > /dev/null
    curl -sf "http://localhost:5802/set_slope_predict?enabled=true&threshold=0.010&window=3" > /dev/null
    curl -sf "http://localhost:5802/set_odom_progress_hold?enabled=false" > /dev/null
    echo "  Config: PROD (Gate 3n full stack)"
}

run_bag_rep() {
    local config_tag="$1"
    local rep="$2"
    local tag="${SHORT}_${config_tag}_R${rep}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- config=$config_tag  rep=$rep ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1

    (cd "$REPO_DIR" && python3.12 $CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB") \
        > "$log_dir/client.log" 2>&1 &
    local client_pid=$!
    sleep 3

    # echo "  playing /workspace/rosbag/$BAG at rate=0.5..."  # gds container
#     ros2 bag play "/workspace/rosbag/$BAG" --rate 0.5 > "$log_dir/bag.log" 2>&1  # played manually from gds container
    sleep 2

    curl -sf http://localhost:5802/async_metrics > "$log_dir/metrics.json"
    kill $client_pid 2>/dev/null || true

    python3 - <<PYEOF
import json
with open("$log_dir/metrics.json") as f: d = json.load(f)
bg   = d.get("background_s2_runs", 0)
ft   = d.get("fresh_traj_outputs", 0)
fa   = d.get("fresh_action_outputs", 0)
skip = d.get("temporal_cache_skip_ratio", 0)
hz   = d.get("joint_req_hz", 0)
n    = ft + fa; ar = fa / max(1, n) * 100
print(f"  bg={bg}  skip={skip:.1f}%  action_rate={ar:.1f}%  (ft={ft} fa={fa})  hz={hz:.2f}")
PYEOF
}

start_server "$LOG_BASE/server.log"

# Run NOCACHE 3×
echo ""
echo "=== NOCACHE (3 reps) ==="
configure_nocache
for rep in 1 2 3; do
    run_bag_rep "NOCACHE" "$rep"
done

# Run PROD 3×
echo ""
echo "=== PROD (3 reps) ==="
configure_prod
for rep in 1 2 3; do
    run_bag_rep "PROD" "$rep"
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3q — VARIANCE ANALYSIS"
echo "=================================================="

python3 - <<PYEOF
import json, os, math, itertools

LOG = "/tmp/gate3q_variance"
SHORT = "073623"
N_REPS = 3

def load(path):
    with open(path) as f: return json.load(f)

def cramers_v(d1, d2):
    ft_a = d1.get("fresh_traj_outputs", 0); fa_a = d1.get("fresh_action_outputs", 0)
    ft_c = d2.get("fresh_traj_outputs", 0);  fa_c = d2.get("fresh_action_outputs", 0)
    n_a = ft_a + fa_a; n_c = ft_c + fa_c
    if n_a == 0 or n_c == 0: return None
    n = n_a + n_c
    obs = [[ft_a, fa_a], [ft_c, fa_c]]
    rs = [n_a, n_c]; cs = [ft_a+ft_c, fa_a+fa_c]
    chi2 = sum((obs[i][j] - rs[i]*cs[j]/n)**2 / max(rs[i]*cs[j]/n, 1e-9)
               for i in range(2) for j in range(2))
    return math.sqrt(chi2 / n)

# Load all runs
nocache_runs = []
prod_runs = []
for rep in range(1, N_REPS + 1):
    nc_path = f"{LOG}/{SHORT}_NOCACHE_R{rep}/metrics.json"
    pr_path = f"{LOG}/{SHORT}_PROD_R{rep}/metrics.json"
    if os.path.exists(nc_path): nocache_runs.append(load(nc_path))
    if os.path.exists(pr_path): prod_runs.append(load(pr_path))

# Action rates
print("\n--- Action rates per run ---")
for i, d in enumerate(nocache_runs, 1):
    ft = d.get("fresh_traj_outputs", 0); fa = d.get("fresh_action_outputs", 0)
    n = ft + fa; ar = fa / max(1, n) * 100
    print(f"  NOCACHE R{i}: bg={d.get('background_s2_runs',0)}  action_rate={ar:.1f}%  (ft={ft} fa={fa})")
for i, d in enumerate(prod_runs, 1):
    ft = d.get("fresh_traj_outputs", 0); fa = d.get("fresh_action_outputs", 0)
    n = ft + fa; ar = fa / max(1, n) * 100
    skip = d.get("temporal_cache_skip_ratio", 0)
    print(f"  PROD    R{i}: bg={d.get('background_s2_runs',0)}  skip={skip:.1f}%  action_rate={ar:.1f}%  (ft={ft} fa={fa})")

# Pairwise V within no-cache runs (intra-baseline noise floor)
print("\n--- V_intra_nocache (pairs of independent no-cache runs) ---")
v_intra_nc = []
for i, j in itertools.combinations(range(len(nocache_runs)), 2):
    v = cramers_v(nocache_runs[i], nocache_runs[j])
    if v is not None:
        v_intra_nc.append(v)
        print(f"  NOCACHE R{i+1} vs R{j+1}: V={v:.4f}")
if v_intra_nc:
    print(f"  Mean V_intra_nocache = {sum(v_intra_nc)/len(v_intra_nc):.4f}")
    print(f"  Max  V_intra_nocache = {max(v_intra_nc):.4f}")

# Pairwise V within PROD runs
print("\n--- V_intra_prod (pairs of independent PROD runs) ---")
v_intra_pr = []
for i, j in itertools.combinations(range(len(prod_runs)), 2):
    v = cramers_v(prod_runs[i], prod_runs[j])
    if v is not None:
        v_intra_pr.append(v)
        print(f"  PROD R{i+1} vs R{j+1}: V={v:.4f}")
if v_intra_pr:
    print(f"  Mean V_intra_prod = {sum(v_intra_pr)/len(v_intra_pr):.4f}")
    print(f"  Max  V_intra_prod = {max(v_intra_pr):.4f}")

# V_cross: each PROD vs each NOCACHE (cross-condition V)
print("\n--- V_cross (PROD vs NOCACHE, each pair) ---")
v_cross = []
for i, dp in enumerate(prod_runs, 1):
    for j, dn in enumerate(nocache_runs, 1):
        v = cramers_v(dp, dn)
        if v is not None:
            v_cross.append(v)
            print(f"  PROD R{i} vs NOCACHE R{j}: V={v:.4f}")
if v_cross:
    print(f"  Mean V_cross = {sum(v_cross)/len(v_cross):.4f}")
    print(f"  Max  V_cross = {max(v_cross):.4f}")

# Summary
print("\n--- Summary ---")
status = "PASS" if v_intra_nc and max(v_intra_nc) < 0.05 else "REVIEW"
print(f"  V_intra_nocache max = {max(v_intra_nc) if v_intra_nc else 'N/A':.4f}  {'< 0.05 (meaningful noise floor)' if v_intra_nc and max(v_intra_nc) < 0.05 else '>= 0.05 (large noise floor)'}")
print(f"  V_intra_prod max    = {max(v_intra_pr) if v_intra_pr else 'N/A':.4f}")
print(f"  V_cross max         = {max(v_cross) if v_cross else 'N/A':.4f}")
print(f"  Status: {status}")

if v_intra_nc and v_cross:
    snr = max(v_cross) / max(max(v_intra_nc), 1e-6)
    print(f"  Signal-to-noise (V_cross_max / V_intra_nc_max) = {snr:.2f}×")

print()
print("--- LaTeX variance table ---")
print(r"\begin{tabular}{lcc}")
print(r"\toprule")
print(r"Comparison & Mean $V$ & Max $V$ \\")
print(r"\midrule")
if v_intra_nc:
    print(f"Intra-baseline (NOCACHE vs NOCACHE) & {sum(v_intra_nc)/len(v_intra_nc):.4f} & {max(v_intra_nc):.4f} \\\\")
if v_intra_pr:
    print(f"Intra-PROD (PROD vs PROD) & {sum(v_intra_pr)/len(v_intra_pr):.4f} & {max(v_intra_pr):.4f} \\\\")
if v_cross:
    print(f"Cross-condition (PROD vs NOCACHE) & {sum(v_cross)/len(v_cross):.4f} & {max(v_cross):.4f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
PYEOF

echo ""
echo "GATE3Q_DONE"
