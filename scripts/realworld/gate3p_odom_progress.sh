#!/usr/bin/env bash
# Gate 3p — I-058 Odometry-Progress Hold Sweep
#
# Sweeps displacement thresholds θ_d ∈ {0.3, 0.5, 0.7, 1.0, 1.5} m.
# Production stack (Gate 3n settings) is active; I-058 is toggled ON top.
# Baseline: Gate 3n PROD results (reused from /tmp/gate3n_prod/*_PROD/metrics.json)
#
# Hypothesis: forcing S2 after the robot travels θ_d meters provides a
# spatial refresh signal that preserves action-rate distribution vs. baseline.
#
# Pass criterion: V(PROD+I058 vs NOCACHE) ≤ 0.10 all 3 bags
# Key metric: odom_progress_bypasses per bag (how often spatial trigger fires)
set -e
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3p_odom"
NOCACHE_LOG="/tmp/gate3n_prod"  # reuse Gate 3n no-cache baseline
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)

THRESHOLDS=(0.3 0.5 0.7 1.0 1.5)

echo "=================================================="
echo " Gate 3p — I-058 Odometry-Progress Hold Sweep"
echo " Thresholds: ${THRESHOLDS[*]} meters"
echo " Production stack (Gate 3n) + I-058 on top"
echo " Baseline: Gate 3n no-cache (reused)"
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

configure_production_plus_i058() {
    local thr="$1"
    # Gate 3n production stack
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=15" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_trajectory_adaptive_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_serve_count_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true" > /dev/null
    curl -sf "http://localhost:5802/set_slope_predict?enabled=true&threshold=0.010&window=3" > /dev/null
    # I-058 on top
    curl -sf "http://localhost:5802/set_odom_progress_hold?enabled=true&threshold=${thr}" > /dev/null
    echo "  PROD + I-058 θ_d=${thr}m"
}

run_bag() {
    local bag="$1"
    local thr="$2"
    local short="${bag#*_}"
    local tag_thr="${thr//./_}"
    local tag="${short}_OD${tag_thr}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  θ_d=${thr}m ---"

    curl -sf http://localhost:5802/reset_metrics > "$log_dir/reset.json"
    sleep 1

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
mh   = d.get("max_hold_bypasses", 0)
sp   = d.get("slope_predict_bypasses", 0)
op   = d.get("odom_progress_bypasses", 0)
skip = d.get("temporal_cache_skip_ratio", 0)
hz   = d.get("joint_req_hz", 0)
n    = ft + fa; ar = fa / max(1, n) * 100
nat  = bg - aa - mh - sp - op
print(f"  bg={bg}  skip={skip:.1f}%  nat={nat}  AA={aa}  MH={mh}  SP={sp}  OP={op}  hz={hz:.2f}")
print(f"  action_rate={ar:.1f}%  (ft={ft} fa={fa})")
PYEOF
}

start_server "$LOG_BASE/server.log"

for thr in "${THRESHOLDS[@]}"; do
    echo ""
    echo "=== θ_d=${thr}m ==="
    configure_production_plus_i058 "$thr"
    for bag in "${BAGS[@]}"; do
        run_bag "$bag" "$thr"
    done
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3p — ODOMETRY-PROGRESS ANALYSIS"
echo "=================================================="

python3 - <<PYEOF
import json, os, math

LOG = "/tmp/gate3p_odom"
NOCACHE_LOG = "/tmp/gate3n_prod"
BAGS = ["073623", "061841", "063047"]
THRESHOLDS = ["0.3", "0.5", "0.7", "1.0", "1.5"]

def load(path):
    with open(path) as f: return json.load(f)

def cramers_v(d, da):
    ft_a = da.get("fresh_traj_outputs", 0); fa_a = da.get("fresh_action_outputs", 0)
    ft_c = d.get("fresh_traj_outputs", 0);  fa_c = d.get("fresh_action_outputs", 0)
    n_a = ft_a + fa_a; n_c = ft_c + fa_c
    if n_a == 0 or n_c == 0: return None
    n = n_a + n_c
    obs = [[ft_a, fa_a], [ft_c, fa_c]]
    rs = [n_a, n_c]; cs = [ft_a+ft_c, fa_a+fa_c]
    chi2 = sum((obs[i][j] - rs[i]*cs[j]/n)**2 / max(rs[i]*cs[j]/n, 1e-9)
               for i in range(2) for j in range(2))
    return math.sqrt(chi2 / n)

print(f"\n{'θ_d (m)':>10} {'Skip%':>7} {'V_073623':>10} {'V_061841':>10} {'V_063047':>10} {'V_max':>8} {'OP_mean':>9} {'Status':>8}")
print("-" * 90)

for thr in THRESHOLDS:
    tag_thr = thr.replace(".", "_")
    skip_vals = []; v_row = {}; op_vals = []
    for b in BAGS:
        p_path = f"{LOG}/{b}_OD{tag_thr}/metrics.json"
        a_path = f"{NOCACHE_LOG}/{b}_NOCACHE/metrics.json"
        if not (os.path.exists(p_path) and os.path.exists(a_path)): continue
        d = load(p_path); da = load(a_path)
        v = cramers_v(d, da)
        skip = d.get("temporal_cache_skip_ratio", 0)
        op = d.get("odom_progress_bypasses", 0)
        if v is not None:
            skip_vals.append(skip); v_row[b] = v; op_vals.append(op)
    if not skip_vals:
        print(f"  {thr:>8}m  NOT AVAILABLE")
        continue
    v_max = max(v_row.values())
    skip_mean = sum(skip_vals)/len(skip_vals)
    op_mean = sum(op_vals)/len(op_vals)
    status = "PASS" if v_max <= 0.10 else "FAIL"
    def fv(b): return f"{v_row[b]:.4f}" if b in v_row else "  N/A"
    print(f"  {thr:>8}m {skip_mean:>7.1f}%  {fv('073623'):>10} {fv('061841'):>10} {fv('063047'):>10} {v_max:>8.4f} {op_mean:>9.1f}  {status}")

print()
print("--- LaTeX table rows (θ_d & Skip% & V_073623 & V_061841 & V_063047 & V_max & OP_mean & Status) ---")
for thr in THRESHOLDS:
    tag_thr = thr.replace(".", "_")
    skip_vals = []; v_row = {}; op_vals = []
    for b in BAGS:
        p_path = f"{LOG}/{b}_OD{tag_thr}/metrics.json"
        a_path = f"{NOCACHE_LOG}/{b}_NOCACHE/metrics.json"
        if not (os.path.exists(p_path) and os.path.exists(a_path)): continue
        d = load(p_path); da = load(a_path)
        v = cramers_v(d, da)
        skip = d.get("temporal_cache_skip_ratio", 0)
        op = d.get("odom_progress_bypasses", 0)
        if v is not None: skip_vals.append(skip); v_row[b] = v; op_vals.append(op)
    if not skip_vals: continue
    v_max = max(v_row.values())
    skip_mean = sum(skip_vals)/len(skip_vals)
    op_mean = sum(op_vals)/len(op_vals)
    chk = "\\checkmark" if v_max <= 0.10 else "\\times"
    v073 = v_row.get("073623", 0); v061 = v_row.get("061841", 0); v063 = v_row.get("063047", 0)
    bold = "\\mathbf{" if v_max <= 0.10 else ""
    endbold = "}" if v_max <= 0.10 else ""
    print(f"{bold}{thr}{endbold} & {skip_mean:.1f}\\% & {v073:.4f} & {v061:.4f} & {v063:.4f} & {v_max:.4f} & {op_mean:.1f} & {chk} \\\\")
PYEOF

echo ""
echo "GATE3P_DONE"
