#!/usr/bin/env bash
# Gate 3h — I-051: Trajectory-Length-Adaptive Max Hold
# Hypothesis: coupling max_hold to cached trajectory length (dynamic_max_hold =
# min(cap, len(trajectory) * multiplier) increases skip% beyond Gate 3f's 88%
# while keeping V≤0.10 all bags.  Long plans have more "plan consumption frames"
# before they go stale; short plans refresh more aggressively.
#
# Conditions (all with τ=0.92, action_aware=on, I-050 active):
#   BASELINE  — I-051 off, max_hold=10 (reproduce Gate 3f D+I-050)
#   TRAJ_M7   — I-051 on, multiplier=7, cap=50 (primary hypothesis)
#   TRAJ_M5   — I-051 on, multiplier=5, cap=50 (conservative variant)
#   TRAJ_M10  — I-051 on, multiplier=10, cap=70 (aggressive variant)
#
# Pass criterion: V(TRAJ_Mx vs BASELINE) ≤ 0.10 all 3 bags
# Bonus: skip% > 88% (improvement over Gate 3f)
#
# Run inside container: docker exec -it vlnav_internvla_server bash
# Usage: bash scripts/realworld/gate3h_traj_adaptive.sh
set -e
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3h_traj"
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)

echo "=================================================="
echo " Gate 3h — I-051 Trajectory-Length-Adaptive Max Hold"
echo " Conditions: BASELINE  TRAJ_M5  TRAJ_M7  TRAJ_M10"
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

configure_baseline() {
    # Gate 3f / D+I-050 config: fixed max_hold=10, I-051 off
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=10" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_trajectory_adaptive_hold?enabled=false" > /dev/null
    echo "  condition=BASELINE (max_hold=10, I-051 off)"
}

configure_traj() {
    local mult="$1"
    local cap="$2"
    local label="$3"
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=10" > /dev/null
    curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
    curl -sf "http://localhost:5802/set_adaptive_max_hold?enabled=false" > /dev/null
    curl -sf "http://localhost:5802/set_trajectory_adaptive_hold?enabled=true&multiplier=${mult}&cap=${cap}" > /dev/null
    echo "  condition=${label} (multiplier=${mult}, cap=${cap})"
}

run_bag() {
    local bag="$1"
    local condition="$2"
    local short="${bag#*_}"
    local tag="${short}_${condition}"
    local log_dir="$LOG_BASE/$tag"
    mkdir -p "$log_dir"

    echo ""
    echo "--- bag=$short  condition=$condition ---"

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
mh_c = d.get("max_hold_bypasses", 0)
skip = d.get("temporal_cache_skip_ratio", 0)
hz   = d.get("joint_req_hz", 0)
avg_tl = d.get("avg_trajectory_length", 0)
n    = ft + fa
ar   = fa / max(1, n) * 100
nat  = bg - aa - mh_c
print(f"  bg={bg}  skip={skip:.1f}%  hz={hz:.2f}")
print(f"  natural={nat}  AA={aa}  MH={mh_c}")
print(f"  action_rate={ar:.1f}%  avg_traj_len={avg_tl:.1f}")
PYEOF
}

start_server "$LOG_BASE/server.log"

for condition in BASELINE TRAJ_M5 TRAJ_M7 TRAJ_M10; do
    echo ""
    echo "=== Condition: $condition ==="
    case "$condition" in
        BASELINE)    configure_baseline ;;
        TRAJ_M5)     configure_traj 5  50 "TRAJ_M5" ;;
        TRAJ_M7)     configure_traj 7  50 "TRAJ_M7" ;;
        TRAJ_M10)    configure_traj 10 70 "TRAJ_M10" ;;
    esac
    for bag in "${BAGS[@]}"; do
        run_bag "$bag" "$condition"
    done
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3h — ANALYSIS"
echo "=================================================="

python3 - <<PYEOF
import json, os, math

LOG = "/tmp/gate3h_traj"
BAGS = ["073623", "061841", "063047"]
CONDITIONS = ["BASELINE", "TRAJ_M5", "TRAJ_M7", "TRAJ_M10"]

def load(path):
    with open(path) as f: return json.load(f)

def bg(d):   return d.get("background_s2_runs", 0)
def fa(d):   return d.get("fresh_action_outputs", 0)
def ft(d):   return d.get("fresh_traj_outputs", 0)
def aa(d):   return d.get("action_aware_bypasses", 0)
def mh_c(d): return d.get("max_hold_bypasses", 0)
def skip(d): return d.get("temporal_cache_skip_ratio", 0)
def avg_tl(d): return d.get("avg_trajectory_length", 0)

# Per-bag table
for b in BAGS:
    print(f"\n--- Bag {b} ---")
    print(f"{'Cond':<12} {'bg':>6} {'skip%':>7} {'AA':>5} {'MH':>5} {'nat':>4} {'ar%':>6} {'avg_tl':>7}")
    for cond in CONDITIONS:
        path = f"{LOG}/{b}_{cond}/metrics.json"
        if not os.path.exists(path):
            print(f"  {cond:<10} (no data)"); continue
        d = load(path)
        nat = bg(d) - aa(d) - mh_c(d)
        n = ft(d) + fa(d)
        ar = fa(d) / max(1, n) * 100
        print(f"{cond:<12} {bg(d):>6} {skip(d):>6.1f}% {aa(d):>5} {mh_c(d):>5} {nat:>4} {ar:>5.1f}%  {avg_tl(d):>5.1f}")

# Cramér's V (each TRAJ condition vs BASELINE)
print("\n--- Cramér's V (TRAJ_Mx vs BASELINE) ---")
print(f"{'Cond':<12} {'073623':>10} {'061841':>10} {'063047':>10} {'worst':>10} {'gate'}")
for cond in ["TRAJ_M5", "TRAJ_M7", "TRAJ_M10"]:
    vs = []
    for b in BAGS:
        path_c = f"{LOG}/{b}_{cond}/metrics.json"
        path_b = f"{LOG}/{b}_BASELINE/metrics.json"
        if not (os.path.exists(path_c) and os.path.exists(path_b)):
            vs.append(None); continue
        dc = load(path_c); db = load(path_b)
        ft_b = ft(db); fa_b = fa(db); ft_c = ft(dc); fa_c = fa(dc)
        n_b = ft_b + fa_b; n_c = ft_c + fa_c
        if n_b == 0 or n_c == 0: vs.append(None); continue
        n = n_b + n_c
        obs = [[ft_b, fa_b], [ft_c, fa_c]]
        rs = [n_b, n_c]; cs = [ft_b+ft_c, fa_b+fa_c]
        chi2 = sum((obs[i][j] - rs[i]*cs[j]/n)**2 / max(rs[i]*cs[j]/n, 1e-9)
                   for i in range(2) for j in range(2))
        vs.append(math.sqrt(chi2 / n))
    vstr = [f"{v:.4f}" if v is not None else "  N/A" for v in vs]
    worst = max((v for v in vs if v is not None), default=None)
    wstr = f"{worst:.4f}" if worst is not None else "  N/A"
    status = "PASS ✓" if worst is not None and worst <= 0.10 else "FAIL ✗"
    print(f"{cond:<12}  {'  '.join(vstr)}  {wstr:>8}  {status}")

# Skip% improvement summary
print("\n--- Skip% improvement over BASELINE ---")
for cond in ["TRAJ_M5", "TRAJ_M7", "TRAJ_M10"]:
    deltas = []
    for b in BAGS:
        pb = f"{LOG}/{b}_BASELINE/metrics.json"
        pc = f"{LOG}/{b}_{cond}/metrics.json"
        if not (os.path.exists(pb) and os.path.exists(pc)): continue
        deltas.append(skip(load(pc)) - skip(load(pb)))
    if deltas:
        print(f"  {cond}: Δskip = {sum(deltas)/len(deltas):+.1f}% (avg over {len(deltas)} bags)")
PYEOF

echo ""
echo "GATE3H_DONE"
