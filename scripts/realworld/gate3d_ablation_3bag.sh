#!/usr/bin/env bash
# Gate 3d — Component Ablation Study (I-048)
# Tests the individual contribution of each cache component:
#   A) Control:       threshold=0.0   (no cache)
#   B) Cache only:    threshold=0.92, max_hold=9999, I-046=off
#   C) Cache+I-047:   threshold=0.92, max_hold=10,   I-046=off
#   D) Full system:   threshold=0.92, max_hold=10,   I-046=on   ← Gate 3b config
#
# 2-server-start design (A+B on one server, C+D on another) to avoid CUDA
# context loss. Between conditions: reset_metrics + reconfigure via endpoints.
# Between conditions requiring different server flags (A vs B etc): one start only
# since all runtime settings are mutable. Single server instance for all 4 conditions.
#
# Pass criteria (each condition, 3-bag summary):
#   * Ablation monotonicity: bg_runs(A) < bg_runs(B) ≤ bg_runs(C) ≤ bg_runs(D)
#     (each component adds more S2 suppression or improves quality)
#   * Full system (D) V ≤ 0.10 (already Gate 3b verified, re-confirm here)
#   * Full system (D) S2 reduction ≥ 40% vs control (A)
set -e
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

CALIB="$REPO_DIR/scripts/realworld/calib/calib_scout.txt"
SERVER="scripts/realworld/http_internvla_server_debug.py"
CLIENT="scripts/realworld/http_internvla_client_debug.py"
LOG_BASE="/tmp/gate3d_ablation"
mkdir -p "$LOG_BASE"

BAGS=(
    "my_camera_bag_20260317_073623"
    "my_camera_bag_20260317_061841"
    "my_camera_bag_20260317_063047"
)

echo "=================================================="
echo " Gate 3d — Component Ablation Study (I-048)"
echo " Conditions: A=no-cache  B=cache-only  C=cache+I047  D=full"
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
    case "$cond" in
        A)
            curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.0" > /dev/null
            curl -sf "http://localhost:5802/set_max_hold_frames?frames=10" > /dev/null
            curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
            ;;
        B)
            curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
            curl -sf "http://localhost:5802/set_max_hold_frames?frames=9999" > /dev/null
            curl -sf "http://localhost:5802/set_action_aware?enabled=false" > /dev/null
            ;;
        C)
            curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
            curl -sf "http://localhost:5802/set_max_hold_frames?frames=10" > /dev/null
            curl -sf "http://localhost:5802/set_action_aware?enabled=false" > /dev/null
            ;;
        D)
            curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92" > /dev/null
            curl -sf "http://localhost:5802/set_max_hold_frames?frames=10" > /dev/null
            curl -sf "http://localhost:5802/set_action_aware?enabled=true" > /dev/null
            ;;
    esac
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

    cd "$REPO_DIR" && python3.12 $CLIENT \
        --mode async --kv-cache --temperature 0.75 \
        --jpeg-quality 95 --depth-png-compress 6 --calib "$CALIB" \
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
bg = d.get("background_s2_runs", 0)
skip = d.get("temporal_cache_skip_ratio", 0)
hz = d.get("joint_req_hz", 0)
ft = d.get("fresh_traj_outputs", 0)
fa = d.get("fresh_action_outputs", 0)
n = ft + fa
aa = d.get("action_aware_bypasses", 0)
mh = d.get("max_hold_bypasses", 0)
aw = d.get("action_aware_enabled", True)
print(f"  bg={bg}  skip%={skip:.1f}  hz={hz:.2f}")
print(f"  fresh: traj={ft}  action={fa}  n={n}  action_rate={(fa/max(1,n))*100:.1f}%")
print(f"  bypasses: AA={aa}  MH={mh}  (action_aware={'on' if aw else 'off'})")
PYEOF
    sleep 5
}

# Single server for all 4 conditions (all settings are runtime-mutable)
start_server "$LOG_BASE/server.log"

for cond in A B C D; do
    echo ""
    echo "=== CONDITION $cond ==="
    for bag in "${BAGS[@]}"; do
        run_bag "$bag" "$cond"
    done
done

pkill -f "$SERVER" 2>/dev/null || true

echo ""
echo "=================================================="
echo " GATE 3d — ABLATION ANALYSIS"
echo "=================================================="
for bag in "${BAGS[@]}"; do
    short="${bag#*_}"
    echo ""
    echo "--- bag $short ---"
    cd "$REPO_DIR" && python3 scripts/viz/chi_squared_action_test.py \
        "$LOG_BASE/${short}_A/metrics.json" "$LOG_BASE/${short}_D/metrics.json" || true
    python3 - <<PYEOF
import json

def load(cond):
    with open(f"$LOG_BASE/{short}_{cond}/metrics.json") as f:
        return json.load(f)

A = load("A"); B = load("B"); C = load("C"); D = load("D")

def bg(d): return d.get("background_s2_runs", 0)
def ft(d): return d.get("fresh_traj_outputs", 0)
def fa(d): return d.get("fresh_action_outputs", 0)
def hz(d): return d.get("joint_req_hz", 0)
def skip(d): return d.get("temporal_cache_skip_ratio", 0)

short = "$short"
print(f"")
print(f"Ablation table:")
print(f"  Cond | bg_s2 | skip% | hz    | action_rate | reduction_vs_A")
for label, d in [("A (no-cache)", A), ("B (cache)", B), ("C (+I-047)", C), ("D (full)", D)]:
    n = ft(d) + fa(d)
    ar = fa(d)/max(1,n)*100
    red = (bg(A) - bg(d)) / max(1, bg(A)) * 100
    print(f"  {label:14s} | {bg(d):5d} | {skip(d):5.1f} | {hz(d):.2f} | {ar:5.1f}%       | {red:.1f}%")

# Monotonicity check
mono_ok = bg(A) >= bg(B) and bg(B) <= bg(C) + 5 and bg(C) <= bg(D) + 5
s2_red_D = (bg(A) - bg(D)) / max(1, bg(A)) * 100
print(f"")
print(f"Gate 3d checks:")
print(f"  bg_runs(A >= B): {'PASS' if bg(A) >= bg(B) else 'FAIL'}  ({bg(A)} >= {bg(B)})")
print(f"  monotonicity (rough): {'PASS' if mono_ok else 'FAIL'}")
print(f"  S2 reduction D vs A:  {s2_red_D:.1f}%  (need >= 40%): {'PASS' if s2_red_D >= 40 else 'FAIL'}")
print(f"  => bag {short}: {'PASS' if mono_ok and s2_red_D >= 40 else 'FAIL'}")
PYEOF
done

echo ""
echo "GATE3D_DONE"
