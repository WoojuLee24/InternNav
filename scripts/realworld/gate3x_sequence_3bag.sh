#!/usr/bin/env bash
# Gate 3X — Phase 3X offline proxy analysis (I-111/I-112/I-113)
# Runs NOCACHE vs PROD on all 3 bags at rate=0.5 and analyses
# per-request response-type sequences for DTW/KL/counterfactual metrics.
#
# Pass criterion (any one):
#   I-111: KL(PROD‖NOCACHE) < 0.10
#   I-112: DTW_norm < 0.10
#   I-113: counterfactual gap < 3pp
#
# Usage (inside container):
#   bash scripts/realworld/gate3x_sequence_3bag.sh
set -euo pipefail
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

LOG_BASE="/tmp/gate3x_sequence"
mkdir -p "$LOG_BASE"

BAGS=(073623 061841 063047)
BAG_DIR="/workspace/rosbag"
SERVER_LOG="$LOG_BASE/server.log"

# ---- helpers ----------------------------------------------------------------
wait_server() {
    echo "[gate3x] Waiting for server on :5802..."
    until curl -sf http://localhost:5802/async_metrics > /dev/null 2>&1; do sleep 2; done
    echo "[gate3x] Server ready."
}

configure_nocache() {
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.0"
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=0"
    curl -sf "http://localhost:5802/set_action_aware?enabled=false"
    curl -sf "http://localhost:5802/set_ema_fingerprint?enabled=false"
    curl -sf "http://localhost:5802/set_slope_predict?enabled=false"
    echo "[gate3x] Config: NOCACHE"
}

configure_prod() {
    curl -sf "http://localhost:5802/set_temporal_threshold?threshold=0.92"
    curl -sf "http://localhost:5802/set_max_hold_frames?frames=15"
    curl -sf "http://localhost:5802/set_action_aware?enabled=true"
    curl -sf "http://localhost:5802/set_ema_fingerprint?enabled=true&alpha=0.10&transition_reset=true"
    curl -sf "http://localhost:5802/set_slope_predict?enabled=true&threshold=0.010&window=3"
    echo "[gate3x] Config: PROD (Gate 3n stack)"
}

run_bag() {
    local bag=$1 tag=$2 log_dir=$3
    mkdir -p "$log_dir"
    curl -sf "http://localhost:5802/reset_metrics" > "$log_dir/reset.json"
    echo "[gate3x] Playing bag $bag ($tag)..."
    ros2 bag play "$BAG_DIR/my_camera_bag_20260317_$bag" --rate 0.5 \
        --topics /camera/camera/color/image_raw \
                 /camera/camera/aligned_depth_to_color/image_raw \
                 /gdq/msg/gdq_odom 2>&1 | tee "$log_dir/bag.log" || true
    sleep 3
    curl -sf "http://localhost:5802/async_metrics" > "$log_dir/metrics.json"
    echo "[gate3x] Saved $log_dir/metrics.json"
    python3 -c "
import json; d=json.load(open('$log_dir/metrics.json'))
seq=d.get('response_sequence',[])
print(f'  requests={d.get(\"total_requests\",0)}  bg={d.get(\"background_s2_runs\",0)}  skip={d.get(\"temporal_cache_skip_ratio\",0.0):.1f}%  seq_len={len(seq)}')
"
}

# ---- start server -----------------------------------------------------------
echo "[gate3x] Starting server..."
python3 scripts/realworld/http_internvla_server_debug.py \
    --mode async --temperature 0.75 --kv-cache \
    --calib scripts/realworld/calib/calib_scout.txt \
    --pre-warm-frames 3 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_server

# ---- NOCACHE runs -----------------------------------------------------------
echo ""
echo "========================================"
echo " NOCACHE runs (τ=0, MH=0, AA=off)"
echo "========================================"
configure_nocache
for BAG in "${BAGS[@]}"; do
    run_bag "$BAG" "NOCACHE" "$LOG_BASE/${BAG}_NOCACHE"
done

# ---- PROD runs --------------------------------------------------------------
echo ""
echo "========================================"
echo " PROD runs (Gate 3n stack)"
echo "========================================"
configure_prod
for BAG in "${BAGS[@]}"; do
    run_bag "$BAG" "PROD" "$LOG_BASE/${BAG}_PROD"
done

# ---- analysis ---------------------------------------------------------------
echo ""
echo "========================================"
echo " Gate 3X Analysis"
echo "========================================"
for BAG in "${BAGS[@]}"; do
    python3 scripts/viz/parse_gate3x.py \
        "$LOG_BASE/${BAG}_NOCACHE" \
        "$LOG_BASE/${BAG}_PROD"
done

echo ""
echo "========================================"
echo " LaTeX table rows"
echo "========================================"
echo "\\begin{tabular}{lccccccc}"
echo "  \\toprule"
echo "  Bag & Skip\\% & TR\_NC & TR\_PROD & KL & DTW & Gap & Result \\\\"
echo "  \\midrule"
for BAG in "${BAGS[@]}"; do
    python3 scripts/viz/parse_gate3x.py \
        "$LOG_BASE/${BAG}_NOCACHE" \
        "$LOG_BASE/${BAG}_PROD" 2>/dev/null | grep "^\s*${BAG}" || true
done
echo "  \\bottomrule"
echo "\\end{tabular}"

echo ""
echo "GATE3X_DONE"
kill "$SERVER_PID" 2>/dev/null || true
