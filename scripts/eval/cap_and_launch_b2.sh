#!/bin/bash
# cap_and_launch_b2.sh — Watch ChronoNav eval, cap at 500 eps, launch B2 A/B.
#
# HARDSTOP RULES (from session):
#   SR ≥ 62% at ≥500 eps → cap container, write result.json
#   SR < 62% at ≥200 eps → kill container immediately
#
# Run this in a terminal (can be tmux) and leave it.
# It polls progress_rank0.json every 60 seconds.
#
# Usage:
#   bash scripts/eval/cap_and_launch_b2.sh
#   bash scripts/eval/cap_and_launch_b2.sh --dry-run   # print what would happen

set -euo pipefail

DRY_RUN=false
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=true; done

CONTAINER=vln_ce_mini_chrononav
PROGRESS=/mnt/nvme0/vln_habitat/logs/vln_ce_mini/chrononav_baseline/progress_rank0.json
RESULT_OUT=/mnt/nvme0/vln_habitat/logs/vln_ce_mini/chrononav_baseline/result.json
TIMING_OUT=/mnt/nvme0/vln_habitat/logs/vln_ce_mini/chrononav_baseline/timing_rank0.json
LOG=/tmp/cap_and_launch_b2_monitor.log
CAP_EPS=500
KILL_THRESHOLD_EPS=200
KILL_THRESHOLD_SR=0.62

INTERNAV=/home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev
CONTAINER_IMAGE=vlnav/habitat-eval:rebuilt
GPU_ARGS="--gpus all"
ROS_MOUNT="-v /mnt/nvme0/vln_habitat:/vln_habitat"
CODE_MOUNT="-v ${INTERNAV}:/workspace/InternNav"

B2_UNCACHED_CFG=scripts/eval/configs/habitat_vln_ce_b2_uncached_cfg.py
B2_CACHED_CFG=scripts/eval/configs/habitat_vln_ce_b2_cached_cfg.py

echo "[$(date '+%H:%M:%S')] cap_and_launch_b2.sh started" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] Watching: $PROGRESS" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] Cap at: $CAP_EPS eps" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] Kill if SR < ${KILL_THRESHOLD_SR} at >= ${KILL_THRESHOLD_EPS} eps" | tee -a "$LOG"

# ── Parse progress ──────────────────────────────────────────────────────────
parse_progress() {
    python3 -c "
import json, sys
lines = open('$PROGRESS').read().strip().split('\n')
rows = [json.loads(l) for l in lines if l.strip()]
n = len(rows)
sr = sum(r['success'] for r in rows) / n if n else 0.0
spl = sum(r['spl'] for r in rows) / n if n else 0.0
ne = sum(r['ne'] for r in rows) / n if n else 0.0
print(f'{n},{sr:.4f},{spl:.4f},{ne:.4f}')
" 2>/dev/null || echo "0,0,0,0"
}

# ── Write result.json ───────────────────────────────────────────────────────
write_result() {
    local n=$1 sr=$2 spl=$3 ne=$4 status=$5
    python3 -c "
import json, os
result = {
    'model': 'ChronoNav (τ=0.92, 4-bit, system2)',
    'checkpoint': '/checkpoints/InternVLA-N1-DualVLN',
    'n_episodes': $n,
    'SR_3m': $sr,
    'SPL': $spl,
    'NE': $ne,
    'status': '$status',
    'config': '$B2_UNCACHED_CFG',
    'progress_json': '$PROGRESS',
}
json.dump(result, open('$RESULT_OUT', 'w'), indent=2)
print(f'[result] written to $RESULT_OUT')
" | tee -a "$LOG"
}

# ── Main watch loop ─────────────────────────────────────────────────────────
while true; do
    if [[ ! -f "$PROGRESS" ]]; then
        echo "[$(date '+%H:%M:%S')] progress file not found, waiting..." | tee -a "$LOG"
        sleep 60
        continue
    fi

    IFS=',' read -r N SR SPL NE <<< "$(parse_progress)"
    echo "[$(date '+%H:%M:%S')] ep=$N SR=${SR} SPL=${SPL} NE=${NE}" | tee -a "$LOG"

    # Hardstop: kill if SR < threshold at >=200 eps
    if [[ $N -ge $KILL_THRESHOLD_EPS ]]; then
        if python3 -c "exit(0 if float('$SR') < $KILL_THRESHOLD_SR else 1)" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] HARDSTOP: SR=$SR < $KILL_THRESHOLD_SR at n=$N — killing container" | tee -a "$LOG"
            write_result "$N" "$SR" "$SPL" "$NE" "KILLED_LOW_SR"
            $DRY_RUN || docker stop "$CONTAINER" 2>/dev/null || true
            break
        fi
    fi

    # Cap at 500 eps
    if [[ $N -ge $CAP_EPS ]]; then
        echo "[$(date '+%H:%M:%S')] CAP REACHED: n=$N SR=$SR — stopping container" | tee -a "$LOG"
        write_result "$N" "$SR" "$SPL" "$NE" "CAPPED_500"
        $DRY_RUN || docker stop "$CONTAINER" 2>/dev/null || true
        echo "[$(date '+%H:%M:%S')] Container stopped." | tee -a "$LOG"
        break
    fi

    sleep 60
done

# ── After cap: launch B2 runs ────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] === Launching B2 A/B eval ===" | tee -a "$LOG"

run_b2() {
    local cfg=$1 name=$2 port=$3
    echo "[$(date '+%H:%M:%S')] Starting B2 $name ($cfg)" | tee -a "$LOG"
    if $DRY_RUN; then
        echo "[DRY-RUN] would run: docker run ... python scripts/eval/eval.py --config $cfg"
        return 0
    fi
    docker run --rm $GPU_ARGS --network host \
        $CODE_MOUNT $ROS_MOUNT \
        -e PYTHONPATH=/workspace/InternNav \
        --name "vln_ce_b2_${name}" \
        "$CONTAINER_IMAGE" \
        bash -c "cd /workspace/InternNav && pip install -e . -q && python scripts/eval/eval.py --config $cfg" \
        2>&1 | tee "/tmp/b2_${name}.log"
    echo "[$(date '+%H:%M:%S')] B2 $name done (exit=$?)" | tee -a "$LOG"
}

run_b2 "$B2_UNCACHED_CFG" "uncached" "2339"
run_b2 "$B2_CACHED_CFG" "cached" "2338"

echo "" | tee -a "$LOG"
echo "[$(date '+%H:%M:%S')] === B2 A/B complete ===" | tee -a "$LOG"
echo "Paths to drop to debug-duilvn-metrics-validation:" | tee -a "$LOG"
echo "  ChronoNav:  $RESULT_OUT" | tee -a "$LOG"
echo "  B2 uncached: /mnt/nvme0/vln_habitat/logs/vln_ce_b2/uncached/progress_rank0.json" | tee -a "$LOG"
echo "  B2 cached:   /mnt/nvme0/vln_habitat/logs/vln_ce_b2/cached/progress_rank0.json" | tee -a "$LOG"
echo "  B2 uncached timing: /mnt/nvme0/vln_habitat/logs/vln_ce_b2/uncached/timing_rank0.json" | tee -a "$LOG"
echo "  B2 cached timing:   /mnt/nvme0/vln_habitat/logs/vln_ce_b2/cached/timing_rank0.json" | tee -a "$LOG"
