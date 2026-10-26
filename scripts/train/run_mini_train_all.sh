#!/usr/bin/env bash
# run_mini_train_all.sh — Sequential mini-model training + VLN-CE eval
#
# Trains all 4 models sequentially on GPU 0, then runs Habitat val_unseen eval.
# Prerequisites:
#   docker build -f scripts/cloud/Dockerfile.nexusnav -t vlnav/nexusnav-train:local .
#
# Usage:  bash scripts/train/run_mini_train_all.sh [--skip-build] [--eval-only]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE="${ROOT_DIR}/scripts/train/docker-compose.mini-train.yml"
RESULTS_DIR="${ROOT_DIR}/results/mini_measured"
LOG_DIR="/mnt/nvme0/vln_habitat/logs"
SKIP_BUILD=false
EVAL_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=true ;;
        --eval-only)  EVAL_ONLY=true ;;
    esac
done

mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

echo "======================================================="
echo "  NexusNav Mini-Model Training + Eval Pipeline"
echo "  $(date)"
echo "  GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Free:  $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "======================================================="

# ── Step 0: GPU memory gate ─────────────────────────────────────────────────
GPU_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ -n "${GPU_FREE_MIB}" ] && [ "${GPU_FREE_MIB}" -lt 6000 ]; then
    echo "WARNING: Only ${GPU_FREE_MIB} MiB GPU free (need ≥6000 MiB for 4-bit 7B training)."
    echo "The Habitat eval is using the GPU. Wait for it to finish or stop vln_ce_chrononav_80pct."
    echo ""
    echo "To check what's using GPU:"
    echo "  nvidia-smi"
    echo "  docker ps | grep vlnav"
    echo ""
    echo "Continuing anyway — training will OOM if eval is still running."
fi

# ── Step 1: Build image ──────────────────────────────────────────────────────
if [ "${SKIP_BUILD}" = false ] && [ "${EVAL_ONLY}" = false ]; then
    echo ""
    echo "[build] Building vlnav/nexusnav-train:local ..."
    echo "[build] This takes ~45 minutes (pulling nvcr.io/nvidia/pytorch:23.12-py3)"
    cd "${ROOT_DIR}"
    docker build \
        -f scripts/cloud/Dockerfile.nexusnav \
        -t vlnav/nexusnav-train:local \
        --progress=plain \
        . 2>&1 | tee "${LOG_DIR}/docker_build_nexusnav.log"
    echo "[build] DONE"
fi

if [ "${EVAL_ONLY}" = true ]; then
    echo "[run] --eval-only: skipping training, running eval only"
    goto_eval=true
else
    goto_eval=false
fi

MODELS=("navdp" "dualvln" "chrononav" "nexusnav")

if [ "${EVAL_ONLY}" = false ]; then
    # ── Step 2: Train each model sequentially ─────────────────────────────────
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "======================================================="
        echo "  Training: ${MODEL}   $(date)"
        echo "======================================================="

        # Remove any leftover container
        docker rm -f "vlnav_train_${MODEL}" 2>/dev/null || true

        docker compose \
            -f "${COMPOSE}" \
            -p "vlnav_mini_train" \
            run --rm "${MODEL}"

        EXIT_CODE=$?
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "ERROR: ${MODEL} training failed (exit ${EXIT_CODE}). Check ${LOG_DIR}/train_${MODEL}_mini.log"
        else
            echo "[${MODEL}] Training complete."
        fi
    done
fi

# ── Step 3: Collect results ──────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  Collecting training results"
echo "======================================================="

cat > "${RESULTS_DIR}/collect_results.py" << 'PYEOF'
import json, glob, os
from pathlib import Path

results = {}
log_dir = "/mnt/nvme0/vln_habitat/logs"
ckpt_root = "/home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev/checkpoints"

models = {
    "navdp":     ("navdp_vln_ce_mini",     "train_navdp_mini.log"),
    "dualvln":   ("dualsystem_vln_ce_mini", "train_dualvln_mini.log"),
    "chrononav": ("chrononav_vln_ce_mini",  "train_chrononav_mini.log"),
    "nexusnav":  ("nexusnav_vln_ce_mini",   "train_nexusnav_mini.log"),
}

for model_key, (ckpt_name, log_file) in models.items():
    log_path = os.path.join(log_dir, log_file)
    ckpt_dir = os.path.join(ckpt_root, ckpt_name, "ckpts")

    # Parse training loss from log
    train_loss = None
    val_sr = None
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "train_loss" in line.lower() or "loss:" in line.lower():
                try:
                    import re
                    m = re.search(r'loss[:\s=]+([0-9.]+)', line, re.I)
                    if m:
                        train_loss = float(m.group(1))
                        break
                except:
                    pass
            if "val_seen" in line.lower() and "sr" in line.lower():
                try:
                    import re
                    m = re.search(r'SR[:\s=]+([0-9.]+)', line, re.I)
                    if m:
                        val_sr = float(m.group(1))
                except:
                    pass

    # Check if checkpoint exists
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt")) if os.path.exists(ckpt_dir) else []

    results[model_key] = {
        "ckpt_dir": ckpt_dir,
        "n_checkpoints": len(ckpts),
        "train_loss_final": train_loss,
        "val_seen_sr": val_sr,
        "log_path": log_path,
        "log_exists": os.path.exists(log_path),
    }

out_path = "/home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev/results/mini_measured/training_summary.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
PYEOF

python3 "${RESULTS_DIR}/collect_results.py"

echo ""
echo "======================================================="
echo "  Mini training pipeline complete!"
echo "  Results: ${RESULTS_DIR}/training_summary.json"
echo ""
echo "  Next step: run val_unseen Habitat eval for each model:"
echo "  docker run --rm --gpus all ... vlnav/habitat-eval:rebuilt ..."
echo "======================================================="
