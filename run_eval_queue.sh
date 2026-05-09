#!/bin/bash
# run_eval_queue.sh
# Reads queue_eval.txt and runs single-GPU eval on the best checkpoint,
# logging to the same wandb run as training.
#
# queue_eval.txt format:
#   # [done]      bash scripts/train/qwenvl_train/...  → eval pending
#   # [eval done] bash scripts/train/qwenvl_train/...  → eval complete (skip)
#
# Output dir is inferred as: checkpoints/<subdir>/<name>_* (latest match)

QUEUE_FILE="queue_eval.txt"
LOG_FILE="run_eval_queue.log"
EVAL_SCRIPT="scripts/eval/bash/eval_dual_system_mini_8gpu.sh"
SYSTEM2_CKPT="/home/irteam/data-vol2/checkpoints/InternVLA-N1-System2"
CKPT_BASE="/home/irteam/data-vol2/checkpoints"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "===== run_eval_queue.sh started (PID=$$) ====="

while true; do
    # Find first "# [done]" line that is NOT "[eval done]"
    LINE_NUM=$(grep -n -m1 '^\s*#\s*\[done\]' "$QUEUE_FILE" 2>/dev/null | cut -d: -f1)

    if [ -z "$LINE_NUM" ]; then
        printf "\r[$(date '+%Y-%m-%d %H:%M:%S')] Queue empty, waiting 30s..."
        sleep 30
        continue
    fi

    NEXT=$(sed -n "${LINE_NUM}p" "$QUEUE_FILE" | sed 's/^\s*#\s*\[done\]\s*//')

    log "========================================"
    log "Processing: $NEXT"
    log "========================================"

    # Mark as in-progress to avoid double execution on restart
    sed -i "${LINE_NUM}s|.*|# [eval running] ${NEXT}|" "$QUEUE_FILE"

    # Infer output_dir from script path: scripts/train/qwenvl_train/<sub>/<name>.sh
    # → checkpoints/<sub>/<name>_* (pick latest)
    script_path=$(echo "$NEXT" | sed 's|bash scripts/train/qwenvl_train/||' | sed 's|\.sh$||')
    OUTPUT_DIR=$(ls -td "${CKPT_BASE}/${script_path}_"[0-9]* 2>/dev/null | head -1)

    if [ -z "${OUTPUT_DIR}" ]; then
        log "No checkpoint directory found for ${script_path}, skipping."
        sed -i "${LINE_NUM}s|.*|# [eval skip] ${NEXT}|" "$QUEUE_FILE"
        continue
    fi
    log "Output dir: ${OUTPUT_DIR}"

    # 1. Find best checkpoint via best_model_checkpoint field in last trainer_state.json
    best_ckpt=$(python3 -c "
import json, glob, os, re
state_files = glob.glob('${OUTPUT_DIR}/checkpoint-*/trainer_state.json')
if not state_files:
    exit(1)
state_files.sort(key=lambda f: int(re.search(r'checkpoint-(\d+)', f).group(1)))
last_state = json.load(open(state_files[-1]))
best_path = last_state.get('best_model_checkpoint', '')
if best_path:
    ckpt_name = os.path.basename(best_path)
    best = os.path.join('${OUTPUT_DIR}', ckpt_name)
    if os.path.exists(best):
        print(best)
" 2>/dev/null)

    if [ -z "${best_ckpt}" ]; then
        log "No best checkpoint found in ${OUTPUT_DIR}, skipping."
        sed -i "${LINE_NUM}s|.*|# [eval skip] ${NEXT}|" "$QUEUE_FILE"
        continue
    fi
    log "Best checkpoint: ${best_ckpt}"

    # 2. Copy required config files
    cp "${OUTPUT_DIR}/preprocessor_config.json" "${best_ckpt}/" 2>/dev/null || true
    cp "${SYSTEM2_CKPT}/chat_template.json" "${best_ckpt}/" 2>/dev/null || true

    # 3. Extract wandb run ID from train.log
    wandb_run_id=$(grep -o 'run-[0-9]*_[0-9]*-[a-z0-9]*' "${OUTPUT_DIR}/train.log" 2>/dev/null \
        | head -1 | sed 's/^run-[0-9]*_[0-9]*-//')

    if [ -n "${wandb_run_id}" ]; then
        log "Resuming wandb run: ${wandb_run_id}"
        export WANDB_RUN_ID="${wandb_run_id}"
        export WANDB_RESUME="allow"
    else
        log "Warning: wandb run ID not found, eval will create a new run."
        unset WANDB_RUN_ID
        unset WANDB_RESUME
    fi

    # 4. run_name = output_dir without CKPT_BASE prefix
    run_name="${OUTPUT_DIR#${CKPT_BASE}/}"

    # 5. Run eval (8 GPU)
    bash "${EVAL_SCRIPT}" \
        --model_path "${best_ckpt}" \
        --wandb_run_name "${run_name}" \
        2>&1 | tee -a "$LOG_FILE" "${best_ckpt}/habitat_test.log"
    eval_exit=${PIPESTATUS[0]}

    if [ "${eval_exit}" -eq 0 ]; then
        sed -i "${LINE_NUM}s|.*|# [eval done] ${NEXT}|" "$QUEUE_FILE"
    else
        sed -i "${LINE_NUM}s|.*|# [eval failed] ${NEXT}|" "$QUEUE_FILE"
        log "Eval failed (exit ${eval_exit}): ${OUTPUT_DIR}"
    fi
    log "Finished: ${OUTPUT_DIR}"
    log "========================================"
done
