#!/bin/bash
# 8-GPU distributed eval for Dual System (mini dataset)

CONFIG=scripts/eval/configs/habitat_dual_system_mini_cfg.py
MODEL_PATH=""
QUIET=""
WANDB_RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --quiet)
            QUIET="--quiet"
            shift
            ;;
        --wandb_run_name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

CONFIG_BASENAME=$(basename "$CONFIG" .py)
CONFIG_PREFIX=$(echo "$CONFIG_BASENAME" | sed 's/_cfg$//')
mkdir -p logs
EVAL_LOG="logs/${CONFIG_PREFIX}_eval.log"

CMD="torchrun --nproc_per_node=8 --master_port=2333 scripts/eval/eval.py --config ${CONFIG} ${QUIET}"
if [ -n "${MODEL_PATH}" ]; then
    CMD="${CMD} --model_path ${MODEL_PATH}"
fi
if [ -n "${WANDB_RUN_NAME}" ]; then
    CMD="${CMD} --wandb_run_name ${WANDB_RUN_NAME}"
fi

echo "[Eval] Config: ${CONFIG}"
echo "[Eval] Model path: ${MODEL_PATH:-from config}"
echo "[Eval] Log: ${EVAL_LOG}"

if [ -n "${MODEL_PATH}" ]; then
    ${CMD} 2>&1 | tee "${EVAL_LOG}" "${MODEL_PATH}/habitat_test.log"
else
    ${CMD} 2>&1 | tee "${EVAL_LOG}"
fi
