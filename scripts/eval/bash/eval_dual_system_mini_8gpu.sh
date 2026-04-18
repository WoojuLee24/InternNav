#!/bin/bash
# 8-GPU distributed eval for Dual System (mini dataset)

CONFIG=scripts/eval/configs/habitat_dual_system_mini_cfg.py
MODEL_PATH=""
QUIET=""

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

echo "[Eval] Config: ${CONFIG}"
echo "[Eval] Model path: ${MODEL_PATH:-from config}"
echo "[Eval] Log: ${EVAL_LOG}"

${CMD} 2>&1 | tee "${EVAL_LOG}"
