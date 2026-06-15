#!/bin/bash
# 8-GPU distributed Habitat eval for Dual System with BEV visual input (mini).
# Based on eval_dual_system_mini_8gpu.sh; adds --bev_s1_mode / --bev_s2_mode,
# which eval.py forwards onto agent.model_settings (BEV config only).

CONFIG=scripts/eval/configs/habitat_dual_system_mini_h200_bev_cfg.py
MODEL_PATH=""
QUIET=""
WANDB_RUN_NAME=""
BEV_S1_MODE="bev"
BEV_S2_MODE="fpv"

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
        --bev_s1_mode)
            BEV_S1_MODE="$2"
            shift 2
            ;;
        --bev_s2_mode)
            BEV_S2_MODE="$2"
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
EVAL_LOG="logs/${CONFIG_PREFIX}_s1${BEV_S1_MODE}_s2${BEV_S2_MODE}_eval.log"

CMD="torchrun --nproc_per_node=8 --master_port=2333 scripts/eval/eval.py --config ${CONFIG} ${QUIET} --bev_s1_mode ${BEV_S1_MODE} --bev_s2_mode ${BEV_S2_MODE}"
if [ -n "${MODEL_PATH}" ]; then
    CMD="${CMD} --model_path ${MODEL_PATH}"
fi
if [ -n "${WANDB_RUN_NAME}" ]; then
    CMD="${CMD} --wandb_run_name ${WANDB_RUN_NAME}"
fi

echo "[Eval] Config: ${CONFIG}  (bev_s1_mode=${BEV_S1_MODE}, bev_s2_mode=${BEV_S2_MODE})"
echo "[Eval] Model path: ${MODEL_PATH:-from config}"
echo "[Eval] Log: ${EVAL_LOG}"

if [ -n "${MODEL_PATH}" ]; then
    ${CMD} 2>&1 | tee "${EVAL_LOG}" "${MODEL_PATH}/habitat_test.log"
else
    ${CMD} 2>&1 | tee "${EVAL_LOG}"
fi
