#!/bin/bash
# ============================================================
#  Ablation training script — grid search  (1 node, all GPUs)
#  Usage:  bash scripts/train/base_train/start_train_ablation_1node.sh
#  Edit the "GRID" and "FIXED SETTINGS" blocks below and re-run.
#  All (num_workers × prefetch_factor)
#  combinations are run sequentially; results are summarised at end.
# ============================================================

# ---- BASE CONFIG -------------------------------------------
MODEL=navdp_ablation_1node         # base config: navdp_1gpu or navdp_1node
SESSION_TAG=ablation_$(date +%Y%m%d_%H%M%S)

# ---- GRID: values to sweep ---------------------------------
NUM_WORKERS_LIST=(4 8)      # e.g. (2 4 8 16)
PREFETCH_FACTOR_LIST=(1 2 4)   # e.g. (1 2 4)

# ---- FIXED SETTINGS ----------------------------------------
# Comment out any line to use the value from the base config file.
SCENE_SCALE=0.01           # data fraction: 0.01 (tiny) | 0.1 | 1.0
BATCH_SIZE=256
PERSISTENT_WORKERS=True    # True | False
PRELOAD=False              # True | False
BF16=False                 # True | False
TF32=False                 # True | False

# ---- GPU ---------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
BASE_PORT=29500   # each run uses BASE_PORT + RUN_INDEX to avoid port conflicts

# ---- ENV ---------------------------------------------------
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- BUILD FIXED EXTRA ARGS --------------------------------
FIXED_ARGS=""
[[ -v SCENE_SCALE ]]        && FIXED_ARGS="$FIXED_ARGS --scene-scale $SCENE_SCALE"
[[ -v BATCH_SIZE ]]         && FIXED_ARGS="$FIXED_ARGS --batch-size $BATCH_SIZE"
[[ -v PERSISTENT_WORKERS ]] && FIXED_ARGS="$FIXED_ARGS --persistent-workers $PERSISTENT_WORKERS"
[[ -v PRELOAD ]]            && FIXED_ARGS="$FIXED_ARGS --preload $PRELOAD"
[[ -v BF16 ]]               && FIXED_ARGS="$FIXED_ARGS --bf16 $BF16"
[[ -v TF32 ]]               && FIXED_ARGS="$FIXED_ARGS --tf32 $TF32"

# ---- GRID LOOP ---------------------------------------------
LOG_DIR="logs/ablation/${SESSION_TAG}"
mkdir -p "$LOG_DIR"

RESULTS=()   # accumulates "name exit_code wall_time" per run
RUN_INDEX=0
TOTAL_RUNS=$(( ${#NUM_WORKERS_LIST[@]} * ${#PREFETCH_FACTOR_LIST[@]} ))

for NW in "${NUM_WORKERS_LIST[@]}"; do
  for PF in "${PREFETCH_FACTOR_LIST[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    NAME="${SESSION_TAG}__nw${NW}_pf${PF}"
    EXTRA_ARGS="$FIXED_ARGS --num-workers $NW --prefetch-factor $PF"

    echo ""
    echo "============================================================"
    echo "  Run $RUN_INDEX / $TOTAL_RUNS"
    echo "  Name             : $NAME"
    echo "  num_workers      : $NW"
    echo "  prefetch_factor  : $PF"
    echo "  Extra args       : $EXTRA_ARGS"
    echo "  Start time       : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    LOGFILE="$LOG_DIR/${NAME}.log"
    LAUNCH_START=$(date +%s)
    PORT=$((BASE_PORT + RUN_INDEX))

    echo "Using torchrun to start $MODEL training, using $NUM_GPUS GPUs (CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES)"
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=$PORT \
        scripts/train/base_train/train.py \
        --name "$NAME" \
        --model-name "$MODEL" \
        $EXTRA_ARGS 2>&1 | tee "$LOGFILE"
    EXIT_CODE=${PIPESTATUS[0]}

    LAUNCH_END=$(date +%s)
    TOTAL=$((LAUNCH_END - LAUNCH_START))
    WALL=$(printf '%02d:%02d:%02d' $((TOTAL/3600)) $((TOTAL%3600/60)) $((TOTAL%60)))

    # Extract last 10 [Speed] lines for quick reporting
    SPEED_LAST=$(grep '\[Speed\] step=' "$LOGFILE" | tail -1)
    SPEED_10=$(grep '\[Speed\] step=' "$LOGFILE" | tail -10)
    echo "${SPEED_10:-n/a}" > "$LOG_DIR/${NAME}_speed.log"

    echo "------------------------------------------------------------"
    echo "  Wall time    : ${TOTAL}s  ($WALL)"
    echo "  Exit code    : $EXIT_CODE"
    echo "  Speed (last 10 steps):"
    echo "${SPEED_10:-    n/a}"
    echo "  Log          : $LOGFILE"
    echo "------------------------------------------------------------"

    RESULTS+=("$NAME  exit=$EXIT_CODE  wall=${WALL}  ${SPEED_LAST:-speed=n/a}")
  done
done

# ---- SUMMARY -----------------------------------------------
echo ""
echo "============================================================"
echo "  GRID SEARCH SUMMARY  ($SESSION_TAG)"
echo "============================================================"
for R in "${RESULTS[@]}"; do
  echo "  $R"
done
echo "============================================================"

# Return non-zero if any run failed
for R in "${RESULTS[@]}"; do
  [[ "$R" == *"exit=0"* ]] || exit 1
done
exit 0
