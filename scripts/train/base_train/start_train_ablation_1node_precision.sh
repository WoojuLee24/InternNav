#!/bin/bash
# ============================================================
#  Ablation training script — precision & dataloader grid
#  Grid: bf16 × tf32 × num_workers
#  Usage:  bash scripts/train/base_train/start_train_ablation_1node_precision.sh
#
#  Wall time reporting:
#    wall_total : torchrun 실행 시작 ~ 종료 (init + 학습 전체)
#    wall_steps : 첫 번째 [Speed] step 로그 ~ 마지막 [Speed] step 로그
# ============================================================

# ---- BASE CONFIG -------------------------------------------
MODEL=navdp_ablation_1node
SESSION_TAG=ablation_precision_$(date +%Y%m%d_%H%M%S)

# ---- GRID: values to sweep ---------------------------------
BF16_LIST=(False True)
TF32_LIST=(False True)
NUM_WORKERS_LIST=(4 8)

# ---- FIXED SETTINGS ----------------------------------------
SCENE_SCALE=0.01
BATCH_SIZE=256
PERSISTENT_WORKERS=True
PRELOAD=False
PREFETCH_FACTOR=2

# ---- GPU ---------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
BASE_PORT=29600

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
[[ -v PREFETCH_FACTOR ]]    && FIXED_ARGS="$FIXED_ARGS --prefetch-factor $PREFETCH_FACTOR"

# ---- Helper: extract epoch seconds from a log line ---------
# Handles both:
#   [2026-03-30 13:15:00.123] ...   (custom print override)
#   2026-03-30 13:15:00 ...          (MyLogger asctime)
parse_ts_epoch() {
    local ts
    ts=$(echo "$1" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1)
    [[ -n "$ts" ]] && date -d "$ts" +%s 2>/dev/null || echo ""
}

fmt_wall() {
    local secs=$1
    printf '%02d:%02d:%02d' $((secs/3600)) $((secs%3600/60)) $((secs%60))
}

# ---- GRID LOOP ---------------------------------------------
LOG_DIR="logs/ablation/${SESSION_TAG}"
mkdir -p "$LOG_DIR"

RESULTS=()
RUN_INDEX=0
TOTAL_RUNS=$(( ${#BF16_LIST[@]} * ${#TF32_LIST[@]} * ${#NUM_WORKERS_LIST[@]} ))

for BF16 in "${BF16_LIST[@]}"; do
  for TF32 in "${TF32_LIST[@]}"; do
    for NW in "${NUM_WORKERS_LIST[@]}"; do
      RUN_INDEX=$((RUN_INDEX + 1))
      NAME="${SESSION_TAG}__bf16${BF16}_tf32${TF32}_nw${NW}"
      EXTRA_ARGS="$FIXED_ARGS --bf16 $BF16 --tf32 $TF32 --num-workers $NW"

      echo ""
      echo "============================================================"
      echo "  Run $RUN_INDEX / $TOTAL_RUNS"
      echo "  Name             : $NAME"
      echo "  bf16             : $BF16"
      echo "  tf32             : $TF32"
      echo "  num_workers      : $NW"
      echo "  Extra args       : $EXTRA_ARGS"
      echo "  Start time       : $(date '+%Y-%m-%d %H:%M:%S')"
      echo "============================================================"

      LOGFILE="$LOG_DIR/${NAME}.log"
      PORT=$((BASE_PORT + RUN_INDEX))

      # ---- wall_total: torchrun 시작 ~ 종료 ------------------
      LAUNCH_START=$(date +%s)

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
      TOTAL_SECS=$((LAUNCH_END - LAUNCH_START))
      WALL_TOTAL=$(fmt_wall $TOTAL_SECS)

      # ---- wall_steps: 첫 step ~ 마지막 step -----------------
      FIRST_STEP_LINE=$(grep '\[Speed\] step=' "$LOGFILE" | head -1)
      LAST_STEP_LINE=$(grep '\[Speed\] step=' "$LOGFILE" | tail -1)

      FIRST_STEP_EPOCH=$(parse_ts_epoch "$FIRST_STEP_LINE")
      LAST_STEP_EPOCH=$(parse_ts_epoch "$LAST_STEP_LINE")

      if [[ -n "$FIRST_STEP_EPOCH" && -n "$LAST_STEP_EPOCH" && "$FIRST_STEP_EPOCH" -lt "$LAST_STEP_EPOCH" ]]; then
          STEP_SECS=$((LAST_STEP_EPOCH - FIRST_STEP_EPOCH))
          WALL_STEPS=$(fmt_wall $STEP_SECS)
      else
          STEP_SECS="n/a"
          WALL_STEPS="n/a"
      fi

      # ---- Speed summary -------------------------------------
      SPEED_LAST=$(grep '\[Speed\] step=' "$LOGFILE" | tail -1)
      SPEED_10=$(grep '\[Speed\] step=' "$LOGFILE" | tail -10)
      echo "${SPEED_10:-n/a}" > "$LOG_DIR/${NAME}_speed.log"

      echo "------------------------------------------------------------"
      echo "  wall_total (init+학습) : ${TOTAL_SECS}s  ($WALL_TOTAL)"
      echo "  wall_steps (step만)    : ${STEP_SECS}s  ($WALL_STEPS)"
      echo "  Exit code              : $EXIT_CODE"
      echo "  Speed (last 10 steps)  :"
      echo "${SPEED_10:-    n/a}"
      echo "  Log                    : $LOGFILE"
      echo "------------------------------------------------------------"

      RESULTS+=("$NAME  exit=$EXIT_CODE  total=${WALL_TOTAL}  steps=${WALL_STEPS}  ${SPEED_LAST:-speed=n/a}")
    done
  done
done

# ---- SUMMARY -----------------------------------------------
echo ""
echo "============================================================"
echo "  GRID SEARCH SUMMARY  ($SESSION_TAG)"
echo "  Grid: bf16=${BF16_LIST[*]}  tf32=${TF32_LIST[*]}  nw=${NUM_WORKERS_LIST[*]}"
echo "  Columns: name | exit | wall_total (init+학습) | wall_steps (step만) | last speed"
echo "============================================================"
for R in "${RESULTS[@]}"; do
  echo "  $R"
done
echo "============================================================"

for R in "${RESULTS[@]}"; do
  [[ "$R" == *"exit=0"* ]] || exit 1
done
exit 0
