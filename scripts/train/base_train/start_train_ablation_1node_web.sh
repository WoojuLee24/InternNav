#!/bin/bash
# ============================================================
#  Ablation training script — WebDataset vs npy cache 속도 비교
#  Grid: use_webdataset × num_workers
#  Usage:  bash scripts/train/base_train/start_train_ablation_1node_web.sh
#
#  Wall time reporting:
#    wall_total : torchrun 실행 시작 ~ 종료 (init + 학습 전체)
#    wall_steps : 첫 번째 [Speed] step 로그 ~ 마지막 [Speed] step 로그
#
#  Prerequisite:
#    python scripts/train/base_train/convert_to_webdataset.py \
#        --src-dir ~/data-vol1/InternData-N1-v0.5-mini_backup/vln_n1/traj_data/matterport3d_d435i \
#        --out-dir ~/data-vol1/InternData-N1-v0.5-mini/vln_n1/webdataset_shards/matterport3d_d435i \
#        --workers 8
# ============================================================

# ---- BASE CONFIG -------------------------------------------
MODEL=navdp_ablation_1node
SESSION_TAG=ablation_web_$(date +%Y%m%d_%H%M%S)

# ---- WEBDATASET SHARD DIR ----------------------------------
SHARD_DIR="${HOME}/data-vol1/InternData-N1-v0.5-mini/vln_n1/webdataset_shards/matterport3d_d435i"

# ---- GRID: values to sweep ---------------------------------
# USE_WEB=True  → NavDP_WebDataset (episode-packed shards)
# USE_WEB=False → NavDP_Base_Datset with npy_cache (baseline)
USE_WEB_LIST=(True False)
NUM_WORKERS_LIST=(4 8)

# ---- FIXED SETTINGS ----------------------------------------
SCENE_SCALE=0.01
BATCH_SIZE=256
PERSISTENT_WORKERS=True
PREFETCH_FACTOR=2
BF16=False
TF32=False

# ---- GPU ---------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
BASE_PORT=29800

# ---- ENV ---------------------------------------------------
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Helper: extract epoch seconds from a log line ---------
parse_ts_epoch() {
    local ts
    ts=$(echo "$1" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1)
    [[ -n "$ts" ]] && date -d "$ts" +%s 2>/dev/null || echo ""
}

fmt_wall() {
    local secs=$1
    printf '%02d:%02d:%02d' $((secs/3600)) $((secs%3600/60)) $((secs%60))
}

# ---- LOG DIR -----------------------------------------------
LOG_DIR="logs/ablation/${SESSION_TAG}"
mkdir -p "$LOG_DIR"

RESULTS=()
RUN_INDEX=0
TOTAL_RUNS=$(( ${#USE_WEB_LIST[@]} * ${#NUM_WORKERS_LIST[@]} ))

for USE_WEB in "${USE_WEB_LIST[@]}"; do
  for NW in "${NUM_WORKERS_LIST[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    NAME="${SESSION_TAG}__web${USE_WEB}_nw${NW}"

    # Build extra args
    EXTRA_ARGS=""
    EXTRA_ARGS="$EXTRA_ARGS --scene-scale $SCENE_SCALE"
    EXTRA_ARGS="$EXTRA_ARGS --batch-size $BATCH_SIZE"
    EXTRA_ARGS="$EXTRA_ARGS --persistent-workers $PERSISTENT_WORKERS"
    EXTRA_ARGS="$EXTRA_ARGS --prefetch-factor $PREFETCH_FACTOR"
    EXTRA_ARGS="$EXTRA_ARGS --bf16 $BF16"
    EXTRA_ARGS="$EXTRA_ARGS --tf32 $TF32"
    EXTRA_ARGS="$EXTRA_ARGS --num-workers $NW"
    EXTRA_ARGS="$EXTRA_ARGS --use-webdataset $USE_WEB"
    if [[ "$USE_WEB" == "True" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --webdataset-shard-dir $SHARD_DIR"
        # webdataset does not use npy-cache path
        EXTRA_ARGS="$EXTRA_ARGS --use-npy-cache False"
    else
        EXTRA_ARGS="$EXTRA_ARGS --use-npy-cache True"
    fi

    echo ""
    echo "============================================================"
    echo "  Run $RUN_INDEX / $TOTAL_RUNS"
    echo "  Name             : $NAME"
    echo "  use_webdataset   : $USE_WEB"
    echo "  num_workers      : $NW"
    echo "  Extra args       : $EXTRA_ARGS"
    echo "  Start time       : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    LOGFILE="$LOG_DIR/${NAME}.log"
    PORT=$((BASE_PORT + RUN_INDEX))

    LAUNCH_START=$(date +%s)

    echo "Using torchrun to start $MODEL training, using $NUM_GPUS GPUs"
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

    # wall_steps: 첫 step ~ 마지막 step
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

# ---- SUMMARY -----------------------------------------------
echo ""
echo "============================================================"
echo "  GRID SEARCH SUMMARY  ($SESSION_TAG)"
echo "  Grid: use_webdataset=${USE_WEB_LIST[*]}  nw=${NUM_WORKERS_LIST[*]}"
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
