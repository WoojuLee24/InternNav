#!/bin/bash
# ============================================================
#  최적화 전/후 속도 비교 스크립트
#
#  비교 대상:
#    original  : HEAD 버전 (브루트포스 거리 계산 + Image.open in process_pixel_goal)
#    optimized : working tree 버전 (cKDTree + orig_img_size 캐시)
#
#  Fixed config: use_npy_cache=True, num_workers=4
#   (이전 ablation에서 최적 설정으로 확인된 값)
#
#  Usage:
#    bash scripts/train/base_train/start_compare_optimization.sh
# ============================================================

set -e

DATASET_FILE="internnav/dataset/navdp_lerobot_dataset.py"
OPTIMIZED_TMP="/tmp/navdp_dataset_optimized_$$.py"
ORIGINAL_TMP="/tmp/navdp_dataset_original_$$.py"

# ---- CONFIG ------------------------------------------------
MODEL=navdp_ablation_1node
SESSION_TAG=compare_opt_$(date +%Y%m%d_%H%M%S)
SCENE_SCALE=0.01
BATCH_SIZE=256
NUM_WORKERS=4
PERSISTENT_WORKERS=True
PRELOAD=False
PREFETCH_FACTOR=2
BF16=False
TF32=False
USE_NPY_CACHE=True

# ---- GPU ---------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
BASE_PORT=29900

# ---- ENV ---------------------------------------------------
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- LOG DIR -----------------------------------------------
LOG_DIR="logs/compare/${SESSION_TAG}"
mkdir -p "$LOG_DIR"

# ---- Helper ------------------------------------------------
parse_ts_epoch() {
    local ts
    ts=$(echo "$1" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1)
    [[ -n "$ts" ]] && date -d "$ts" +%s 2>/dev/null || echo ""
}

fmt_wall() {
    local secs=$1
    printf '%02d:%02d:%02d' $((secs/3600)) $((secs%3600/60)) $((secs%60))
}

cleanup() {
    # 스크립트 종료 시 원본 복원 보장
    if [[ -f "$OPTIMIZED_TMP" ]]; then
        echo "[cleanup] Restoring optimized file..."
        cp "$OPTIMIZED_TMP" "$DATASET_FILE"
        rm -f "$OPTIMIZED_TMP" "$ORIGINAL_TMP"
    fi
}
trap cleanup EXIT

# ---- SAVE: 현재 (optimized) working tree 버전 백업 ---------
cp "$DATASET_FILE" "$OPTIMIZED_TMP"

# ---- EXTRACT: HEAD (original) 버전 추출 --------------------
git show HEAD:"$DATASET_FILE" > "$ORIGINAL_TMP"

echo ""
echo "============================================================"
echo "  최적화 전/후 속도 비교"
echo "  Session : $SESSION_TAG"
echo "  Config  : npy_cache=$USE_NPY_CACHE  nw=$NUM_WORKERS  bs=$BATCH_SIZE"
echo "  GPUs    : $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "============================================================"

RESULTS=()
PORT=$BASE_PORT

run_variant() {
    local VARIANT="$1"   # "original" or "optimized"
    local VARIANT_FILE="$2"

    PORT=$((PORT + 1))
    NAME="${SESSION_TAG}__${VARIANT}"
    LOGFILE="$LOG_DIR/${NAME}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "  Variant    : $VARIANT"
    echo "  Start time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    # 해당 버전으로 파일 교체
    cp "$VARIANT_FILE" "$DATASET_FILE"

    LAUNCH_START=$(date +%s)

    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=$PORT \
        scripts/train/base_train/train.py \
        --name "$NAME" \
        --model-name "$MODEL" \
        --scene-scale "$SCENE_SCALE" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --use-npy-cache "$USE_NPY_CACHE" \
        --persistent-workers "$PERSISTENT_WORKERS" \
        --preload "$PRELOAD" \
        --prefetch-factor "$PREFETCH_FACTOR" \
        --bf16 "$BF16" \
        --tf32 "$TF32" \
        2>&1 | tee "$LOGFILE"
    local EXIT_CODE=${PIPESTATUS[0]}

    LAUNCH_END=$(date +%s)
    TOTAL_SECS=$((LAUNCH_END - LAUNCH_START))
    WALL_TOTAL=$(fmt_wall $TOTAL_SECS)

    FIRST_STEP_LINE=$(grep '\[Speed\] step=' "$LOGFILE" | head -1)
    LAST_STEP_LINE=$(grep '\[Speed\] step='  "$LOGFILE" | tail -1)

    FIRST_STEP_EPOCH=$(parse_ts_epoch "$FIRST_STEP_LINE")
    LAST_STEP_EPOCH=$(parse_ts_epoch  "$LAST_STEP_LINE")

    if [[ -n "$FIRST_STEP_EPOCH" && -n "$LAST_STEP_EPOCH" && "$FIRST_STEP_EPOCH" -lt "$LAST_STEP_EPOCH" ]]; then
        STEP_SECS=$((LAST_STEP_EPOCH - FIRST_STEP_EPOCH))
        WALL_STEPS=$(fmt_wall $STEP_SECS)
    else
        STEP_SECS="n/a"
        WALL_STEPS="n/a"
    fi

    SPEED_LAST=$(grep '\[Speed\] step=' "$LOGFILE" | tail -1)
    SPEED_10=$(grep '\[Speed\] step='   "$LOGFILE" | tail -10)
    echo "${SPEED_10:-n/a}" > "$LOG_DIR/${NAME}_speed.log"

    echo ""
    echo "  [${VARIANT}] wall_total : ${TOTAL_SECS}s  ($WALL_TOTAL)"
    echo "  [${VARIANT}] wall_steps : ${STEP_SECS}s   ($WALL_STEPS)"
    echo "  [${VARIANT}] exit       : $EXIT_CODE"
    echo "  [${VARIANT}] Speed (last 10 steps):"
    echo "${SPEED_10:-    n/a}"

    RESULTS+=("variant=${VARIANT}  exit=$EXIT_CODE  total=${WALL_TOTAL}  steps=${WALL_STEPS}  ${SPEED_LAST:-speed=n/a}")
}

# ---- RUN 1: original (HEAD) --------------------------------
run_variant "original"  "$ORIGINAL_TMP"

# ---- RUN 2: optimized (working tree) -----------------------
run_variant "optimized" "$OPTIMIZED_TMP"

# ---- SUMMARY -----------------------------------------------
echo ""
echo "============================================================"
echo "  OPTIMIZATION COMPARISON SUMMARY  ($SESSION_TAG)"
echo "  npy_cache=$USE_NPY_CACHE  num_workers=$NUM_WORKERS  batch_size=$BATCH_SIZE"
echo "  Columns: variant | exit | wall_total | wall_steps | last speed"
echo "============================================================"
for R in "${RESULTS[@]}"; do
    echo "  $R"
done
echo "============================================================"
echo ""

# speedup 계산 (wall_steps 기준)
ORIG_STEPS=$(echo "${RESULTS[0]}" | grep -oP 'steps=\K[\d:]+')
OPT_STEPS=$(echo  "${RESULTS[1]}" | grep -oP 'steps=\K[\d:]+')
echo "  original  wall_steps : $ORIG_STEPS"
echo "  optimized wall_steps : $OPT_STEPS"
echo ""
echo "  [비교 기준]"
echo "  wall_steps : DataLoader + forward/backward 포함 전체 step 시간"
echo "               (init 제외, GPU 연산 포함이므로 dataloader 효과가 희석될 수 있음)"
echo "============================================================"
