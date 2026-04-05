#!/bin/bash
# ============================================================
#  Ablation training script — flexible grid search (1node 8GPU)
#  Usage:  bash start_train_1node_ablation.sh
#
#  각 파라미터를 _LIST 배열로 설정:
#    값 여러 개 → grid sweep   e.g. NUM_WORKERS_LIST=(4 8 16)
#    값 하나    → 전체 고정     e.g. BATCH_SIZE_LIST=(32)
#    빈 배열   → base config 사용  e.g. BF16_LIST=()
#
#  모든 조합(cartesian product)을 순차 실행하고 결과를 요약.
# ============================================================

# ---- ARGS --------------------------------------------------
TIMING_FLAG=""
for arg in "$@"; do
    [[ "$arg" == "--time" ]] && TIMING_FLAG="--enable-timing"
done

# ---- BASE CONFIG -------------------------------------------
MODEL=navdp_1node_ablation         # base config: navdp_1gpu or navdp_1node
SESSION_TAG=ablation_$(date +%Y%m%d_%H%M%S)

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


# ---- GRID: values to sweep ---------------------------------
# 여러 값 → sweep, 값 하나 → 고정, 빈 배열 () → base config 사용
SCENE_SCALE_LIST=(0.1)       # e.g. (0.01 0.1 1.0)
NUM_WORKERS_LIST=(4)        # e.g. (2 4 8 16)
PREFETCH_FACTOR_LIST=(1)      # e.g. (1 2 4)
BATCH_SIZE_LIST=(32)          # e.g. (16 32 64)
PERSISTENT_WORKERS_LIST=(True) # e.g. (True False)
PRELOAD_LIST=(False)          # e.g. (True False)
BF16_LIST=(False)             # e.g. (True False)
TF32_LIST=(False)             # e.g. (True False)
USE_SCIPY_KDTREE_LIST=(True)  # e.g. (True False)
USE_NPY_OBSTACLE_LIST=(True False)  # e.g. (True False)
USE_NPZ_PARQUET_LIST=(True False)  # e.g. (True False)
USE_KDTREE_CACHE_LIST=(True False)  # e.g. (True False)
USE_PARQUET_CACHE_LIST=(True False)  # e.g. (True False)


# ---- GENERATE COMBINATIONS via Python ----------------------
# Python이 cartesian product를 계산하여 각 줄에 "NAME_SUFFIX|--flag val ..." 형태로 출력
mapfile -t COMBOS < <(python3 - <<PYEOF
import itertools

SKIP = '__skip__'

def to_list(s):
    s = s.strip()
    return s.split() if s else [SKIP]

# (파라미터명, 짧은약어, CLI flag, 값 목록)
params = [
    ('num_workers',        'nw',   '--num-workers',        to_list('${NUM_WORKERS_LIST[*]}')),
    ('prefetch_factor',    'pf',   '--prefetch-factor',    to_list('${PREFETCH_FACTOR_LIST[*]}')),
    ('batch_size',         'bs',   '--batch-size',         to_list('${BATCH_SIZE_LIST[*]}')),
    ('scene_scale',        'ss',   '--scene-scale',        to_list('${SCENE_SCALE_LIST[*]}')),
    ('persistent_workers', 'pw',   '--persistent-workers', to_list('${PERSISTENT_WORKERS_LIST[*]}')),
    ('preload',            'pl',   '--preload',            to_list('${PRELOAD_LIST[*]}')),
    ('bf16',               'bf16', '--bf16',               to_list('${BF16_LIST[*]}')),
    ('tf32',               'tf32', '--tf32',               to_list('${TF32_LIST[*]}')),
    ('use_scipy_kdtree',   'kdt',  '--use-scipy-kdtree',   to_list('${USE_SCIPY_KDTREE_LIST[*]}')),
    ('use_npy_obstacle',   'npy',  '--use-npy-obstacle',   to_list('${USE_NPY_OBSTACLE_LIST[*]}')),
    ('use_npz_parquet',    'npzp', '--use-npz-parquet',    to_list('${USE_NPZ_PARQUET_LIST[*]}')),
    ('use_kdtree_cache',   'kdc',  '--use-kdtree-cache',   to_list('${USE_KDTREE_CACHE_LIST[*]}')),
    ('use_parquet_cache',  'pqc',  '--use-parquet-cache',  to_list('${USE_PARQUET_CACHE_LIST[*]}')),
]

lists = [p[3] for p in params]

for combo in itertools.product(*lists):
    name_parts, arg_parts = [], []
    for (name, short, flag, _), val in zip(params, combo):
        if val != SKIP:
            name_parts.append(f'{short}{val}')
            arg_parts.append(f'{flag} {val}')
    name_suffix = '__'.join(name_parts)
    extra_args  = ' '.join(arg_parts)
    print(f'{name_suffix}|{extra_args}')
PYEOF
)


# ---- GRID LOOP ---------------------------------------------
LOG_DIR="logs/ablation/${SESSION_TAG}"
mkdir -p "$LOG_DIR"

RESULTS=()
RUN_INDEX=0
TOTAL_RUNS=${#COMBOS[@]}

for COMBO in "${COMBOS[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))

    NAME_SUFFIX="${COMBO%%|*}"
    EXTRA_ARGS="${COMBO#*|}"
    NAME="${SESSION_TAG}__${NAME_SUFFIX}"

    echo ""
    echo "============================================================"
    echo "  Run $RUN_INDEX / $TOTAL_RUNS"
    echo "  Name       : $NAME"
    echo "  Extra args : $EXTRA_ARGS"
    echo "  Start time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    LOGFILE="$LOG_DIR/${NAME}.log"
    LAUNCH_START=$(date +%s)
    PORT=$((BASE_PORT + RUN_INDEX))

    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=$PORT \
        scripts/train/base_train/train_ablation.py \
        --name "$NAME" \
        --model-name "$MODEL" \
        $TIMING_FLAG \
        $EXTRA_ARGS 2>&1 | tee "$LOGFILE"
    EXIT_CODE=${PIPESTATUS[0]}

    LAUNCH_END=$(date +%s)
    TOTAL=$((LAUNCH_END - LAUNCH_START))
    WALL=$(printf '%02d:%02d:%02d' $((TOTAL/3600)) $((TOTAL%3600/60)) $((TOTAL%60)))

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
