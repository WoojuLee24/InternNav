#!/bin/bash
# ============================================================
#  DataLoader 병목 측정 — I/O vs CPU (CubicSpline) 분리
#  Grid: use_npy_cache × num_workers
#  각 설정에서 N batches 수행 후 throughput + 타이밍 breakdown 출력
#
#  Usage:  bash scripts/train/base_train/start_benchmark_dataloader.sh
#
#  [해석]
#    worker stdout에 출력되는 [TimedNavDP] 라인:
#      io=Xms    → process_image + process_depth 시간
#      cpu=Xms   → process_actions × 2 (CubicSpline 포함)
#      other=Xms → xyz_to_xyt, 거리 계산, tensor 변환 등
#
#    io >> cpu → I/O 병목 → WebDataset/npy 전략 유효
#    cpu >> io → CPU 병목 → num_workers 증가, CubicSpline 최적화 필요
# ============================================================

# ---- FIXED SETTINGS ----------------------------------------
MODEL=navdp_ablation_1node
SCENE_SCALE=0.01
BATCH_SIZE=256
N_BATCHES=30
WARMUP_BATCHES=5
PREFETCH_FACTOR=2
PERSISTENT_WORKERS=True

# ---- GRID --------------------------------------------------
NPY_CACHE_LIST=(True False)
NUM_WORKERS_LIST=(4 8)

# ---- LOG DIR -----------------------------------------------
SESSION_TAG=benchmark_$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/benchmark/${SESSION_TAG}"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""  # GPU 사용 안 함 (DataLoader만 측정)

RESULTS=()
RUN_INDEX=0
TOTAL_RUNS=$(( ${#NPY_CACHE_LIST[@]} * ${#NUM_WORKERS_LIST[@]} ))

for NPY in "${NPY_CACHE_LIST[@]}"; do
  for NW in "${NUM_WORKERS_LIST[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    NAME="${SESSION_TAG}__npy${NPY}_nw${NW}"
    LOGFILE="${LOG_DIR}/${NAME}.log"

    echo ""
    echo "============================================================"
    echo "  Run $RUN_INDEX / $TOTAL_RUNS"
    echo "  use_npy_cache : $NPY"
    echo "  num_workers   : $NW"
    echo "  Start time    : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    START_TS=$(date +%s)

    python scripts/train/base_train/benchmark_dataloader.py \
        --model-name "$MODEL" \
        --num-workers "$NW" \
        --batch-size "$BATCH_SIZE" \
        --n-batches "$N_BATCHES" \
        --warmup-batches "$WARMUP_BATCHES" \
        --scene-scale "$SCENE_SCALE" \
        --use-npy-cache "$NPY" \
        --prefetch-factor "$PREFETCH_FACTOR" \
        --persistent-workers "$PERSISTENT_WORKERS" \
        2>&1 | tee "$LOGFILE"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    # 결과 파싱: Throughput mean
    THROUGHPUT=$(grep 'Throughput.*mean' "$LOGFILE" | awk '{print $(NF-1)}' | head -1)
    BATCH_MEAN=$(grep 'Batch time.*mean' "$LOGFILE" | awk '{print $(NF-1)}' | head -1)

    # worker 타이밍 (마지막 출력)
    IO_MS=$(grep '\[TimedNavDP\]' "$LOGFILE" | tail -1 | grep -oP 'io=\K[\d.]+')
    CPU_MS=$(grep '\[TimedNavDP\]' "$LOGFILE" | tail -1 | grep -oP 'cpu\(CubicSpline\)=\K[\d.]+')
    TOT_MS=$(grep '\[TimedNavDP\]' "$LOGFILE" | tail -1 | grep -oP 'total=\K[\d.]+')

    echo ""
    echo "------------------------------------------------------------"
    echo "  use_npy_cache : $NPY   num_workers : $NW"
    echo "  throughput    : ${THROUGHPUT:-n/a} samples/s"
    echo "  batch_time    : ${BATCH_MEAN:-n/a} ms (mean)"
    echo "  [worker]  io  : ${IO_MS:-n/a} ms/sample"
    echo "  [worker]  cpu : ${CPU_MS:-n/a} ms/sample  (CubicSpline)"
    echo "  [worker]  tot : ${TOT_MS:-n/a} ms/sample"
    echo "  exit=$EXIT_CODE  elapsed=${ELAPSED}s"
    echo "------------------------------------------------------------"

    RESULTS+=("npy=$NPY nw=$NW | throughput=${THROUGHPUT:-?} s/s | io=${IO_MS:-?}ms cpu=${CPU_MS:-?}ms tot=${TOT_MS:-?}ms")
  done
done

# ---- SUMMARY -----------------------------------------------
echo ""
echo "============================================================"
echo "  BENCHMARK SUMMARY  ($SESSION_TAG)"
echo "  Columns: config | throughput | io(ms) | cpu(ms) | total(ms)"
echo "============================================================"
for R in "${RESULTS[@]}"; do
  echo "  $R"
done
echo "============================================================"
echo ""
echo "  [판단 기준]"
echo "  io >> cpu  → I/O 병목 → WebDataset 유효"
echo "  cpu >> io  → CPU 병목 → num_workers 증가 or CubicSpline 최적화"
echo "============================================================"
