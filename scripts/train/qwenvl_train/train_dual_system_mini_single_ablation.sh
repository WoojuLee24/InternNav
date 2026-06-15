#!/bin/bash
# ============================================================
#  Ablation training script for Dual System (InternVLA-N1)
#  Single GPU (RTX 5090) — no SLURM
#
#  Usage:
#    bash train_dual_system_ablation.sh          # run all combos
#    bash train_dual_system_ablation.sh --dry-run # print combos only
#
#  파라미터를 _LIST 배열로 설정:
#    값 여러 개 → grid sweep   e.g. LR_LIST=(1e-4 5e-5)
#    값 하나    → 전체 고정     e.g. BATCH_SIZE_LIST=(1)
#    빈 배열 () → base config 기본값 사용 (해당 flag 전달 안 함)
# ============================================================

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

# ---- BASE CONFIG -------------------------------------------
SESSION_TAG=ablation_$(date +%Y%m%d_%H%M%S)
BASE_PORT=29500

deepspeed=scripts/train/qwenvl_train/zero2.json
llm=Qwen/Qwen2.5-VL-3B-Instruct
system2_ckpt=${S2_CKPT:-checkpoints/InternVLA-N1-System2}
data_root=/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_ce

# ---- ENV ---------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- GRID: 값 여러 개 → sweep, 값 하나 → 고정, () → 기본값 사용 ---
LR_LIST=(1e-4)                        # e.g. (1e-4 5e-5 1e-5)
BATCH_SIZE_LIST=(1)                   # e.g. (1 2)
SYSTEM1_LIST=(nextdit_async)          # e.g. (nextdit_async navdp_async nextdit)
NUM_HISTORY_LIST=(8)                  # e.g. (4 8 16)
PREDICT_STEP_NUM_LIST=(32)            # e.g. (16 32)
NUM_FUTURE_STEPS_LIST=(4)             # e.g. (2 4 8)
DATA_AUGMENTATION_LIST=(True)         # e.g. (True False)
PIXEL_GOAL_ONLY_LIST=(True)           # e.g. (True False)
VLN_DATASETS_LIST=("r2r_125cm_0_30,") # e.g. ("r2r_125cm_0_30," "r2r_125cm_0_30,r2r_60cm_15_15,")
MAX_PIXELS_LIST=(65536)               # e.g. (65536 262144)
TUNE_MM_LLM_LIST=()                   # e.g. (True False) — () uses trainer default (False)
TUNE_MM_MLP_LIST=()                   # e.g. (True False)
TUNE_MM_VISION_LIST=()                # e.g. (True False)

# ---- GENERATE COMBINATIONS via Python ----------------------
mapfile -t COMBOS < <(python3 - <<PYEOF
import itertools

SKIP = '__skip__'

def to_list(s):
    s = s.strip()
    return s.split() if s else [SKIP]

params = [
    ('lr',                 'lr',    '--learning_rate',        to_list('${LR_LIST[*]}')),
    ('bs',                 'bs',    '--per_device_train_batch_size', to_list('${BATCH_SIZE_LIST[*]}')),
    ('s1',                 's1',    '--system1',              to_list('${SYSTEM1_LIST[*]}')),
    ('nh',                 'nh',    '--num_history',          to_list('${NUM_HISTORY_LIST[*]}')),
    ('psn',                'psn',   '--predict_step_num',     to_list('${PREDICT_STEP_NUM_LIST[*]}')),
    ('nfs',                'nfs',   '--num_future_steps',     to_list('${NUM_FUTURE_STEPS_LIST[*]}')),
    ('aug',                'aug',   '--data_augmentation',    to_list('${DATA_AUGMENTATION_LIST[*]}')),
    ('pgo',                'pgo',   '--pixel_goal_only',      to_list('${PIXEL_GOAL_ONLY_LIST[*]}')),
    ('ds',                 'ds',    '--vln_dataset_use',      to_list('${VLN_DATASETS_LIST[*]}')),
    ('mp',                 'mp',    '--max_pixels',           to_list('${MAX_PIXELS_LIST[*]}')),
    ('tune_llm',           'tllm',  '--tune_mm_llm',          to_list('${TUNE_MM_LLM_LIST[*]}')),
    ('tune_mlp',           'tmlp',  '--tune_mm_mlp',          to_list('${TUNE_MM_MLP_LIST[*]}')),
    ('tune_vis',           'tvis',  '--tune_mm_vision',       to_list('${TUNE_MM_VISION_LIST[*]}')),
]

lists = [p[3] for p in params]

for combo in itertools.product(*lists):
    name_parts, arg_parts = [], []
    for (name, short, flag, _), val in zip(params, combo):
        if val != SKIP:
            safe_val = val.rstrip(',')  # strip trailing comma for run name
            name_parts.append(f'{short}{safe_val}')
            arg_parts.append(f'{flag} {val}')
    name_suffix = '__'.join(name_parts)
    extra_args  = ' '.join(arg_parts)
    print(f'{name_suffix}|{extra_args}')
PYEOF
)

TOTAL_RUNS=${#COMBOS[@]}

if $DRY_RUN; then
    echo "=== DRY RUN: $TOTAL_RUNS combinations ==="
    IDX=0
    for COMBO in "${COMBOS[@]}"; do
        IDX=$((IDX + 1))
        NAME_SUFFIX="${COMBO%%|*}"
        EXTRA_ARGS="${COMBO#*|}"
        echo "  [$IDX/$TOTAL_RUNS] ${SESSION_TAG}__${NAME_SUFFIX}"
        echo "         args: $EXTRA_ARGS"
    done
    exit 0
fi

# ---- GRID LOOP ---------------------------------------------
LOG_DIR="logs/ablation/${SESSION_TAG}"
mkdir -p "$LOG_DIR"

RESULTS=()
RUN_INDEX=0

for COMBO in "${COMBOS[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))

    NAME_SUFFIX="${COMBO%%|*}"
    EXTRA_ARGS="${COMBO#*|}"
    RUN_NAME="${SESSION_TAG}__${NAME_SUFFIX}"
    OUTPUT_DIR="checkpoints/${RUN_NAME}"
    PORT=$((BASE_PORT + RUN_INDEX))

    echo ""
    echo "============================================================"
    echo "  Run $RUN_INDEX / $TOTAL_RUNS"
    echo "  Name       : $RUN_NAME"
    echo "  Extra args : $EXTRA_ARGS"
    echo "  Start time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    LOGFILE="$LOG_DIR/${RUN_NAME}.log"
    START_TS=$(date +%s)

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
        --nproc_per_node=1 --nnodes=1 --node_rank=0 \
        --master_addr=localhost --master_port=${PORT} \
        internnav/trainer/internvla_n1_trainer.py \
        --deepspeed ${deepspeed} \
        --model_name_or_path "${system2_ckpt}" \
        --data_root ${data_root} \
        --data_flatten False \
        --tune_mm_vision False \
        --tune_mm_mlp False \
        --tune_mm_llm False \
        --bf16 \
        --sample_step 4 \
        --resize_h 384 \
        --resize_w 384 \
        --min_pixels 3136 \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs 1.0 \
        --gradient_accumulation_steps 1 \
        --weight_decay 0 \
        --warmup_ratio 0.003 \
        --max_grad_norm 1 \
        --lr_scheduler_type "cosine_with_min_lr" \
        --lr_scheduler_kwargs '{"min_lr": 1e-05}' \
        --logging_steps 1 \
        --model_max_length 1024 \
        --gradient_checkpointing True \
        --dataloader_num_workers 0 \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps 5000 \
        --save_total_limit 2 \
        --report_to none \
        --run_name "${RUN_NAME}" \
        $EXTRA_ARGS 2>&1 | tee "$LOGFILE"
    EXIT_CODE=${PIPESTATUS[0]}

    END_TS=$(date +%s)
    TOTAL=$((END_TS - START_TS))
    WALL=$(printf '%02d:%02d:%02d' $((TOTAL/3600)) $((TOTAL%3600/60)) $((TOTAL%60)))

    echo "------------------------------------------------------------"
    echo "  Wall time  : ${TOTAL}s  ($WALL)"
    echo "  Exit code  : $EXIT_CODE"
    echo "  Log        : $LOGFILE"
    echo "  Checkpoint : $OUTPUT_DIR"
    echo "------------------------------------------------------------"

    RESULTS+=("${RUN_NAME}  exit=${EXIT_CODE}  wall=${WALL}")
done

# ---- SUMMARY -----------------------------------------------
echo ""
echo "============================================================"
echo "  ABLATION SUMMARY  ($SESSION_TAG)"
echo "============================================================"
for R in "${RESULTS[@]}"; do
    echo "  $R"
done
echo "============================================================"

for R in "${RESULTS[@]}"; do
    [[ "$R" == *"exit=0"* ]] || exit 1
done
exit 0
