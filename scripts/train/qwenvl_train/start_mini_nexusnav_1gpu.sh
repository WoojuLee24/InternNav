#!/bin/bash
# NexusNav-V3 mini fine-tuning on R2R-CE 5-scene data.
# Single GPU (RTX 3080 Ti 12GB) via ZeRO-3 CPU offload.
#
# NexusNav training = InternVLA-N1 (DualVLN base) fine-tuning on CE mini data.
# The NexusNav S2 innovations (S2KVCache, VisionEmbeddingCache τ=0.92,
# WorldPredictor, ActionTokenVocab) are inference-time modules — they attach
# on top of the fine-tuned checkpoint at eval time.
#
# Usage:
#   bash scripts/train/qwenvl_train/start_mini_nexusnav_1gpu.sh [data_root] [ckpt_out_dir]
#
# Defaults:
#   data_root  = data/InternData-N1-v0.5-mini/vln_ce
#   ckpt_out   = checkpoints/nexusnav_vln_ce_mini

set -euo pipefail

DATA_ROOT=${1:-data/InternData-N1-v0.5-mini/vln_ce}
CKPT_OUT=${2:-checkpoints/nexusnav_vln_ce_mini}

# Base: same DualVLN checkpoint — NexusNav builds on top at inference
CKPT=/home/kemal/VLNav/VLNav/checkpoints/InternVLA-N1-DualVLN

DEEPSPEED=scripts/train/qwenvl_train/zero3_offload.json

LR=2e-5
VISION_LR=5e-6
BATCH=1
GRAD_ACCUM=4
MAX_PIXELS=65536
MIN_PIXELS=3136
EPOCHS=10

VLN_DATASETS=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30

RUN_NAME=nexusnav_vln_ce_mini_$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${CKPT_OUT}

echo "=== NexusNav-V3 Mini Training (1-GPU) ==="
echo "  base checkpoint : $CKPT"
echo "  data_root       : $DATA_ROOT"
echo "  output_dir      : $OUTPUT_DIR"
echo "  epochs          : $EPOCHS  batch=$BATCH  grad_accum=$GRAD_ACCUM"
echo "  NOTE: S2KVCache / VisionEmbCache / ActionTokenVocab attach at eval."
echo ""

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run \
    --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29502 \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${DEEPSPEED} \
    --model_name_or_path "${CKPT}" \
    --vln_dataset_use ${VLN_DATASETS} \
    --data_root ${DATA_ROOT} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    \
    --num_history 4 \
    --data_augmentation True \
    --resize_h 256 \
    --resize_w 256 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
    \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH} \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_pixels ${MAX_PIXELS} \
    --min_pixels ${MIN_PIXELS} \
    --val_ratio 0.1 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate ${LR} \
    --vision_tower_lr ${VISION_LR} \
    --weight_decay 0 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${RUN_NAME} \
    --report_to none \
    2>&1 | tee logs/mini_nexusnav_1gpu_$(date +%Y%m%d_%H%M%S).log
