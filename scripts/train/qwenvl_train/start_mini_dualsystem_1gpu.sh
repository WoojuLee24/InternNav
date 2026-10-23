#!/bin/bash
# Single-GPU mini fine-tuning of InternVLA-N1 DualVLN S2 on R2R-CE 5-scene data.
# Runs on RTX 3080 Ti (12GB) using ZeRO-3 offload + gradient checkpointing.
#
# Usage:
#   bash scripts/train/qwenvl_train/start_mini_dualsystem_1gpu.sh [data_root] [ckpt_out_dir]
#
# Defaults:
#   data_root  = data/InternData-N1-v0.5-mini/vln_ce
#   ckpt_out   = checkpoints/dualsystem_vln_ce_mini
#
# For ChronoNav: temporal cache is inference-time only — train DualVLN the same way.

set -euo pipefail

DATA_ROOT=${1:-data/InternData-N1-v0.5-mini/vln_ce}
CKPT_OUT=${2:-checkpoints/dualsystem_vln_ce_mini}

# Checkpoint to fine-tune from (InternVLA-N1-DualVLN 7B on this machine)
CKPT=/home/kemal/VLNav/VLNav/checkpoints/InternVLA-N1-DualVLN

# DeepSpeed ZeRO-3 + CPU offload for 7B on 12GB GPU
DEEPSPEED=scripts/train/qwenvl_train/zero3_offload.json

# Training hypers: reduced for 12GB single GPU
LR=2e-5
VISION_LR=5e-6
BATCH=1
GRAD_ACCUM=4          # effective batch = 4
MAX_PIXELS=65536       # reduced from 313600 for memory
MIN_PIXELS=3136
EPOCHS=10              # 10 epochs for mini data (120 eps × 4 camera angles = ~480 clips)

# Dataset: all 4 camera-angle variants present in mini data
VLN_DATASETS=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30

RUN_NAME=dualsystem_vln_ce_mini_$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${CKPT_OUT}

echo "=== Mini DualVLN S2 Training (1-GPU) ==="
echo "  checkpoint : $CKPT"
echo "  data_root  : $DATA_ROOT"
echo "  output_dir : $OUTPUT_DIR"
echo "  epochs     : $EPOCHS  batch=$BATCH  grad_accum=$GRAD_ACCUM"
echo ""

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run \
    --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29501 \
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
    2>&1 | tee logs/mini_dualsystem_1gpu_$(date +%Y%m%d_%H%M%S).log
