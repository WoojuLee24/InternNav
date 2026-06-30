#!/bin/bash
# System2 training (Qwen3-VL-32B) — Kubernetes-compatible
# Diff from qwen3_8b.sh: llm=Qwen3-VL-32B-Instruct, zero3, batch_size=1, grad_accum=16
#
# Prerequisites before running:
#   1. Upgrade transformers: pip install "transformers>=4.57.0"
#   2. Add Qwen3-VL support to internvla_n1_trainer.py:
#      - import Qwen3VLForConditionalGeneration
#      - add elif "qwen3" branch (similar to qwen2.5 branch)
#   3. Add "qwen3vl" model_type branch to internvla_n1_lerobot_dataset.py
#      (get_rope_index_25 is likely reusable for Qwen3-VL)

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$((RANDOM % 101 + 20001))}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}

# DeepSpeed configuration (ZeRO-3 required for 32B)
deepspeed=scripts/train/qwenvl_train/zero3.json

# Model configuration
llm=$HOME/data-vol2/checkpoints/Qwen/Qwen3-VL-32B-Instruct

# Training hyperparameters
# 1node x 8GPU x bs1 x accum16 = 128 effective batch
lr=2e-5
vision_tower_lr=5e-6
batch_size=1
grad_accum_steps=16
max_pixels=313600
min_pixels=3136

# Validation configuration
val_ratio=0.1
val_interval_steps=500

# Dataset configuration
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30

# Data path
data_root=${1:-/home/irteam/git/InternNav/data/InternData-N1/vln_ce}

# Output configuration
run_suffix=qwen3_32b_$(date +%Y%m%d_%H%M%S)
run_name=s2_backbone/${run_suffix}
output_dir=$HOME/data-vol2/checkpoints/s2_backbone/${run_suffix}

mkdir -p ${output_dir}

torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --vln_dataset_use ${vln_datasets} \
    --data_root ${data_root} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
    \
    --output_dir ${output_dir} \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --val_ratio ${val_ratio} \
    --eval_strategy "steps" \
    --eval_steps ${val_interval_steps} \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --vision_tower_lr ${vision_tower_lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb \
    2>&1 | tee ${output_dir}/train.log
