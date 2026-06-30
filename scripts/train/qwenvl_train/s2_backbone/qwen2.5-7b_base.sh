#!/bin/bash
# Kubernetes-compatible version (no SLURM required)

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$((RANDOM % 101 + 20001))}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}

# DeepSpeed configuration
deepspeed=scripts/train/qwenvl_train/zero2.json

# Model configuration
llm=$HOME/data-vol2/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct

# Training hyperparameters
lr=2e-5
vision_tower_lr=5e-6
batch_size=4
grad_accum_steps=4  # 1 node x 8 GPUs x 4 x 4 = 128 effective batch (same as 4nodes)
max_pixels=313600
min_pixels=3136

# Validation configuration
val_ratio=0.1          # fraction of data for validation (0.0 to disable)
val_interval_steps=500  # evaluate every N steps

# Dataset configuration
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30

# Data path (override with: bash train_system2_1node_k8s.sh /path/to/data)
data_root=${1:-/home/irteam/git/InternNav/data/InternData-N1/vln_ce}

# Output configuration
run_suffix=7b_$(date +%Y%m%d_%H%M%S)
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
