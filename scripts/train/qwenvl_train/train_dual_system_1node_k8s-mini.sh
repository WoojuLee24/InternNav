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
llm=Qwen/Qwen2.5-VL-7B-Instruct

# Training hyperparameters
lr=1e-4
batch_size=2
grad_accum_steps=8  # 1 node x 8 GPUs x 2 x 8 = 128 effective batch (same as 8 nodes)
max_pixels=313600
min_pixels=3136

# Dataset configuration
vln_datasets=r2r_125cm_0_30%30,r2r_60cm_15_15%30

# Data path (override with: bash train_dual_system_1node_k8s.sh /path/to/data)
data_root=${1:-/home/irteam/git/InternNav/data/InternData-N1-v0.5-mini/vln_ce} # ${1:-/ws/src/InternNav/data/InternData-N1/vln_ce}

# Output configuration
run_name=InternVLA-N1-DualVLN-v0.5-mini
output_dir=checkpoints/${run_name}
# system 1 options: nextdit_async, navdp_async, nextdit
system1=nextdit_async

system2_ckpt=checkpoints/InternVLA-N1-System2

torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${system2_ckpt}" \
    --vln_dataset_use ${vln_datasets} \
    --data_root ${data_root} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --bf16 \
    \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only True \
    --system1 ${system1} \
    \
    --output_dir ${output_dir} \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1e-05}' \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb
