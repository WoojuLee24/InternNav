#!/bin/bash
# Single GPU debug training for System2 (RTX 5090, no SLURM)

# DeepSpeed configuration
deepspeed=scripts/train/qwenvl_train/zero3_offload.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct

# Training hyperparameters
lr=2e-5
vision_tower_lr=5e-6
batch_size=1
grad_accum_steps=1
max_pixels=65536
min_pixels=3136

# Dataset configuration
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30

# Data path
data_root=/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_ce

# Output configuration
run_name=InternVLA-N1-System2-debug
output_dir=checkpoints/${run_name}

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run \
    --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
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
    --output_dir ${output_dir} \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate ${lr} \
    --vision_tower_lr ${vision_tower_lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to none
