#!/bin/bash
# Single GPU debug training for Dual System (RTX 5090, no SLURM)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# DeepSpeed configuration
deepspeed=scripts/train/qwenvl_train/zero2.json

# Training hyperparameters
lr=1e-4
batch_size=1
grad_accum_steps=1
max_pixels=65536
min_pixels=3136

# Dataset configuration
vln_datasets=r2r_125cm_0_30%1

# Data path (override with: bash train_dual_system_debug_single_gpu.sh /path/to/data)
data_root=${1:-/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_ce}

# Output configuration
run_name=InternVLA-N1-DualVLN-single
output_dir=checkpoints/${run_name}
# system 1 options: nextdit_async, navdp_async, nextdit
system1=nextdit_async

system2_ckpt=checkpoints/InternVLA-N1-System2

torchrun --nnodes=1 --nproc_per_node=1 \
    --node_rank=0 \
    --master_addr=localhost --master_port=29501 \
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
    --num_train_epochs 1.0 \
    --max_steps 100 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --val_ratio 0. \
    --val_max_samples 1024 \
    --eval_strategy "steps" \
    --eval_steps 2 \
    --save_strategy "steps" \
    --save_steps 2 \
    --save_total_limit 2 \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --load_best_model_at_end False \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1e-05}' \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --run_name ${run_name} \
    --report_to wandb

# Auto-eval on best checkpoint after training
best_ckpt=$(python -c "
import json, glob, os
state_files = glob.glob('${output_dir}/checkpoint-*/trainer_state.json')
best = None
best_loss = float('inf')
for f in state_files:
    s = json.load(open(f))
    ckpt = os.path.dirname(f)
    loss = s.get('best_metric')
    if loss is not None and loss < best_loss:
        best_loss = loss
        best = ckpt
if best:
    print(best)
" 2>/dev/null)

if [ -n "${best_ckpt}" ]; then
    echo "[Eval] Best checkpoint: ${best_ckpt}"
    cp "${output_dir}/preprocessor_config.json" "${best_ckpt}/" 2>/dev/null || true
    cp "${system2_ckpt}/chat_template.json" "${best_ckpt}/" 2>/dev/null || true
    bash scripts/eval/bash/eval_dual_system_mini_single.sh --model_path "${best_ckpt}" --quiet
else
    echo "[Eval] No best checkpoint found, skipping eval."
fi
