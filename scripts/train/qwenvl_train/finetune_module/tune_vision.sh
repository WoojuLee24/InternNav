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

# Training hyperparameters
lr=1e-4
batch_size=4
grad_accum_steps=4
max_pixels=313600
min_pixels=3136

# Validation configuration
val_ratio=0.1
val_interval_steps=$((100 * 4 / batch_size))

# Dataset configuration
vln_datasets=r2r_125cm_0_30%30,r2r_60cm_15_15%30

# Data path
data_root=${1:-/home/irteam/git/InternNav/data/InternData-N1-v0.5-mini/vln_ce}

# Output configuration
run_name=finetune_module/tune_vision_$(date +%Y%m%d_%H%M%S)
output_dir=/home/irteam/data-vol2/checkpoints/${run_name}
system1=nextdit_async
system2_ckpt=/home/irteam/data-vol2/checkpoints/InternVLA-N1-System2

# [ablation] tune_mm_vision: True (baseline: False)

mkdir -p ${output_dir}

torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${system2_ckpt}" \
    --vln_dataset_use ${vln_datasets} \
    --data_root ${data_root} \
    --data_flatten False \
    --tune_mm_vision True \
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
    --val_ratio ${val_ratio} \
    --eval_strategy "steps" \
    --eval_steps ${val_interval_steps} \
    --save_strategy "steps" \
    --save_steps ${val_interval_steps} \
    --save_total_limit 2 \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --load_best_model_at_end True \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1e-05}' \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb \
    2>&1 | tee ${output_dir}/train.log

if [ "${NODE_RANK}" = "0" ]; then
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
        wandb_run_id=$(ls -td "wandb/run-"* 2>/dev/null | head -1 | xargs basename | sed 's/^run-[0-9]*_[0-9]*-//')
        if [ -n "${wandb_run_id}" ]; then
            export WANDB_RUN_ID="${wandb_run_id}"
            export WANDB_RESUME="allow"
        fi
        bash scripts/eval/bash/eval_dual_system_mini_8gpu.sh --model_path "${best_ckpt}" --quiet --wandb_run_name "${run_name}" 2>&1 | tee ${output_dir}/test.log
    else
        echo "[Eval] No best checkpoint found, skipping eval."
    fi
fi
