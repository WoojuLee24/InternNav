#!/bin/bash
# Speed benchmark: batch_size x dataloader_num_workers
# Loops over all combinations and reports train_samples_per_second.
# Results are saved to speed_benchmark_results.csv

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}

deepspeed=scripts/train/qwenvl_train/zero2.json
data_root=${1:-/home/irteam/git/InternNav/data/InternData-N1-v0.5-mini/vln_ce}
vln_datasets=r2r_125cm_0_30%30,r2r_60cm_15_15%30
system2_ckpt=checkpoints/InternVLA-N1-System2
system1=nextdit_async
max_pixels=313600
min_pixels=3136

# [benchmark] small dataset + fixed steps for quick throughput measurement
MAX_STEPS=50           # first ~5 steps include CUDA warmup; average over 50 is stable
TRAIN_MAX_SAMPLES=3200 # MAX_STEPS x max_batch_size(8) x n_gpus(8): ensures 50 steps fit in 1 epoch
                       # avoids DataLoader restart overhead penalizing larger batch sizes
GRAD_ACCUM=1           # fixed to isolate batch_size effect

output_dir=checkpoints/speed_benchmark
log_dir=${output_dir}/logs
results_log=${output_dir}/results.csv

mkdir -p ${log_dir}
echo "batch_size,num_workers,train_samples_per_sec,train_steps_per_sec" > ${results_log}

for batch_size in 2 4 8; do
    for num_workers in 4 8 16; do
        echo ""
        echo "=== batch_size=${batch_size}, num_workers=${num_workers} ==="

        # Re-randomize port for each run to avoid port-already-in-use
        MASTER_PORT=$((RANDOM % 1000 + 20001))
        log_file=${log_dir}/bs${batch_size}_nw${num_workers}.log

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
            --num_train_epochs 100 \
            --max_steps ${MAX_STEPS} \
            --train_max_samples ${TRAIN_MAX_SAMPLES} \
            --per_device_train_batch_size ${batch_size} \
            --gradient_accumulation_steps ${GRAD_ACCUM} \
            --max_pixels ${max_pixels} \
            --min_pixels ${min_pixels} \
            --val_ratio 0 \
            --save_strategy "no" \
            --learning_rate 1e-4 \
            --warmup_ratio 0 \
            --weight_decay 0 \
            --max_grad_norm 1 \
            --lr_scheduler_type "constant" \
            --logging_steps 10 \
            --model_max_length 8192 \
            --gradient_checkpointing True \
            --dataloader_num_workers ${num_workers} \
            --run_name "speed_benchmark/bs${batch_size}_nw${num_workers}" \
            --report_to none \
            2>&1 | tee ${log_file}

        # HF Trainer prints: {'train_samples_per_second': 9.698, ...}
        sps=$(grep -oP "'train_samples_per_second':\s*\K[\d.]+" ${log_file} | tail -1)
        tps=$(grep -oP "'train_steps_per_second':\s*\K[\d.]+" ${log_file} | tail -1)

        echo "${batch_size},${num_workers},${sps:-N/A},${tps:-N/A}" | tee -a ${results_log}
    done
done

echo ""
echo "===== Benchmark Results ====="
column -t -s',' ${results_log}
echo ""
echo "Results saved to : ${results_log}"
echo "Per-run logs saved to : ${log_dir}/"
