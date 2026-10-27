#!/usr/bin/env bash
# run_mini_docker_train.sh — Direct docker run training for mini models
# Bypasses docker compose (which has replace=thor_gd_vln issue on this host)
# Usage: bash scripts/train/run_mini_docker_train.sh [dualvln|chrononav|nexusnav]

set -uo pipefail

MODEL=${1:-dualvln}
# Calibration cap (user 2026-10-27): full 5-epoch run = ~8-10 days/model at ~76s/it (NVMe optimizer offload),
# which blows the Nov 1 freeze. ~600 steps ≈ 12.7h/model gives a real measured fine-tuned ckpt for the comparison
# (calibration test: "does fine-tuning help vs zero-shot?"). Override with MAX_STEPS=... env to train longer.
MAX_STEPS=${MAX_STEPS:-600}
IMAGE=vlnav/nexusnav-train:local
LOG_DIR=/mnt/nvme0/vln_habitat/logs
CKPT_ROOT=/home/kemal/VLNav/VLNav/checkpoints
MINI_DATA_ROOT=/workspace/InternNav-dev/data/InternData-N1-v0.5-mini/vln_ce
# NVMe ZeRO-Infinity (optimizer->nvme, param->cpu): fits 94GB no-swap host (~60GB peak).
# The CPU-offload variant (zero3_offload_nopin.json) OOM-kills (SIGKILL -9) here: full-param
# 7B AdamW CPU offload needs ~100-112GB RAM. Do NOT revert without adding swap/NVMe.
DS_CFG=scripts/train/qwenvl_train/zero3_nvme_offload.json

# Validate image exists
docker inspect ${IMAGE} > /dev/null 2>&1 || { echo "ERROR: ${IMAGE} not found"; exit 1; }

# GPU gate
GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ -n "${GPU_FREE}" ] && [ "${GPU_FREE}" -lt 6000 ]; then
    echo "WARNING: Only ${GPU_FREE} MiB GPU free. Training may OOM."
fi

# Port selection by model
case ${MODEL} in
    dualvln)   PORT=29501; EPOCHS=5; OUT=dualsystem_vln_ce_mini; NUM_HIST=4; LR=2e-5; VLR=5e-6; GACC=4 ;;
    chrononav) PORT=29502; EPOCHS=5; OUT=chrononav_vln_ce_mini;  NUM_HIST=4; LR=2e-5; VLR=5e-6; GACC=4 ;;
    nexusnav)  PORT=29503; EPOCHS=3; OUT=nexusnav_vln_ce_mini;   NUM_HIST=8; LR=1e-5; VLR=2e-6; GACC=8 ;;
    *) echo "Unknown model: ${MODEL}. Use: dualvln|chrononav|nexusnav"; exit 1 ;;
esac

LOG_FILE=${LOG_DIR}/train_${MODEL}_mini.log
mkdir -p ${LOG_DIR} /mnt/nvme0/ds_offload

echo "======================================================="
echo "  Mini ${MODEL} training"
echo "  GPU free: ${GPU_FREE} MiB"
echo "  Log: ${LOG_FILE}"
echo "  $(date)"
echo "======================================================="

# Remove stale container if exists
docker rm -f vlnav_train_${MODEL} 2>/dev/null || true

docker run --rm \
  --name vlnav_train_${MODEL} \
  --runtime nvidia \
  --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e BITSANDBYTES_NOWELCOME=1 \
  -e "PYTHONPATH=/workspace/InternNav-dev:/workspace/InternNav:/workspace/InternNav/src/diffusion-policy" \
  -e TRITON_CACHE_DIR=/tmp/triton_cache \
  -e TRITON_OVERRIDE_ARCH=sm86 \
  -e TRANSFORMERS_CACHE=/tmp/hf_cache \
  -e HF_HOME=/tmp/hf_cache \
  -e TORCH_EXTENSIONS_DIR=/tmp/torch_ext \
  -e TORCH_COMPILE_DISABLE=1 \
  -v /home/kemal/VLNav/VLNav/workspaces/model/InternNav-dev:/workspace/InternNav-dev:rw \
  -v /home/kemal/VLNav/VLNav/workspaces/model/InternNav:/workspace/InternNav:ro \
  -v ${CKPT_ROOT}:/checkpoints:ro \
  -v ${CKPT_ROOT}:/workspace/InternNav-dev/checkpoints:ro \
  -v /mnt/nvme0/vln_habitat:/workspace/InternNav/data:rw \
  -v /mnt/nvme0/vln_habitat/logs:/workspace/InternNav/logs:rw \
  -v /mnt/nvme0/ds_offload:/ds_offload:rw \
  -w /workspace/InternNav-dev \
  ${IMAGE} \
  "mkdir -p /workspace/InternNav/data/checkpoints/${OUT} /workspace/InternNav/logs &&
   echo '[${MODEL}] Starting training...' &&
   python -m torch.distributed.run \
     --nproc_per_node=1 --nnodes=1 --node_rank=0 \
     --master_addr=localhost --master_port=${PORT} \
     internnav/trainer/internvla_n1_trainer.py \
     --deepspeed ${DS_CFG} \
     --model_name_or_path /checkpoints/InternVLA-N1-DualVLN \
     --vln_dataset_use r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30 \
     --data_root ${MINI_DATA_ROOT} \
     --data_flatten False \
     --tune_mm_vision False --tune_mm_mlp True --tune_mm_llm False \
     --bf16 \
     --num_history ${NUM_HIST} --data_augmentation True \
     --resize_h 256 --resize_w 256 \
     --sample_step 4 --num_future_steps 2 --predict_step_num 16 \
     --pixel_goal_only False --system1 none \
     --output_dir /workspace/InternNav/data/checkpoints/${OUT} \
     --num_train_epochs ${EPOCHS} --max_steps ${MAX_STEPS} \
     --per_device_train_batch_size 1 \
     --per_device_eval_batch_size 2 \
     --gradient_accumulation_steps ${GACC} \
     --max_pixels 16384 --min_pixels 3136 \
     --val_ratio 0.1 --eval_strategy no \
     --save_strategy steps --save_steps 200 --save_total_limit 2 \
     --learning_rate ${LR} --vision_tower_lr ${VLR} \
     --weight_decay 0 --warmup_ratio 0.05 --max_grad_norm 1 \
     --lr_scheduler_type cosine --logging_steps 1 \
     --model_max_length 1024 --gradient_checkpointing True \
     --dataloader_num_workers 2 \
     --run_name ${MODEL}_vln_ce_mini \
     --report_to none \
   2>&1 | tee /workspace/InternNav/logs/train_${MODEL}_mini.log" \
  2>&1 | tee ${LOG_FILE}

EXIT=$?
echo "[${MODEL}] Docker run exited with code: ${EXIT}"
exit ${EXIT}
