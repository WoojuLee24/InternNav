#!/bin/bash
# BEV S1 debug training via the inference-side provider module.
# Single GPU, single dataset (r2r_125cm_0_30), fixed seed — deterministic order.
#
# This uses internvla_n1_bev_provider_trainer.py so training BEV is computed by
# the SAME BEVProcessor the eval path uses (bit-for-bit parity, see
# tests/unit_test/test_bev_provider_training.py). Baseline + legacy BEV files
# are untouched.
#
# Usage:
#   bash scripts/train/qwenvl_train/bev/train_bev_provider_debug.sh [data_root] [bev_s1_mode]
#
#   bev_s1_mode : fpv | bev | fpv_bev   (default: bev)
#   data_root   : defaults to /ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_ce

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT=${1:-/ws/src/InternNav/data/InternData-N1-v0.5-mini/vln_ce}
BEV_S1_MODE=${2:-bev}

/workspace/isaaclab/_isaac_sim/python.sh internnav/trainer/internvla_n1_bev_provider_trainer.py --bev_s1_mode ${BEV_S1_MODE} --bev_image_type rgb --bev_depth_source gt --deepspeed scripts/train/qwenvl_train/zero2.json --model_name_or_path checkpoints/InternVLA-N1-System2 --vln_dataset_use r2r_125cm_0_30 --data_root ${DATA_ROOT} --seed 42 --num_history 4 --system1 nextdit_async --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --output_dir checkpoints/debug/bev_s1_provider --num_train_epochs 1.0 --eval_strategy no --save_strategy steps --save_steps 500 --learning_rate 1e-4 --bf16 --dataloader_num_workers 0 --report_to none --logging_steps 1
