"""DualVLN paper reproduction — Stage 1 (System-2 pretraining) on the FULL dataset.

Equivalent to `main:scripts/train/qwenvl_train/train_system2.sh`: full finetune of
Qwen2.5-VL-7B-Instruct (vision tower at a lower LR) on 8 dataset-settings at 100% sampling,
System 1 disabled (`system1="none"`). The output checkpoint plays the role of the released
`InternVLA-N1-System2` and is consumed by `dualvln_stage2_from_stage1.py`.

    # train + auto-eval (S2 + ShortestPathFollower on R2R val_unseen; --no-eval to train only)
    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200

    # evaluate an existing checkpoint
    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-train --model-path <checkpoints_root>/baseline/dualvln_stage1_internvla-n1-system2_<TIMESTAMP>

Argv parity with train_system2.sh (verified by baseline/verify_params_stage1.py):
effective batch 128 (main: 2 x 1 x 64 GPUs, here: 4 x 4 x 8 GPUs), lr 2e-5 + vision_tower_lr 5e-6,
plain cosine (no min_lr kwargs), epochs 2.0, save_steps 5000 / limit 5, no validation split.
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

REPO = "/home/irteam/git/InternNav"
FULL_VLN_CE = f"{REPO}/data/InternData-N1/vln_ce"

STAGE1_INIT = "Qwen/Qwen2.5-VL-7B-Instruct"

# the substring routes stage2's model loader (internvla_n1_trainer.py branches on
# 'internvla-n1-system2' in model_name_or_path) -> local stage1 output loads with zero trainer changes
EXP_NAME = "baseline/dualvln_stage1_internvla-n1-system2"
assert "internvla-n1-system2" in EXP_NAME

# runner overrides PARAMS from TRAIN_MACHINE unconditionally -> init model MUST be set here too
TRAIN_MACHINE = {
    "h200": {**base.TRAIN_MACHINE["h200"], "data_root": FULL_VLN_CE, "system2_ckpt": STAGE1_INIT},
}

PARAMS = replace(
    base.PARAMS,
    # ---- train: main's train_system2.sh, verbatim ----
    # 8 settings at 100% sampling (scalevln excluded, as in main)
    vln_datasets=(
        "r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,"
        "rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30"
    ),
    data_root=FULL_VLN_CE,
    system2_ckpt=STAGE1_INIT,        # TRAIN_MACHINE overrides too; stated for readability
    lr=2e-5,
    vision_tower_lr=5e-6,            # separate LR group for "visual" params
    # full-VLM finetune OOMs at batch 16/GPU on H200; 4 x 4 x 8 GPUs keeps main's eff batch 128 (2 x 1 x 64)
    batch_size=4,
    grad_accum_steps=4,
    lr_scheduler_type="cosine",
    lr_scheduler_min_lr=None,        # plain cosine rejects min_lr
    tune_mm_vision=True,
    tune_mm_mlp=True,
    tune_mm_llm=True,                # full VLM finetune (stage2 freezes all three)
    pixel_goal_only=False,           # turn + stop samples too (as main)
    system1="none",                  # System-2 only
    num_train_epochs=2.0,
    val_ratio=0.0,                   # main: no validation
    save_interval_steps=5000,        # main: save_steps 5000
    save_total_limit=5,              # main
    save_only_model=False,           # keep optimizer state for resume
    dataloader_num_workers=8,        # main
    # ---- eval: S2 + ShortestPathFollower (mode auto-derived from system1="none") ----
    # README target (R2R val_unseen): NE 4.25 / OS 68.3 / SR 60.9 / SPL 55.2
    # ld30 (2 x 15 = 30 deg) matches the evaluator's hard-coded 30-deg unprojection (vln_r2r.yaml's tilt 30 would not)
    eval_config_path="scripts/eval/configs/vln_r2r_full_ld30.yaml",
    eval_render_gpu_offset=1,        # driver 580.126.16: same-GPU GL+CUDA aborts/corrupts frames
)

# Everything else inherits base.PARAMS and already matches train_system2.sh
# (verified by baseline/verify_params_stage1.py).

eval_cfg = base.make_eval_cfg(PARAMS)
