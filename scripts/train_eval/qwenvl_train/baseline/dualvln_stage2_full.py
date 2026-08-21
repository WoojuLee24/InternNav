"""DualVLN paper reproduction — Stage 2 (dual-system / System-1 training) on the FULL dataset.

Equivalent to `main:scripts/train/qwenvl_train/train_dual_system.sh`, which starts from the
released InternVLA-N1-System2 checkpoint and trains only System 1 (the VLM stays frozen).
Stage 1 (System-2 pretraining) is NOT reproduced — the released checkpoint is used as-is.

Target (README, VLN-CE R2R val_unseen, "InternVLA-N1 (Dual System) DualVLN"):
    NE 4.05 / OS 70.7 / SR 64.3 / SPL 58.5

ONE config drives both train and eval: `Params` is the single source for the trainer argv and
for the eval `model_settings`, so num_history / resize_* / predict_step_num / num_future_steps
can never drift between the two.

    # A) sanity-gate the eval pipeline on the released checkpoint (~3.5h, no training)
    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-train --model-path /home/irteam/git/InternNav/checkpoints/InternVLA-N1-DualVLN --debug-dir logs/baseline/ld30

    # B) train (~3.5 days on 8xH200)
    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-eval

    # C) evaluate the trained checkpoint
    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-train --model-path <checkpoints_root>/baseline/dualvln_stage2_full_<TIMESTAMP>

Train+eval also works as a SINGLE invocation (drop --no-eval): with val_ratio=0 there is no
`best_metric`, so runner falls back to the final model at the top level of output_dir (the
"last checkpoint" that main's recipe evaluates). B/C above remain useful for re-running eval
on an existing checkpoint.
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

REPO = "/home/irteam/git/InternNav"
# repo `data/` is a symlink to data-vol1, which holds the EXTRACTED dataset.
# (data-vol2/InternData-N1/vln_ce/traj_data/* is still un-extracted <scene>.tar.gz archives.)
FULL_VLN_CE = f"{REPO}/data/InternData-N1/vln_ce"

# runner.py applies TRAIN_MACHINE[machine]'s data_root/system2_ckpt/nproc_per_node on top of
# PARAMS unconditionally, and prefers a TRAIN_MACHINE defined in the experiment config module
# over default_config's. So redirect data_root here (the shared preset points at the mini
# dataset) and inherit system2_ckpt / checkpoints_root / nproc_per_node unchanged.
# Doing it here rather than via the --data-root CLI flag means a forgotten flag cannot
# silently train on the mini dataset.
TRAIN_MACHINE = {
    "h200": {**base.TRAIN_MACHINE["h200"], "data_root": FULL_VLN_CE},
}

EXP_NAME = "baseline/dualvln_stage2_full"

PARAMS = replace(
    base.PARAMS,
    # ---- train: main's train_dual_system.sh ----
    # 6 dataset-settings at 30% each (r2r x2, rxr x2, scalevln x2), verbatim from main.
    vln_datasets=(
        "r2r_125cm_0_30%30,r2r_60cm_15_15%30,"
        "rxr_125cm_0_30%30,rxr_60cm_15_15%30,"
        "scalevln_125cm_0_30%30,scalevln_60cm_30_30%30"
    ),
    data_root=FULL_VLN_CE,       # (TRAIN_MACHINE overrides this too; stated for readability)
    val_ratio=0.0,               # main: eval_strategy=no -> train on 100% of the data
    save_interval_steps=5000,    # main: save_steps 5000 (default 25 would make validation/saving
                                 # dominate a ~14k-step run)
    save_total_limit=5,          # main
    save_only_model=False,       # main default; keeps optimizer state so a 3.5-day run can resume
    dataloader_num_workers=8,    # main
    # ---- eval: same Params object -> train==eval for num_history/resize/predict_step_num ----
    # full data root + look-down 15 deg x 2 LOOKDOWN steps = 30 deg, i.e. the same effective
    # settings as main. See scripts/eval/configs/vln_r2r_full_ld30.yaml for the rationale.
    eval_config_path="scripts/eval/configs/vln_r2r_full_ld30.yaml",
    eval_render_gpu_offset=1,    # driver 580.126.16: same-GPU GL+CUDA aborts/corrupts frames
)

# Everything else is inherited from base.PARAMS and already matches train_dual_system.sh:
#   lr 1e-4, cosine_with_min_lr (min_lr 1e-5), warmup 0.003, weight_decay 0, max_grad_norm 1,
#   batch_size 16 x 8 GPUs = effective 128 (main: 2 x 64), num_train_epochs 3.0,
#   num_history 8, resize 384x384, sample_step 4, num_future_steps 4, predict_step_num 32,
#   max_pixels 313600, min_pixels 3136, model_max_length 8192, bf16, gradient_checkpointing,
#   pixel_goal_only True, system1 nextdit_async, tune_mm_vision/mlp/llm all False (VLM frozen).

eval_cfg = base.make_eval_cfg(PARAMS)
