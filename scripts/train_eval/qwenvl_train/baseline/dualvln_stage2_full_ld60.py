"""A/B control for dualvln_stage2_full.py: eval look-down 60 deg instead of 30 deg.

Identical training setup and identical eval data root; the ONLY difference is the
look_up/look_down tilt_angle in the Habitat eval yaml:

    dualvln_stage2_full.py -> vln_r2r_full_ld30.yaml  (tilt 15, 2 x LOOKDOWN = 30 deg)
    this file              -> vln_r2r_full_ld60.yaml  (tilt 30, 2 x LOOKDOWN = 60 deg)

30 deg is what main did (habitat 0.2.x `simulator.tilt_angle: 15`) and what the training
data implies (look-down = pitch_1 -> pitch_2, at most 0 -> 30 deg). 60 deg is what every
ablation run on this branch has measured, after the 0.3.x port added an explicit
per-action `tilt_angle: 30` that habitat 0.3.3 would otherwise have defaulted to 15.

Run both against the released DualVLN checkpoint to find out empirically which one
reproduces the paper numbers (NE 4.05 / OS 70.7 / SR 64.3 / SPL 58.5), before spending
days on training. Give each run its own output dir so they do not share progress.json:

    cp -al checkpoints/InternVLA-N1-DualVLN checkpoints/baseline/released_ld60
    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-train --model-path /home/irteam/git/InternNav/checkpoints/baseline/released_ld60

(`--debug-dir logs/baseline/ld60` also isolates the output, but runner.py turns wandb off
whenever --debug-dir is given.)
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import dualvln_stage2_full

import default_config as base  # noqa: E402
import dualvln_stage2_full as full  # noqa: E402

TRAIN_MACHINE = full.TRAIN_MACHINE

EXP_NAME = "baseline/dualvln_stage2_full_ld60"
# Must be set explicitly: eval_config_path=None would fall back to EVAL_MACHINE["h200"]'s
# vln_r2r_mini.yaml, which changes the DATA ROOT as well as the angle -- turning the A/B
# into a two-variable comparison.
PARAMS = replace(full.PARAMS, eval_config_path="scripts/eval/configs/vln_r2r_full_ld60.yaml")
eval_cfg = base.make_eval_cfg(PARAMS)
