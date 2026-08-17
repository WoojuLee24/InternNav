"""A/B control for dualvln_stage2_full.py: eval look-down 60 deg instead of 30 deg.

Identical training setup; the ONLY difference is the Habitat eval yaml.

    eval_config_path=None -> EVAL_MACHINE["h200"]["config_path"] = vln_r2r_mini.yaml,
    whose look_up/look_down tilt_angle is 30, and habitat_vln_evaluator.py steps LOOKDOWN
    twice -> 60 deg. That is what all 12 previous ablation runs on this branch measured.

dualvln_stage2_full.py uses vln_r2r_ld30.yaml (tilt_angle 15 -> 30 deg total), which is what
main did and what the training data's pitch_1->pitch_2 delta implies.

Run both against the released DualVLN checkpoint to find out empirically which one reproduces
the paper numbers (NE 4.05 / OS 70.7 / SR 64.3 / SPL 58.5), before spending days on training:

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-train --model-path /home/irteam/git/InternNav/checkpoints/InternVLA-N1-DualVLN --debug-dir logs/baseline/ld60

--debug-dir keeps the two runs' progress.json/result.json apart; without it both would write
into the checkpoint directory and collide.
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
PARAMS = replace(full.PARAMS, eval_config_path=None)  # None -> vln_r2r_mini.yaml (60 deg)
eval_cfg = base.make_eval_cfg(PARAMS)
