"""S1-FPV baseline with pixel_goal input (skips generate_latents).

S2 outputs pixel [col, row] -> normalized [x, y] in [0,1] -> pixel_cond_projector
-> NextDiT z_latents conditioning (replaces VLM latent tokens).

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/bev/base_s1.fpv_s2.fpv_rgb_gt.pixel_goal.py --machine 5090
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import default_config as base

EXP_NAME = "bev/base_s1.fpv_s2.fpv_rgb_gt.pixel_goal"
PARAMS = replace(
    base.PARAMS,
    bev=True,
    bev_s1_mode="fpv",
    bev_s2_mode_eval="fpv",
    bev_image_type="rgb",
    bev_depth_source="gt",
    use_pixel_goal=True,
)
eval_cfg = base.make_eval_cfg(PARAMS)
