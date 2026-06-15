"""S1-FPV no-op baseline (BEV pipeline ON, mode fpv). == bev/bev_s1_fpv_base.sh.

Same BEV provider trainer + habitat_vln_bev evaluator as bev_s1_bev.py, but
bev_s1_mode='fpv' so S1 keeps original FPV behaviour — the control run to compare
against bev_s1_bev.py. All other hyperparameters inherit the b4 baseline.

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --eval-target 5090 --nproc 1
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

EXP_NAME = "bev/base_s1.fpv_s2.fpv_rgb_gt"
PARAMS = replace(
    base.PARAMS,
    bev=True,
    bev_s1_mode="fpv",       # no-op baseline
    bev_s2_mode_eval="fpv",
    bev_image_type="rgb",
    bev_depth_source="gt",
)
eval_cfg = base.make_eval_cfg(PARAMS)
