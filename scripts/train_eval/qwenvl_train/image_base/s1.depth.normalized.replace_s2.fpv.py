"""S1 view=fpv, type=depth, mode=normalized, combine=replace (S2 stays fpv no-op).

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine 5090 --no-eval --max-steps 3 --debug-dir logs/input_v0.1/image_base/s1.depth.normalized.replace_s2.fpv
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

EXP_NAME = "image_base/s1.depth.normalized.replace_s2.fpv"
PARAMS = replace(
    base.PARAMS,
    image_provider=True,
    s1_image_view='fpv',
    s1_image_type='depth',
    s1_image_mode='normalized',
    s1_combine_mode='replace',
    bev_depth_source='gt',
)
eval_cfg = base.make_eval_cfg(PARAMS)
