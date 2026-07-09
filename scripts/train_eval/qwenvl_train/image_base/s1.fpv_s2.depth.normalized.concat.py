"""S1 stays fpv, S2 view=fpv, type=depth, mode=normalized, combine=replace (dataset-level training image injection).

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine 5090 --no-eval --max-steps 3 --debug-dir logs/260707_s1.fpv_s2.depth.normalized
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

EXP_NAME = "image_base/s1.fpv_s2.depth.normalized.concat"
PARAMS = replace(
    base.PARAMS,
    image_provider=True,
    s2_image_view='fpv',
    s2_image_type='depth',
    s2_image_mode='normalized',
    s2_combine_mode='concat',
    bev_depth_source='gt',
)
eval_cfg = base.make_eval_cfg(PARAMS)
