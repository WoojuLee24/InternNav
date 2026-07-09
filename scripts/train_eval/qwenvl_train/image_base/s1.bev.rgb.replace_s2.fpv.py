"""S1 view=bev, type=rgb, mode=raw, combine=replace (S2 stays fpv no-op).

Exercises the unified provider's view='bev' delegation path (100% inherited
BEVImageProvider logic) through the NEW s1_image_* argv/trainer wiring, for
combo-coverage of scripts/train_eval/qwenvl_train/runner.py.

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine 5090 --no-eval --max-steps 3 --debug-dir logs/260707_s1.bev.rgb.replace_s2.fpv
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

EXP_NAME = "image_base/s1.bev.rgb.replace_s2.fpv"
PARAMS = replace(
    base.PARAMS,
    image_provider=True,
    s1_image_view="bev",
    s1_image_type="rgb",
    s1_image_mode="raw",
    s1_combine_mode="replace",
    bev_depth_source="gt",
)
eval_cfg = base.make_eval_cfg(PARAMS)
