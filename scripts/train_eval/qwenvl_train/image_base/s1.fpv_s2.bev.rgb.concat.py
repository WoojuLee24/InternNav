"""S1 stays fpv, S2 view=bev (FPV→BEV projection), type=rgb, combine=replace."""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import default_config as base  # noqa: E402

EXP_NAME = "image_base/s1.fpv_s2.bev.rgb.concat"
PARAMS = replace(
    base.PARAMS,
    image_provider=True,
    s2_image_view='bev',
    s2_image_type='rgb',
    s2_image_mode='raw',
    s2_combine_mode='concat',
    bev_depth_source='gt',
)
eval_cfg = base.make_eval_cfg(PARAMS)
