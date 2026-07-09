"""S1 stays fpv, S2 gets an fpv+depth (colormap) image at eval time (Habitat).

Also exercises S2's GT-depth training-time image injection (dataset-level;
see internvla_n1_lerobot_dataset.py) since s2_combine_mode='replace' is set —
train_argv() will pass the s2_image_* flags regardless of the S1 trainer used.

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine 5090 --no-train --debug-dir logs/260707_s2_depth_eval
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

EXP_NAME = "image_base/s1.fpv_s2.depth.colormap.concat"
PARAMS = replace(
    base.PARAMS,
    image_provider=True,
    s2_image_view="fpv",
    s2_image_type="depth",
    s2_image_mode="colormap",
    s2_combine_mode="concat",
    bev_depth_source="gt",
)
eval_cfg = base.make_eval_cfg(PARAMS)
