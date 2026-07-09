"""Unified image-provider no-op baseline (image_provider ON, all axes at default).

Proves image_provider=True with every axis left at its default is byte-identical
to the plain image_provider=False path — the control run to diff against for
every other config in this directory.

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine 5090 --no-eval --debug-dir logs/260707_base_s1.fpv_s2.fpv
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

EXP_NAME = "image_base/base_s1.fpv_s2.fpv"
PARAMS = replace(
    base.PARAMS,
    image_provider=True,
    s1_image_view="fpv",
    s1_image_type="rgb",
    s1_image_mode="raw",
    s1_combine_mode="none",
    s2_image_view="fpv",
    s2_image_type="rgb",
    s2_image_mode="raw",
    s2_combine_mode="none",
)
eval_cfg = base.make_eval_cfg(PARAMS)
