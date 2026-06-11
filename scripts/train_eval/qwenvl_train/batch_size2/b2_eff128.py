"""batch_size variant config: bs=2, grad_accum=8 (eff 128). == b2_eff128.sh.

Declarative config for `runner.py --config`: imports the default config and
overrides only batch_size / grad_accum_steps.

    python scripts/train_eval/qwenvl_train/runner.py --config <this file>
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

EXP_NAME = "batch_size2/b2_eff128"
PARAMS = replace(base.PARAMS, batch_size=2, grad_accum_steps=8)
eval_cfg = base.make_eval_cfg(PARAMS)
