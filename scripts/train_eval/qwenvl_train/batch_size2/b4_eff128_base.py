"""batch_size baseline config (b4): bs=4, grad_accum=4 (eff 128). == b4_eff128_base.sh.

Declarative config for `runner.py --config`. This is the baseline, so it uses the
default config's PARAMS unchanged (it equals running runner.py with no --config).

    python scripts/train_eval/qwenvl_train/runner.py --config <this file>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import default_config

import default_config as base  # noqa: E402

EXP_NAME = "batch_size2/b4_eff128_base"
PARAMS = base.PARAMS  # baseline, no override
eval_cfg = base.make_eval_cfg(PARAMS)
