"""`dualvln_stage1_full_gemm.py` + every new logging/verification flag on (smoke tests only).

Only the flags differ from the parent (data / hyperparameters / eval yaml inherited, val_ratio stays 0):
  eval_metrics_schema    metrics_schema.py recompute + exact comparison -> schema_check_<machine>.jsonl
  eval_decision_metrics  per-S2-call decision records in raw/ + dec_* in progress / result rows
  decision_metrics       System2 decision metrics during training -> train/dec_* (internvla_n1_metrics_trainer.py)
Its own EXP_NAME keeps the eval logs in logs/baseline_ablation_dualvln_stage1_full_gemm_check/, apart
from the real evaluations of the same checkpoint.

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-train --model-path <ckpt> --max-steps 3
"""

import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))            # import default_config
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "baseline"))
sys.path.insert(0, _HERE)                             # import the parent config

import default_config as base  # noqa: E402
import dualvln_stage1_full_gemm as parent  # noqa: E402

TRAIN_MACHINE = parent.TRAIN_MACHINE  # re-export (else it silently falls back to the mini dataset)
EXP_NAME = "baseline_ablation/dualvln_stage1_full_gemm_check"
PARAMS = replace(parent.PARAMS, eval_metrics_schema=True, eval_decision_metrics=True, decision_metrics=True)
eval_cfg = base.make_eval_cfg(PARAMS)
