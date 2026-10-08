"""DualVLN-mini stage1 — 학습 labeling 'v218' x 평가 labeling 'v218'. 평가만 (train.v218_eval.gt 셀이 학습한 ckpt 를 --model-path 로)

recipe/데이터 차이는 dualvln_mini_base.py 참고.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/dualvln_mini_s1.train.v218_eval.v218.py --machine h200 --no-train --model-path <ckpt>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # dualvln_mini_base

import default_config as base  # noqa: E402
import dualvln_mini_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = dualvln_mini_base.stage1("v218", "v218")
eval_cfg = base.make_eval_cfg(PARAMS)
