"""DualVLN-mini **half** stage1 — 학습 labeling 'v278fix_g4temp-070' x 평가 labeling 'gt'. 학습 + 자동 평가.

stage1 이 1 epoch 인 것 외에는 relabel/dualvln_mini_s1.train.<L>_eval.gt.py 와 같다 (dualvln_mini_half_base.py 참고).
make_cells.py 로 생성됨.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half/dualvln_mini_half_s1.train.v278fix_g4temp-070_eval.gt.py --machine h200
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # dualvln_mini_half_base

import default_config as base  # noqa: E402
import dualvln_mini_half_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = dualvln_mini_half_base.stage1("v278fix_g4temp-070", "gt")
eval_cfg = base.make_eval_cfg(PARAMS)
