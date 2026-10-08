"""DualVLN-mini **half** stage2 — 학습 labeling 'gt' x 평가 labeling 'v278fix_g4temp-100'. 평가만 (학습은 train.<L>_eval.gt 셀).

stage1 이 1 epoch 인 것 외에는 relabel/dualvln_mini_s2.train.<L>_eval.gt.py 와 같다 (dualvln_mini_half_base.py 참고).
make_cells.py 로 생성됨.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half/dualvln_mini_half_s2.train.gt_eval.v278fix_g4temp-100.py --machine h200 --no-train --model-path $(ls -dt /home/irteam/data-vol2/checkpoints/relabel_half/dualvln_mini_half_s2.train.gt_eval.gt_*/ | head -1)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # dualvln_mini_half_base

import default_config as base  # noqa: E402
import dualvln_mini_half_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS, STAGE1_CKPT = dualvln_mini_half_base.stage2("gt", "v278fix_g4temp-100")
eval_cfg = base.make_eval_cfg(PARAMS)
