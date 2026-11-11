"""노은역 S2 학습 (relabel_half_noeun-v01) — 데이터 'noeun-both-b1', wait_mode 'wait', 반경 STOP 2.0 m. 학습만.

s1.noeun-both-b1_wait-wait.py (기준, r=0) 와 stop_radius_m 만 다르다. noeun_base.stage1_stop_radius 참고. make_cells.py 로 생성됨.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/s1.noeun-both-b1_wait-wait_stopr2.0.py --machine h200 --no-eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # noeun_base

import default_config as base  # noqa: E402
import noeun_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = noeun_base.stage1_stop_radius("noeun-both-b1", "wait", 2.0)
eval_cfg = base.make_eval_cfg(PARAMS)
