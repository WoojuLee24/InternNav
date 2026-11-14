"""노은역 랜드마크 S2 학습 (relabel_half_noeun_lm-v01) — 데이터 'noeun-lm1200-b1'. 학습만 (노은역 평가는 다른 노드).

make_cells_landmark.py 로 생성됨. 학습 설정은 noeun_base.stage1 과 같고 EXP_NAME 폴더만 relabel_half_noeun_lm-v01.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/s1.noeun-lm1200-b1.py --machine h200 --no-eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # noeun_base

import default_config as base  # noqa: E402
import noeun_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = noeun_base.stage1("noeun-lm1200-b1", "wait")
EXP_NAME = EXP_NAME.replace(noeun_base.VERSION + "/", "relabel_half_noeun_lm-v01/", 1)
eval_cfg = base.make_eval_cfg(PARAMS)
