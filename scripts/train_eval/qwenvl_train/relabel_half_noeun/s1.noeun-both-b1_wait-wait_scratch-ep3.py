"""노은역 S2 학습 scratch (relabel_half_noeun-v01-scratch) — 데이터 'noeun-both-b1', wait_mode 'wait', Qwen2.5-VL-7B-Instruct 출발 3 epoch, 매 epoch 저장. 학습만.

noeun_base.stage1_scratch 참고. make_cells.py 로 생성됨.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/s1.noeun-both-b1_wait-wait_scratch-ep3.py --machine h200 --no-eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # noeun_base

import default_config as base  # noqa: E402
import noeun_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = noeun_base.stage1_scratch("noeun-both-b1", "wait", 3)
eval_cfg = base.make_eval_cfg(PARAMS)
