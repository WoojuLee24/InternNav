"""노은역 S2 학습 scratch (relabel_half_noeun-v01-scratch) — 데이터 'noeun-hunav-b1', wait_mode 'exclude', Qwen2.5-VL-7B-Instruct 출발 1 epoch. 학습만.

noeun_base.stage1_scratch 참고. make_cells.py 로 생성됨.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/s1.noeun-hunav-b1_wait-exclude_scratch-ep1.py --machine h200 --no-eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # noeun_base

import default_config as base  # noqa: E402
import noeun_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = noeun_base.stage1_scratch("noeun-hunav-b1", "exclude", 1)
eval_cfg = base.make_eval_cfg(PARAMS)
