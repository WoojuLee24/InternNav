"""노은역 S2 학습 (relabel_half_noeun-v01) — 데이터 'noeun-hunav-post', wait_mode 'exclude'. 학습만 (노은역 평가는 다른 노드).

noeun_base.py 참고 (공개 InternVLA-N1-System2 출발, r2r_60cm_15_15, 1 epoch). make_cells.py 로 생성됨.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/s1.noeun-hunav-post_wait-exclude.py --machine h200 --no-eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # noeun_base

import default_config as base  # noqa: E402
import noeun_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = noeun_base.stage1("noeun-hunav-post", "exclude")
eval_cfg = base.make_eval_cfg(PARAMS)
