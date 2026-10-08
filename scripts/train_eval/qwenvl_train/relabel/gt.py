"""대조군 — 학습·평가 모두 기존 GT labeling (기존 동작과 동일)

생성물 — build_relabel_dataset.py --labels gt --emit config

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/gt.py --machine 5090
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # relabel_base

import default_config as base  # noqa: E402
import relabel_base  # noqa: E402  (같은 폴더 — eval_yaml 헬퍼)

EXP_NAME = "relabel/gt"
# 학습 쪽은 train_label 이 data_root 를 `<root>_<label>` 로 바꾼다.
# 평가 쪽은 eval_config_path 에 relabel yaml 을 직접 지정한다 (None = 기존 GT yaml).
# 평가 로그는 runner 가 logs/<EXP_NAME slug>/ 로 이미 나누므로 셀끼리 안 섞인다.
PARAMS = replace(base.PARAMS, train_label="gt", eval_config_path=relabel_base.eval_yaml("gt"))
eval_cfg = base.make_eval_cfg(PARAMS)
