"""DualVLN-mini **half** stage1 — 학습 labeling 'b1_facts-memo-v1_knobs-v2_delivery-joint3-v1' x 평가 labeling 'b1_facts-memo-v1_knobs-v2_delivery-joint3-v1', **gradient clipping 켬**. 평가만(②) — 학습은 train.<L>-clip_eval.gt 셀.

train.b1_facts-memo-v1_knobs-v2_delivery-joint3-v1_eval.gt 셀(2026-10-24)이 step 864·1074 grad 스파이크(227·758)로 발산(① 18.8 · ② 26.1)해서 다시 돌리는 셀.
원인: zero2.json 에 gradient_clipping 이 없어 max_grad_norm=1.0 이 DeepSpeed 에서 무시됨 (모든 2x2 셀 공통).
여기만 zero2_clip.json(gradient_clipping=auto -> 1.0) 을 쓴다. 나머지는 원래 셀과 같다. 출력 이름의 학습 라벨에 '-clip' 을 붙여 원래 ckpt 와 섞이지 않게 한다.

    python scripts/train_eval/qwenvl_train/runner.py --machine h200 --config scripts/train_eval/qwenvl_train/relabel_half/dualvln_mini_half_s1.train.b1_facts-memo-v1_knobs-v2_delivery-joint3-v1-clip_eval.b1_facts-memo-v1_knobs-v2_delivery-joint3-v1.py --no-train --model-path $(ls -dt /home/irteam/data-vol2/checkpoints/relabel_half/dualvln_mini_half_s1.train.b1_facts-memo-v1_knobs-v2_delivery-joint3-v1-clip_eval.gt_internvla-n1-system2_*/ | head -1)
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # dualvln_mini_half_base

import default_config as base  # noqa: E402
import dualvln_mini_half_base  # noqa: E402

LABEL = "b1_facts-memo-v1_knobs-v2_delivery-joint3-v1"
EXP_NAME, TRAIN_MACHINE, PARAMS = dualvln_mini_half_base.stage1(LABEL, "b1_facts-memo-v1_knobs-v2_delivery-joint3-v1")
EXP_NAME = EXP_NAME.replace(f"train.{LABEL}_eval", f"train.{LABEL}-clip_eval")
PARAMS = replace(PARAMS, deepspeed="scripts/train/qwenvl_train/zero2_clip.json")
eval_cfg = base.make_eval_cfg(PARAMS)
