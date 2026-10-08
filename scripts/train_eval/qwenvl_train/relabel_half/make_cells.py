#!/usr/bin/env python3
"""relabel_half 셀 생성기 (2x2). label L 마다 stage 1/2 각각:
  train.L_eval.gt   학습 + GT 자동 평가 (이 셀만 학습한다)
  train.L_eval.L    평가만: --no-train --model-path <train.L_eval.gt ckpt>
  train.gt_eval.L   평가만: --no-train --model-path <train.gt_eval.gt ckpt>
평가 yaml 은 scripts/eval/configs/relabel/vln_r2r_mini_<L>.yaml (build_relabel_dataset.py --emit yaml).

    python3 scripts/train_eval/qwenvl_train/relabel_half/make_cells.py gt v218fix v278fix v278fix_stop-gtdist ...

label 이름에 '.' 은 쓸 수 없다 (셀 파일명이 '.' 으로 train/eval 을 나눈다). 이미 있는 셀은 내용이 같을 때만 통과.
학습 데이터는 data/InternData-N1-v0.5-mini/vln_ce_<label> (build_relabel_dataset.py --labels <label> --emit data) 가 있어야 한다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = '''"""DualVLN-mini **half** stage{s} — 학습 labeling '{label}' x 평가 labeling '{ev}'. {role}

stage1 이 1 epoch 인 것 외에는 relabel/dualvln_mini_s{s}.train.<L>_eval.gt.py 와 같다 (dualvln_mini_half_base.py 참고).
make_cells.py 로 생성됨.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half/{fname} --machine h200{extra}
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # dualvln_mini_half_base

import default_config as base  # noqa: E402
import dualvln_mini_half_base  # noqa: E402

{ret} = dualvln_mini_half_base.stage{s}("{label}", "{ev}")
eval_cfg = base.make_eval_cfg(PARAMS)
'''
RET = {1: "EXP_NAME, TRAIN_MACHINE, PARAMS", 2: "EXP_NAME, TRAIN_MACHINE, PARAMS, STAGE1_CKPT"}

CKPT = "/home/irteam/data-vol2/checkpoints/relabel_half/dualvln_mini_half_s{s}.train.{label}_eval.gt{marker}_*"


def cell(s, label, ev):
    assert "." not in label and "." not in ev, (label, ev)
    fname = f"dualvln_mini_half_s{s}.train.{label}_eval.{ev}.py"
    if ev == "gt":
        role, extra = "학습 + 자동 평가.", ""
    else:
        marker = "_internvla-n1-system2" if s == 1 else ""
        role = "평가만 (학습은 train.<L>_eval.gt 셀)."
        extra = f" --no-train --model-path $(ls -dt {CKPT.format(s=s, label=label, marker=marker)}/ | head -1)"
    text = TEMPLATE.format(s=s, label=label, ev=ev, fname=fname, ret=RET[s], role=role, extra=extra)
    path = os.path.join(HERE, fname)
    if os.path.exists(path) and open(path).read() != text:
        sys.exit(f"{fname} exists with different content")
    open(path, "w").write(text)
    print(fname)


for label in sys.argv[1:]:
    for s in (1, 2):
        cell(s, label, "gt")
        if label != "gt":
            cell(s, label, label)   # train L x eval L
            cell(s, "gt", label)    # train gt x eval L
