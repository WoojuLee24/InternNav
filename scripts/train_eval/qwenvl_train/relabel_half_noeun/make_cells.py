#!/usr/bin/env python3
"""relabel_half_noeun-v01 셀 생성기 — 노은역 S2 학습 14 개 (공개 System2 출발, 1 epoch, 평가 없음)
+ scratch 28 개 (Qwen2.5-VL-7B-Instruct 출발, 1 epoch · 3 epoch(매 epoch 저장)) -> relabel_half_noeun-v01-scratch.

  본실험 : noeun-{v6,hunav,both}-{b1,post} x wait          (6)
  대조군 : noeun-{hunav,both}-{b1,post}   x exclude        (4, 같은 데이터에서 WAIT 만 없음)
  충돌가능: noeun-{hunav,both}-{b1,post}   x visible        (4, 사람과 충돌 가능한 대기 프레임만 WAIT)
v6 에는 대기 프레임이 없어 wait 셀이 기본 로더와 샘플 목록까지 같다.

    python3 scripts/train_eval/qwenvl_train/relabel_half_noeun/make_cells.py
이미 있는 셀은 내용이 같을 때만 통과.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = '''"""노은역 S2 학습 (relabel_half_noeun-v01) — 데이터 '{label}', wait_mode '{mode}'. 학습만 (노은역 평가는 다른 노드).

noeun_base.py 참고 (공개 InternVLA-N1-System2 출발, r2r_60cm_15_15, 1 epoch). make_cells.py 로 생성됨.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/{fname} --machine h200 --no-eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # noeun_base

import default_config as base  # noqa: E402
import noeun_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = noeun_base.stage1("{label}", "{mode}")
eval_cfg = base.make_eval_cfg(PARAMS)
'''

CELLS = [(f"noeun-{p}-{l}", "wait") for p in ("v6", "hunav", "both") for l in ("b1", "post")] + \
        [(f"noeun-{p}-{l}", m) for m in ("exclude", "visible") for p in ("hunav", "both") for l in ("b1", "post")]


def cell(label, mode):
    fname = f"s1.{label}_wait-{mode}.py"
    text = TEMPLATE.format(label=label, mode=mode, fname=fname)
    path = os.path.join(HERE, fname)
    if os.path.exists(path) and open(path).read() != text:
        sys.exit(f"{fname} exists with different content")
    open(path, "w").write(text)
    print(fname)


SCRATCH_TEMPLATE = '''"""노은역 S2 학습 scratch (relabel_half_noeun-v01-scratch) — 데이터 '{label}', wait_mode '{mode}', Qwen2.5-VL-7B-Instruct 출발 {ep} epoch{every}. 학습만.

noeun_base.stage1_scratch 참고. make_cells.py 로 생성됨.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/{fname} --machine h200 --no-eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # noeun_base

import default_config as base  # noqa: E402
import noeun_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = noeun_base.stage1_scratch("{label}", "{mode}", {ep})
eval_cfg = base.make_eval_cfg(PARAMS)
'''


def scratch_cell(label, mode, ep):
    fname = f"s1.{label}_wait-{mode}_scratch-ep{ep}.py"
    text = SCRATCH_TEMPLATE.format(label=label, mode=mode, ep=ep, fname=fname, every=", 매 epoch 저장" if ep > 1 else "")
    path = os.path.join(HERE, fname)
    if os.path.exists(path) and open(path).read() != text:
        sys.exit(f"{fname} exists with different content")
    open(path, "w").write(text)
    print(fname)


STOPR_TEMPLATE = '''"""노은역 S2 학습 (relabel_half_noeun-v01) — 데이터 '{label}', wait_mode '{mode}', 반경 STOP {r} m. 학습만.

s1.{label}_wait-{mode}.py (기준, r=0) 와 stop_radius_m 만 다르다. noeun_base.stage1_stop_radius 참고. make_cells.py 로 생성됨.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel_half_noeun/{fname} --machine h200 --no-eval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # noeun_base

import default_config as base  # noqa: E402
import noeun_base  # noqa: E402

EXP_NAME, TRAIN_MACHINE, PARAMS = noeun_base.stage1_stop_radius("{label}", "{mode}", {r})
eval_cfg = base.make_eval_cfg(PARAMS)
'''
STOPR_CELLS = [("noeun-both-b1", "wait")]   # 반경 STOP (2026-11-10): 기준 셀 하나 x noeun_base.STOP_RADII


def stopr_cell(label, mode, r):
    fname = f"s1.{label}_wait-{mode}_stopr{r}.py"
    text = STOPR_TEMPLATE.format(label=label, mode=mode, r=r, fname=fname)
    path = os.path.join(HERE, fname)
    if os.path.exists(path) and open(path).read() != text:
        sys.exit(f"{fname} exists with different content")
    open(path, "w").write(text)
    print(fname)


sys.path.insert(0, HERE)
import noeun_base  # noqa: E402

for label, mode in CELLS:
    cell(label, mode)
for ep in (1, 3):                      # scratch (2026-10-30): 같은 14 조합
    for label, mode in CELLS:
        scratch_cell(label, mode, ep)
for label, mode in STOPR_CELLS:
    for r in noeun_base.STOP_RADII:
        stopr_cell(label, mode, r)
