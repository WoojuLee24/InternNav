"""relabel 2x2 의 **stage1 절반 학습** 판. ``relabel/dualvln_mini_base.py`` 와 다른 점은 둘뿐이다:

  1. stage1 ``num_train_epochs`` 2.0 -> 1.0 (13,810 -> 6,905 step, ~33h -> ~16.5h).
     ``max_steps`` 대신 epoch 을 줄인다: max_steps>0 은 runner 에서 smoke 로 취급돼 중간
     eval/save/load-best 가 꺼지고, cosine LR 도 줄어든 길이에 맞춰 끝까지 감쇠해야 비교가 공정하다.
  2. 출력 이름 접두사 ``relabel_half/dualvln_mini_half_s{1,2}`` — 기존 full-length 셀과 ckpt 가 섞이지 않게.
     stage2 는 같은 train_label 의 **half** stage1 ckpt 에서 출발한다 (stage2 recipe 는 그대로, 3 epoch).

데이터·하이퍼파라미터·평가 yaml 은 전부 dualvln_mini_base 를 그대로 쓴다.
"""

import glob
import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_QWENVL = os.path.dirname(_HERE)
for _p in (_QWENVL, os.path.join(_QWENVL, "relabel")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dualvln_mini_base as full  # noqa: E402

S1_EPOCHS = 1.0  # full recipe = 2.0


def exp_name(stage, train_label, eval_label):
    if stage == 1:
        return f"relabel_half/dualvln_mini_half_s1.train.{train_label}_eval.{eval_label}_{full.S2_MARKER}"
    elif stage == 2:
        return f"relabel_half/dualvln_mini_half_s2.train.{train_label}_eval.{eval_label}"
    else:
        assert False, f"unreachable stage={stage!r}"


def stage1(train_label, eval_label):
    _, train_machine, params = full.stage1(train_label, eval_label)
    assert params.num_train_epochs == 2.0, params.num_train_epochs  # 부모 recipe 가 바뀌면 여기서 멈춘다
    return exp_name(1, train_label, eval_label), train_machine, replace(params, num_train_epochs=S1_EPOCHS)


def stage2(train_label, eval_label):
    # full.stage2 는 STAGE1_CKPT env 를 최우선으로 쓴다. 비어 있으면 half stage1 중 최신으로 채운다
    # (그대로 두면 full-length stage1 을 집어 간다).
    if not os.environ.get("STAGE1_CKPT"):
        pattern = f"{full.CHECKPOINTS_ROOT}/{exp_name(1, train_label, 'gt')}_*"
        ckpt = max(glob.glob(pattern), default=None)
        if ckpt is None:
            raise FileNotFoundError(f"half stage1 ckpt 없음: {pattern}")
        os.environ["STAGE1_CKPT"] = ckpt
    _, train_machine, params, ckpt = full.stage2(train_label, eval_label)
    assert "/relabel_half/" in ckpt, f"half 가 아닌 stage1 에서 출발하려 한다: {ckpt}"
    return exp_name(2, train_label, eval_label), train_machine, params, ckpt
