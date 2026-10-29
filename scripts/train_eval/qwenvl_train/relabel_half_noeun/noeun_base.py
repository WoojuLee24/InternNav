"""노은역 S2(stage1) 학습 (relabel_half_noeun-v01) 의 공용 헬퍼. ``relabel_half/dualvln_mini_half_base.stage1`` 과 다른 점:

  1. 출발점 ``system2_ckpt`` : Qwen2.5-VL-7B -> 공개 InternVLA-N1-System2 (fine-tune). 경로에 'internvla-n1-system2'(소문자 비교) 가 있어
     trainer 가 InternVLAN1ForCausalLM 으로 읽는다. runner 가 TRAIN_MACHINE 값으로 덮어쓰므로 둘 다 바꾼다.
  2. ``vln_datasets`` : R2R 카메라 4 종 -> ``r2r_60cm_15_15`` 하나 (노은역은 60cm_15deg 만 렌더). sampling 100 %.
  3. ``train_label`` = ``noeun-<paths>-<label>`` -> ``<mini>/vln_ce_noeun-<paths>-<label>`` (nav3d/noeun/64_build_train.py).
  4. ``wait_mode`` : wait | exclude | visible (default_config.Params.wait_mode; internvla_n1_wait_trainer.py).
     visible = 사람과 충돌 가능한 대기 프레임만 WAIT (``WAIT_KEEP`` = nav3d/noeun/66_wait_keep.py 출력, 자동 판정 + 사람 검토).
  5. 평가 없음: 노은역 평가는 다른 노드(Isaac). 셀은 ``--no-eval`` 로 돌린다 (eval_cfg 는 runner 가 import 만 한다).
1 epoch · 하이퍼파라미터는 그대로 (relabel_half stage1).
"""

import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_QWENVL = os.path.dirname(_HERE)
for _p in (_QWENVL, os.path.join(_QWENVL, "relabel_half"), os.path.join(_QWENVL, "relabel")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dualvln_mini_half_base as half  # noqa: E402

VERSION = "relabel_half_noeun-v01"
PUBLIC_SYSTEM2 = "/home/irteam/data-vol2/checkpoints/InternVLA-N1-System2"
VLN_DATASETS = "r2r_60cm_15_15"
WAIT_MODES = ("wait", "exclude", "visible")
WAIT_KEEP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_QWENVL))),
                         "scripts", "dataset_converters", "nav3d", "noeun", "wait", "wait_keep_v1.json")


def exp_name(train_label, wait_mode):
    return f"{VERSION}/s1.{train_label}_wait-{wait_mode}_{half.full.S2_MARKER}"


def stage1(train_label, wait_mode):
    """(EXP_NAME, TRAIN_MACHINE, PARAMS)."""
    assert train_label.startswith("noeun-") and "." not in train_label, train_label
    assert wait_mode in WAIT_MODES, wait_mode
    _, train_machine, params = half.stage1(train_label, "gt")
    assert "internvla-n1-system2" in PUBLIC_SYSTEM2.lower()
    train_machine = {"h200": {**train_machine["h200"], "system2_ckpt": PUBLIC_SYSTEM2}}
    params = replace(params, system2_ckpt=PUBLIC_SYSTEM2, vln_datasets=VLN_DATASETS, train_label=train_label,
                     wait_mode=wait_mode, wait_keep=WAIT_KEEP if wait_mode == "visible" else "", eval_config_path=None)
    assert params.num_train_epochs == half.S1_EPOCHS == 1.0, params.num_train_epochs
    return exp_name(train_label, wait_mode), train_machine, params
