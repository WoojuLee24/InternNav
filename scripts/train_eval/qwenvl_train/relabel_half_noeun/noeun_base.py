"""노은역 S2(stage1) 학습 (relabel_half_noeun-v01) 의 공용 헬퍼. ``relabel_half/dualvln_mini_half_base.stage1`` 과 다른 점:

  1. 출발점 ``system2_ckpt`` : Qwen2.5-VL-7B -> 공개 InternVLA-N1-System2 (fine-tune). runner 가 TRAIN_MACHINE 값으로 덮어쓰므로 둘 다 바꾼다.
     ⚠️ trainer 는 경로 문자열로 모델 클래스를 고른다: 'internvla-n1-system2' 가 있으면 InternVLAN1ForCausalLM -> system1='none'(S2 만)
     이면 initialize_vision_modules 가 NotImplementedError (2026-10-29 14 개 전부 이것으로 실패). 공개 System2 는 원래
     Qwen2_5_VLForConditionalGeneration (S2 전용) 이라 'qwen2.5' 가 든 링크 경로(``PUBLIC_SYSTEM2``)로 넘겨 Qwen 클래스로 읽는다
     (R2R stage1 · habitat system2 평가와 같은 클래스).
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
PUBLIC_SYSTEM2 = "/home/irteam/data-vol2/checkpoints/init_qwen2.5vl/public_s2"   # -> /home/irteam/data-vol2/checkpoints/InternVLA-N1-System2 (링크)
VLN_DATASETS = "r2r_60cm_15_15"
WAIT_MODES = ("wait", "exclude", "visible")
WAIT_KEEP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_QWENVL))),
                         "scripts", "dataset_converters", "nav3d", "noeun", "wait", "wait_keep_v1.json")


def exp_name(train_label, wait_mode):
    return f"{VERSION}/s1.{train_label}_wait-{wait_mode}_{half.full.S2_MARKER}"


# scratch (2026-10-30 사용자): Qwen2.5-VL-7B-Instruct 에서 출발 (R2R stage1 과 같은 출발점), 1 epoch 와 3 epoch.
# 3 epoch 는 epoch 1 · 2 · 3 마다 저장 (save_steps = 한 epoch step 수, 아래 표 = 공개 ckpt 1 epoch run 의 max_steps 실측; 데이터·batch·GPU 수가 같으면 같다).
# optimizer 는 저장하지 않는다 (save_only_model=True, ckpt 약 16 GB; 중간 재개 불가).
SCRATCH_VERSION = "relabel_half_noeun-v01-scratch"
SCRATCH_INIT = "Qwen/Qwen2.5-VL-7B-Instruct"     # HF 캐시 (R2R stage1 과 같은 값; baseline/dualvln_stage1_full.STAGE1_INIT)
STEPS_PER_EPOCH = {("v6", "wait"): 213,
                   ("hunav", "exclude"): 216, ("hunav", "visible"): 222, ("hunav", "wait"): 226,
                   ("both", "exclude"): 429, ("both", "visible"): 435, ("both", "wait"): 439}


def scratch_exp_name(train_label, wait_mode, epochs):
    return f"{SCRATCH_VERSION}/s1.{train_label}_wait-{wait_mode}_ep{epochs}_{half.full.S2_MARKER}"


def stage1_scratch(train_label, wait_mode, epochs):
    """Qwen2.5-VL-7B-Instruct 출발 · epochs (1 | 3). 3 이면 매 epoch 저장. 나머지는 stage1 과 같다."""
    assert epochs in (1, 3), epochs
    name, train_machine, params = stage1(train_label, wait_mode)
    assert "qwen2.5" in SCRATCH_INIT.lower() and "internvla-n1-system2" not in SCRATCH_INIT.lower()
    paths = train_label.split("-")[1]                 # noeun-<paths>-<label>
    spe = STEPS_PER_EPOCH[(paths, wait_mode)]
    train_machine = {"h200": {**train_machine["h200"], "system2_ckpt": SCRATCH_INIT}}
    params = replace(params, system2_ckpt=SCRATCH_INIT, num_train_epochs=float(epochs), save_only_model=True,
                     save_interval_steps=spe if epochs > 1 else params.save_interval_steps, save_total_limit=max(epochs, 1))
    return scratch_exp_name(train_label, wait_mode, epochs), train_machine, params


# 반경 STOP (2026-11-10): stage1 과 모든 인자가 같고 stop_radius_m 만 다르다 (기준 r=0 = stage1 셀 그대로).
# internnav/dataset/internvla_n1_lerobot_dataset_stop_radius.py
STOP_RADII = (0.5, 1.0, 2.0, 3.0)   # 2.0 · 3.0 추가 (2026-11-11, r=1.0 결과가 좋아서)


def stop_radius_exp_name(train_label, wait_mode, stop_radius_m):
    return f"{VERSION}/s1.{train_label}_wait-{wait_mode}_stopr{stop_radius_m}_{half.full.S2_MARKER}"


def stage1_stop_radius(train_label, wait_mode, stop_radius_m):
    """stage1 + stop_radius_m (> 0)."""
    assert stop_radius_m > 0, stop_radius_m
    _, train_machine, params = stage1(train_label, wait_mode)
    params = replace(params, stop_radius_m=float(stop_radius_m))
    return stop_radius_exp_name(train_label, wait_mode, stop_radius_m), train_machine, params


def stage1(train_label, wait_mode):
    """(EXP_NAME, TRAIN_MACHINE, PARAMS)."""
    assert train_label.startswith("noeun-") and "." not in train_label, train_label
    assert wait_mode in WAIT_MODES, wait_mode
    _, train_machine, params = half.stage1(train_label, "gt")
    assert "qwen2.5" in PUBLIC_SYSTEM2.lower() and "internvla-n1-system2" not in PUBLIC_SYSTEM2.lower(), PUBLIC_SYSTEM2  # Qwen 클래스로 읽게
    train_machine = {"h200": {**train_machine["h200"], "system2_ckpt": PUBLIC_SYSTEM2}}
    params = replace(params, system2_ckpt=PUBLIC_SYSTEM2, vln_datasets=VLN_DATASETS, train_label=train_label,
                     wait_mode=wait_mode, wait_keep=WAIT_KEEP if wait_mode == "visible" else "", eval_config_path=None)
    assert params.num_train_epochs == half.S1_EPOCHS == 1.0, params.num_train_epochs
    return exp_name(train_label, wait_mode), train_machine, params
