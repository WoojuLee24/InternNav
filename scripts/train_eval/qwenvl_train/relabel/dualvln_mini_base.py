"""DualVLN stage1 -> stage2 main recipe 를 **mini(R2R 전용)** 데이터에 옮긴 relabel 2x2 실험의 공용 헬퍼.

부모는 full 재현에 쓰는 channels_last 변종이다:
  - stage1: ``baseline_ablation/dualvln_stage1_full_cl.py`` (Qwen2.5-VL-7B full finetune, 2 epoch)
  - stage2: ``baseline_ablation/dualvln_stage2_full_cl.py`` (VLM frozen, System1 학습, 3 epoch)

부모와 다른 점은 데이터 쪽 3가지뿐이다 (하이퍼파라미터·eval GPU 배치는 전부 상속):
  1. ``data_root``  : full ``InternData-N1/vln_ce`` -> mini ``InternData-N1-v0.5-mini/vln_ce``
                      (mini 의 R2R traj_data 는 full 과 byte 단위로 같다. mini 에는 rxr/scalevln 이 없다)
  2. ``vln_datasets``: 부모 목록에서 ``r2r_*`` 만 남긴다 (sampling ``%N`` 도 그대로)
  3. 평가 yaml     : full_ld30 -> mini_ld30 (GT) / ``relabel/vln_r2r_mini_<label>.yaml`` (ld30 원본에서 생성)

2x2 노브는 기존 relabel 과 같다: 학습 ``train_label``, 평가 ``eval_config_path``.
셀 파일은 ``dualvln_mini_s{1,2}.train.<L>_eval.<E>.py`` 이고, 학습은 ``eval.gt`` 셀만 한다
(나머지 셀은 같은 ckpt 를 ``--no-train --model-path`` 로 평가만 한다).
"""

import glob
import os
import shutil
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_QWENVL = os.path.dirname(_HERE)
for _p in (_QWENVL, os.path.join(_QWENVL, "baseline"), os.path.join(_QWENVL, "baseline_ablation"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import default_config as base  # noqa: E402
import relabel_base  # noqa: E402

MINI_VLN_CE = base.TRAIN_MACHINE["h200"]["data_root"]  # .../InternData-N1-v0.5-mini/vln_ce
assert MINI_VLN_CE.endswith("InternData-N1-v0.5-mini/vln_ce"), MINI_VLN_CE
CHECKPOINTS_ROOT = base.TRAIN_MACHINE["h200"]["checkpoints_root"]
RELEASED_SYSTEM2 = base.TRAIN_MACHINE["h200"]["system2_ckpt"]  # chat_template.json 보충용

# stage1 출력 디렉토리 이름에 반드시 들어가야 한다 (internvla_n1_trainer.py 가 이 substring 으로
# 모델 클래스를 고른다. 없으면 stage2 가 조용히 Qwen2VL 로 로드된다).
S2_MARKER = "internvla-n1-system2"


def r2r_only(vln_datasets):
    """부모의 vln_datasets 에서 r2r_* 만 남긴다 (sampling 접미사 %N 유지)."""
    kept = [d for d in vln_datasets.split(",") if d.strip().startswith("r2r_")]
    assert kept, f"r2r 데이터셋이 없다: {vln_datasets!r}"
    return ",".join(kept)


def exp_name(stage, train_label, eval_label):
    """셀의 EXP_NAME. 출력 디렉토리(학습 셀)와 평가 로그 디렉토리(logs/<slug>/) 이름이 된다."""
    if stage == 1:
        return f"relabel/dualvln_mini_s1.train.{train_label}_eval.{eval_label}_{S2_MARKER}"
    elif stage == 2:
        return f"relabel/dualvln_mini_s2.train.{train_label}_eval.{eval_label}"
    else:
        assert False, f"unreachable stage={stage!r}"


def stage1(train_label, eval_label):
    """(EXP_NAME, TRAIN_MACHINE, PARAMS) — 부모 stage1 recipe, 데이터만 mini R2R."""
    import dualvln_stage1_full_cl as parent

    name = exp_name(1, train_label, eval_label)
    assert S2_MARKER in name
    # runner 가 TRAIN_MACHINE 의 data_root/system2_ckpt 를 PARAMS 위에 무조건 덮어쓰므로 여기서도 mini 로.
    # (train_label 은 그 뒤에 runner 가 적용해 <mini>/vln_ce_<label> 로 바꾼다)
    train_machine = {"h200": {**parent.TRAIN_MACHINE["h200"], "data_root": MINI_VLN_CE}}
    params = replace(
        parent.PARAMS,
        vln_datasets=r2r_only(parent.PARAMS.vln_datasets),
        data_root=MINI_VLN_CE,
        train_label=train_label,
        eval_config_path=relabel_base.eval_yaml(eval_label, "h200"),
    )
    return name, train_machine, params


def find_stage1_ckpt(train_label):
    """stage2 초기값: STAGE1_CKPT env > 같은 train_label 로 학습한 stage1 중 최신. 없으면 FileNotFoundError."""
    pattern = f"{CHECKPOINTS_ROOT}/{exp_name(1, train_label, 'gt')}_*"
    ckpt = os.environ.get("STAGE1_CKPT") or max(glob.glob(pattern), default=None)
    if not ckpt or not os.path.isfile(os.path.join(ckpt, "config.json")):
        raise FileNotFoundError(
            f"stage1 ckpt 없음 (STAGE1_CKPT env, 다음으로 최신 {pattern}; got {ckpt!r}, top-level config.json 없음). "
            f"먼저 학습: runner.py --config scripts/train_eval/qwenvl_train/relabel/"
            f"dualvln_mini_s1.train.{train_label}_eval.gt.py --machine h200"
        )
    assert S2_MARKER in ckpt, f"stage1 ckpt 경로에 {S2_MARKER!r} 가 없다: {ckpt}"
    # 부모(dualvln_stage2_from_stage1.py) 와 같은 안전장치: 예전 stage1 출력엔 chat_template.json 이 없다
    tmpl = os.path.join(ckpt, "chat_template.json")
    if not os.path.isfile(tmpl):
        shutil.copy(os.path.join(RELEASED_SYSTEM2, "chat_template.json"), tmpl)
    return ckpt


def stage2(train_label, eval_label):
    """(EXP_NAME, TRAIN_MACHINE, PARAMS, STAGE1_CKPT) — 부모 stage2 recipe, 초기값은 같은 label 의 stage1."""
    import dualvln_stage2_full_cl as parent

    ckpt = find_stage1_ckpt(train_label)
    train_machine = {"h200": {**parent.TRAIN_MACHINE["h200"], "data_root": MINI_VLN_CE, "system2_ckpt": ckpt}}
    params = replace(
        parent.PARAMS,
        vln_datasets=r2r_only(parent.PARAMS.vln_datasets),
        data_root=MINI_VLN_CE,
        system2_ckpt=ckpt,
        train_label=train_label,
        eval_config_path=relabel_base.eval_yaml(eval_label, "h200"),
    )
    return exp_name(2, train_label, eval_label), train_machine, params, ckpt
