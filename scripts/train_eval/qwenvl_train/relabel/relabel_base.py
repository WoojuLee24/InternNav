"""relabel 실험 config 들의 공용 헬퍼.

이 폴더의 config 는 instruction labeling 만 바꾸는 실험이다. 노브는 둘뿐이다:

- 학습: ``Params.train_label`` — data_root 를 ``<root>_<label>`` 로 바꾼다.
- 평가: ``Params.eval_config_path`` — relabel yaml 을 직접 가리킨다 (None = 기존 GT yaml).

둘을 따로 두어야 학습 GT/new x 평가 GT/new 의 2x2 가 된다. 평가 로그 분리는
``runner.run_eval`` 의 ``logs/<exp_slug>/`` 가 이미 해 주므로 여기서 할 일이 없다.

`load_exp` 가 필요한 이유
-------------------------
기존 실험 config 파일명에 점이 들어있어(`bev/base_s1.fpv_s2.fpv_rgb_gt.py`)
일반 `import` 로는 못 읽는다. 경로로 직접 로드한다.

BEV 실험 위에 relabel 을 얹는 예::

    from dataclasses import replace
    import default_config, relabel_base

    src = relabel_base.load_exp("bev/base_s1.fpv_s2.fpv_rgb_gt.py")
    EXP_NAME = "relabel/bev.base_train.v6_eval.v6"
    PARAMS = replace(src.PARAMS, train_label="v6", eval_config_path=relabel_base.eval_yaml("v6"))
    eval_cfg = default_config.make_eval_cfg(PARAMS)
"""

import importlib.util
import os
import sys

QWENVL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if QWENVL_DIR not in sys.path:
    sys.path.insert(0, QWENVL_DIR)

# 머신별 habitat yaml. build_relabel_dataset.py 가 이 이름으로 relabel 복사본을 만든다.
BASE_YAML = {"5090": "vln_r2r_mini_5090", "h200": "vln_r2r_mini"}
# label='gt' 셀의 평가 yaml. relabel 복사본과 data_path 한 줄만 달라야 2x2 가 문장만 비교한다.
# h200 relabel 복사본은 ld30 원본에서 만들어지므로(build_relabel_dataset.EVAL_YAMLS) 짝도 ld30 이다.
# 5090 은 ld30 GT yaml 이 없어 기존 기본 yaml(None) 을 쓰고, 복사본도 그 기본 yaml 에서 만들어진다.
GT_YAML = {"h200": "scripts/eval/configs/vln_r2r_mini_ld30.yaml"}
RELABEL_YAML_DIR = "scripts/eval/configs/relabel"


def eval_yaml(label, machine=None):
    """relabel 평가 yaml 경로. label='gt' 면 GT_YAML[machine] (없으면 None = 기존 GT yaml 그대로).

    machine 을 생략하면 runner 가 심어 둔 ``TRAIN_EVAL_TARGET`` 을 쓴다 —
    ``default_config.make_eval_cfg`` 와 같은 출처라 둘이 어긋날 수 없다.
    h1(Isaac)은 habitat yaml 을 안 쓰므로 None 이다.
    """
    machine = machine or os.environ.get("TRAIN_EVAL_TARGET", "h200")
    if label == "gt":
        return GT_YAML.get(machine)
    stem = BASE_YAML.get(machine)
    if stem is None:  # h1 등 habitat 이 아닌 타깃
        return None
    return f"{RELABEL_YAML_DIR}/{stem}_{label}.yaml"


def load_exp(rel_path):
    """`scripts/train_eval/qwenvl_train/` 기준 상대경로로 실험 config 모듈을 읽는다."""
    path = os.path.join(QWENVL_DIR, rel_path)
    name = "exp_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
