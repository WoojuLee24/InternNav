"""System1+System2 (stage2, mode=dual_system) checkpoint 를 v218 labeling 으로 평가만 한다 (학습 없음).

부모 `baseline_ablation/dualvln_stage2_from_stage1_gemm.py` 와 다른 점은 평가 yaml 한 줄뿐이다:
    vln_r2r_full_ld30.yaml -> relabel/vln_r2r_mini_v218.yaml
두 yaml 은 ld30 으로 같고, mini/full 의 평가 데이터(val_unseen, 90 scene)는 byte 단위로 같으므로
부모 config 로 낸 GT 결과와 문장만 다른 비교가 된다.

    python scripts/train_eval/qwenvl_train/runner.py --config scripts/train_eval/qwenvl_train/relabel/eval_v218.dualvln_stage2_from_stage1_gemm.py --machine h200 --no-train --model-path <ckpt>
"""

import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # default_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # relabel_base

import default_config as base  # noqa: E402
import relabel_base  # noqa: E402

_src = relabel_base.load_exp("baseline_ablation/dualvln_stage2_from_stage1_gemm.py")
EXP_NAME = "relabel/eval_v218.dualvln_stage2_from_stage1_gemm"
TRAIN_MACHINE = _src.TRAIN_MACHINE
PARAMS = replace(_src.PARAMS, eval_config_path=relabel_base.eval_yaml("v218", "h200"))
eval_cfg = base.make_eval_cfg(PARAMS)
