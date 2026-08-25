"""`baseline/dualvln_stage2_from_stage1.py` + patch_embed을 등가 GEMM으로 (forward 비트 동일) (`patch_embed_impl="gemm"`).

부모와 유일한 차이는 그 한 줄이다. 데이터셋/하이퍼파라미터/eval yaml은 전부 상속한다.

forward 출력은 비트 단위로 동일하다(torch.equal, max_abs_diff=0). VLM이 freeze라
patch_embed gradient가 아예 계산되지 않으므로 학습 동작이 완전히 보존된다.

runner 30-step 실측 (8xH200): conv 10.19 s/it -> gemm 5.18 s/it (1.97배). 3.5일 -> 1.8일.
분석: .claude/memory/understanding_qwen25vl_patch_embed_conv3d_bottleneck.md

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-eval
"""

import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_QWENVL = os.path.dirname(_HERE)
sys.path.insert(0, _QWENVL)                          # import default_config
sys.path.insert(0, os.path.join(_QWENVL, "baseline"))  # import 부모 config (eval.py는 이 dir를 안 넣어준다)

import default_config as base  # noqa: E402
import dualvln_stage2_from_stage1 as full  # noqa: E402

TRAIN_MACHINE = full.TRAIN_MACHINE  # 재-export 필수 (빼면 조용히 mini 데이터셋으로 떨어진다)

EXP_NAME = "baseline_ablation/dualvln_stage2_from_stage1_gemm"
PARAMS = replace(full.PARAMS, patch_embed_impl="gemm")
eval_cfg = base.make_eval_cfg(PARAMS)
