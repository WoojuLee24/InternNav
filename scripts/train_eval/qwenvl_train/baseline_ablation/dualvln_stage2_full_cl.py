"""`baseline/dualvln_stage2_full.py` + patch_embed을 channels_last_3d 입력으로 (cuDNN 경로) (`patch_embed_impl="channels_last"`).

부모와 유일한 차이는 그 한 줄이다. 데이터셋/하이퍼파라미터/eval yaml은 전부 상속한다.

주의: cuDNN 커널의 누적 순서가 달라 **forward 출력이 부모와 다르다**(maxdiff ~1.56e-2).
loss와 gradient도 따라서 달라진다. gemm보다 6배 느리다 — "conv 구현이 틀린 게 아니라 라우팅이
문제였다"를 확인하는 대조군 용도이며, 기존 동작 보존이 목표라면 gemm 변종을 쓸 것.

runner 30-step 실측 (8xH200): gemm 대비 6배 느림 (patch_embed fwd+bwd 4.02 ms vs 24.71 ms). 미측정.
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
import dualvln_stage2_full as full  # noqa: E402

TRAIN_MACHINE = full.TRAIN_MACHINE  # 재-export 필수 (빼면 조용히 mini 데이터셋으로 떨어진다)

EXP_NAME = "baseline_ablation/dualvln_stage2_full_cl"
PARAMS = replace(full.PARAMS, patch_embed_impl="channels_last")
eval_cfg = base.make_eval_cfg(PARAMS)
