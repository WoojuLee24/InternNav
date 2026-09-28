"""`baseline/dualvln_stage2_full.py` + patch_embed을 channels_last_3d 입력으로 (cuDNN 경로) (`patch_embed_impl="channels_last"`).

부모와 유일한 차이는 그 한 줄이다. 데이터셋/하이퍼파라미터/eval yaml은 전부 상속한다.

## 왜 이 변종을 쓰는가 (권장 경로)

부모의 `conv`는 **PyTorch 2.9.x 회귀**(pytorch#174051, NVIDIA 포럼 355210)에 걸린다:
bf16 + NCDHW + 큰 batch + 1x1x1 출력 조합에서 PyTorch가 cuDNN을 아예 시도하지 않고
ATen `slow_conv_dilated3d`로 떨어뜨린다. 입력 레이아웃만 NDHWC로 바꾸면 그 게이트를 통과한다.

H200 실측 (N=34,480 = stage1 bs4의 micro-batch 패치 수, bf16):

    구성                              backend          fwd+bwd     wgrad 오차(fp64 기준)
    conv (부모, torch 2.9)            SlowDilated3d   2566.89 ms   1.901e-01   <- 깨짐
    conv + channels_last (이 config)  Cudnn              5.61 ms   1.656e-03   <- 정상
    gemm                              -                  0.30 ms   1.656e-03
    conv (torch 2.10)                 Cudnn              6.25 ms   1.656e-03

**핵심: 이 변종의 출력은 `torch 2.10 + 원본 conv`와 비트 단위로 완전히 동일하다.**
둘 다 같은 cuDNN 커널을 타기 때문이다 (입력 고정 후 forward/wgrad 전부 `torch.equal` 확인).
즉 환경을 전혀 바꾸지 않고 torch 2.10으로 올린 것과 같은 결과를 얻는다.
원저자도 torch 2.8 이하에서 cuDNN 경로로 학습했을 것이므로 이쪽이 재현에 가깝다.

`gemm`은 cuBLAS 커널이라 반올림이 달라 비트 동일하지 않다(정확도는 동일).
속도 차이 5.61 ms vs 0.30 ms는 전체 step 9,020 ms에서 0.06%라 실용적으로 무의미하다.

## 주의

forward 출력이 부모(torch 2.9의 깨진 conv)와는 다르다 — 상대차 ~1.7e-3, bf16 저장 수준이며
**정확도는 이쪽이 정상이고 부모가 틀린 것이다**(wgrad 19% vs 0.17%).

torch를 2.10으로 올리면 이 config 없이 원본 `conv` 그대로 같은 결과가 나온다.
설치 스크립트: `scripts/train_eval/qwenvl_train/install_torch210.sh`

분석 전문: `.claude/memory/understanding_qwen25vl_patch_embed_conv3d_bottleneck.md`

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
# eval: 모델 rank를 GPU 0-3, habitat GL 렌더를 GPU 4-7로 완전 분리 (겹치는 GPU 없음).
# 부모의 eval_render_gpu_offset=1 + nproc 8은 8개 GPU 전부가 CUDA와 GL을 동시에 호스팅해
# libnvidia-eglcore에서 SIGABRT가 난다 (stage1 mode='system2'에서 결정적 재현 확인).
PARAMS = replace(
    full.PARAMS,
    patch_embed_impl="channels_last",
    eval_nproc=4,
    eval_render_gpu_offset=4,
)
eval_cfg = base.make_eval_cfg(PARAMS)
