"""`baseline/dualvln_stage1_full.py` + patch_embed을 fp32 누산 등가 GEMM으로 (`patch_embed_impl="gemm_fp32"`).

부모와 유일한 차이는 그 한 줄이다. 데이터셋/하이퍼파라미터/eval yaml은 전부 상속한다.

## 왜 이 변종인가

bf16 conv는 cuDNN을 못 타고 ATen `slow_conv_dilated3d`로 떨어져 배치 차원을 순차 루프로 돌고,
그 안에서 wgrad를 bf16으로 N회 누적해 **fp64 정확해 대비 14% 오차**를 낸다.
bf16 GEMM은 이를 0.17%로 줄이지만 그 값은 **grad를 bf16으로 저장하는 한계**라 같은 dtype에서는
더 못 줄인다. 정밀도를 더 올리려면 연산 dtype을 fp32로 올려야 한다.

H200 실측 (N=34,480 = stage1 bs4의 micro-batch 패치 수):

    mode            forward      wgrad 오차(모델 전체 fp32 기준)
    conv (부모)     1258.13 ms   1.40e-01   <- ATen fallback, bf16 누적이 깨짐
    gemm            0.153 ms     1.66e-03   <- bf16 grad 저장 한계 (이미 최적)
    conv_fp32       3.12 ms      2.94e-04   <- cuDNN fp32 커널
    gemm_fp32       2.72 ms      1.05e-06

**그러나 bf16 파라미터에서는 fp32 변종의 정확도 이득이 없다** (CPU 측정, N=8192):
    gemm 1.655e-03  vs  gemm_fp32 1.655e-03  (둘 다 grad dtype=bf16)
병목은 누산이 아니라 `w.grad`를 bf16으로 저장하는 것이다. 위 표의 fp32 수치는
모델 전체가 fp32였던 측정이라 이 맥락에 그대로 적용되지 않는다.
DeepSpeed ZeRO-2가 fp32 gradient 버퍼를 쓰면 달라질 수 있으나 미측정.

forward 누산만 fp32로 올린 변종. 비용은 micro-step당 +2.6 ms(step당 0.1%)로 무시할 수준이지만,
**측정상 wgrad 정확도 이득은 없다**(위 참조). gemm과의 차이를 확인하는 실험용이다.

## 주의

forward 출력이 부모(bf16 conv)와 **비트 동일하지 않다** — 더 정확한 쪽으로 달라진다.
(비트 동일을 원하면 `dualvln_stage1_full_gemm.py`를 쓸 것.)
train/eval 양쪽에 같은 값이 적용되므로 정합성은 유지된다.

stage1은 ViT를 학습하므로(`tune_mm_vision=True`) 이 차이가 patch_embed weight gradient에 실제로 반영된다.
stage2는 VLM freeze라 patch_embed gradient가 아예 없어 forward만 달라진다.

분석 전문: `.claude/memory/understanding_qwen25vl_patch_embed_conv3d_bottleneck.md`

    python scripts/train_eval/qwenvl_train/runner.py --config <this file> --machine h200 --no-eval
"""

import os
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_QWENVL = os.path.dirname(_HERE)
sys.path.insert(0, _QWENVL)                            # import default_config
sys.path.insert(0, os.path.join(_QWENVL, "baseline"))  # import 부모 config (eval.py는 이 dir를 안 넣어준다)

import default_config as base  # noqa: E402
import dualvln_stage1_full as full  # noqa: E402

TRAIN_MACHINE = full.TRAIN_MACHINE  # 재-export 필수 (빼면 조용히 mini 데이터셋으로 떨어진다)

EXP_NAME = "baseline_ablation/dualvln_stage1_full_gemm_fp32"
# stage2(dualvln_stage2_from_stage1)는 두 가지를 이름으로 판단한다:
#   1) glob  : {checkpoints_root}/baseline/dualvln_stage1_internvla-n1-system2_*
#   2) 모델 클래스: internvla_n1_trainer.py가 model_name_or_path에 이 substring이 있는지로
#      InternVLAN1ForCausalLM을 고른다. 없으면 조용히 Qwen2VL(2.5도 아님)로 로드된다.
# EXP_NAME이 규약을 벗어나면 stage2가 이 출력을 못 쓴다 -> 규약에 맞는 심볼릭 링크를
# baseline/ 아래 만들거나 STAGE1_CKPT로 직접 지정해야 한다.
# eval: 모델 rank를 GPU 0-3, habitat GL 렌더를 GPU 4-7로 완전 분리한다.
# 부모의 eval_render_gpu_offset=1 + nproc 8은 8개 GPU 전부가 CUDA와 GL을 동시에 호스팅해
# local_rank 6이 libnvidia-eglcore에서 SIGABRT로 죽는다 (결정적 재현 확인, mode='system2').
PARAMS = replace(
    full.PARAMS,
    patch_embed_impl="gemm_fp32",
    eval_nproc=4,
    eval_render_gpu_offset=4,
)
eval_cfg = base.make_eval_cfg(PARAMS)
