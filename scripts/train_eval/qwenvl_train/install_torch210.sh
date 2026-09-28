#!/usr/bin/env bash
# torch 2.9.0 -> 2.10.0 업그레이드 (방안 A).
#
# ============================================================================
# 왜 필요한가
# ============================================================================
# Qwen2.5-VL ViT의 patch_embed는 nn.Conv3d인데, **PyTorch 2.9.x 회귀** 때문에
# bf16 + NCDHW + 큰 batch + 1x1x1 출력 조합에서 cuDNN을 아예 시도하지 않고
# ATen slow_conv_dilated3d로 떨어진다. 그 fallback은 배치를 순차 루프로 돌아
# 느리고, wgrad를 저정밀로 누적해 부정확하다.
#
#   보고: https://github.com/pytorch/pytorch/issues/174051   (2.10에서 해결)
#         https://forums.developer.nvidia.com/t/355210       (H100, 동일 조건)
#         https://github.com/pytorch/pytorch/issues/166643   (bf16 conv3d 메모리)
#
# H200 실측 (N=34,480 = stage1 bs4의 micro-batch 패치 수, bf16):
#
#   환경                    backend          fwd+bwd     wgrad 오차(fp64 기준)
#   torch 2.9.0 (현재)      SlowDilated3d   2566.89 ms   1.901e-01   <- 깨짐
#   torch 2.10.0            Cudnn              6.25 ms   1.656e-03   <- 정상
#
# 즉 코드를 한 줄도 안 고치고 속도 411배, 정확도 115배가 회복된다.
#
# ============================================================================
# 지금 당장은 필요 없다
# ============================================================================
# `patch_embed_impl="channels_last"` (방안 B)가 **torch 2.10의 결과와 비트 단위로
# 완전히 동일**하다 — 둘 다 같은 cuDNN 커널을 탄다 (입력 고정 후 forward/wgrad
# 전부 torch.equal 확인). 환경 변경 없이 같은 결과를 얻으므로 B로 진행 중이다.
#
# 이 스크립트는 나중에 A로 갈 때(원본 conv 코드를 그대로 쓰고 싶을 때)를 위한 것이다.
#
# ============================================================================
# 주의 — 되돌리기 어려운 변경이다
# ============================================================================
#  * flash-attn 2.8.3 은 "+cu130torch2.9" 로 **torch 2.9에 고정 빌드**되어 있다.
#    torch를 올리면 import 시 깨지므로 반드시 재설치해야 한다.
#  * torch 2.10.0 의 PyPI 기본 빌드는 **cu128**이다 (현재는 cu130).
#    드라이버 580.126.16 은 둘 다 지원하지만 CUDA 마이너 버전이 바뀐다.
#  * 실행 중인 학습/평가가 있으면 절대 돌리지 말 것.
#
# 사용법:
#   bash scripts/train_eval/qwenvl_train/install_torch210.sh --check    # 사전 점검만
#   bash scripts/train_eval/qwenvl_train/install_torch210.sh --dry-run  # 명령만 출력
#   bash scripts/train_eval/qwenvl_train/install_torch210.sh --yes      # 실제 설치
set -uo pipefail

PY="${PY:-/usr/bin/python}"
MODE="${1:---check}"
TORCH_VER="2.10.0"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
run()  { if [ "$MODE" = "--dry-run" ]; then echo "  [dry-run] $*"; else echo "  + $*"; "$@"; fi }

# --------------------------------------------------------------------------
say "[1/5] 현재 환경"
$PY - <<'EOF'
import torch
print(f"  torch      : {torch.__version__}")
print(f"  cuda       : {torch.version.cuda}")
print(f"  cudnn      : {torch.backends.cudnn.version()}")
print(f"  gpu        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
EOF
$PY -m pip list 2>/dev/null | grep -iE "^(torch|torchvision|torchaudio|flash.attn|flash_attn|deepspeed|triton|transformers|nvidia-cudnn) " | sed 's/^/  /'

# --------------------------------------------------------------------------
say "[2/5] 안전 점검"
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
QUEUE=$(ps -eo cmd | grep -cE "[r]un_queue.sh" || true)
echo "  GPU 점유 프로세스 : $BUSY"
echo "  run_queue.sh      : $QUEUE"
if [ "$BUSY" -gt 0 ] || [ "$QUEUE" -gt 0 ]; then
    echo "  >> 실행 중인 작업이 있다. 중단하고 다시 시도할 것." >&2
    [ "$MODE" = "--yes" ] && exit 1
fi

# --------------------------------------------------------------------------
say "[3/5] 되돌리기용 현재 버전 기록"
STAMP=$(date +%Y%m%d_%H%M%S)
FREEZE="/home/irteam/data-vol2/checkpoints/pip_freeze_before_torch210_${STAMP}.txt"
if [ "$MODE" = "--yes" ]; then
    $PY -m pip freeze > "$FREEZE" 2>/dev/null && echo "  저장: $FREEZE"
    echo "  되돌리려면: $PY -m pip install -r $FREEZE"
else
    echo "  [skip] --yes 일 때만 저장 (경로: $FREEZE)"
fi

# --------------------------------------------------------------------------
say "[4/5] 설치"
if [ "$MODE" = "--check" ]; then
    echo "  [check 모드] 설치하지 않음. 실제 실행은 --yes."
else
    # torch + 짝이 맞는 torchvision (pip이 호환 버전을 고르게 둔다)
    run $PY -m pip install "torch==${TORCH_VER}" torchvision
    # flash-attn: torch 2.9 고정 빌드라 반드시 재설치. --no-build-isolation 은
    # 이미 설치된 torch 로 빌드하게 해 wheel 이 없을 때의 소스 빌드를 가능하게 한다.
    run $PY -m pip install --force-reinstall --no-build-isolation flash-attn
fi

# --------------------------------------------------------------------------
say "[5/5] 검증"
if [ "$MODE" = "--yes" ]; then
    $PY - <<'EOF'
import sys
import torch, torch.nn.functional as F
print(f"  torch {torch.__version__} / cuda {torch.version.cuda} / cudnn {torch.backends.cudnn.version()}")
ok = True
try:
    import flash_attn; print(f"  flash_attn {flash_attn.__version__}  OK")
except Exception as e:
    print(f"  flash_attn 실패: {type(e).__name__}: {e}"); ok = False
try:
    import deepspeed; print(f"  deepspeed {deepspeed.__version__}  OK")
except Exception as e:
    print(f"  deepspeed 실패: {type(e).__name__}: {e}"); ok = False

# 핵심 검증: patch_embed 형상에서 conv 가 cuDNN 을 타는가
C,T,P,E,N = 3,2,14,1280,34480
x = torch.randn(N, C*T*P*P, device="cuda", dtype=torch.bfloat16).view(N,C,T,P,P)
W = (torch.randn(E,C,T,P,P, device="cuda", dtype=torch.bfloat16)*0.02)
be = str(torch._C._select_conv_backend(x, W, None, [T,P,P], [0,0,0], [1,1,1], False, [0,0,0], 1)).split('.')[-1]
print(f"  conv backend : {be}   (Cudnn 이어야 성공, SlowDilated3d 면 실패)")
ok &= (be == "Cudnn")
print("\n  => " + ("성공: 원본 conv 그대로 쓸 수 있다. config 에서 patch_embed_impl 을 빼도 된다."
                   if ok else "실패: 위 항목을 확인할 것. 되돌리려면 pip freeze 파일 사용."))
sys.exit(0 if ok else 1)
EOF
else
    echo "  [skip] --yes 일 때만 실행"
fi
