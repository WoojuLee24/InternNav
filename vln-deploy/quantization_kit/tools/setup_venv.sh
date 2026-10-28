#!/usr/bin/env bash
# 경량화 venv quantization_kit/convenv_{x86,aarch64} 를 만든다 (docs/command.md 1-2). vln-deploy/ 에서 실행:
#   x86 (5090): isaacsim 이미지 안에서  bash quantization_kit/tools/setup_venv.sh
#               Isaac 인터프리터(3.11)로 만들고, requirements_x86.txt 의 검증된 버전으로 설치한다
#   Thor      : 장치에서                bash quantization_kit/tools/setup_venv.sh
#               Jetson SBSA cu130 torch + 같은 패키지 (버전은 Thor 용 휠에 맞춰 고정하지 않음). TensorRT 는 apt
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."      # vln-deploy/
ARCH=$([ "$(uname -m)" = aarch64 ] && echo aarch64 || echo x86)
V=quantization_kit/convenv_$ARCH
[ -e "$V" ] && { echo "$V 가 이미 있음 — 다시 만들려면 먼저 옮기거나 지운다"; exit 1; }

if [ $ARCH = x86 ]; then
    [ -x /isaac-sim/kit/python/bin/python3 ] || { echo "isaacsim 이미지 안에서 실행 (Isaac python 3.11 필요)"; exit 1; }
    /isaac-sim/kit/python/bin/python3 -m venv $V      # 시스템 3.12 로 만들면 Isaac 쪽에서 import 가 안 된다
    $V/bin/pip install --upgrade pip
    $V/bin/pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.0' 'torchvision==0.22.0'   # cu128 이어야 sm_120
    $V/bin/pip install -r quantization_kit/tools/requirements_x86.txt
else
    python3 -m venv $V
    $V/bin/pip install --upgrade pip
    $V/bin/pip install torch --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130   # Thor: SBSA, jp7 아님
    $V/bin/pip install diffusers safetensors transformers einops accelerate numpy pillow timm opencv-python-headless \
        onnx onnxscript pyzmq msgpack pandas pyarrow huggingface_hub
    echo "TensorRT 는 pip 휠이 Tegra 를 거부한다: sudo apt install -y tensorrt python3-libnvinfer libzmq3-dev"
fi
$V/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
