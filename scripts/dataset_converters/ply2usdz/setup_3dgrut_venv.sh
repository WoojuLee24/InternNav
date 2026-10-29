#!/bin/bash
# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI).
# All rights reserved. 이용 조건은 NOTICE 파일을 따른다.
# 3DGRUT(ply_to_usd) 를 conda 없이 venv 로 설치한다. conda 이미지 3dgrut:cuda128 과 같은 구성:
#   3dgrut 16b7b19 · python 3.11 · torch 2.9.1+cu128 · kaolin c2da967(소스 빌드) · fused-ssim 98126b7 · 나머지는 requirements-3dgrut.txt
# 빌드 도구(nvcc 12.8, gcc 13, Isaac python 3.11)는 gd_vln isaac 이미지에서 빌려 쓴다. 호스트에서 실행 (10~30분):
#   ./setup_3dgrut_venv.sh [DEST]          기본 DEST=~/git/3dgrut-venv  ->  DEST/src (소스), DEST/venv
# 사용 (컨테이너 안 경로 /opt/3dgrut 고정 — venv 는 절대경로라 옮기면 안 된다):
#   docker run --rm --gpus '"device=0"' -e PATH=/opt/3dgrut/venv/bin:/usr/local/cuda-12.8/bin:/usr/bin:/bin -v DEST:/opt/3dgrut -v "$(pwd)":/usd2usdz \
#     dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090 python -m threedgrut.export.scripts.ply_to_usd <in.ply> --output_file <out_nurec.usdz>
#   PATH 에 venv/bin 을 앞에 둔다 (conda activate 와 같은 효과: 실행 중 C++ 확장 JIT 빌드가 venv 의 ninja 를 쓴다)
set -euo pipefail
DEST=${1:-$HOME/git/3dgrut-venv}
IMAGE=${IMAGE:-dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090}
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -e "$DEST/venv" ] && { echo "$DEST/venv 가 이미 있음 — 다시 만들려면 먼저 옮기거나 지운다"; exit 1; }
mkdir -p "$DEST"
docker run --rm --entrypoint bash -v "$DEST":/opt/3dgrut -v "$HERE/requirements-3dgrut.txt":/tmp/req.txt:ro "$IMAGE" -c '
set -euo pipefail
cd /opt/3dgrut
[ -d src ] || git clone --recursive https://github.com/nv-tlabs/3dgrut.git src
git -C src checkout -q 16b7b19bf79e66991280d244f8b6300a32d7fff4 && git -C src submodule update --init --recursive -q
# ply_to_usd 는 PLY 를 3DGRUT 학습 좌표계(Y-down)로 보고 축 변환 행렬을 붙인다. PortalCam PLY 는 이미 Z-up 이라
# 그 행렬을 빼서 기존 ETRI 데이터처럼 단위 행렬로 쓰게 한다 (빼지 않으면 장면이 90° 눕는다)
sed -i "s/corrected_matrix = normalizing_inverse @ default_conv_tf/corrected_matrix = normalizing_inverse/" src/threedgrut/export/usd_util.py
grep -q "corrected_matrix = normalizing_inverse$" src/threedgrut/export/usd_util.py
/isaac-sim/kit/python/bin/python3 -m venv venv
P=/opt/3dgrut/venv/bin/python
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0" FORCE_CUDA=1 MAX_JOBS=8
export CPATH=/isaac-sim/kit/python/include/python3.11        # Isaac python 의 INCLUDEPY 가 없는 경로를 가리킨다
$P -m pip install -q --upgrade pip
$P -m pip install -q torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
$P -m pip install -q --no-deps -r /tmp/req.txt
rm -rf /tmp/kaolin && git clone -q --recursive https://github.com/NVIDIAGameWorks/kaolin.git /tmp/kaolin
git -C /tmp/kaolin checkout -q c2da967b9e0d8e3ebdbd65d3e8464d7e39005203
sed -i "s!AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type()!AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type()!g" /tmp/kaolin/kaolin/csrc/render/spc/raytrace_cuda.cu
(cd /tmp/kaolin && IGNORE_TORCH_VER=1 $P -m pip install --no-deps --no-build-isolation .)
$P -m pip install --no-deps --no-build-isolation git+https://github.com/rahul-goel/fused-ssim.git@98126b7781f9e563234c92d2bf08ee0994f4f175
$P -m pip install --no-deps --no-build-isolation -e src
# 3DGUT 렌더러 C++ 확장을 미리 빌드해 패키지 폴더에 둔다 (conda 이미지도 src/threedgut_tracer/lib3dgut_cc.so 를 들고 있다).
# 실행 중 JIT 빌드는 torch 2.9 에서 import 가 안 되고(sys.path 에 빌드 폴더가 없다) 컨테이너가 지워지면 사라진다.
export PATH=/opt/3dgrut/venv/bin:$PATH TORCH_EXTENSIONS_DIR=/tmp/torch_ext
(cd /tmp && $P -c "from threedgrut.export.scripts.ply_to_usd import load_default_config as c; from threedgut_tracer.setup_3dgut import setup_3dgut as s
try: s(c())
except Exception as e: print(\"[3dgrut venv] lib3dgut_cc built (import here skipped:\", type(e).__name__, \")\")")
cp "$(find /tmp/torch_ext -name lib3dgut_cc.so | head -1)" src/threedgut_tracer/lib3dgut_cc.so
$P -c "import torch, kaolin, threedgrut; print(\"[3dgrut venv]\", torch.__version__, kaolin.__version__, threedgrut.__file__)"
chown -R '"$(id -u):$(id -g)"' /opt/3dgrut
'
