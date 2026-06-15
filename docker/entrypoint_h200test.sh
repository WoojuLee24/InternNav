#!/bin/bash
set -e

# emptyDir 마운트가 이미지 내 디렉토리를 덮어쓰므로,
# 컨테이너 시작 시점에 다시 한번 필요한 하위 디렉토리/권한을 보정 (root 권한으로 항상 가능)
mkdir -p \
  "$HOME/.cache/ov" \
  "$HOME/.cache/pip" \
  "$HOME/.cache/nvidia/GLCache" \
  "$HOME/.nv/ComputeCache" \
  "$HOME/isaac-sim/kit/cache" \
  "$HOME/.nvidia-omniverse/logs" \
  "$HOME/.local/share/ov/data"

chmod -R 1777 \
  "$HOME/.cache" \
  "$HOME/.nv" \
  "$HOME/isaac-sim" \
  "$HOME/.nvidia-omniverse" \
  "$HOME/.local" 2>/dev/null || true

exec /workspace/isaaclab/_isaac_sim/python.sh -m jupyterlab \
  --notebook-dir="$HOME" \
  --ip=0.0.0.0 \
  --no-browser \
  --allow-root \
  --port=8888 \
  --LabApp.token='' \
  --LabApp.password='' \
  --LabApp.allow_origin='*' \
  --LabApp.base_url="${NB_PREFIX}"