#!/bin/bash
# A5000 isaac 셸: VLN-PE(InternUtopia) · nav3d 렌더 · 경량화(vln-deploy/docs/command.md) · sim 스택 S1/S2.
# 컨테이너는 계속 떠 있게 만들고(sleep infinity, scripts/sim 등이 docker exec 로 쓴다) 셸을 붙인다. -d 면 띄우기만 한다.
# 이미 있으면 그대로 쓰고(멈춰 있으면 start), 이미지가 없으면 먼저 빌드한다.
set -e
IMAGE=dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-a5000
NAME=internnav-isaacsim
HERE="$(cd "$(dirname "$0")" && pwd)"
WS=/home/dnwn24/gd_project/modules/gd_vln/workspace
C=$HOME/docker/isaac-sim      # Isaac Sim 캐시 (셰이더·확장). 지우면 첫 실행이 느려질 뿐

shell() { [ "${1:-}" = -d ] && { echo "[run] $NAME 실행 중"; exit 0; }; exec docker exec -it "$NAME" bash; }
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then docker start "$NAME" >/dev/null; shell "$@"; fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then   # 없으면 이 폴더의 Dockerfile 로 빌드 (git 커밋을 label 로 남긴다)
    docker build -t "$IMAGE" --label gdvln.git="$(git -C "$HERE" rev-parse --short HEAD)" -f "$HERE/Dockerfile-${IMAGE#*:}" "$HERE"
fi
xhost +local:root >/dev/null 2>&1 || true
docker run --name "$NAME" --shm-size 32g --entrypoint bash -d --network=host --ipc=host --gpus all --runtime=nvidia \
    -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority \
    -v $C/cache/kit:/isaac-sim/kit/cache:rw \
    -v $C/cache/ov:/root/.cache/ov:rw \
    -v $C/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v $C/cache/computecache:/root/.nv/ComputeCache:rw \
    -v $C/logs:/root/.nvidia-omniverse/logs:rw \
    -v $C/data:/root/.local/share/ov/data:rw \
    -v $C/documents:/root/Documents:rw \
    -v $WS:/ws:rw \
    -v /home/dnwn24/data:/datasets:ro \
    -v /media/TrainDataset:/ws/src/InternNav/data:rw \
    -v /media/LTDataset/checkpoints:/ws/src/InternNav/checkpoints:ro \
    -v /media/TrainDataset/InternData-N1-v0.5-mini/scene_data/mp3d_pe:/isaac-sim/Matterport3D/data/v1/scans:ro \
    -w /ws/src/InternNav \
    "$IMAGE" -c "sleep infinity" >/dev/null
shell "$@"
