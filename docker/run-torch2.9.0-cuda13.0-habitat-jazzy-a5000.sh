#!/bin/bash
# A5000 habitat 셸: 학습 · habitat 평가 · HF 기준값 · sim 연동(sim_docker / gdm_docker 와 host network, ROS_DOMAIN_ID=42).
# 컨테이너는 계속 떠 있게 만들고(sleep infinity, scripts/sim 등이 docker exec 로 쓴다) 셸을 붙인다. -d 면 띄우기만 한다.
# 이미 있으면 그대로 쓰고(멈춰 있으면 start), 이미지가 없으면 먼저 빌드한다.
set -e
IMAGE=dnwn24/internnav:torch2.9.0-cuda13.0-habitat-jazzy-a5000
NAME=internnav-habitat
HERE="$(cd "$(dirname "$0")" && pwd)"
WS=/home/dnwn24/gd_project/modules/gd_vln/workspace
GDM_SRC=/home/dnwn24/gd_project/modules/gdm_docker/workspace/src

shell() { [ "${1:-}" = -d ] && { echo "[run] $NAME 실행 중"; exit 0; }; exec docker exec -it "$NAME" bash; }
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then docker start "$NAME" >/dev/null; shell "$@"; fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then   # 없으면 이 폴더의 Dockerfile 로 빌드 (git 커밋을 label 로 남긴다)
    docker build -t "$IMAGE" --label gdvln.git="$(git -C "$HERE" rev-parse --short HEAD)" -f "$HERE/Dockerfile-${IMAGE#*:}" "$HERE"
fi
xhost +local:root >/dev/null 2>&1 || true
# root(이 컨테이너) ↔ uid 1000(sim/gdm) 사이는 Fast DDS SHM 으로 데이터가 안 건너온다 -> UDP 강제
docker run --name "$NAME" --shm-size 32g --entrypoint bash -d --network=host --ipc=host --gpus all --runtime=nvidia \
    -e DISPLAY="$DISPLAY" -e ROS_DOMAIN_ID=42 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/.Xauthority:/root/.Xauthority \
    -v $WS:/ws:rw \
    -v /home/dnwn24/data:/datasets:ro \
    -v /media/TrainDataset:/ws/src/InternNav/data:rw \
    -v /media/LTDataset/checkpoints:/checkpoints:ro \
    -v /media/LTDataset/checkpoints:/ws/src/InternNav/checkpoints:ro \
    -v $GDM_SRC/gd_ifc_pkg:/ws/ros_ws/src/gd_ifc_pkg:ro \
    -w /ws/src/InternNav \
    "$IMAGE" -c "sleep infinity" >/dev/null
shell "$@"
