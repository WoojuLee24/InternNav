#!/bin/bash

# 1. 호스트의 X 서버 접근 허용 (GUI 실행용)
xhost +local:docker > /dev/null

# 2. 컨테이너 이름 설정 (관리 편의성)
CONTAINER_NAME="internnav_dev"

# 3. Docker 실행
docker run -ti --rm \
    --name $CONTAINER_NAME \
    --gpus all \
    --shm-size 32g \
    --network host \
    --ipc host \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/workspace/InternNav \
    dnwn24/internnav:torch2.9.0-cuda13.0-issac-ros2-jazzy-3080ti \
    /bin/bash

# 4. 컨테이너 종료 후 X 서버 권한 복구 (선택 사항)
# xhost -local:docker > /dev/null
