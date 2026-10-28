#!/bin/bash
# VLN 전용 rviz2 를 연다 (호스트에서 실행). gd_vln 이미지에는 rviz2 가 없어서, rviz2·robot_description·
# DISPLAY 가 이미 있는 gdm 컨테이너에 설정 파일을 복사해 띄운다 (gdm repo 는 수정하지 않음).
# 사용: scripts/sim/rviz/open_rviz.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
C=${GDM_CONTAINER:-gdm_docker-gdm-1}
docker cp "$HERE/vln.rviz" "$C:/tmp/vln.rviz"
docker exec -d "$C" bash -lc 'rviz2 -d /tmp/vln.rviz --ros-args -p use_sim_time:=true > /tmp/vln_rviz.log 2>&1'
echo "[open_rviz] rviz2 started in $C (log: docker exec $C cat /tmp/vln_rviz.log)"
