#!/bin/bash
# gd_vln 컨테이너 안에서 gd_ifc_pkg(gdm 메시지)를 /ws/ros_ws 에 빌드하고 환경을 source 한다.
# 사용: source scripts/sim/setup_ros_ws.sh   (이미 빌드돼 있으면 source 만)
source /opt/ros/jazzy/setup.bash
WS=/ws/ros_ws
if [[ ! -f $WS/install/gd_ifc_pkg/share/gd_ifc_pkg/package.xml || $WS/src/gd_ifc_pkg/msg -nt $WS/install/gd_ifc_pkg ]]; then
    (cd $WS && colcon build --packages-select gd_ifc_pkg 2>&1 | tail -2)
fi
source $WS/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
# 이 컨테이너는 root, sim/gdm 은 uid 1000 이라 Fast DDS 공유메모리(SHM) transport 로는 discovery 만 되고
# 데이터가 안 온다 (2026-10-05 실측: topic list 는 보이나 echo/hz 무응답). UDP 만 쓰게 강제한다.
export FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}
