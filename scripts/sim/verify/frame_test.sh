#!/bin/bash
# bridge 좌표계 일관성 테스트 (모델·Isaac Sim 불필요): fake_sim(map->odom 오프셋) + stub_server(고정 응답) + bridge.
# 사용(gd_vln 컨테이너 안): scripts/sim/verify/frame_test.sh [seconds per case]
# 각 (pose_source, map_odom) 조합에서 goal(/gdg/data/gp)·path(/vln/trajectory)를 frame_check.py 로 판정한다.
# (set -u 금지: /opt/ros/jazzy/setup.bash 가 unbound 변수를 참조)
SECS=${1:-15}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$ROOT"
OUT=scripts/sim/verify/out/frame_$(date +%y%m%d_%H%M%S); mkdir -p $OUT
export PYTHONDONTWRITEBYTECODE=1
source scripts/sim/setup_ros_ws.sh; export ROS_DOMAIN_ID=${PLUMB_DOMAIN:-77}   # 별도 domain: 실제 sim/gdm 과 섞이지 않게
EP=data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/17DRP5sb8fy
PORT=5891
python3 scripts/sim/verify/stub_server.py --port $PORT > $OUT/stub.log 2>&1 &
SRV=$!
for POSE in odom tf; do
  for MO in "[0.0,0.0,0.0]" "[2.0,-1.0,30.0]"; do
    for MODE in pixel_goal trajectory; do
      TAG=${POSE}_$(echo $MO | tr -d '[] ' | tr ',' '_')_$MODE
      python3 scripts/sim/verify/fake_sim.py --ros-args -p episode_dir:=$EP -p episode:=0 -p "map_odom:=$MO" > $OUT/fake_sim_$TAG.log 2>&1 &
      FS=$!; sleep 2
      python3 scripts/sim/vln_bridge_node.py --ros-args -p mode:=$MODE -p pose_source:=$POSE -p server_url:=http://127.0.0.1:$PORT -p instruction:="frame test" > $OUT/bridge_$TAG.log 2>&1 &
      BR=$!
      KIND=$([ $MODE = pixel_goal ] && echo goal || echo path)
      R=$(python3 scripts/sim/verify/frame_check.py --ros-args -p kind:=$KIND -p seconds:=$SECS.0 2> $OUT/check_$TAG.log | tail -1)
      echo "$TAG  $R" | tee -a $OUT/summary.txt
      kill $BR $FS; wait $BR $FS 2>/dev/null
    done
  done
done
kill $SRV; wait $SRV 2>/dev/null
echo "[frame_test] out: $OUT"
