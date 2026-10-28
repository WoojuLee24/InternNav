#!/bin/bash
# ② 시각화 테스트 (모델·Isaac Sim 불필요): fake_sim + stub_server(고정 pixel 5곳, 2프레임 지연) + bridge(trajectory 모드로 이동).
# 사용(gd_vln 컨테이너 안): scripts/sim/verify/overlay_test.sh [seconds per case]
# 정면 프레임 / 30° look-down 프레임 두 경우에 대해 초록 원 위치와 배경 프레임을 overlay_check.py 로 판정한다.
# (set -u 금지: /opt/ros/jazzy/setup.bash 가 unbound 변수를 참조)
SECS=${1:-20}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$ROOT"
OUT=scripts/sim/verify/out/overlay_$(date +%y%m%d_%H%M%S); mkdir -p $OUT
export PYTHONDONTWRITEBYTECODE=1
source scripts/sim/setup_ros_ws.sh; export ROS_DOMAIN_ID=${PLUMB_DOMAIN:-77}   # 별도 domain: 실제 sim/gdm 과 섞이지 않게
EP=data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/17DRP5sb8fy
PORT=5892
PIX="240,320 40,20 40,619 459,20 459,619"   # 중앙 + 네 모서리 (row,col; 위쪽은 라벨 글자를 피해 40)
for CASE in fpv lookdown; do
  SAVE=$OUT/frames_$CASE
  python3 scripts/sim/verify/stub_server.py --port $PORT --pixels $PIX --lag 2 --save_dir $SAVE > $OUT/stub_$CASE.log 2>&1 &
  SRV=$!
  LD=$([ $CASE = lookdown ] && echo true || echo false)
  python3 scripts/sim/verify/fake_sim.py --ros-args -p episode_dir:=$EP -p episode:=0 -p lookdown:=$LD > $OUT/fake_sim_$CASE.log 2>&1 &
  FS=$!; sleep 2
  BARGS="-p mode:=trajectory -p server_url:=http://127.0.0.1:$PORT -p instruction:=overlay_test"
  [ $CASE = lookdown ] && BARGS="$BARGS -p lookdown_ns:=/camera_lookdown/camera"
  python3 scripts/sim/vln_bridge_node.py --ros-args $BARGS > $OUT/bridge_$CASE.log 2>&1 &
  BR=$!
  R=$(python3 scripts/sim/verify/overlay_check.py --ros-args -p save_dir:=$SAVE -p seconds:=$SECS.0 2> $OUT/check_$CASE.log | tail -1)
  echo "$CASE  $R" | tee -a $OUT/summary.txt
  kill $BR $FS $SRV; wait $BR $FS $SRV 2>/dev/null
done
echo "[overlay_test] out: $OUT"
