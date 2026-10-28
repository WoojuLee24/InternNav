#!/bin/bash
# ④ 전달 테스트 (실제 sim + gdm ./lpp-sim 필요, 모델 불필요): stub_server(--once, 고정 pixel 1개) + bridge(Mode A).
# 사용(gd_vln 컨테이너 안): scripts/sim/verify/gdm_test.sh [seconds per case] [lookdown 0|1]
# 각 pixel 마다 시작 위치로 teleport -> 목표 1개 전송 -> gdm_check.py 가 gp 해석·계획 끝점·도착·cmd_vel 발행자를 잰다.
# (set -u 금지: /opt/ros/jazzy/setup.bash 가 unbound 변수를 참조)
SECS=${1:-40}; LDON=${2:-0}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$ROOT"
OUT=scripts/sim/verify/out/gdm_$(date +%y%m%d_%H%M%S); mkdir -p $OUT
export PYTHONDONTWRITEBYTECODE=1
source scripts/sim/setup_ros_ws.sh
PORT=5893
for PIX in 400,320 330,200 330,440; do   # 정면 아래 / 왼쪽 앞 / 오른쪽 앞 (row,col)
  TAG=${PIX/,/_}_ld$LDON
  python3 - <<'PY'
import math, time, rclpy
from geometry_msgs.msg import Pose2D
from gd_ifc_pkg.msg import GDGGuidancePoint as GP
rclpy.init(); n = rclpy.create_node("gdm_test_tp"); p = n.create_publisher(Pose2D, "/sim/teleport", 1)
s = n.create_publisher(GP, "/gdg/data/gp", 10)  # FORCE_TO_STOP: drop the previous case's goal before teleporting
t0 = time.time()
while s.get_subscription_count() == 0 and time.time() - t0 < 10:
    rclpy.spin_once(n, timeout_sec=0.1)
g = GP(); g.gp_type.append(GP.FORCE_TO_STOP_GP); g.etype.append(GP.EDGE_BASIC); s.publish(g)
rclpy.spin_once(n, timeout_sec=0.5)
t0 = time.time()
while p.get_subscription_count() == 0 and time.time() - t0 < 10:  # wait for discovery, else the message is lost
    rclpy.spin_once(n, timeout_sec=0.1)
for _ in range(3):
    p.publish(Pose2D(x=0.0, y=0.0, theta=math.radians(-90.0))); rclpy.spin_once(n, timeout_sec=0.3)
print("teleport subscribers", p.get_subscription_count())
PY
  sleep 6   # settle after teleport (run_e2e.py)
  python3 scripts/sim/verify/stub_server.py --port $PORT --pixels $PIX --once > $OUT/stub_$TAG.log 2>&1 &
  SRV=$!
  BARGS="-p use_sim_time:=true -p mode:=pixel_goal -p server_url:=http://127.0.0.1:$PORT -p instruction:=gdm_test -p log_dir:=$OUT/bridge_$TAG"
  [ $LDON = 1 ] && BARGS="$BARGS -p lookdown_ns:=/camera_lookdown/camera"
  python3 scripts/sim/vln_bridge_node.py --ros-args $BARGS > $OUT/bridge_$TAG.log 2>&1 &
  BR=$!
  R=$(cd scripts/sim/verify && python3 gdm_check.py --ros-args -p use_sim_time:=true -p seconds:=$SECS.0 2> $ROOT/$OUT/check_$TAG.log | tail -1)
  echo "$TAG  $R" | tee -a $OUT/summary.txt
  kill $BR $SRV; wait $BR $SRV 2>/dev/null
done
echo "[gdm_test] out: $OUT"
