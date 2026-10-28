#!/bin/bash
# fake_sim + vln_server + vln_bridge_node 를 한 컨테이너에서 띄워 배선을 검증한다 (Isaac Sim 불필요).
# 사용(gd_vln 컨테이너 안): scripts/sim/verify/plumbing_test.sh <model_path> <quant> <seconds per mode>
# (set -u 금지: /opt/ros/jazzy/setup.bash 가 unbound 변수를 참조)
CKPT=${1:-/checkpoints/InternVLA-N1-DualVLN}; QUANT=${2:-bf16}; SECS=${3:-60}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$ROOT"
OUT=scripts/sim/verify/out/plumbing_$(date +%y%m%d_%H%M%S); mkdir -p $OUT
export PYTHONDONTWRITEBYTECODE=1 ROS_DOMAIN_ID=${PLUMB_DOMAIN:-77}   # 별도 domain: 실제 sim/gdm 과 섞이지 않게
source scripts/sim/setup_ros_ws.sh; export ROS_DOMAIN_ID=${PLUMB_DOMAIN:-77}
EP=data/InternData-N1-v0.5-mini/vln_ce/traj_data/r2r/17DRP5sb8fy
python3 scripts/sim/vln_server.py --model_path $CKPT --quant $QUANT > $OUT/server.log 2>&1 &
SRV=$!
until curl -sf localhost:5801/health >/dev/null; do sleep 3; kill -0 $SRV 2>/dev/null || { echo server died; tail $OUT/server.log; exit 1; }; done
for MODE in pixel_goal trajectory; do
  python3 scripts/sim/verify/fake_sim.py --ros-args -p episode_dir:=$EP -p episode:=0 > $OUT/fake_sim_$MODE.log 2>&1 &
  FS=$!; sleep 3
  INSTR=$(python3 -c "import json;print([json.loads(l) for l in open('$EP/meta/episodes.jsonl')][0]['tasks'][0].strip())")
  python3 scripts/sim/vln_bridge_node.py --ros-args -p mode:=$MODE -p log_dir:=$OUT/bridge_$MODE -p instruction:="$INSTR" > $OUT/bridge_$MODE.log 2>&1 &
  BR=$!
  sleep $((SECS / 2))
  { echo "== $MODE"; for t in /gdq/msg/cmd_vel /cmd_vel /gdg/data/gp /vln/trajectory; do ros2 topic info -v $t 2>/dev/null | grep -E "^Publisher count|Node name" | tr '\n' ' '; echo " <- $t"; done; } >> $OUT/topic_info.txt
  timeout 5 ros2 topic echo --once /vln/trajectory nav_msgs/msg/Path > $OUT/traj_$MODE.txt 2>&1
  sleep $((SECS / 2))
  kill $BR $FS; wait $BR $FS 2>/dev/null
done
kill $SRV; wait $SRV 2>/dev/null
for f in $OUT/bridge_*/vln_*.csv; do echo "== $f"; python3 scripts/sim/verify/analyze_bridge_log.py $f; done > $OUT/analysis.txt 2>&1
echo "[plumbing] out: $OUT"; cat $OUT/topic_info.txt
