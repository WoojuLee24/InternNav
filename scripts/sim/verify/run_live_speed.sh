#!/bin/bash
# 실제 sim 에서 40 s 직진 주행으로 단일 서버 vs S1/S2 분리(async, MPS 유/무) 비교 (Mode B, gdm planner off).
# gd_vln 컨테이너 안에서: scripts/sim/verify/run_live_speed.sh [model_path]
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$ROOT"
CK=${1:-/checkpoints/InternVLA-N1-DualVLN}
source scripts/sim/setup_ros_ws.sh >/dev/null
L=scripts/sim/verify/out/live
pgrep -f "^python3 scripts/sim/web_viz.py" >/dev/null || { nohup python3 scripts/sim/web_viz.py --ros-args -p use_sim_time:=true > $L/web_viz.log 2>&1 & sleep 5; }
run_probe() {
  D=$L/probe_$1; rm -rf $D; mkdir -p $D
  nohup python3 scripts/sim/vln_bridge_node.py --ros-args -p use_sim_time:=true -p mode:=trajectory -p log_dir:=$D > $D/bridge.log 2>&1 &
  BP=$!; sleep 6; echo "== $1"; python3 scripts/sim/verify/live_speed_probe.py $D 40 2>&1 | grep "{"; kill -INT $BP; sleep 3
}
scripts/sim/run_dual.sh stop >/dev/null 2>&1
nohup python3 scripts/sim/vln_server.py --model_path $CK --quant bf16 > $L/probe_mono_server.log 2>&1 &
until curl -sf localhost:5801/health >/dev/null; do sleep 3; done
run_probe mono
scripts/sim/run_dual.sh start $CK async 0 0 >/dev/null 2>&1;   run_probe split_async_nomps
scripts/sim/run_dual.sh start $CK async 50 50 >/dev/null 2>&1; run_probe split_async_mps50_50
scripts/sim/run_dual.sh start $CK async 70 30 >/dev/null 2>&1; run_probe split_async_mps70_30
echo "== done"
