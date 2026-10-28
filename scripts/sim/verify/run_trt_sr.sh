#!/bin/bash
# Closed-loop SR: S1 = original PyTorch checkpoint (HF bf16, 10 steps) vs reproduced TensorRT FP16 engines
# (vln-deploy lite S1 + quantization_kit/out/trt_engines, 10 / 5 steps). S2 = HF bf16 for all, dual_server async,
# Mode B (gdm planner off: ./lpp-sim-vln), clear-area episodes, REPS repeats.  Run on the HOST (drives 2 containers).
#   scripts/sim/verify/run_trt_sr.sh [reps]
set -u
REPS=${1:-2}
GD=internnav-habitat
IS_IMG=dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090
W=/home/universe/gd_project/modules/gd_vln/workspace
EPS=scripts/sim/verify/e2e_episodes_etri_indoor_clear.json
L=scripts/sim/verify/out/live

gd() { docker exec $GD bash -c "cd /ws/src/InternNav && export VLN_PY=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1 && $*"; }
stop_s1() {
  docker rm -f vln-s1-trt >/dev/null 2>&1
  for p in $(docker exec $GD pgrep -f "^\S*python\S* -u -m deploy\.s1_module\.service"); do docker exec $GD kill -9 $p; done
  sleep 2
}
wait_log() { until docker exec $GD grep -q "$2" $1 2>/dev/null; do sleep 3; docker exec $GD grep -q Traceback $1 2>/dev/null && { echo "FAILED: $1"; docker exec $GD tail -5 $1; exit 1; }; done; }

gd "scripts/sim/run_dual.sh stop" >/dev/null 2>&1
gd "nohup bash vln-deploy/deploy/s2_module/run.sh --s2-backend hf --hf-model-path /checkpoints/InternVLA-N1-DualVLN > $L/sr_s2.log 2>&1 &"
wait_log /ws/src/InternNav/$L/sr_s2.log "S2_SERVICE READY"
echo "[sr] S2 (HF bf16) ready"

for cfg in hf10 trt10 trt5; do
  stop_s1
  for p in $(docker exec $GD pgrep -f "^python3 scripts/sim/dual_server\.py"); do docker exec $GD kill -9 $p; done
  docker exec $GD rm -f /ws/src/InternNav/$L/sr_s1.log
  case $cfg in
    hf10) gd "nohup bash vln-deploy/deploy/s1_module/run.sh --s1-backend hf --hf-model-path /checkpoints/InternVLA-N1-DualVLN > $L/sr_s1.log 2>&1 &" ;;
    trt10|trt5)
      STEPS=${cfg#trt}
      docker run -d --rm --name vln-s1-trt --entrypoint bash --gpus all --runtime=nvidia --network=host --ipc=host \
        -v $W:/ws -v /media/LTDataset/checkpoints:/ws/src/InternNav/checkpoints:ro -w /ws/src/InternNav/vln-deploy $IS_IMG \
        -c "PYTHONDONTWRITEBYTECODE=1 bash deploy/s1_module/run.sh --s1-steps $STEPS --no-gpu-prep --trt-onnx-dir quantization_kit/out/onnx --trt-engine-dir quantization_kit/out/trt_engines > /ws/src/InternNav/$L/sr_s1.log 2>&1" >/dev/null ;;
    *) echo "unknown cfg $cfg"; exit 2 ;;
  esac
  wait_log /ws/src/InternNav/$L/sr_s1.log "S1_SERVICE READY"
  echo "[sr] $cfg S1: $(docker exec $GD grep 'S1_SERVICE READY' /ws/src/InternNav/$L/sr_s1.log)"
  gd "nohup python3 scripts/sim/dual_server.py --mode async --model_name s1=$cfg > $L/sr_dual_$cfg.log 2>&1 &"
  until docker exec $GD curl -sf localhost:5801/health >/dev/null; do sleep 1; done
  for r in $(seq 1 $REPS); do
    gd "source scripts/sim/setup_ros_ws.sh >/dev/null; python3 scripts/sim/verify/run_e2e.py --mode trajectory --tag sr_s1_${cfg}_rep$r --episodes $EPS" \
      | grep '^{' | python3 -c "import sys,json;[print('   ',d['id'],'OK' if d.get('success') else 'x','%.1f m'%d['final_dist_to_goal_m']) for d in map(json.loads,sys.stdin)]"
  done
done
stop_s1
echo "[sr] done"
