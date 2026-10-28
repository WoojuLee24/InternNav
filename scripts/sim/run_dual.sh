#!/bin/bash
# vln-deploy S2/S1 modules with the HF backend (+ optional CUDA MPS SM partition) + sim adapter dual_server (:5801).
# Run inside the gd_vln container (root). vln_bridge_node / verify/bench_dual.py talk to :5801 as to vln_server.py.
# The GGUF/TRT backends of the same modules are untouched (they run with vln-deploy's convenv in the isaacsim image).
#
# 사용: scripts/sim/run_dual.sh <start|stop> [model_path] [mode sync|async] [s2_pct] [s1_pct] [quant]
#   s2_pct/s1_pct : CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (SM 상한 %). 둘 다 0 이면 MPS 데몬 없이 일반 time-slicing.
#                   한쪽만 0 이면 그쪽은 MPS client 로 상한 없이 실행.
#   예) scripts/sim/run_dual.sh start /checkpoints/InternVLA-N1-DualVLN async 70 30
# 로그: scripts/sim/verify/out/live/dual_{s2,s1,server}.log
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
CMD=${1:?start|stop}; CKPT=${2:-/checkpoints/InternVLA-N1-DualVLN}; MODE=${3:-async}; S2P=${4:-0}; S1P=${5:-0}
QUANT=${6:-bf16}
L=$ROOT/scripts/sim/verify/out/live; mkdir -p $L
DEPLOY=$ROOT/vln-deploy
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log PYTHONDONTWRITEBYTECODE=1
export VLN_PY=$(command -v python3)   # deploy/common/env.sh: HF backends use this image's torch python

stop_all() {
  # anchored patterns: only the service processes themselves, never a shell whose command line mentions them
  for p in $(pgrep -f "^\S*python\S* -u -m deploy\.s[12]_module\.service|^python3 scripts/sim/(dual_server|vln_server)\.py"); do
    kill -9 $p 2>/dev/null
  done
  sleep 2
  pgrep -f nvidia-cuda-mps-control >/dev/null && { echo quit | nvidia-cuda-mps-control; sleep 1; }
}

case "$CMD" in
  stop) stop_all; echo "[run_dual] stopped"; exit 0 ;;
  start) ;;
  *) echo "unknown cmd $CMD" >&2; exit 2 ;;
esac

stop_all
if [[ "$S2P" != 0 || "$S1P" != 0 ]]; then
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && echo "[run_dual] MPS daemon started (S2 ${S2P}% / S1 ${S1P}% SMs; 0 = uncapped)"
fi
mps_env() { [[ "$1" != 0 ]] && echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$1" || echo ""; }
env $(mps_env $S2P) nohup bash $DEPLOY/deploy/s2_module/run.sh --s2-backend hf --hf-model-path $CKPT --hf-quant $QUANT \
    > $L/dual_s2.log 2>&1 &
env $(mps_env $S1P) nohup bash $DEPLOY/deploy/s1_module/run.sh --s1-backend hf --hf-model-path $CKPT \
    > $L/dual_s1.log 2>&1 &
for w in s2 s1; do
  until grep -q "SERVICE READY" $L/dual_$w.log 2>/dev/null; do
    sleep 3; grep -q Traceback $L/dual_$w.log 2>/dev/null && { echo "[run_dual] $w module failed"; tail -20 $L/dual_$w.log; exit 1; }
  done
  grep "SERVICE READY" $L/dual_$w.log
done
nohup python3 scripts/sim/dual_server.py --mode $MODE --model_name $CKPT > $L/dual_server.log 2>&1 &
until curl -sf localhost:5801/health >/dev/null; do sleep 1; done
echo "[run_dual] dual_server up: $(curl -s localhost:5801/health)"
