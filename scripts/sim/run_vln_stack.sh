#!/bin/bash
# VLN model stack for the sim (HOST script — drives two containers):
#   S2  vln-deploy S2 module, HF backend (bf16)      -> gd_vln container (internnav-habitat)
#   S1  vln-deploy S1 module, TensorRT FP16 10 step   -> isaacsim container (vln-s1-trt), reproduced engines
#       (or HF PyTorch backend in the gd_vln container with S1=hf)
#   dual_server (async|sync, HTTP :5801)               -> gd_vln container; vln_bridge_node / bench talk to it
#
# 사용: scripts/sim/run_vln_stack.sh <start|stop> [model_path] [mode]
#   env S1=trt|hf (기본 trt), S1_STEPS=10, S2=hf|gguf (기본 hf), S2_QUANT=bf16|nf4|int8 (hf),
#       GGUF_TEXT/GGUF_MMPROJ (gguf; 기본 = 재현 Q8_0 decoder + mmproj f16), S2_RESIZE=384 (gguf), S2_HF_PARITY=1 (gguf: HF 해상도 392/644x476 + latent EOS; 0=기존 112정렬),
#       S2_EMBD_CACHE=512 (gguf: vision 임베딩 캐시 항목 수, history 프레임 미리 encode; 0=끔), MPS_S2=0, MPS_S1=0
set -u
CMD=${1:?start|stop}; CKPT=${2:-/checkpoints/InternVLA-N1-DualVLN}; MODE=${3:-async}
S1=${S1:-trt}; S1_STEPS=${S1_STEPS:-10}; S2_QUANT=${S2_QUANT:-bf16}; MPS_S2=${MPS_S2:-0}; MPS_S1=${MPS_S1:-0}
S2=${S2:-hf}; S2_RESIZE=${S2_RESIZE:-384}; S2_HF_PARITY=${S2_HF_PARITY:-1}; S2_EMBD_CACHE=${S2_EMBD_CACHE:-512}
# S1_CTR / S1_HEALTH_PORT: S1 container name / HealthBeacon port (change only if a stale S1 container still holds them)
S1_CTR=${S1_CTR:-vln-s1-trt}; S1_HEALTH_PORT=${S1_HEALTH_PORT:-5614}
if [[ $S2_HF_PARITY == 1 ]]; then GGUF_RES_ARGS="--hf-parity"; else GGUF_RES_ARGS="--lookdown-full-res"; fi
GGUF_TEXT=${GGUF_TEXT:-quantization_kit/out/gguf/InternVLA-S2-text-Q8_0-latent-embf16.gguf}
GGUF_MMPROJ=${GGUF_MMPROJ:-quantization_kit/out/gguf/InternVLA-S2-mmproj-f16.gguf}
GD=${GD_CONTAINER:-internnav-habitat}
IS_IMG=dnwn24/internnav:torch2.7.0-cuda12.8-isaacsim5.1.0-jazzy-5090
W=/home/universe/gd_project/modules/gd_vln/workspace
L=scripts/sim/verify/out/live
MPS_DIR=/tmp/vln-mps   # host dir shared by both containers (MPS pipe) when MPS is on

gd() { docker exec $GD bash -c "cd /ws/src/InternNav && export VLN_PY=/usr/bin/python3 PYTHONDONTWRITEBYTECODE=1 && $*"; }
stop_all() {
  docker rm -f $S1_CTR vln-s2-gguf >/dev/null 2>&1
  for p in $(docker exec $GD pgrep -f "^\S*python\S* -u -m deploy\.s[12]_module\.service|^python3 scripts/sim/(dual_server|vln_server)\.py"); do
    docker exec $GD kill -9 $p
  done
  docker exec $GD bash -c "pgrep -f nvidia-cuda-mps-control >/dev/null && echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPS_DIR nvidia-cuda-mps-control" >/dev/null 2>&1
  sleep 2
}
wait_log() {
  until docker exec $GD grep -q "$2" /ws/src/InternNav/$1 2>/dev/null; do
    sleep 3
    docker exec $GD grep -q Traceback /ws/src/InternNav/$1 2>/dev/null && { echo "[stack] FAILED: $1"; docker exec $GD tail -8 /ws/src/InternNav/$1; exit 1; }
  done
}

case "$CMD" in
  stop) stop_all; echo "[stack] stopped"; exit 0 ;;
  start) ;;
  *) echo "unknown cmd $CMD" >&2; exit 2 ;;
esac
stop_all

MPS_ENV=""
if [[ "$MPS_S2" != 0 || "$MPS_S1" != 0 ]]; then
  [[ "$S1" == trt ]] && { echo "[stack] MPS with S1=trt needs the MPS pipe shared across containers ($MPS_DIR is not mounted in $GD) -> use S1=hf for MPS"; exit 2; }
  gd "mkdir -p $MPS_DIR && CUDA_MPS_PIPE_DIRECTORY=$MPS_DIR CUDA_MPS_LOG_DIRECTORY=$MPS_DIR nvidia-cuda-mps-control -d"
  MPS_ENV="CUDA_MPS_PIPE_DIRECTORY=$MPS_DIR"
  echo "[stack] MPS daemon on (S2 ${MPS_S2}% / S1 ${MPS_S1}%)"
fi
pct() { [[ "$1" != 0 ]] && echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$1" || echo ""; }

docker exec $GD rm -f /ws/src/InternNav/$L/stack_s1.log /ws/src/InternNav/$L/stack_s2.log
case "$S2" in
  hf) gd "mkdir -p $L; env $MPS_ENV $(pct $MPS_S2) nohup bash vln-deploy/deploy/s2_module/run.sh --s2-backend hf --hf-model-path $CKPT --hf-quant $S2_QUANT > $L/stack_s2.log 2>&1 &" ;;
  gguf)   # llama.cpp runner + vln-deploy convenv live in the isaacsim image
    docker run -d --rm --name vln-s2-gguf --entrypoint bash --gpus all --runtime=nvidia --network=host --ipc=host \
      -v $W:/ws -v /media/LTDataset/checkpoints:/ws/src/InternNav/checkpoints:ro -w /ws/src/InternNav/vln-deploy $IS_IMG \
      -c "PYTHONDONTWRITEBYTECODE=1 bash deploy/s2_module/run.sh --gguf-text $GGUF_TEXT --gguf-mmproj $GGUF_MMPROJ --resize $S2_RESIZE $GGUF_RES_ARGS --embd-cache $S2_EMBD_CACHE > /ws/src/InternNav/$L/stack_s2.log 2>&1" >/dev/null ;;
  *) echo "unknown S2=$S2" >&2; exit 2 ;;
esac
case "$S1" in
  trt)
    docker run -d --rm --name $S1_CTR -e VLN_PORT_S1_HEALTH=$S1_HEALTH_PORT --entrypoint bash --gpus all --runtime=nvidia --network=host --ipc=host \
      -v $W:/ws -v /media/LTDataset/checkpoints:/ws/src/InternNav/checkpoints:ro -w /ws/src/InternNav/vln-deploy $IS_IMG \
      -c "PYTHONDONTWRITEBYTECODE=1 bash deploy/s1_module/run.sh --s1-steps $S1_STEPS --no-gpu-prep --trt-onnx-dir quantization_kit/out/onnx --trt-engine-dir quantization_kit/out/trt_engines > /ws/src/InternNav/$L/stack_s1.log 2>&1" >/dev/null ;;
  hf)
    gd "env $MPS_ENV $(pct $MPS_S1) nohup bash vln-deploy/deploy/s1_module/run.sh --s1-backend hf --hf-model-path $CKPT > $L/stack_s1.log 2>&1 &" ;;
  *) echo "unknown S1=$S1" >&2; exit 2 ;;
esac
wait_log $L/stack_s2.log "S2_SERVICE READY"; echo "[stack] S2: $(docker exec $GD grep 'S2_SERVICE READY' /ws/src/InternNav/$L/stack_s2.log)"
wait_log $L/stack_s1.log "S1_SERVICE READY"; echo "[stack] S1: $(docker exec $GD grep 'S1_SERVICE READY' /ws/src/InternNav/$L/stack_s1.log)"
gd "nohup python3 scripts/sim/dual_server.py --mode $MODE --model_name '$CKPT s1=$S1' > $L/stack_dual.log 2>&1 &"
until docker exec $GD curl -sf localhost:5801/health >/dev/null; do sleep 1; done
echo "[stack] dual_server ($MODE) up on :5801 — S1=$S1 (${S1_STEPS} steps if trt), S2=$S2 ($([[ $S2 == hf ]] && echo $S2_QUANT || echo "$GGUF_TEXT @${S2_RESIZE}"))"
