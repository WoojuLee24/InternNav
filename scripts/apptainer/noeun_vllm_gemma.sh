#!/bin/bash
# 6000pro 에서 Gemma-4-31B-it (bf16) vLLM 서버 — 라벨링 시험용 (62b_label_landmark.py --base_url http://127.0.0.1:8100/v1).
# 설정은 configs/model/gemma31b-bf16.yaml 의 managed 와 같다 (served name gemma-4-31B-it · max_model_len 16384 · image 16 · mem 0.9).
# GPU 는 0 · 1 만 쓴다 (2 · 3 은 다른 사람). 사용: bash scripts/apptainer/noeun_vllm_gemma.sh [gpu=0] [port=8100]
# 끝내기: pkill -f "served-model-name gemma-4-31B-it"
set -u
GPU=${1:-0}
PORT=${2:-8100}
case "$GPU" in 0|1) ;; *) echo "GPU 는 0 또는 1 만"; exit 2;; esac
SIF=/data/users/dnwn24/sif/vllm-openai-v0.24.0.sif
CK=/data/users/dnwn24/checkpoints/VLM/gemma-4-31B-it
LOG=/data/users/dnwn24/src/InternNav/logs/noeun/vllm_gemma_gpu${GPU}.log
mkdir -p "$(dirname "$LOG")" /data/users/dnwn24/apptainer_cache/vllm_home
echo "[vllm] GPU $GPU port $PORT log $LOG"
CUDA_VISIBLE_DEVICES=$GPU nohup apptainer exec --nv --cleanenv --env CUDA_VISIBLE_DEVICES=$GPU --env HOME=/vhome \
  -B "$CK":/model:ro -B /data/users/dnwn24/apptainer_cache/vllm_home:/vhome "$SIF" \
  vllm serve /model --served-model-name gemma-4-31B-it --port "$PORT" --tensor-parallel-size 1 --data-parallel-size 1 \
  --max-model-len 16384 --limit-mm-per-prompt '{"image": 16}' --gpu-memory-utilization 0.9 > "$LOG" 2>&1 &
echo "[vllm] pid $!"
