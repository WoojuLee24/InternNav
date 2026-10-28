#!/bin/bash
# 도구 3: dataset 정답 대조. 배포 스택(S1/S2 module, run_vln_stack.sh로 실행 중)에 sync dual_server(:5894)를 하나 더 붙여
# VLN-CE 에피소드를 재생하고 dataset_check.py로 학습 GT와 비교한다. 실행 중에는 bridge(주행)를 돌리지 않는다(S2 module 공유).
# S1/S2 module은 ZMQ PUSH/PULL이라 dual_server가 둘 붙으면 응답이 나뉜다 -> :5801 dual_server를 먼저 내리고, 끝나면 다시 띄운다.
# 사용(gd_vln 컨테이너 안): scripts/sim/verify/dataset_test.sh [episodes] [tag]
EPS=${1:-30}
TAG=${2:-q8trt}
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$ROOT"
OUT=scripts/sim/verify/out/dataset_${TAG}_$(date +%y%m%d_%H%M%S); mkdir -p $OUT
export PYTHONDONTWRITEBYTECODE=1
PORT=5894
if pgrep -f "^python3 scripts/sim/dual_server.py" > /dev/null; then
  echo "[dataset_test] 다른 dual_server가 실행 중입니다(S1/S2 응답이 나뉨). 먼저 종료하세요: pkill -f '^python3 scripts/sim/dual_server.py'" >&2; exit 1
fi
python3 scripts/sim/dual_server.py --mode sync --port $PORT --model_name dataset_check > $OUT/dual_sync.log 2>&1 &
SRV=$!
for _ in $(seq 60); do curl -s localhost:$PORT/health > /dev/null && break; sleep 1; done
python3 scripts/sim/verify/dataset_check.py --server http://127.0.0.1:$PORT --episodes $EPS --out $OUT 2>&1 | tee $OUT/check.log | grep -v "^\[dataset_check\] .*frames, S2"
kill $SRV; wait $SRV 2>/dev/null
echo "[dataset_test] out: $OUT"
