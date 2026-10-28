#!/bin/bash
# 단일 서버(vln_server) vs vln-deploy S1/S2 분리(HF backend, sync/async, MPS SM 상한) 동일성 + 속도 측정.
# gd_vln 컨테이너 안에서: scripts/sim/verify/run_dual_bench.sh [model_path]   결과: scripts/sim/verify/out/bench/
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$ROOT"
CK=${1:-/checkpoints/InternVLA-N1-DualVLN}
B="python3 scripts/sim/verify/bench_dual.py"
L=scripts/sim/verify/out/live; mkdir -p $L
mono() {  # monolithic HTTP server
  scripts/sim/run_dual.sh stop >/dev/null 2>&1
  nohup python3 scripts/sim/vln_server.py --model_path $CK --quant bf16 > $L/bench_mono_server.log 2>&1 &
  until curl -sf localhost:5801/health >/dev/null; do sleep 3; done
}
echo "=== [1/3] equivalence (back-to-back, period 0) ==="
mono;                                                  $B run --tag eq_mono --period 0
scripts/sim/run_dual.sh start $CK sync 0 0 >/dev/null; $B run --tag eq_split_sync --period 0
scripts/sim/run_dual.sh start $CK sync 70 30 >/dev/null; $B run --tag eq_split_sync_mps70_30 --period 0
echo "--- cmp mono vs split sync";            $B cmp scripts/sim/verify/out/bench/eq_mono.npy scripts/sim/verify/out/bench/eq_split_sync.npy
echo "--- cmp mono vs split sync MPS 70/30";  $B cmp scripts/sim/verify/out/bench/eq_mono.npy scripts/sim/verify/out/bench/eq_split_sync_mps70_30.npy
echo "=== [2/3] latency @ bridge rate (period 0.3 s) ==="
mono;                                                    $B run --tag lat_mono --period 0.3
scripts/sim/run_dual.sh start $CK sync 0 0 >/dev/null;   $B run --tag lat_split_sync --period 0.3
scripts/sim/run_dual.sh start $CK async 0 0 >/dev/null;  $B run --tag lat_split_async --period 0.3
scripts/sim/run_dual.sh start $CK async 70 30 >/dev/null; $B run --tag lat_split_async_mps70_30 --period 0.3
scripts/sim/run_dual.sh start $CK async 50 50 >/dev/null; $B run --tag lat_split_async_mps50_50 --period 0.3
scripts/sim/run_dual.sh start $CK async 80 20 >/dev/null; $B run --tag lat_split_async_mps80_20 --period 0.3
echo "=== [3/3] done ==="
