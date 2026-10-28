#!/bin/bash
# 여러 (모델, look-down on/off) 조합으로 run_e2e.py 를 순서대로 돌린다 (gd_vln 컨테이너 안, sim+gdm 실행 중).
# 사용: scripts/sim/verify/run_matrix.sh <mode> <episodes.json> <tag_suffix> "<model>:<ld on|off>" ...
#   예: scripts/sim/verify/run_matrix.sh pixel_goal scripts/sim/verify/e2e_episodes_etri_indoor_clear.json clear dualvln:off dualvln:on navdp:on wodagger:on
# 모델이 바뀔 때만 서버(vln_server.py, bf16)를 재시작한다. 결과: out/e2e/<model>_bf16_<ld>_<suffix>/<mode>/
MODE=$1; EPS=$2; SUF=$3; shift 3
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$ROOT"
source scripts/sim/setup_ros_ws.sh >/dev/null
declare -A CK=([dualvln]=/checkpoints/InternVLA-N1-DualVLN [navdp]=/checkpoints/InternVLA-N1-w-NavDP [wodagger]=/checkpoints/InternVLA-N1-wo-dagger)
cur=$(curl -s -m 3 localhost:5801/health | python3 -c "import sys,json;print(json.load(sys.stdin)['model'])" 2>/dev/null)
for item in "$@"; do
  M=${item%%:*}; LD=${item##*:}; CKPT=${CK[$M]}
  if [[ "$cur" != "$CKPT" ]]; then
    pkill -f "^python3 scripts/sim/vln_server.py"; sleep 5
    nohup python3 scripts/sim/vln_server.py --model_path $CKPT --quant bf16 > scripts/sim/verify/out/live/server_$M.log 2>&1 &
    until curl -sf localhost:5801/health >/dev/null; do sleep 5; pgrep -f "^python3 scripts/sim/vln_server.py" >/dev/null || { echo "[matrix] server $M died"; tail -5 scripts/sim/verify/out/live/server_$M.log; continue 2; }; done
    cur=$CKPT
  fi
  BA=""; [[ "$LD" == on ]] && BA="lookdown_ns:=/camera_lookdown/camera"
  echo "[matrix] === $M lookdown=$LD mode=$MODE"
  python3 scripts/sim/verify/run_e2e.py --mode $MODE --tag ${M}_bf16_ld${LD}_${SUF} --episodes $EPS --bridge_args "$BA"
done
echo "[matrix] done"
