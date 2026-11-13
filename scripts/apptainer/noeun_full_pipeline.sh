#!/bin/bash
# 노은역 데이터 전체 파이프라인 — 22 경로 선별부터 학습·평가까지. [호스트, 순차]
#
#   22 select_paths (3600)
#     -> [새] 고속 렌더 -> stop_cue -> 필터   (쓸모없는 목적지를 **걷기·렌더 전에** 버린다 = 시간 절약)
#     -> 31 floor_to_mesh -> 32 straighten_narrow -> 41 make_walk_inputs -> 42 walk_rbq
#        -> 43 replace_fallen -> 44 shift_starts -> 45 straighten_stairs
#     -> 51 render_rbq
#     -> 61 build_zones -> 62 label(Gemma-4-31B + stop_cue) -> 63 build_labeled -> 64 build_train
#     -> 평가 80 / 학습 1200 분리 -> 학습 2 셀 (400 · 1200) + 평가 80
#
# 단계마다 logs/pipeline/<단계>.log 를 남기고 성공하면 done_<단계> 를 찍는다 -> 다시 돌리면 끝난 단계는 건너뛴다.
# 실패하면 **그 자리에서 멈춘다** (뒤 단계가 깨진 입력을 먹지 않게).
#
#   bash scripts/apptainer/noeun_full_pipeline.sh
#   REDO=render bash scripts/apptainer/noeun_full_pipeline.sh   # 그 단계 done 을 지우고 다시
set -u
cd /data/users/dnwn24/src/InternNav
P=logs/pipeline; mkdir -p $P
AX=scripts/apptainer/noeun_apptainer.sh
D4=scripts/dataset_converters/nav3d/logs/noeun/paths3600
N=scripts/dataset_converters/nav3d/noeun
MESH=data/scene_data/noeun_station/noeun_station_mid.ply
ISAAC_PRE="cd /ws/src/InternNav && export PYTHONPATH=/ws/src/InternNav CUDA_VISIBLE_DEVICES=0 ISAAC_ACTIVE_GPU=0 NOEUN_AGENT_SERVER=0 NOEUN_HEADLESS=1"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $P/pipeline.log; }
[ -n "${REDO:-}" ] && rm -f "$P/done_${REDO}" && say "REDO ${REDO}"

run(){  # run <이름> <명령...>
  local n=$1; shift
  if [ -f "$P/done_$n" ]; then say "SKIP  $n"; return 0; fi
  say "START $n"
  "$@" >> "$P/$n.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { say "FAIL  $n rc=$rc — 중단. 로그: $P/$n.log"; exit $rc; }
  touch "$P/done_$n"; say "DONE  $n"
}

# ── 1) 22 경로 선별 (이미 돌고 있으면 기다린다) ─────────────────────────────
if [ ! -f "$P/done_select" ]; then
  say "START select"
  while pgrep -f 22_select_paths >/dev/null; do sleep 30; done
  if ls $D4/*paths*.json >/dev/null 2>&1; then touch "$P/done_select"; say "DONE  select"
  else say "FAIL  select — $D4 에 결과 없음"; exit 1; fi
fi
PJ=$(ls $D4/*paths*.json 2>/dev/null | head -1)
say "경로 json: $PJ"

# ── 2) [새] 고속 렌더 -> stop_cue -> 필터 ──────────────────────────────────
# 걷기·렌더 전에 "멈출 대상이 2 m 안에 있는 목적지" 만 남긴다. 3600 -> 약 1700 예상.
run goalsnap bash -c "$AX isaac bash -c '$ISAAC_PRE && /workspace/isaaclab/_isaac_sim/python.sh scripts/eval/noeun_goal_snap_fast.py --out /ws/src/InternNav/logs/pipeline/goal_snap_all'"
run stopcue  bash -c "$AX habitat python3 $N/68_stop_cues.py --snaps logs/pipeline/goal_snap_all --val $PJ --out logs/pipeline/stop_cues"

# ── 3) 31~45 보정 + RBQ 걷기 ───────────────────────────────────────────────
run s31 bash -c "$AX habitat python3 $N/31_floor_to_mesh.py --paths_json $PJ --mesh $MESH --out $D4/paths_meshfloor.json"
run s32 bash -c "$AX habitat python3 $N/32_straighten_narrow.py --paths_json $D4/paths_meshfloor.json --mesh $MESH --out $D4/paths_meshfloor_straight.json"
run s41 bash -c "$AX habitat python3 $N/41_make_walk_inputs.py --paths_json $D4/paths_meshfloor_straight.json --out $D4/eps_noeun.json"
run s42 bash -c "$AX isaac bash -c '$ISAAC_PRE && /workspace/isaaclab/_isaac_sim/python.sh $N/42_walk_rbq.py --episodes_json /ws/src/InternNav/$D4/eps_noeun.json --tag noeun_3600 --workers 2'"

say "=== 여기까지 자동. 43~45 · 51 · 61~64 는 42 결과를 보고 이어서 ==="
