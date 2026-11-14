#!/bin/bash
# 노은역 랜드마크 경로 (74b + 74c) -> 31 -> 32 -> 41 -> 44 -> 42 걷기 (GPU 0·1) -> 고르기 -> 51 사진 (vln_pe · vln_ce). [호스트, 6000pro]
#
#   45 (계단 곧게 펴기)는 쓰지 않는다: 샘플 28 ep 에서 성공 경로 2 개를 넘어뜨리고 고친 것 0 (2026-11-14).
#   32 는 --pre_m/--post_m 1.5: 개찰구 출구에서 몸 뒤끝이 걸려 넘어지던 것 (샘플 ep 5) 해결.
#   고르기: split 마다 episode_id 순서로 앞에서부터 성공한 것 eval 80 · eval_seen 80 · train 1200.
#   사진: GPU 0 = vln_pe train · GPU 1 = vln_ce train -> eval · eval_seen 두 형식.
#
# 단계마다 $P/<단계>.log, 성공하면 $P/done_<단계>. 다시 돌리면 끝난 단계는 건너뛴다. 실패하면 멈춘다.
#   bash scripts/apptainer/noeun_landmark_pipeline.sh
#   REDO=walk bash scripts/apptainer/noeun_landmark_pipeline.sh
set -u
cd /data/users/dnwn24/src/InternNav
export SIF_DIR=/data/users/dnwn24/sif
D=logs/noeun/paths_landmark_ext
P=$D/pipeline; mkdir -p $P $D/walk
AX=scripts/apptainer/noeun_apptainer.sh
N=scripts/dataset_converters/nav3d/noeun
SB=/home/dnwn24/workspace/sandbox/internnav
G0=$(cat logs/env_cmp/isaac_walk_cmd.txt)        # GPU 0 · /ws/assets · /pylib 연결
G1=$(cat logs/env_cmp/isaac_walk_cmd_gpu1.txt)   # GPU 1
OUT=data/noeun_landmark_v1                        # 컨테이너 안 data = /home/dnwn24/workspace/data
ENVS='export PYTHONPATH=$PYTHONPATH:/pylib NOEUN_AGENT_SERVER=0 NOEUN_HEADLESS=1'
N_EVAL=${N_EVAL:-80}; N_SEEN=${N_SEEN:-80}; N_TRAIN=${N_TRAIN:-1200}

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $P/pipeline.log; }
[ -n "${REDO:-}" ] && rm -f "$P/done_${REDO}" && say "REDO ${REDO}"
run(){
  local n=$1; shift
  if [ -f "$P/done_$n" ]; then say "SKIP  $n"; return 0; fi
  say "START $n"
  "$@" >> "$P/$n.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { say "FAIL  $n rc=$rc — 중단. 로그: $P/$n.log"; exit $rc; }
  touch "$P/done_$n"; say "DONE  $n"
}

# 0) 74c 가 끝나야 시작
while ps -eo args | grep -q "[7]4c_extend_paths"; do sleep 60; done
[ -s $D/noeun_station_paths.json ] || { say "FAIL 74c 결과 없음"; exit 1; }

# 1) 31 · 32 · 41 · 44 (CPU)
run s31 bash -c "$AX habitat python3 $N/31_floor_to_mesh.py --paths_json $D/noeun_station_paths.json --out $D/paths_meshfloor.json"
run s32 bash -c "$AX habitat python3 $N/32_straighten_narrow.py --paths_json $D/paths_meshfloor.json --out $D/paths_meshfloor_straight.json --pre_m 1.5 --post_m 1.5"
run s41 bash -c "$AX habitat python3 $N/41_make_walk_inputs.py --paths_json $D/paths_meshfloor_straight.json --out $D/eps.json"
run s44 bash -c "$AX habitat python3 $N/44_shift_starts.py --paths_json $D/paths_meshfloor_straight.json --eps_json $D/eps.json --out_tag v5"
PJ=$D/paths_meshfloor_straight_v5.json; EJ=$D/eps_noeun_v5.json

# 2) 42 걷기 — 에피소드를 번갈아 GPU 0 / GPU 1 로 나눠 동시에 (각 worker 3 × threads 4)
run split python3 -c "
import json; d=json.load(open('$EJ'))
for g in (0, 1):
    json.dump(dict(d, episodes=d['episodes'][g::2]), open('$D/eps_g%d.json' % g, 'w'))
print(len(d['episodes']))"
walk(){
  local g=$1 C=$2
  bash -c "$C $SB bash -c '$ENVS; python3 $N/42_walk_rbq.py --episodes_json $D/eps_g$g.json --paths_json $PJ --log_dir $D/walk --tag g$g --workers 3 --timeout_s 60000'" >> $P/walk_g$g.log 2>&1
}
if [ ! -f $P/done_walk ]; then
  say "START walk (GPU 0 · 1)"
  walk 0 "$G0" & p0=$!; sleep 90; walk 1 "$G1" & p1=$!
  wait $p0; r0=$?; wait $p1; r1=$?
  [ $r0 -ne 0 -o $r1 -ne 0 ] && { say "FAIL  walk rc=$r0/$r1"; exit 1; }
  touch $P/done_walk; say "DONE  walk"
fi

# 3) 고르기: split 마다 episode_id 순서로 성공한 것 앞에서부터
run select python3 -c "
import json, collections
R = {}
for g in (0, 1):
    W = json.load(open('$D/walk/walk_g%d.json' % g)); bg = W.get('background')
    for r in W['results']: R[int(r['key'].split(':')[1])] = r
E = {e['episode_id']: e for e in json.load(open('$PJ'))['episodes']}
need = {'eval': $N_EVAL, 'eval_seen': $N_SEEN, 'train': $N_TRAIN}
sel, stat = collections.defaultdict(list), collections.defaultdict(collections.Counter)
for i in sorted(E):
    s = E[i]['split']; r = R.get(i)
    stat[s]['walked'] += r is not None; stat[s]['ok'] += bool(r and r['success'])
    if r and r['success'] and len(sel[s]) < need[s]: sel[s].append(r)
for s, rs in sel.items():
    json.dump({'background': bg, 'results': rs}, open('$D/walk/sel_%s.json' % s, 'w'))
    print(s, len(rs), '/', need[s], dict(stat[s]))
assert all(len(sel[s]) == need[s] for s in need), '성공 경로가 모자란다'
"

# 4) 51 사진 — GPU 0: vln_pe train · GPU 1: vln_ce train -> eval · eval_seen (두 형식)
render(){   # render <C> <split> <fmt>
  local C=$1 s=$2 f=$3 tag=${3}_${2}
  [ -f $P/done_r_$tag ] && { say "SKIP  r_$tag"; return 0; }
  say "START r_$tag"
  bash -c "$C $SB bash -c '$ENVS; /workspace/isaaclab/_isaac_sim/python.sh $N/51_render_rbq.py --walk_json $D/walk/sel_$s.json --format $f --out_root $OUT/${f}_$s --log_dir $D/render_log/$tag'" >> $P/r_$tag.log 2>&1 \
    && grep -q "렌더 .* ep" $P/r_$tag.log && touch $P/done_r_$tag && say "DONE  r_$tag" || { say "FAIL  r_$tag"; return 1; }
}
( render "$G0" train vlnpe ) & q0=$!
( render "$G1" train vlnce && render "$G1" eval vlnpe && render "$G1" eval vlnce \
  && render "$G1" eval_seen vlnpe && render "$G1" eval_seen vlnce ) & q1=$!
wait $q0; a0=$?; wait $q1; a1=$?
[ $a0 -ne 0 -o $a1 -ne 0 ] && { say "FAIL  render $a0/$a1"; exit 1; }
say "=== 끝: $OUT/{vlnpe,vlnce}_{train,eval,eval_seen} ==="
