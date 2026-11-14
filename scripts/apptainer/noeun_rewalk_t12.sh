#!/bin/bash
# 노은역 1,752 ep 다시 걷기 (2026-11-14) — 좁은 통로 보정 + 회전 기준 12° (왔다갔다 41.6% -> 13%, 48 ep 시험)
#   6000pro: 앞쪽 1,402 ep (eps_front_g{0,1}.json) 를 GPU 0·1 에 worker 4 개씩.  A5000: 뒤쪽 350 ep (eps_tail.json).
#   결과: logs/noeun/paths_landmark_ext/walk_t12/walk_f{0,1}.json   (A5000 은 walk_tail.json 으로 받아 같은 폴더에 둔다)
#   bash scripts/apptainer/noeun_rewalk_t12.sh [front|tail]   (tail = A5000 이 못 하면 6000pro 가 뒤쪽을 걷는다)
set -u
cd /data/users/dnwn24/src/InternNav
G0=$(cat logs/env_cmp/isaac_walk_cmd.txt); G1=$(cat logs/env_cmp/isaac_walk_cmd_gpu1.txt); SB=/home/dnwn24/workspace/sandbox/internnav
D=logs/noeun/paths_landmark_ext; W=$D/walk_t12; PJ=$D/paths_meshfloor_straight_v5.json
ENVS='export PYTHONPATH=$PYTHONPATH:/pylib NOEUN_AGENT_SERVER=0 NOEUN_HEADLESS=1'
OPT='--narrow_only --steer_gain 1.5 --turn_threshold_deg 12'
w(){ bash -c "$1 $SB bash -c '$ENVS; python3 scripts/dataset_converters/nav3d/noeun/42_walk_rbq.py --episodes_json $W/$2.json --paths_json $PJ --log_dir $W --tag $3 --workers $4 --timeout_s 60000 -- $OPT'" > $W/$3.log 2>&1; }
case ${1:-front} in
  front) w "$G0" eps_front_g0 f0 4 & sleep 150; w "$G1" eps_front_g1 f1 4 & wait; echo "front $(date)" >> $W/done.txt ;;
  tail)  w "$G0" eps_tail tail 4 & wait; echo "tail $(date)" >> $W/done.txt ;;
esac
