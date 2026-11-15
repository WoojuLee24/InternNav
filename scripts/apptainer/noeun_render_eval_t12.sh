#!/bin/bash
# 노은역 랜드마크 평가 A · B 의 vln_ce 사진 (회전 기준 12° 걷기 결과, 2026-11-15). 학습과 같은 51b_render_rbq_shard.py · 같은 설정. [호스트, 6000pro]
#   평가 A = sel_eval.json (안 본 랜드마크 80) · 평가 B = sel_eval_seen.json (본 랜드마크 · 새 출발점 80)
#   bash scripts/apptainer/noeun_render_eval_t12.sh <gpu 0|1> <eval|eval_seen> [ep_start] [ep_end]
#   env OUT_NAME (기본 vlnce_<split>_t12 → data/noeun_landmark_v1/<OUT_NAME>)
#   조립: scripts/apptainer/noeun_apptainer.sh habitat python3 scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json logs/noeun/paths_landmark_ext/walk_t12/sel_<split>.json --out_root data/noeun_landmark_v1/<OUT_NAME> --log_dir logs/noeun/paths_landmark_ext/render_log/<OUT_NAME> --assemble
set -u
cd /data/users/dnwn24/src/InternNav
g=$1; sp=$2; s=${3:-0}; e=${4:-0}
case $sp in eval|eval_seen) ;; *) echo "split=$sp (eval|eval_seen)"; exit 1;; esac
WJ=logs/noeun/paths_landmark_ext/walk_t12/sel_${sp}.json
ON=${OUT_NAME:-vlnce_${sp}_t12}
C=$(cat logs/env_cmp/isaac_walk_cmd$([ "$g" = 1 ] && echo _gpu1).txt)
SB=/home/dnwn24/workspace/sandbox/internnav
L=logs/noeun/paths_landmark_ext/render_log/$ON; mkdir -p $L
bash -c "$C $SB bash -c 'export PYTHONPATH=\$PYTHONPATH:/pylib NOEUN_AGENT_SERVER=0 NOEUN_HEADLESS=1; /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json $WJ --out_root data/noeun_landmark_v1/$ON --log_dir $L --ep_start $s --ep_end $e --skip_existing'" >> $L/shard_${s}_${e}.log 2>&1
echo "shard $s-$e rc=$? $(date)" >> $L/done.txt
