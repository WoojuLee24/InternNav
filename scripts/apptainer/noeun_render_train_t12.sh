#!/bin/bash
# 노은역 학습용 vln_ce 사진 — 회전 기준 12° 다시 걷기 결과 (2026-11-14). 51b_render_rbq_shard.py 로 ep 범위만 찍는다. [호스트, 6000pro]
#   (21:35) A5000 렌더는 RGB 차이 1.9/255 이상이라 안 쓴다 -> 6000pro 만: GPU0 0-398 → 757-963 · GPU1 398-757 → 963-1200 (이어서 자동)
#   나머지 1082-1200 은 뒤쪽 걷기를 합친 sel_train.json 으로 찍는다 (WJ=... 로 바꿔서).
#   bash scripts/apptainer/noeun_render_train_t12.sh <gpu 0|1> <ep_start> <ep_end>
#   조립: scripts/apptainer/noeun_apptainer.sh habitat python3 scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json logs/noeun/paths_landmark_ext/walk_t12/sel_train.json --out_root data/noeun_landmark_v1/vlnce_train_t12 --log_dir logs/noeun/paths_landmark_ext/render_log/vlnce_train_t12 --assemble
set -u
cd /data/users/dnwn24/src/InternNav
g=$1; s=$2; e=$3
WJ=${WJ:-logs/noeun/paths_landmark_ext/walk_t12/sel_train.json}
C=$(cat logs/env_cmp/isaac_walk_cmd$([ "$g" = 1 ] && echo _gpu1).txt)
SB=/home/dnwn24/workspace/sandbox/internnav
L=logs/noeun/paths_landmark_ext/render_log/vlnce_train_t12; mkdir -p $L
bash -c "$C $SB bash -c 'export PYTHONPATH=\$PYTHONPATH:/pylib NOEUN_AGENT_SERVER=0 NOEUN_HEADLESS=1; /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json $WJ --out_root data/noeun_landmark_v1/vlnce_train_t12 --log_dir $L --ep_start $s --ep_end $e --skip_existing'" >> $L/shard_${s}_${e}.log 2>&1
echo "shard $s-$e rc=$? $(date)" >> $L/done.txt
