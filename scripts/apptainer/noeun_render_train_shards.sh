#!/bin/bash
# 노은역 학습용 vln_ce 사진을 조각(ep 범위)으로 나눠 찍는다 — 51b_render_rbq_shard.py. [호스트, 6000pro]
#   조각 (sel_train 성공 ep 1200, 프레임 수로 나눔. a5000 = 6000pro 의 0.3 배 속도라 GPU 하나당 0.3 몫):
#     6000pro GPU0 0-395 · GPU1 395-753 · a5000 GPU0 753-856 · GPU1 856-960 · GPU2 960-1086 · GPU3 1086-1200
#   bash scripts/apptainer/noeun_render_train_shards.sh <gpu 0|1> <ep_start> <ep_end>
#   다 찍은 뒤 (a5000 사진 · part npz 를 같은 폴더로 받은 다음) 조립:
#   scripts/apptainer/noeun_apptainer.sh habitat python3 scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json logs/noeun/paths_landmark_ext/walk/sel_train.json --out_root data/noeun_landmark_v1/vlnce_train --log_dir logs/noeun/paths_landmark_ext/render_log/vlnce_train_shard --assemble
set -u
cd /data/users/dnwn24/src/InternNav
export SIF_DIR=/data/users/dnwn24/sif
g=$1; s=$2; e=$3
C=$(cat logs/env_cmp/isaac_walk_cmd$([ "$g" = 1 ] && echo _gpu1).txt)
SB=/home/dnwn24/workspace/sandbox/internnav
L=logs/noeun/paths_landmark_ext/render_log/vlnce_train_shard; mkdir -p $L
bash -c "$C $SB bash -c 'export PYTHONPATH=\$PYTHONPATH:/pylib NOEUN_AGENT_SERVER=0 NOEUN_HEADLESS=1; /workspace/isaaclab/_isaac_sim/python.sh scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json logs/noeun/paths_landmark_ext/walk/sel_train.json --out_root data/noeun_landmark_v1/vlnce_train --log_dir $L --ep_start $s --ep_end $e --skip_existing'" >> $L/shard_${s}_${e}.log 2>&1
