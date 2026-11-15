#!/bin/bash
# 노은역 랜드마크 평가 A · B 를 한 번에: 사진 (GPU 1) → 조립 → 검사 → 라벨 4 종 (GPU 0 Gemma :8100) → 평가 데이터 8 개 → 로더 검사. [호스트, 6000pro]
#   nohup bash scripts/apptainer/noeun_build_eval_t12.sh > /dev/null 2>&1 &
#   로그 logs/noeun/paths_landmark_ext/render_log/build_eval_t12.log (끝나면 DONE)
#   env SPLITS (eval eval_seen) · GPU (1) · BASE_URL (http://127.0.0.1:8100/v1) · MINI (data/noeun_landmark_v1/mini_eval)
set -u
cd /data/users/dnwn24/src/InternNav
export SIF_DIR=/data/users/dnwn24/sif
SPLITS=${SPLITS:-"eval eval_seen"}; GPU=${GPU:-1}; BASE_URL=${BASE_URL:-http://127.0.0.1:8100/v1}
D=logs/noeun/paths_landmark_ext; W=$D/walk_t12
HD=/home/dnwn24/workspace/data/noeun_landmark_v1          # 호스트 경로 (= 컨테이너 data/noeun_landmark_v1)
MINI=${MINI:-$HD/mini_eval}
LOG=$D/render_log/build_eval_t12.log; mkdir -p $D/render_log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

for sp in $SPLITS; do
  ON=vlnce_${sp}_t12; L=$D/render_log/$ON; OUT=$HD/$ON
  # 1) 사진
  say "$sp 사진 시작 (GPU $GPU)"
  bash scripts/apptainer/noeun_render_eval_t12.sh $GPU $sp 0 0
  grep -q "shard 0-0 rc=0" $L/done.txt || { say "FAIL $sp 렌더 rc"; exit 1; }
  say "$(grep -a '^\[51b\] 렌더' $L/shard_0_0.log | tail -1)"
  # 2) 조립
  scripts/apptainer/noeun_apptainer.sh habitat python3 scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json $W/sel_${sp}.json --out_root data/noeun_landmark_v1/$ON --log_dir $L --assemble >> $LOG 2>&1 || { say "FAIL $sp 조립"; exit 1; }
  # 3) 검사: 사진 수 = sel 의 성공 ep 프레임 합
  V=$OUT/traj_data/r2r/noeunstation
  nr=$(ls $V/videos/chunk-000/observation.images.rgb.60cm_15deg | wc -l); nd=$(ls $V/videos/chunk-000/observation.images.depth.60cm_15deg | wc -l)
  np_=$(find $V/data -name '*.parquet' | wc -l)
  read exp ne < <(python3 -c "import json;r=[x for x in json.load(open('$W/sel_${sp}.json'))['results'] if x['success']];print(sum(len(x['frames']) for x in r), len(r))")
  say "$sp 검사 rgb $nr depth $nd (기대 $exp) · parquet $np_ (기대 $ne)"
  [ "$nr" = "$exp" ] && [ "$nd" = "$exp" ] && [ "$np_" = "$ne" ] || { say "FAIL $sp 수가 안 맞음"; exit 1; }
  mkdir -p $OUT/noeun_meta
  cp $W/sel_${sp}.json $W/stats.json $D/paths_meshfloor_straight_v5.json $L/vlnce/render_meta.json $OUT/noeun_meta/
done

# 4) 라벨 4 종 x split (학습과 같은 설정 → 같은 gen_hash)
for sp in $SPLITS; do
  for m in b1 post; do for sc in none noeun-paths-v1; do
    scripts/apptainer/noeun_apptainer.sh habitat /usr/bin/python scripts/dataset_converters/nav3d/noeun/62b_label_landmark.py --data_root $HD/vlnce_${sp}_t12 --split $sp --methods $m --base_url $BASE_URL --workers 8 --landmark_nouns noeun-v1 --rooms noeun-v2 --stop_cue $sc >> $LOG.label 2>&1 || { say "FAIL 라벨 $sp $m $sc"; exit 1; }
    say "$(grep -a '오류 수' $LOG.label | tail -1) ($sp $m $sc)"
  done; done
done

# 5) 평가 데이터 8 개 (64b 가 gen_hash 확인) + 로더 검사
for sp in $SPLITS; do
  python3 scripts/dataset_converters/nav3d/noeun/64b_build_train_landmark.py --split $sp --src $HD/vlnce_${sp}_t12 --labels_root $HD/instruction_labels/nouns-noeun-v1 --mini $MINI >> $LOG 2>&1 || { say "FAIL 64b $sp"; exit 1; }
done
for d in $MINI/vln_ce_noeun-lmeval*; do
  r=$(scripts/apptainer/noeun_apptainer.sh habitat /usr/bin/python scripts/dataset_converters/nav3d/checks/loader_check.py --vlnce "$(basename $d)=$d/traj_data/r2r=60,15,15" 2>&1 | grep -a "\[08 vln_ce\]")
  say "$r"
done
echo DONE >> $LOG
