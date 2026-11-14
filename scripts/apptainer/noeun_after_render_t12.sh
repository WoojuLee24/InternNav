#!/bin/bash
# 노은역 학습 vln_ce 사진 (12° 걷기) 4 조각이 다 끝나면: 조립 -> 검사 -> tar -> Gdrive 업로드. [호스트, 6000pro]
#   H200 에서 instruction labeling 에 쓴다.
#   bash scripts/apptainer/noeun_after_render_t12.sh   (nohup 으로 띄워 두면 조각이 끝날 때까지 기다린다)
set -u
cd /data/users/dnwn24/src/InternNav
export SIF_DIR=/data/users/dnwn24/sif
D=logs/noeun/paths_landmark_ext; W=$D/walk_t12; L=$D/render_log/vlnce_train_t12
OUT=/home/dnwn24/workspace/data/noeun_landmark_v1/vlnce_train_t12
NAME=vln_ce_noeun_landmark_t12_train
TAR=/home/dnwn24/workspace/data/noeun_landmark_v1/$NAME.tar
R=gdrive:InternNav/data
LOG=$L/after_render.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

# 1) 4 조각 모두 rc=0 으로 끝날 때까지
for s in 0-398 398-757 757-963 963-1200; do
  until grep -q "shard $s rc=" $L/done.txt 2>/dev/null; do sleep 60; done
  grep -q "shard $s rc=0" $L/done.txt || { say "FAIL 조각 $s rc!=0"; exit 1; }
done
say "4 조각 끝"
for s in 0_398 398_757 757_963 963_1200; do
  grep -q "^\[51b\] 렌더" $L/shard_$s.log || { say "FAIL shard_$s 끝줄 없음"; exit 1; }
  say "$(grep '^\[51b\] 렌더' $L/shard_$s.log | tail -1)"
done

# 2) 조립 (parquet · meta) — 모든 ep 의 pose 가 있어야 된다
scripts/apptainer/noeun_apptainer.sh habitat python3 scripts/dataset_converters/nav3d/noeun/51b_render_rbq_shard.py --walk_json $W/sel_train.json --out_root data/noeun_landmark_v1/vlnce_train_t12 --log_dir $L --assemble >> $LOG 2>&1 || { say "FAIL 조립"; exit 1; }
say "조립 끝"

# 3) 검사: 사진 수 = 159,022 (rgb · depth), parquet 1,200
V=$OUT/traj_data/r2r/noeunstation
nr=$(ls $V/videos/chunk-000/observation.images.rgb.60cm_15deg | wc -l); nd=$(ls $V/videos/chunk-000/observation.images.depth.60cm_15deg | wc -l)
np_=$(find $V/data -name '*.parquet' | wc -l)
exp=$(python3 -c "import json;print(sum(len(r['frames']) for r in json.load(open('$W/sel_train.json'))['results']))")
say "검사 rgb $nr depth $nd (기대 $exp) · parquet $np_ (기대 1200)"
[ "$nr" = "$exp" ] && [ "$nd" = "$exp" ] && [ "$np_" = 1200 ] || { say "FAIL 수가 안 맞음 — 업로드 안 함"; exit 1; }

# 4) 옆 파일 (경로·고르기 정보) 를 데이터 안에 같이 넣는다
mkdir -p $OUT/noeun_meta
cp $W/sel_train.json $W/stats.json $D/paths_meshfloor_straight_v5.json $OUT/noeun_meta/
cp $L/vlnce/render_meta.json $OUT/noeun_meta/ 2>/dev/null

# 5) tar (jpg · png 는 이미 압축됨 -> 압축 안 함, CPU 아낌) + sha256 -> Gdrive
nice -n 10 tar cf $TAR -C /home/dnwn24/workspace/data/noeun_landmark_v1 vlnce_train_t12 || { say "FAIL tar"; exit 1; }
(cd $(dirname $TAR) && sha256sum $(basename $TAR) > $(basename $TAR).sha256)
say "tar $(du -h $TAR | cut -f1)"
rclone copy $TAR $R/ --transfers 1 --drive-chunk-size 256M >> $LOG 2>&1 && rclone copy $TAR.sha256 $R/ >> $LOG 2>&1 || { say "FAIL 업로드"; exit 1; }
rs=$(rclone lsl $R/$NAME.tar | awk '{print $1}'); ls_=$(stat -c %s $TAR)
[ "$rs" = "$ls_" ] && say "업로드 끝 $R/$NAME.tar ($rs bytes · 크기 같음)" || { say "FAIL 업로드 크기 다름 $rs vs $ls_"; exit 1; }
echo DONE >> $LOG
