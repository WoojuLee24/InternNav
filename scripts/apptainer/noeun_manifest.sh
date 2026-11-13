#!/bin/bash
# 노은역 ABC·DAgger 준비물 지문 — 두 서버(A5000 / 6000 Pro)에서 각각 돌려 diff 한다.
#
#   bash scripts/apptainer/noeun_manifest.sh > manifest_<서버이름>.txt
#   diff manifest_a5000.txt manifest_6000pro.txt
#
# 경로가 서버마다 다르면 환경변수로 준다:
#   NAV=... UTOPIA=... DATA=... CKPT=... bash scripts/apptainer/noeun_manifest.sh
#
# 큰 파일은 md5 가 느리므로 크기만 본다 (safetensors 등). 작은 파일은 md5 까지 본다.
set -u

NAV="${NAV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
UTOPIA="${UTOPIA:-$(dirname "$NAV")/InternUtopia}"
DATA="${DATA:-$NAV/data}"
CKPT="${CKPT:-$NAV/checkpoints}"
CK="relabel_half_noeun-v01/s1.noeun-both-b1_wait-wait_stopr1.0_internvla-n1-system2_20261110_112953"

echo "=== CODE ==="
for r in "$NAV" "$UTOPIA"; do
    if [ -d "$r/.git" ]; then
        cd "$r"
        echo "$(basename "$r") branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse HEAD)"
        echo "$(basename "$r") dirty_tracked=$(git status --porcelain --untracked-files=no | wc -l)"
    else
        echo "$(basename "$r") MISSING"
    fi
done

echo "=== CODE FILES (노은역 관련 스크립트 md5) ==="
cd "$NAV" 2>/dev/null && \
find scripts/dataset_converters/nav3d/noeun scripts/eval/configs/noeun -type f -name '*.py' 2>/dev/null | sort | \
    xargs md5sum 2>/dev/null
for f in scripts/eval/noeun_navmesh_server.py scripts/eval/start_server.py scripts/eval/noeun_vlnpe_snap.py \
         internnav/agent/internvla_n1_s2_agent.py internnav/agent/navmesh_follower.py; do
    [ -f "$NAV/$f" ] && md5sum "$NAV/$f" | sed "s# $NAV/# #"
done

echo "=== DATA (크기 + 작은 파일 md5) ==="
for p in "GS_USDZ/noeun_station_collision.usdz" \
         "data/vln_pe_noeun_nav3d_rbq_v6/raw_data/r2r/train/train.json.gz" \
         "data/vln_pe_noeun_nav3d_rbq_hunav_v1/raw_data/r2r/train/train.json.gz" \
         "data/noeun_eval/labels/v6_train/v6_train.json.gz" \
         "data/noeun_eval/labels/hunav_v1_train/hunav_v1_train.json.gz" \
         "data/noeun_eval/navmesh/noeun_station/noeun_station.navmesh"; do
    f="$DATA/$p"
    if [ -f "$f" ]; then
        sz=$(stat -c %s "$f")
        if [ "$sz" -lt 104857600 ]; then
            echo "$p size=$sz md5=$(md5sum "$f" | cut -d' ' -f1)"
        else
            echo "$p size=$sz md5=(skip,>100MB)"
        fi
    else
        echo "$p MISSING"
    fi
done
echo "labels/v6_train    files=$(find "$DATA/data/noeun_eval/labels/v6_train" -type f 2>/dev/null | wc -l)"
echo "labels/hunav_train files=$(find "$DATA/data/noeun_eval/labels/hunav_v1_train" -type f 2>/dev/null | wc -l)"
echo "navmesh dir        files=$(find "$DATA/data/noeun_eval/navmesh/noeun_station" -type f 2>/dev/null | wc -l)"

echo "=== CHECKPOINT ==="
d="$CKPT/$CK"
if [ -d "$d" ]; then
    echo "files=$(find "$d" -type f | wc -l) bytes=$(du -sb "$d" | cut -f1)"
    for f in config.json generation_config.json model.safetensors.index.json trainer_state.json; do
        [ -f "$d/$f" ] && echo "$f md5=$(md5sum "$d/$f" | cut -d' ' -f1)"
    done
    (cd "$d" && for f in model-0000*.safetensors; do echo "$f size=$(stat -c %s "$f")"; done)
else
    echo "MISSING $d"
fi
