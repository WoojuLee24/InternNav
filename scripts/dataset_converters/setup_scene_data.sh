#!/bin/bash
# mini habitat 평가용 씬(mp3d_ce)을 노드 로컬 data-vol1 에 올바른 위치로 풀고 검증한다.
#
# 왜 필요한가
#   `data` 는 노드마다 로컬 디스크(/home/irteam/data-vol1)라 노드마다 내용이 다르다.
#   mini 평가 yaml 은 scenes_dir=data/InternData-N1-v0.5-mini/scene_data/mp3d_ce 이고
#   scene_id 가 "mp3d/<scene>/<scene>.glb" 라서 실제 경로는 scene_data/mp3d_ce/mp3d/<scene>/ 이다.
#   그런데 mp3d_ce.tar.gz 의 최상위 항목은 "mp3d/" 라서 extract_dataset.sh 처럼 tar 옆에 풀면
#   scene_data/mp3d/ 가 되어 경로가 한 단계 어긋난다. (2026-10-01 node3, 그 전 node4 평가 전멸 원인:
#   ESP_CHECK failed: No Stage Attributes exists for requested scene '.../mp3d_ce/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb')
#
# 하는 일
#   1. 검증: r2r raw_data(val_seen/val_unseen) 가 참조하는 모든 scene 의 .glb/.navmesh 가 있는지
#   2. 빠져 있으면 mp3d_ce.tar.gz 를 mp3d_ce/ 안에 푼다 (로컬 tar 우선, 없으면 data-vol2 의 tar).
#      임시 디렉토리에 푼 뒤 mv 하므로 중간에 끊겨도 반쯤 풀린 mp3d_ce/mp3d 가 남지 않는다.
#   3. 다시 검증. 하나라도 빠지면 exit 1.
#
# Usage (한 줄):
#   bash scripts/dataset_converters/setup_scene_data.sh              # 검증 + 필요하면 풀기
#   bash scripts/dataset_converters/setup_scene_data.sh --check-only # 검증만
#   bash scripts/dataset_converters/setup_scene_data.sh --root <mini root> --src <tar 가 있는 shared scene_data>

set -o pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$REPO/data/InternData-N1-v0.5-mini"
SRC="/home/irteam/data-vol2/InternData-N1-v0.5-mini/scene_data"
CHECK_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        --src) SRC="$2"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        *) echo "알 수 없는 인자: $1" >&2; exit 2 ;;
    esac
done

SCENE_DATA="$ROOT/scene_data"
SCENES_DIR="$SCENE_DATA/mp3d_ce"          # 평가 yaml 의 scenes_dir
TARGET="$SCENES_DIR/mp3d"                  # tar 의 최상위 mp3d/ 가 놓일 곳
RAW="$ROOT/vln_ce/raw_data/r2r"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 평가 split 이 참조하는 scene 이 전부 있는지. 빠진 개수를 출력하고 그 수를 exit code 로 (0=OK).
verify() {
    /usr/bin/python3 - "$SCENES_DIR" "$RAW" <<'EOF'
import gzip, json, os, sys
scenes_dir, raw = sys.argv[1], sys.argv[2]
need = set()
for split in ("val_seen", "val_unseen"):
    p = os.path.join(raw, split, f"{split}.json.gz")
    if not os.path.isfile(p):
        print(f"  [x] {p} 없음"); sys.exit(1)
    need |= {e["scene_id"] for e in json.load(gzip.open(p))["episodes"]}
missing = []
for sid in sorted(need):                       # sid = "mp3d/<scene>/<scene>.glb"
    glb = os.path.join(scenes_dir, sid)
    nav = os.path.splitext(glb)[0] + ".navmesh"
    if not (os.path.isfile(glb) and os.path.getsize(glb) > 0 and os.path.isfile(nav)):
        missing.append(sid)
print(f"  평가 split 이 쓰는 scene {len(need)}개 중 빠진 것 {len(missing)}개 (scenes_dir={scenes_dir})")
for m in missing[:5]:
    print(f"    - {m}")
sys.exit(min(len(missing), 1))
EOF
}

log "검증: $SCENES_DIR"
if verify; then
    log "OK — 풀 필요 없음"
    exit 0
fi
if [ "$CHECK_ONLY" -eq 1 ]; then
    log "FAIL (--check-only 라 풀지 않음)"
    exit 1
fi

TAR="$SCENE_DATA/mp3d_ce.tar.gz"
[ -f "$TAR" ] || TAR="$SRC/mp3d_ce.tar.gz"
[ -f "$TAR" ] || { log "mp3d_ce.tar.gz 를 찾지 못함 ($SCENE_DATA, $SRC)"; exit 1; }
TOP=$(tar -tzf "$TAR" | head -1 | cut -d/ -f1)
[ "$TOP" = "mp3d" ] || { log "예상과 다른 tar 구조: 최상위 '$TOP' (mp3d 여야 함)"; exit 1; }

if [ -e "$TARGET" ]; then
    BAK="$TARGET.incomplete_$(date +%Y%m%d_%H%M%S)"
    log "불완전한 $TARGET 를 $BAK 로 옮김 (지우지 않음)"
    mv "$TARGET" "$BAK" || exit 1
fi
TMP="$SCENES_DIR/.extract_tmp_$$"
mkdir -p "$TMP" || exit 1
log "풀기: $TAR -> $TMP (약 16 GB, 수 분 소요)"
tar -xzf "$TAR" -C "$TMP" || { log "tar 실패"; rm -rf "$TMP"; exit 1; }
mv "$TMP/mp3d" "$TARGET" && rmdir "$TMP" || exit 1

log "재검증"
if verify; then
    log "OK — $TARGET 준비 완료"
    exit 0
fi
log "FAIL — 풀었는데도 scene 이 빠져 있다"
exit 1
