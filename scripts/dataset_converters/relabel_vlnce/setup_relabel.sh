#!/bin/bash
# setup_relabel.sh — labeling 데이터 준비 → self_check → 데이터셋 생성 → 결과 검증을 한 번에 한다.
#
# Usage (repo 어디서 실행해도 된다):
#   bash scripts/dataset_converters/relabel_vlnce/setup_relabel.sh [options]
#
# Options:
#   --src <dir>      labeling 원본 위치 (기본 /home/irteam/data-vol2/vln, Drive 에서 받은 곳)
#   --link           data/vln 을 복사 대신 symlink 로 만든다 (기본: 복사)
#   --labels <l,..>  생성할 label (기본 all = 3개 split 에 모두 있는 것 전부)
#   --emit <e,..>    data / yaml / config (기본 data,yaml)
#   --check-only     1(준비)+2(self_check)+4(검증)만 하고 생성은 건너뛴다
#
# 단계마다 검증하고, 하나라도 실패하면 exit 1 로 멈춘다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILDER="scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py"
PY=/usr/bin/python3

SRC=/home/irteam/data-vol2/vln
MODE=copy
LABELS=all
EMIT=data,yaml
CHECK_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --src) SRC="$2"; shift 2 ;;
        --link) MODE=link; shift ;;
        --labels) LABELS="$2"; shift 2 ;;
        --emit) EMIT="$2"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        -h|--help) sed -n 2,15p "$0"; exit 0 ;;
        *) echo "알 수 없는 인자: $1" >&2; exit 2 ;;
    esac
done

cd "$REPO_ROOT"
DST="data/vln"
GT_ROOT="data/InternData-N1-v0.5-mini/vln_ce"
LABEL_DIR="$DST/mp3d/r2r/v1"
SPLITS="train val_seen val_unseen"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "[FAIL] $*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. data/vln 준비
log "===== 1. data/vln 준비 ($MODE: $SRC -> $DST) ====="
[ -d "$GT_ROOT/traj_data/r2r" ] || fail "GT 가 없다: $GT_ROOT"

if [ -e "$DST" ]; then
    log "$DST 가 이미 있다 ($(readlink -f "$DST")) -> 새로 만들지 않고 원본과 비교만 한다"
elif [ "$MODE" = link ]; then
    [ -d "$SRC" ] || fail "원본이 없다: $SRC"
    ln -s "$SRC" "$DST"
else
    [ -d "$SRC" ] || fail "원본이 없다: $SRC"
    cp -a "$SRC" "$DST"
fi

for s in $SPLITS; do
    [ -f "$LABEL_DIR/$s/$s.json.gz" ] || fail "$LABEL_DIR/$s/$s.json.gz 가 없다"
done
if [ -d "$SRC" ] && [ "$(readlink -f "$SRC")" != "$(readlink -f "$DST")" ]; then
    # 내용까지 비교한다 (81 MB 수준이라 몇 초면 끝난다)
    diff -rq "$SRC" "$DST" >/dev/null || fail "$DST 와 $SRC 의 내용이 다르다 (diff -rq $SRC $DST)"
    log "원본과 동일: $(find -L "$DST" -type f | wc -l) 파일"
fi
bad_gz=0
while IFS= read -r f; do gzip -t "$f" 2>/dev/null || { echo "  깨진 gz: $f"; bad_gz=$((bad_gz+1)); }; done \
    < <(find -L "$LABEL_DIR" -name '*.json.gz')
[ "$bad_gz" -eq 0 ] || fail "깨진 json.gz $bad_gz 개"
log "OK: json.gz 무결성 확인"

# ---------------------------------------------------------------- 2. self_check
log "===== 2. self_check ====="
CHECK_OUT=$($PY "$BUILDER" --self_check 2>&1) || { echo "$CHECK_OUT"; fail "self_check 실패"; }
echo "$CHECK_OUT"
if [ "$LABELS" = all ]; then
    LABEL_LIST=$(echo "$CHECK_OUT" | sed -n 's/.*사용 가능한 label [0-9]*개: //p')
else
    LABEL_LIST=$(echo "$LABELS" | tr ',' ' ')
fi
LABEL_LIST=$(echo "$LABEL_LIST" | tr ' ' '\n' | grep -vx 'gt' | grep -v '^$' | tr '\n' ' ' || true)
[ -n "$LABEL_LIST" ] || [ "$LABELS" = gt ] || fail "대상 label 이 없다"
log "OK: 대상 label: $LABEL_LIST"

# ---------------------------------------------------------------- 3. 생성
if [ "$CHECK_ONLY" -eq 0 ]; then
    log "===== 3. 생성 (--labels $LABELS --emit $EMIT) ====="
    $PY "$BUILDER" --labels "$LABELS" --emit "$EMIT" || fail "생성 실패"
else
    log "===== 3. 생성 건너뜀 (--check-only) ====="
fi

# ---------------------------------------------------------------- 4. 결과 검증
log "===== 4. 결과 검증 ====="
GT_SCENES=$(ls "$GT_ROOT/traj_data/r2r" | wc -l)
GT_EPISODES=$(cat "$GT_ROOT"/traj_data/r2r/*/meta/episodes.jsonl | wc -l)
NFAIL=0
bad() { echo "  [x] $1: $2"; NFAIL=$((NFAIL+1)); }

for L in $LABEL_LIST; do
    OUT="${GT_ROOT}_$L"
    before=$NFAIL
    if [[ ",$EMIT," == *",data,"* ]] || [ "$CHECK_ONLY" -eq 1 ]; then
        [ -d "$OUT" ] || { bad "$L" "$OUT 가 없다"; continue; }

        n=$(ls "$OUT/traj_data/r2r" | wc -l)
        [ "$n" -eq "$GT_SCENES" ] || bad "$L" "scene $n / GT $GT_SCENES"

        n=$(find "$OUT/traj_data" -xtype l | wc -l)
        [ "$n" -eq 0 ] || bad "$L" "깨진 symlink $n 개"

        # data/videos/info.json/episodes_stats.jsonl 은 GT symlink, episodes/tasks.jsonl 은 실파일이어야 한다
        n=0
        for sd in "$OUT"/traj_data/r2r/*/; do
            for x in data videos meta/info.json meta/episodes_stats.jsonl; do [ -L "$sd$x" ] || n=$((n+1)); done
            for x in meta/episodes.jsonl meta/tasks.jsonl; do [ -f "$sd$x" ] && [ ! -L "$sd$x" ] || n=$((n+1)); done
        done
        [ "$n" -eq 0 ] || bad "$L" "symlink/실파일 구조가 어긋난 항목 $n 개"

        n=$(cat "$OUT"/traj_data/r2r/*/meta/episodes.jsonl | wc -l)
        [ "$n" -eq "$GT_EPISODES" ] || bad "$L" "episode $n / GT $GT_EPISODES"

        # 평가 입력: split 파일이 habitat 에서 읽히는 형태인지 (episode 수 = 공식본, word_list 존재)
        msg=$($PY - "$GT_ROOT" "$OUT" <<'EOF'
import gzip, json, sys
gt, out = sys.argv[1:3]
errs = []
for s in ("train", "val_seen", "val_unseen"):
    try:
        new = json.load(gzip.open(f"{out}/raw_data/r2r/{s}/{s}.json.gz"))
    except Exception as e:
        errs.append(f"{s}: 읽기 실패 {e}"); continue
    ref = json.load(gzip.open(f"{gt}/raw_data/r2r/{s}/{s}.json.gz"))
    if len(new["episodes"]) != len(ref["episodes"]):
        errs.append(f"{s}: episode {len(new['episodes'])} / GT {len(ref['episodes'])}")
    if not (new.get("instruction_vocab") or {}).get("word_list"):
        errs.append(f"{s}: word_list 없음")
print("; ".join(errs))
EOF
)
        [ -z "$msg" ] || bad "$L" "raw_data $msg"
    fi
    if [[ ",$EMIT," == *",yaml,"* ]]; then
        for y in "vln_r2r_mini_5090_$L.yaml" "vln_r2r_mini_$L.yaml" \
                 "habitat_dual_system_mini_5090_cfg_$L.py" "habitat_dual_system_mini_h200_cfg_$L.py"; do
            [ -f "scripts/eval/configs/relabel/$y" ] || bad "$L" "평가 config 없음: $y"
        done
    fi
    if [ "$NFAIL" -eq "$before" ]; then echo "  [v] $L"; fi
done

[ "$NFAIL" -eq 0 ] || fail "검증 실패 $NFAIL 건"
log "ALL OK: label $(echo $LABEL_LIST | wc -w)개 / scene $GT_SCENES / episode $GT_EPISODES"
