#!/bin/bash
# 다른 노드에서 data-vol2 의 relabel label 을 받아 평가용 데이터를 다시 만들고 검증한다.
#
#   bash scripts/dataset_converters/relabel_vlnce/sync_relabel_labels.sh            # 이미 빌드된 vln_ce_<L> 만
#   bash scripts/dataset_converters/relabel_vlnce/sync_relabel_labels.sh v278fix    # + 이 label 은 없어도 빌드
#
# 1. git pull (runner.py rc 수정 등)
# 2. data-vol2/relabel_half/labels/{split}/{split}_<L>.json.gz -> data/vln/mp3d/r2r/v1/{split}/ (덮어씀)
# 3. 로컬에 vln_ce_<L> 가 있거나 인자로 준 label 을 build_relabel_dataset.py 로 다시 빌드
#    (평가가 읽는 raw_data 사본에 남은 _generated_instruction 제거. 학습 문장은 그대로)
# 4. 검증: 복사본 == data-vol2 원본, raw_data episode key ⊆ GT key, habitat 로더로 로드,
#    raw_data 문장 == label 문장, runner.py 가 종료코드를 돌려주는지
# 학습/평가는 띄우지 않는다. 실패하면 0 이 아닌 코드로 끝난다.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # InternNav repo root

SRC=/home/irteam/data-vol2/relabel_half/labels
DST=data/vln/mp3d/r2r/v1
MINI=data/InternData-N1-v0.5-mini

echo "== 1. git pull"
git pull --ff-only origin feature/input_v0.1

echo "== 2. copy labels from $SRC"
for s in train val_seen val_unseen; do
    mkdir -p "$DST/$s"
    cp -f "$SRC/$s/${s}"_*.json.gz "$DST/$s/"
done

echo "== 3. rebuild vln_ce_<label>"
BUILT=()
for f in "$SRC"/train/train_*.json.gz; do
    L=$(basename "$f" .json.gz); L=${L#train_}
    if [ -d "$MINI/vln_ce_$L" ] || [[ " $* " == *" $L "* ]]; then
        python3 scripts/dataset_converters/relabel_vlnce/build_relabel_dataset.py --labels "$L" --emit data | grep -a '\[data\]'
        BUILT+=("$L")
    fi
done
for L in "$@"; do [ -f "$SRC/train/train_$L.json.gz" ] || { echo "FAIL: $L 은 $SRC 에 없다"; exit 1; }; done

echo "== 4. verify"
python3 - "$SRC" "$DST" "$MINI" "${BUILT[@]}" <<'EOF'
import filecmp, glob, gzip, json, os, sys
sys.path.insert(0, "scripts/dataset_converters/relabel_vlnce")
from build_relabel_dataset import normalize
from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1

src, dst, mini, built = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
splits = ("train", "val_seen", "val_unseen")
fails = []

n = 0
for f in glob.glob(f"{src}/*/*.json.gz"):
    s = os.path.basename(os.path.dirname(f))
    if not filecmp.cmp(f, f"{dst}/{s}/{os.path.basename(f)}", shallow=False):
        fails.append(f"copy differs: {s}/{os.path.basename(f)}")
    n += 1
print(f"  copies identical to data-vol2: {n} files")

gt_keys = {s: set(json.load(gzip.open(f"{mini}/vln_ce/raw_data/r2r/{s}/{s}.json.gz"))["episodes"][0]) for s in splits}
for L in built:
    for s in splits:
        raw = f"{mini}/vln_ce_{L}/raw_data/r2r/{s}/{s}.json.gz"
        text = gzip.open(raw, "rt").read()
        eps = json.loads(text)["episodes"]
        extra = set().union(*(e.keys() for e in eps)) - gt_keys[s]
        if extra:
            fails.append(f"{L}/{s}: non-GT episode keys {sorted(extra)}")
        label = {e["episode_id"]: normalize(e["instruction"]["instruction_text"])
                 for e in json.load(gzip.open(f"{dst}/{s}/{s}_{L}.json.gz"))["episodes"]}
        if {e["episode_id"]: e["instruction"]["instruction_text"] for e in eps} != label:
            fails.append(f"{L}/{s}: raw_data text != label text")
        try:
            ds = VLNDatasetV1.__new__(VLNDatasetV1)
            ds.episodes = []
            ds.from_json(text)
            assert len(ds.episodes) == len(eps)
        except Exception as e:
            fails.append(f"{L}/{s}: habitat load failed: {e!r}")
    print(f"  {L:24s} raw_data keys / text / habitat load checked")

if "sys.exit(train_and_eval(" not in open("scripts/train_eval/qwenvl_train/runner.py").read():
    fails.append("runner.py has no exit-code fix (git pull 확인)")

if fails:
    print("FAIL"); [print("  -", f) for f in fails]; sys.exit(1)
print(f"OK: {len(built)} labels rebuilt and verified")
EOF

if pgrep -f 'debug/repeat_train.sh' >/dev/null; then
    echo "NOTE: repeat_train.sh 가 돌고 있다 (무한 루프라 큐가 다음 줄로 넘어가지 않는다). 큐를 진행시키려면:"
    echo "  pkill -f 'debug/repeat_train.sh'; sleep 3; pkill -f 'checkpoints/debug/repeat_train'"
fi
