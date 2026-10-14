"""설정 '구조'만 바뀌어 gen_hash 가 달라졌을 때(VLM 요청 내용은 같음), 기존 meta 의 gen_hash 를 현재 값으로 갱신한다.

언제 쓰나: config.py / 자원 yaml 의 배치를 바꿔 hash 계산 입력이 달라졌지만, 실제 VLM 요청(이미지·프롬프트)이
글자 단위로 같다고 확인한 경우에만. 요청이 달라졌으면 쓰지 말고 다시 생성한다.
이전 hash 는 meta 의 `hash_history` 에 남긴다. 실행 중인 job 의 폴더에는 쓰지 않는다.

    /usr/bin/python scripts/dataset_converters/instruction_labeling/label/rehash.py --method b0 --set route.slight=omit --tag val_unseen
"""
import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from annotate import REPO  # noqa: E402
from route_words import CODE_VERSION  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--method', required=True)
ap.add_argument('--set', action='append', default=[])
ap.add_argument('--tag', required=True, help='split 폴더 이름: val_unseen | train | val_unseen_dev ...')
ap.add_argument('--out_root', default='data/InternData-N1-v0.5-mini/instruction_labels')
a = ap.parse_args()
cfg = config.load(a.method, a.set)
h = config.gen_hash(cfg, CODE_VERSION)
n = same = 0
for f in glob.glob(str(REPO / a.out_root / cfg['name'] / a.tag / 'meta' / '*.json')):
    m = json.load(open(f))
    if m['gen_hash'] == h:
        same += 1
        continue
    m.setdefault('hash_history', []).append(m['gen_hash'])
    m['gen_hash'] = h
    json.dump(m, open(f, 'w'), ensure_ascii=False)
    n += 1
print(f'{cfg["name"]}/{a.tag}: hash {h} 갱신 {n}, 이미 같음 {same}')
