"""이미 생성한 label 을 새 방법 이름으로 복사한다 (예: b0_slight-omit_..._rooms-house-v2 -> b1). 다시 생성하지 않는다.

조건: 새 이름의 설정으로 계산한 gen_hash 가 원본 meta 의 gen_hash 와 **모두** 같아야 한다 (생성 내용이 같다는 뜻).
원본 폴더는 건드리지 않는다. 새 폴더가 이미 있으면 덮어쓰지 않고 멈춘다.
복사본의 meta `name` 과 json.gz `_generation_meta` 의 이름·방법·sets 를 새 값으로 바꾸고, `renamed_from` 에 원본 이름을 남긴다.

    /usr/bin/python scripts/dataset_converters/instruction_labeling/label/rename.py --from b0_slight-omit_fewshot-3_keyframe_deg-0_rooms-house-v2 --method b1
    /usr/bin/python scripts/dataset_converters/instruction_labeling/label/rename.py --from <원본 이름> --method b1 --set route.stop_cue=A-grounded-v1
"""
import argparse
import glob
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from annotate import REPO  # noqa: E402
from route_words import CODE_VERSION  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--from', dest='src', required=True, help='원본 label 이름 (instruction_labels/<이름>)')
ap.add_argument('--method', required=True)
ap.add_argument('--set', action='append', default=[])
ap.add_argument('--out_root', default='data/InternData-N1-v0.5-mini/instruction_labels')
a = ap.parse_args()

cfg = config.load(a.method, a.set)
h = config.gen_hash(cfg, CODE_VERSION)
root = REPO / a.out_root
src, dst = root / a.src, root / cfg['name']
assert src.is_dir(), f'원본이 없다: {src}'
assert not dst.exists(), f'이미 있다 (덮어쓰지 않음): {dst}'
tags = sorted(p.name for p in src.iterdir() if p.is_dir())
for tag in tags:  # 먼저 전부 검사하고, 하나라도 다르면 아무것도 쓰지 않는다
    hs = {json.load(open(f))['gen_hash'] for f in glob.glob(str(src / tag / 'meta' / '*.json'))}
    assert hs == {h}, f'{tag}: gen_hash 가 다르다 (원본 {sorted(hs)} / 새 이름 {h}) — 생성 내용이 달라 복사할 수 없다'
for tag in tags:
    split = tag.split('_dev')[0]
    (dst / tag / 'meta').mkdir(parents=True)
    for f in sorted(glob.glob(str(src / tag / 'meta' / '*.json'))):
        m = json.load(open(f))
        m['name'], m['renamed_from'] = cfg['name'], a.src
        (dst / tag / 'meta' / Path(f).name).write_text(json.dumps(m, ensure_ascii=False))
    gz = src / tag / f'{split}_{a.src}.json.gz'
    if gz.is_file():
        d = json.load(gzip.open(gz))
        gm = d.get('_generation_meta', {})
        gm.update({'name': cfg['name'], 'method': a.method, 'sets': list(a.set), 'renamed_from': a.src})
        d['_generation_meta'] = gm
        with gzip.open(dst / tag / f'{split}_{cfg["name"]}.json.gz', 'wt') as fo:
            json.dump(d, fo)
    print(f'{tag}: {a.src} -> {cfg["name"]} (gen_hash {h}, meta {len(glob.glob(str(dst / tag / "meta" / "*.json")))})')
