"""noeun 62 · 라벨 — 노은역 nav3d 데이터(rbq v6 · hunav_v1, train · val_seen)에 b1 · '후' instruction 을 붙인다.

생성 코드는 그대로 쓴다: e2e.generate (b1) / common/label_paths.joint_results ('후', joint3-v1).
nav3d 라벨(n3sg-b1 · n3sg-post)과 다른 점은 노은역이라 바꿀 수밖에 없는 두 가지뿐이다 (2026-10-28 사용자 확인):
  frames.rig=60cm_15deg  (렌더된 rig 가 이것 하나)
  route.rooms=noeun-v1   (.house 가 없어 61_build_zones.py 의 구역으로; rooms.room_sequence 를 noeun_rooms 로 바꿔 끼운다)

입력 item: raw_data/r2r/<split>/<split>.json.gz 의 episode (habitat 좌표) — reference_path 는 RDP(--simplify_m) 로 꺾인 점만 남긴다
(common/label_paths.py 의 nav3d 와 같음). 이미지 · action · pose 는 vln_ce_noeun_nav3d_rbq_<ds>/traj_data/<r2r|r2r_val_seen>.
경로 하나에 형제 --n_sib 개: episode_id = 데이터셋·split 오프셋 + 경로번호 x 10 + 형제번호, trajectory_id = 오프셋 + 경로번호.
vln_ce 와 vln_pe 는 경로가 같아서 라벨은 한 번만 만든다 (63_build_labeled.py 가 두 데이터셋에 붙인다).

출력: <out_root>/<label name>/<ds>_<split>/meta/ep_<id>.json (생성 기록) + <ds>_<split>.json.gz (R2R 형식 + instruction)

사용 (저장소 루트, habitat 환경 python):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/62_label.py --methods b1,post --managed
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/62_label.py --methods b1 --limit 2 --base_url http://127.0.0.1:8100/v1
"""

import argparse
import copy
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
sys.path.insert(0, str(HERE))
import label_paths as LP  # noqa: E402  (label/ · utils/ 경로도 여기서 잡힌다)
import config as LC  # noqa: E402
import e2e  # noqa: E402
import noeun_rooms  # noqa: E402
import rooms  # noqa: E402
from route_words import CODE_VERSION  # noqa: E402
from sources import episode_track  # noqa: E402

METHODS = {  # 이름 -> annotate --set 목록 (nav3d n3sg-b1 · n3sg-post 와 같은 설정)
    'b1': [],
    'post': ['route.facts=memo-v1', 'style.knobs=v2', 'delivery=joint3-v1', 'style.writer=gepa-v3-joint-gemma'],
}
NOEUN_SETS = ['frames.rig=60cm_15deg', 'route.rooms=noeun-v1']
DS_OFF = {'v6': 0, 'hunav_v1': 100000}
SPLIT_DIR = {'train': 'r2r', 'val_seen': 'r2r_val_seen'}
SPLIT_OFF = {'train': 0, 'val_seen': 50000}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', default='b1,post', help='b1 | post (쉼표)')
    ap.add_argument('--datasets', default='v6,hunav_v1')
    ap.add_argument('--splits', default='train,val_seen')
    ap.add_argument('--data_root', default='/home/irteam/data-vol1/nav3d/noeun')
    ap.add_argument('--out_root', default='/home/irteam/data-vol1/nav3d/noeun/instruction_labels')
    ap.add_argument('--model', default='gemma31b-bf16', help='configs/model/<name>.yaml')
    ap.add_argument('--n_sib', type=int, default=3, help='경로 하나당 문장 수 (vln_pe tasks 3 개)')
    ap.add_argument('--simplify_m', type=float, default=0.3, help='RDP 허용 오차 (label_paths 와 같음)')
    ap.add_argument('--limit', type=int, default=0, help='데이터셋·split 당 앞에서 경로 N 개 (0 = 전부)')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--managed', action='store_true', help='vLLM 서버를 띄우고 끝나면 내린다 (model yaml 의 managed)')
    ap.add_argument('--base_url', default='', help='이미 떠 있는 서버 (빈값 = model yaml 의 base_url)')
    ap.add_argument('--gpus', default='', help='--managed 일 때 GPU (빈값 = model yaml)')
    return ap.parse_args()


def items(a, ds: str, split: str) -> list:
    base = Path(a.data_root) / f'vln_ce_noeun_nav3d_rbq_{ds}'
    eps = json.load(gzip.open(base / f'raw_data/r2r/{split}/{split}.json.gz'))['episodes']
    root = base / 'traj_data' / SPLIT_DIR[split]
    off = DS_OFF[ds] + SPLIT_OFF[split]
    assert len(eps) * 10 < 50000 and a.n_sib <= 10
    out = []
    for i, e in enumerate(eps[:a.limit] if a.limit else eps):
        assert int(e['episode_id']) == i, f'{ds}/{split}: raw 순서와 traj_data episode_index 가 다르다 ({e["episode_id"]} != {i})'
        ref = LP.rdp(np.asarray(e['reference_path'], float), a.simplify_m)
        for k in range(a.n_sib):
            ep = {'episode_id': off + i * 10 + k, 'trajectory_id': off + i, 'scene_id': e['scene_id'],
                  'start_position': e['start_position'], 'start_rotation': e['start_rotation'], 'goals': e['goals'],
                  'reference_path': ref.tolist(), 'info': e['info']}
            out.append({'ep': ep, 'scene': 'noeunstation', 'root': root, 'episode_index': i, 'split': f'noeun_{ds}_{split}', 'sib_idx': k,
                        'src': {'source': 'noeun', 'dataset': ds, 'split': split, 'raw_episode_id': i, 'key': e.get('key'), 'kind': e.get('kind')}})
    return out


def run(a, client, method: str, ds: str, split: str):
    cfg = LC.load('b1', METHODS[method] + [f'model={a.model}'] + NOEUN_SETS)
    name, ghash = cfg['name'], LC.gen_hash(cfg, CODE_VERSION)
    its = items(a, ds, split)
    out_dir = Path(a.out_root) / name / f'{ds}_{split}'
    md = out_dir / 'meta'
    md.mkdir(parents=True, exist_ok=True)
    done = set()
    for it in its:
        mp = md / f'ep_{it["ep"]["episode_id"]:06d}.json'
        if mp.is_file():
            assert json.loads(mp.read_text())['gen_hash'] == ghash, f'{mp}: gen_hash 가 다르다 — 설정이 바뀌었다'
            done.add(it['ep']['episode_id'])
    todo = [it for it in its if it['ep']['episode_id'] not in done]
    print(f'[62 {method} {ds}/{split}] {name} gen_hash={ghash} episodes={len(its)} todo={len(todo)}', flush=True)
    t0 = time.time()

    def record(it, r):
        m = {'episode_id': it['ep']['episode_id'], 'scene': it['scene'], 'episode_index': it['episode_index'], 'sib_idx': it['sib_idx'],
             'gen_hash': ghash, 'name': name, 'src': it['src'], 'reference_path_n': len(it['ep']['reference_path']), 'generated': r}
        (md / f'ep_{it["ep"]["episode_id"]:06d}.json').write_text(json.dumps(m, ensure_ascii=False))

    if cfg['delivery']['type'] == 'joint':   # '후': 경로마다 형제 전부를 한 번에 (하나라도 없으면 그 경로 전부 다시)
        need = {it['ep']['trajectory_id'] for it in todo}
        groups = {}
        for it in its:
            if it['ep']['trajectory_id'] in need:
                groups.setdefault(it['ep']['trajectory_id'], []).append(it)

        def work(sib):
            res = LP.joint_results(client, cfg, sib)
            for it in sib:
                record(it, res[it['ep']['episode_id']])
            return [res[it['ep']['episode_id']] for it in sib]
        jobs = list(groups.values())
    elif cfg['delivery']['type'] == 'image':  # b1: 문장마다 한 번 (형제는 episode_id 로 seed 가 다르다)
        def work(it):
            r = e2e.generate(client, cfg, it['ep'], episode_track(it, cfg['frames']['rigs'][0]))
            record(it, r)
            return [r]
        jobs = todo
    else:
        assert False, f'unreachable delivery.type={cfg["delivery"]["type"]!r}'
    n_err = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for k, rs in enumerate(ex.map(work, jobs), 1):
            n_err += sum(r['error'] is not None for r in rs)
            if k % 20 == 0 or k == len(jobs):
                print(f'  {k}/{len(jobs)} err={n_err} {time.time() - t0:.0f}s | {rs[0]["text"][:120]}', flush=True)
    eps = []
    for it in its:
        m = json.loads((md / f'ep_{it["ep"]["episode_id"]:06d}.json').read_text())
        e = copy.deepcopy(it['ep'])
        e['instruction'] = {'instruction_text': m['generated']['text'].strip() + ' ', 'instruction_tokens': []}
        e['sib_idx'], e['raw_episode_id'] = it['sib_idx'], it['episode_index']
        eps.append(e)
    dst = out_dir / f'{ds}_{split}.json.gz'
    with gzip.open(dst, 'wt') as f:
        json.dump({'episodes': eps, '_generation_meta': {'name': name, 'gen_hash': ghash, 'method': method, 'dataset': ds, 'split': split,
                                                         'n_sib': a.n_sib, 'simplify_m': a.simplify_m, 'n': len(eps)}}, f)
    print(f'[62 {method} {ds}/{split}] -> {dst} ({time.time() - t0:.0f}s, 오류 {n_err})', flush=True)
    return n_err


def main():
    a = parse()
    rooms.room_sequence = noeun_rooms.room_sequence   # route.rooms=noeun-v1: .house 대신 노은역 구역 (rooms_line 이 이 이름으로 부른다)
    from openai import OpenAI
    import backend
    mcfg = LC._yaml(LC.CONF / 'model' / f'{a.model}.yaml')
    ctx = backend.managed_server(mcfg, Path(a.out_root) / 'vllm.log', a.gpus or None) if a.managed else nullcontext(a.base_url or mcfg['base_url'])
    with ctx as base_url:
        client = OpenAI(base_url=base_url, api_key=mcfg['api_key'], timeout=600)
        errs = {}
        for method in a.methods.split(','):
            for ds in a.datasets.split(','):
                for split in a.splits.split(','):
                    errs[(method, ds, split)] = run(a, client, method, ds, split)
    print('[62] 오류 수', {'/'.join(k): v for k, v in errs.items()})


if __name__ == '__main__':
    main()
