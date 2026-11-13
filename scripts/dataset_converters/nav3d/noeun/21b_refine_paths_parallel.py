"""noeun 21b · 층간 경로 다듬기를 **여러 프로세스로 나눠** 돌리고 하나로 합친다.

왜: `common/refine_paths.py` 는 병렬 인자가 없고 경로 하나에 5~25 초 걸린다 (200 쌍 = 48 분).
    학습 1200 + 평가 80 을 필터 통과율 고려해 뽑으려면 층간만 1800 쌍 이상이 필요하다 -> 그대로면 7 시간+.
    이 서버는 CPU 128 개이고 refine 은 GPU 를 안 쓰므로, **seed 를 달리한 샤드**로 나눠 동시에 돌린다.

합치기: 샤드마다 `<out>/shard_<k>/scenes/<scene>.json` + `_traj.npz` 가 나온다.
       id 를 샤드 오프셋만큼 밀어 `rows` 를 잇고 npz 키(`<id>_raw` 등)도 같이 민다.
       seed 가 다르면 뽑는 시작·끝 쌍이 달라 서로 다른 경로가 된다 (같은 seed 면 같은 것이 나온다).

    python3 scripts/dataset_converters/nav3d/noeun/21b_refine_paths_parallel.py --total 1800 --shards 24
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--total', type=int, required=True, help='만들 층간 경로 총 쌍 수')
    ap.add_argument('--shards', type=int, default=24, help='동시에 돌릴 프로세스 수 (CPU 여유를 남길 것)')
    ap.add_argument('--scene', default='noeun_station')
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/common/refine_paths')
    ap.add_argument('--navmesh_dir', default='scripts/dataset_converters/nav3d/logs/noeun/12_build_navmesh')
    ap.add_argument('--mesh_root', default='data/scene_data')
    ap.add_argument('--seed0', type=int, default=1000, help='샤드 seed 시작값 (기존 200 쌍은 seed 0 이므로 겹치지 않게)')
    ap.add_argument('--python', default='python3')
    ap.add_argument('--merge_only', action='store_true')
    return ap.parse_args()


def main():
    a = parse()
    out = Path(a.out)
    per = (a.total + a.shards - 1) // a.shards
    shard_dirs = [out / f'shard_{k}' for k in range(a.shards)]

    if not a.merge_only:
        procs = []
        for k, sd in enumerate(shard_dirs):
            sd.mkdir(parents=True, exist_ok=True)
            cmd = [a.python, 'scripts/dataset_converters/nav3d/noeun/21_refine_paths.py',
                   '--n_pairs', str(per), '--seed', str(a.seed0 + k),
                   '--navmesh_dir', a.navmesh_dir, '--mesh_root', a.mesh_root,
                   '--out_dir', str(sd)]
            log = open(sd / 'run.log', 'w')
            procs.append((k, subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT), log))
            print(f'[21b] shard {k}: {per} 쌍 seed {a.seed0 + k}', flush=True)
        t0 = time.time()
        for k, p, log in procs:
            rc = p.wait(); log.close()
            print(f'[21b] shard {k} 끝 rc={rc} ({time.time() - t0:.0f}s)', flush=True)

    # 합치기
    rows, arrays, off = [], {}, 0
    for k, sd in enumerate(shard_dirs):
        jf = sd / 'scenes' / f'{a.scene}.json'
        zf = sd / 'scenes' / f'{a.scene}_traj.npz'
        if not jf.exists():
            print(f'[21b] shard {k} 결과 없음 — 건너뜀', flush=True); continue
        d = json.load(open(jf)); z = np.load(zf)
        n = 0
        for r in d['rows']:
            r = dict(r); old = r['id']; r['id'] = old + off; r['shard'] = k
            rows.append(r); n = max(n, old + 1)
        for key in z.files:
            i, suf = key.split('_', 1)
            arrays[f'{int(i) + off}_{suf}'] = z[key]
        off += n
        print(f'[21b] shard {k}: {len(d["rows"])} 행 (누적 {len(rows)})', flush=True)

    sc = out / 'scenes'; sc.mkdir(parents=True, exist_ok=True)
    meta = {'scene': a.scene, 'kind': 'noeun', 'merged_shards': a.shards, 'total': len(rows)}
    json.dump({'meta': meta, 'rows': rows}, open(sc / f'{a.scene}.json', 'w'))
    np.savez_compressed(sc / f'{a.scene}_traj.npz', **arrays)
    ok = sum(1 for r in rows if r.get('found'))
    print(f'[21b] 합침: {len(rows)} 행 (성공 {ok}) -> {sc}/{a.scene}.json')


if __name__ == '__main__':
    main()
