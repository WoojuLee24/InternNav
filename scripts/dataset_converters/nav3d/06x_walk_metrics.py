"""06x — 06k walk 결과를 같은 자로 잰다 (설정 비교용).

## 지표 (ep 마다 → 모아서)
- success: 06k 판정.
- heading: 전진 프레임의 몸 방향 − 그 자리 경로 진행 방향 (가장 가까운 경로 점 앞뒤 0.5 m) |°|. 전체 p90 · 계단 구간 p90 · 계단 최대.
- start_dev: 출발 뒤 2 m 걸을 때까지 경로에서 벗어난 최대 거리 [m].
- path_dev: 전 구간 경로에서 벗어난 최대 거리 [m].

## 판정 (결과를 보기 전에 정함)
설정 고르기: ① 성공 수가 가장 많은 것 ② 같으면 계단 heading p90 이 가장 작은 것. 단 path_dev 최대 ≤ 0.5 m · start_dev 최대 ≤ 0.5 m 를 지켜야 한다.

사용 (habitat 컨테이너, 저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/06x_walk_metrics.py --walk_json a.json b.json ...
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walk_json', nargs='+', required=True)
    ap.add_argument('--paths_json', default='scripts/dataset_converters/nav3d/logs/04_export_paths/noeun_station/noeun_station_paths_meshfloor_straight_v5.json')
    ap.add_argument('--out', default='scripts/dataset_converters/nav3d/logs/06x_walk_metrics/summary.json')
    return ap.parse_args()


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def seg_dist(P, T):
    a, b = T[:-1], T[1:]
    ab = b - a
    t = np.clip(((P[:, None] - a[None]) * ab[None]).sum(2) / np.maximum((ab ** 2).sum(1), 1e-12)[None], 0, 1)
    return np.linalg.norm(P[:, None] - (a[None] + t[..., None] * ab[None]), axis=2).min(1)


def one(r, e):
    T = np.asarray(e['trajectory'])
    st = np.zeros(len(T), bool)
    for s in e['segments']:
        if s['kind'] == 'stairs':
            st[int(s['i0']):int(s['i1']) + 1] = True
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(T[:, :2], axis=0), axis=1))]
    P = np.array([f['pos'][:2] for f in r['frames']])
    prog = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    hd, hs = [], []
    for f, p in zip(r['frames'][1:], P[1:]):
        if f['action'] != 1:
            continue
        i = int(np.argmin(np.linalg.norm(T[:, :2] - p, axis=1)))
        i0, i1 = np.searchsorted(L, L[i] - 0.5), min(len(T) - 1, np.searchsorted(L, L[i] + 0.5))
        d = T[i1, :2] - T[i0, :2]
        if np.linalg.norm(d) < 1e-3:
            continue
        err = abs(math.degrees((yaw_of(f['quat_wxyz']) - math.atan2(d[1], d[0]) + math.pi) % (2 * math.pi) - math.pi))
        (hs if st[i] else hd).append(err)
    return {'success': bool(r['success']), 'why': r['why'], 'head': hd + hs, 'head_st': hs,
            'start_dev': float(seg_dist(P[prog < 2.0], T[:, :2]).max()), 'path_dev': float(seg_dist(P, T[:, :2]).max())}


def main():
    args = parse()
    E = {str(e['episode_id']): e for e in json.loads(Path(args.paths_json).read_text())['episodes']}
    R = {}
    for wj in args.walk_json:
        rows = [one(r, E[r['key'].split(':')[1]]) for r in json.loads(Path(wj).read_text())['results']]
        h = np.concatenate([x['head'] for x in rows])
        hs = np.concatenate([x['head_st'] for x in rows]) if any(len(x['head_st']) for x in rows) else np.zeros(1)
        R[Path(wj).stem] = {'n': len(rows), 'success': sum(x['success'] for x in rows),
                            'fails': [x['why'] for x in rows if not x['success']],
                            'heading_p90': float(np.percentile(h, 90)), 'stairs_heading_p90': float(np.percentile(hs, 90)),
                            'stairs_heading_max': float(hs.max()), 'start_dev_max': max(x['start_dev'] for x in rows),
                            'path_dev_max': max(x['path_dev'] for x in rows)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, indent=1))
    for k, v in R.items():
        print(f"{k:28s} 성공 {v['success']}/{v['n']} · heading p90 {v['heading_p90']:.1f}° · 계단 p90 {v['stairs_heading_p90']:.1f}° (최대 {v['stairs_heading_max']:.0f}°)"
              f" · start_dev {v['start_dev_max']:.2f} · path_dev {v['path_dev_max']:.2f} m · 실패 {v['fails']}")
    print('->', args.out)


if __name__ == '__main__':
    main()
