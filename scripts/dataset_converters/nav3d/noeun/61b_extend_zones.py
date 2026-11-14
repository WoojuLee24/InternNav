"""noeun 61b · 구역 v2 — 61 의 noeun_zones_v1 에 랜드마크 경로가 새로 간 곳을 더한다 (v1 파일은 그대로).

v1 은 예전 경로(v6 · hunav_v1) 프레임으로만 만들어서, 랜드마크 경로(walk_t12)가 가는 곳 중 구역 점이 3 m(rooms max_dist_m) 밖인 데가 있다:
  · 다른 출구 앞 지상 바깥 (높이 6 ~ 7.6 m) -> 새 구역 **G-D = "street"** (2026-11-15 사용자 결정)
  · B1 · B2 층 가장자리 (예전 경로가 안 간 끝) -> 같은 층 띠 안에서 가장 가까운 v1 구역 점의 구역으로 늘린다 (새 이름 없음)
  · 그 밖의 높이 -> 멈춤 (assert) — 생기면 사람이 보고 정한다
v1 점은 하나도 바꾸지 않는다. 새 점은 --voxel_m 격자로 줄여 붙인다.

사용 (저장소 루트, habitat 환경 python):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/61b_extend_zones.py --walk_json logs/noeun/paths_landmark_ext/walk_t12/walk_all.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
OUTDOOR = ('G-D', 'street', 6.0, 7.6)          # 새 구역 · 이름 · 높이 띠 (habitat y)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--walk_json', required=True, help='걸은 결과 (results[*].frames[*].pos, nav3d z-up trunk)')
    ap.add_argument('--v1', default=str(HERE / 'zones' / 'noeun_zones_v1.json'))
    ap.add_argument('--out', default=str(HERE / 'zones' / 'noeun_zones_v2.json'))
    ap.add_argument('--base_height', type=float, default=0.55, help='trunk − 이 값 = 바닥 (51b · 62b 와 같음)')
    ap.add_argument('--max_dist_m', type=float, default=3.0, help='route.yaml rooms max_dist_m 과 같은 값 — 이보다 멀면 새 점')
    ap.add_argument('--voxel_m', type=float, default=0.5)
    return ap.parse_args()


def main():
    a = parse()
    Z = json.loads(Path(a.v1).read_text())
    P = np.asarray(Z['points'], float)
    zones, names = list(Z['zones']), dict(Z['names'])
    floors = {f: (lo, hi) for f, lo, hi in Z['source']['floors']}
    W = json.loads(Path(a.walk_json).read_text())
    X = np.concatenate([np.array([f['pos'] for f in r['frames']], float) for r in W['results'] if r['success']])
    X[:, 2] -= a.base_height
    H = np.stack([X[:, 0], X[:, 2], -X[:, 1]], 1)                     # nav3d z-up -> habitat y-up (nav3d_utils.ours_to_hab)
    d, _ = cKDTree(P[:, :3]).query(H)
    new = H[d > a.max_dist_m]
    new = np.unique(np.round(new / a.voxel_m) * a.voxel_m, axis=0)
    zid, lo, hi = OUTDOOR[0], OUTDOOR[2], OUTDOOR[3]
    zones.append(zid)
    names[zid] = OUTDOOR[1]
    add, count = [], {}
    for fl, (flo, fhi) in floors.items():                          # 층 가장자리: 같은 층 띠의 가장 가까운 v1 점 구역
        m_new = (new[:, 1] > flo) & (new[:, 1] < fhi)
        m_old = (P[:, 1] > flo) & (P[:, 1] < fhi)
        if m_new.any():
            _, j = cKDTree(P[m_old, :3]).query(new[m_new])
            lab = P[m_old, 3][j].astype(int)
            add += [[*map(float, p), int(z)] for p, z in zip(new[m_new], lab)]
            count[fl] = int(m_new.sum())
    m_out = (new[:, 1] > lo) & (new[:, 1] < hi)
    add += [[*map(float, p), zones.index(zid)] for p in new[m_out]]
    count[zid] = int(m_out.sum())
    rest = len(new) - sum(count.values())
    assert rest == 0, f'층 띠 · 바깥 띠 어디에도 안 드는 새 점 {rest} 개 — 높이를 보고 정해야 한다'
    out = {'names': names, 'zones': zones, 'points': Z['points'] + [[round(p[0], 2), round(p[1], 2), round(p[2], 2), p[3]] for p in add],
           'source': dict(Z['source'], v2={'from': 'noeun_zones_v1.json', 'walk_json': a.walk_json, 'max_dist_m': a.max_dist_m,
                                           'voxel_m': a.voxel_m, 'outdoor': list(OUTDOOR), 'added': count})}
    Path(a.out).write_text(json.dumps(out))
    print(f'[61b] v1 점 {len(P)} + 새 점 {len(add)} {count} -> {a.out}')


if __name__ == '__main__':
    main()
