"""noeun 61 · 구역 — 라벨의 route.rooms 자리에 넣을 노은역 구역 (MP3D .house 대신).

노은역에는 .house 방 주석이 없다. 경로 프레임의 world 좌표(habitat, y-up)로 구역을 나눈다:
  층 = 높이 띠 (B2 승강장 · B1 대합실 · G 지상), 층 사이 = 계단 (평면 위치로 DBSCAN -> C1, C2, …),
  층 안 = 평면 위치 k-means (B2 4 · B1 6 · G 3, x 순서로 A, B, …).
이름은 2026-10-28 사용자 확인 (구역 확인 페이지 https://claude.ai/artifact/XcBq1oUwbaG671KHhKiLh7).

출력 (`--out`): {'names': {구역: 이름}, 'points': [[x, y, z, 구역번호], ...], 'zones': [구역, ...], 'source': {...}}
라벨 쪽(`62_label.py` -> `noeun_rooms.py`)은 경로 점마다 가장 가까운 구역 점의 구역을 쓴다.

사용 (저장소 루트):
    /usr/bin/python scripts/dataset_converters/nav3d/noeun/61_build_zones.py
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import DBSCAN, KMeans

HERE = Path(__file__).resolve().parent
FLOORS = [('B2', -99, -4.6), ('B1', -1.3, 0.4), ('G', 7.6, 99)]   # 경로 높이 히스토그램: B2 ≈ -5.5, B1 ≈ -0.5, G ≈ 8.0
SUB_K = {'B2': 4, 'B1': 6, 'G': 3}
NAMES = {  # 사용자 확인 (2026-10-28)
    'B2-A': 'platform', 'B2-B': 'platform', 'B2-C': 'platform', 'B2-D': 'end of the platform',
    'C1': 'stairs', 'C2': 'stairs', 'C3': 'exit stairs',
    'B1-A': 'exit passage', 'B1-B': 'concourse', 'B1-C': 'concourse', 'B1-D': 'concourse', 'B1-E': 'ticket gates', 'B1-F': 'paid area',
    'G-A': 'street', 'G-B': 'exit', 'G-C': 'exit',
}


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', default='/home/irteam/data-vol1/nav3d/noeun')
    ap.add_argument('--datasets', default='v6,hunav_v1', help='vln_ce_noeun_nav3d_rbq_<이름>')
    ap.add_argument('--rig', default='60cm_15deg')
    ap.add_argument('--every', type=int, default=2, help='구역 점 표본 간격 (프레임)')
    ap.add_argument('--out', default=str(HERE / 'zones' / 'noeun_zones_v1.json'))
    return ap.parse_args()


def frame_xyz(data_root: Path, ds: str, rig: str) -> np.ndarray:
    """모든 에피소드 프레임의 world (x, y, z). pose.<rig> 는 시작 기준 로컬(x 전방, y 왼쪽, z 위)."""
    out = []
    base = data_root / f'vln_ce_noeun_nav3d_rbq_{ds}'
    for split, tdir in [('train', 'r2r'), ('val_seen', 'r2r_val_seen')]:
        eps = json.load(gzip.open(base / f'raw_data/r2r/{split}/{split}.json.gz'))['episodes']
        for i, e in enumerate(eps):
            sx, sy, sz = e['start_position']
            q = e['start_rotation']
            phi = 2 * math.atan2(q[1], q[3])                        # habitat y 축 회전
            fwd = np.array([-math.sin(phi), -math.cos(phi)])        # (x, z)
            left = np.array([-math.cos(phi), math.sin(phi)])
            col = f'pose.{rig}'
            T = pq.read_table(base / f'traj_data/{tdir}/noeunstation/data/chunk-000/episode_{i:06d}.parquet', columns=[col]).to_pydict()[col]
            T = np.array([[np.array(r) for r in x] for x in T])
            t = T[:, :3, 3]
            xz = np.array([sx, sz]) + t[:, :1] * fwd + t[:, 1:2] * left
            out.append(np.stack([xz[:, 0], sy + t[:, 2] - t[0, 2], xz[:, 1]], 1))
    return np.concatenate(out)


def main():
    a = parse()
    R = np.concatenate([frame_xyz(Path(a.data_root), ds, a.rig) for ds in a.datasets.split(',')])
    zone = np.array([''] * len(R), dtype=object)
    fl = np.array([next((n for n, lo, hi in FLOORS if lo <= y < hi), '') for y in R[:, 1]], dtype=object)
    ci = np.where(fl == '')[0]
    lab = DBSCAN(eps=1.5, min_samples=20).fit_predict(R[ci][:, [0, 2]])
    cents = {l: (R[ci][lab == l][:, 1].min(), R[ci][lab == l][:, 0].mean()) for l in set(lab) - {-1}}
    cname = {l: f'C{n + 1}' for n, l in enumerate(sorted(cents, key=lambda l: cents[l]))}
    zone[ci] = [cname.get(l, '') for l in lab]                       # DBSCAN 잡음 점은 구역 없음
    for F, k in SUB_K.items():
        ii = np.where(fl == F)[0]
        km = KMeans(k, n_init=10, random_state=0).fit(R[ii][:, [0, 2]])
        rk = {int(c): n for n, c in enumerate(np.argsort(km.cluster_centers_[:, 0]))}
        zone[ii] = [f'{F}-{chr(65 + rk[int(l)])}' for l in km.labels_]
    zones = sorted(NAMES)
    assert set(zone) - {''} == set(zones), f'구역이 확인한 것과 다르다: {sorted(set(zone) - {""})}'
    keep = [i for i in range(0, len(R), a.every) if zone[i]]
    for z in zones:
        n = int((zone == z).sum())
        print(f'{z:5s} {NAMES[z]:20s} 프레임 {n:6d}  y {R[zone == z][:, 1].min():5.1f}~{R[zone == z][:, 1].max():5.1f}')
    out = {'names': NAMES, 'zones': zones, 'points': [[round(float(R[i, 0]), 2), round(float(R[i, 1]), 2), round(float(R[i, 2]), 2), zones.index(zone[i])] for i in keep],
           'source': {'datasets': a.datasets, 'rig': a.rig, 'floors': FLOORS, 'sub_k': SUB_K, 'every': a.every, 'frames': len(R)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')))
    print(f'-> {a.out} (점 {len(keep)})')


if __name__ == '__main__':
    main()
