"""nav3d 씬 표와 층별 navmesh 점 뽑기 (01 · 03 · 04 · 04d 가 같이 쓴다).

`01_build_navmesh.py` 에서 옮겼다 (코드 그대로). 01 은 이 이름들을 다시 import 해서 그대로 노출한다.
"""

import numpy as np

import nav3d_utils as nu


SCENES = {
    # noeun_station_mid.ply 는 이름과 달리 **역 전체** 충돌 mesh 다 — usdz 안 mesh.ply 와 꼭짓점·면이 완전히 같다
    # (2026-10-13 확인. usdz 의 mesh.usd 에는 xform 이 없어 world = ply 좌표).
    'noeun_station': {'mesh': 'data/scene_data/noeun_station/noeun_station_mid.ply',
                      'compare_2d': 'scripts/dataset_converters/gs_vlnpe/apply_real/esdf/noeun_station_mid.npz',
                      # 3DGS 가 못 살린 투명 유리 — mesh 에 벽 상자를 더한다. 사용자 확인(2026-10-13): 대합실 남쪽 y=-15.5
                      # 칸막이는 유리, 문은 x 0.8~1.95 (틀 하단 1.9 m). 기둥 x -4.25..0.75 사이 판넬은 바닥~0.9 m 가 비어 있었다.
                      'blockers': [{'name': 'glass_B1_south', 'min': [-4.3, -15.55, -0.30], 'max': [0.8, -15.45, 0.95]}]},
}


def sample_storey_points(pf, levels, lv2st, per_bucket=40, tol=0.3, tries=300000, seed=0):
    """무작위 navmesh 점을 (층, 섬) 칸에 모은다 — 작은 층(출구 계단 위 16 m2 등)도 같은 섬이면 뽑히게."""
    pf.seed(seed)
    Lz = np.array([L['z'] for L in levels])
    pts = {}
    for _ in range(tries):
        q = np.asarray(pf.get_random_navigable_point())
        if not np.isfinite(q).all():
            continue
        p = nu.hab_to_ours(q)
        k = int(np.argmin(np.abs(Lz - p[2])))
        if abs(Lz[k] - p[2]) > tol:
            continue
        b = (int(lv2st[k]), int(pf.get_island(q)))
        if len(pts.setdefault(b, [])) < per_bucket:
            pts[b].append(q)
    return pts
