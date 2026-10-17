"""경로 바닥 z 를 navmesh 표면에서 **장면 mesh 바닥**으로 바꾼다 (04b · 06v 공용).

왜: navmesh(Recast)는 mesh 바닥에서 장소마다 −5 ~ +20 cm 떠 있다(06v 실측, MP3D 4 씬). vln_pe GT 의 H1 카메라는
mesh 바닥 위 높이가 모든 에피소드에서 1.656 ± 0.004 m 로 일정했다 = 로봇은 mesh 를 딛는다. 노은역 06b R4 도 같은 원인.

규칙
- 바닥 점: 경로 점(navmesh z + `probe_up`)에서 아래로 쏜 광선이 처음 닿는 mesh 높이. `max_drop` 안에 안 닿으면 버린다.
- 계단 점: navmesh 경사를 그대로 두고, 앞·뒤 바닥 점에서 잰 (mesh − navmesh) 차이를 경로 순서로 선형 보간해 더한다
  (계단 위 광선은 디딤판 높이를 톱니처럼 집어 카메라가 계단마다 튄다).
- 못 잰 바닥 점도 같은 보간으로 채운다.
"""

import numpy as np


def mesh_floor_z(traj: np.ndarray, stairs: np.ndarray, mesh, probe_up: float = 0.5, max_drop: float = 0.8) -> dict:
    """traj (N,3) navmesh z · stairs (N,) bool · mesh (trimesh, 같은 z-up 좌표) -> {'z', 'offset', 'measured'}."""
    traj = np.asarray(traj, dtype=np.float64)
    stairs = np.asarray(stairs, dtype=bool)
    org = traj.copy()
    org[:, 2] += probe_up
    loc, idx, _ = mesh.ray.intersects_location(org, np.tile([0.0, 0.0, -1.0], (len(traj), 1)), multiple_hits=False)
    zm = np.full(len(traj), np.nan)
    zm[idx] = loc[:, 2]
    ok = np.isfinite(zm) & (org[:, 2] - zm <= max_drop) & ~stairs
    off = np.full(len(traj), np.nan)
    off[ok] = zm[ok] - traj[ok, 2]
    if not ok.any():
        return {'z': traj[:, 2].copy(), 'offset': np.zeros(len(traj)), 'measured': ok}
    i = np.arange(len(traj))
    off = np.interp(i, i[ok], off[ok])
    return {'z': traj[:, 2] + off, 'offset': off, 'measured': ok}


def stairs_mask_from_segments(n: int, segments: list) -> np.ndarray:
    """04 paths json 의 segments [{'kind': 'floor'|'stairs', 'i0', 'i1'}] -> (n,) bool."""
    m = np.zeros(n, dtype=bool)
    for s in segments:
        if s['kind'] == 'stairs':
            m[int(s['i0']):int(s['i1']) + 1] = True
        elif s['kind'] == 'floor':
            pass
        else:
            assert False, f"unreachable segment kind={s['kind']!r}"
    return m
